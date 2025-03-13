# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
from nnsight import list as nnsight_list
from nnsight.intervention import Envoy
from nnsight.modeling.language import LanguageModel
from nnsight.util import fetch_attr
from transformers import (
    AutoModel,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto import modeling_auto

from ...typing import LatentActivations
from .splitting_utils import get_layer_by_idx, sort_paths, validate_path, walk_modules
from .transformers_classes import (
    get_supported_hf_transformer_autoclasses,
    get_supported_hf_transformer_generation_autoclasses,
    get_supported_hf_transformer_generation_classes,
)


class InitializationError(ValueError):
    """Raised to signal a problem with model initialization."""


class ActivationSelectionStrategy(Enum):
    """Activation selection strategies for ModelWithSplitPoints.get_activations."""

    ALL = "all"
    FLATTEN = "flatten"

    @staticmethod
    def is_match(a: str | ActivationSelectionStrategy, b: ActivationSelectionStrategy) -> bool:
        return a == b.value if isinstance(a, str) else a == b


class ModelWithSplitPoints(LanguageModel):
    """Code: [:octicons-mark-github-24: model_wrapping/model_with_split_points.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/commons/model_wrapping/model_with_split_points.py)

    Generalized NNsight.LanguageModel wrapper around encoder-only, decoder-only and encoder-decoder language models.
    Handles splitting model at specified locations and activation extraction.

    Inputs can be in the form of:

        * One (`str`) or more (`list[str]`) prompts, including batched prompts (`list[list[str]]`).

        * One (`list[int] or torch.Tensor`) or more (`list[list[int]] or torch.Tensor`) tokenized prompts.

        * Direct model inputs: (`dic[str,Any]`)

    Attributes:
        model_autoclass (type): The [AutoClass](https://huggingface.co/docs/transformers/en/model_doc/auto#natural-language-processing)
            corresponding to the loaded model type.
        split_points (list[str]): Getter/setters for model paths corresponding to split points inside the loaded model.
            Automatically handle validation, sorting and resolving int paths to strings.
        repo_id (str): Either the model id in the HF Hub, or the path from which the model was loaded.
        generator (nnsight.Envoy | None): If the model is generative, a generator is provided to handle multi-step
            inference. None for encoder-only models.
        _model (transformers.PreTrainedModel): Huggingface transformers model wrapped by NNSight.
        _model_paths (list[str]): List of cached valid paths inside `_model`, used to validate `split_points`.
        _split_points (list[str]): List of split points, should be accessed with getter/setter.
    """

    _example_input = "hello"

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_points: str | int | list[str | int] | tuple[str | int],
        *args: tuple[Any],
        model_autoclass: str | type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize a ModelWithSplitPoints object.

        Args:
            model_or_repo_id (str | transformers.PreTrainedModel): One of:

                * A `str` corresponding to the ID of the model that should be loaded from the HF Hub.
                * A `str` corresponding to the local path of a folder containing a compatible checkpoint.
                * A preloaded `transformers.PreTrainedModel` object.
                If a string is provided, a model_autoclass should also be provided.
            split_points (str | Sequence[str] | int | Sequence[int]): One or more to split locations inside the model.
                Either the path is provided explicitly (`str`), or an `int` is used as shorthand for splitting at
                the n-th layer. Example: `split_points='cls.predictions.transform.LayerNorm'` correspond to a split
                after the LayerNorm layer in the MLM head (assuming a `BertForMaskedLM` model in input).
            model_autoclass (Type): Huggingface [AutoClass](https://huggingface.co/docs/transformers/en/model_doc/auto#natural-language-processing)
                corresponding to the desired type of model (e.g. `AutoModelForSequenceClassification`).

                :warning: `model_autoclass` **must be defined** if `model_or_repo_id` is `str`, since the the model class
                    cannot be known otherwise.
            config (PretrainedConfig): Custom configuration for the loaded model.
                If not specified, it will be instantiated with the default configuration for the model.
            tokenizer (PreTrainedTokenizer): Custom tokenizer for the loaded model.
                If not specified, it will be instantiated with the default tokenizer for the model.
        """
        if isinstance(model_or_repo_id, str):  # Repository ID
            if model_autoclass is None:
                raise InitializationError(
                    "Model autoclass not found.\n"
                    "The model class can be omitted if a pre-loaded model is passed to `model_or_repo_id` "
                    "param.\nIf an HF Hub ID is used, the corresponding autoclass must be specified in `model_autoclass`.\n"
                    "Example: ModelWithSplitPoints('bert-base-cased', model_autoclass=AutoModelForMaskedLM, ...)"
                )
            if isinstance(model_autoclass, str):
                supported_autoclasses = get_supported_hf_transformer_autoclasses()
                try:
                    self.model_autoclass: type[AutoModel] = getattr(modeling_auto, model_autoclass)
                except AttributeError:
                    raise InitializationError(
                        f"The specified class {model_autoclass} is not a valid autoclass.\n"
                        f"Supported autoclasses: {', '.join(supported_autoclasses)}"
                    ) from AttributeError
                if model_autoclass not in supported_autoclasses:
                    raise InitializationError(
                        f"The specified autoclass {model_autoclass} is not supported.\n"
                        f"Supported autoclasses: {', '.join(supported_autoclasses)}"
                    )
            else:
                self.model_autoclass: type[AutoModel] = model_autoclass

        # Handles model loading through LanguageModel._load
        super().__init__(
            model_or_repo_id,
            *args,
            config=config,
            tokenizer=tokenizer,  # type: ignore
            automodel=self.model_autoclass,
            **kwargs,
        )
        self._model_paths = list(walk_modules(self._model))
        self.split_points = split_points
        self._model: PreTrainedModel
        if self.repo_id is None:
            self.repo_id = self._model.config.name_or_path
        self.generator: Envoy | None
        if self._model.__class__.__name__ not in get_supported_hf_transformer_generation_classes():
            self.generator = None  # type: ignore

    @property
    def split_points(self) -> list[str]:
        return self._split_points

    @split_points.setter
    def split_points(self, split_points: str | int | list[str | int] | tuple[str | int]) -> None:
        """Split points are automatically validated and sorted upon setting"""
        pre_conversion_split_points = split_points if isinstance(split_points, list | tuple) else [split_points]
        post_conversion_split_points: list[str] = []
        for split in pre_conversion_split_points:
            # Handle conversion of layer idx to full path
            if isinstance(split, int):
                str_split = get_layer_by_idx(split, model_paths=self._model_paths)
            else:
                str_split = split
            post_conversion_split_points.append(str_split)

            # Validate whether the split exists in the model
            validate_path(self._model, str_split)

        # Sort split points to match execution order
        self._split_points: list[str] = sort_paths(post_conversion_split_points, model_paths=self._model_paths)

    def _generate(
        self,
        inputs: BatchEncoding,
        max_new_tokens=1,
        streamer: Any = None,
        **kwargs,
    ):
        if self.generator is None:
            gen_classes = get_supported_hf_transformer_generation_autoclasses()
            raise RuntimeError(
                f"model.generate was called but model class {self._model.__class__.__name__} does not support "
                "generation. Use regular forward passes for inference, or change model_autoclass in the initialization "
                f"to use a generative class. Supported classes: {', '.join(gen_classes)}."
            )
        super()._generate(inputs=inputs, max_new_tokens=max_new_tokens, streamer=streamer, **kwargs)

    def get_activations(
        self,
        inputs: str | list[str] | BatchEncoding,
        select_strategy: str | ActivationSelectionStrategy = ActivationSelectionStrategy.ALL,
        select_indices: int | list[int] | tuple[int] | None = None,
        **kwargs,
    ) -> dict[str, LatentActivations]:
        """Get intermediate activations for all model split points

        Args:
            inputs (str | list[str] | BatchEncoding): Inputs to the model forward pass before or after tokenization.
            select_strategy (str | ActivationSelectionStrategy): Selection strategy for activations.

                Options are:

                * `all`: All sequence activations are returned, keeping the original shape `(batch, seq_len, d_model)`.
                * `flatten`: Every token activation is treated as a separate element - `(batch x seq_len, d_model)`.

            select_indices (list[int] | tuple[int] | None): Specifies indices that should be selected from the
                activation sequence. Can be combined with `select_strategy` to obtain several behaviors. E.g.
                `select_strategy="all", select_indices=mask_idxs` can be used to extract only activations corresponding
                to [MASK] input ids with shape `(batch, len(mask_idxs), d_model)`, or
                `select_strategy="flatten", select_indices=0` can be used to extract activations for the `[CLS]` token
                only across all sequences, with shape `(batch, d_model)`. By default, all positions are selected.

        Returns:
            Dictionary having one key, value pair for each split point defined for the model. Keys correspond to split
                names in `self.split_points`, while values correspond to the extracted activations for the split point
                for the given `inputs`.
        """
        activations_dict = {}
        if not self.split_points:
            raise RuntimeError(
                "No split points are currently defined for the model. "
                "Please set split points before calling get_activations."
            )
        select_indices = [select_indices] if isinstance(select_indices, int) else select_indices

        # Compute activations
        with self.trace(inputs, **kwargs):
            activations = nnsight_list().save()  # type: ignore
            for idx, split_point in enumerate(self.split_points):
                curr_module: Envoy = fetch_attr(self, split_point)
                # Handle case in which module has .output attribute, and .nns_output gets overridden instead
                module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"
                activations.append(getattr(curr_module, module_out_name))
                # Early stopping at the last splitting layer
                if idx == len(self.split_points) - 1:
                    getattr(curr_module, module_out_name).stop()

        # TODO: Ideally we would not need to unpack tuples, check if necessary
        for idx, split in enumerate(self.split_points):
            act = activations.value[idx]
            activations_dict[split] = act[0] if isinstance(act, tuple) else act

        # Validate that activations have the expected type
        for layer, act in activations_dict.items():
            if not isinstance(act, torch.Tensor):
                raise RuntimeError(
                    f"Invalid output for layer '{layer}'. Expected torch.Tensor activation, got {type(act)}: {act}"
                )
            if len(activations_dict[layer].shape) != 3:
                raise RuntimeError(
                    f"Invalid output for layer '{layer}'. Expected a torch.Tensor activation with shape"
                    f"(batch_size, sequence_length, model_dim), got {activations_dict[layer].shape}"
                )

            # Apply selection rule
            if select_indices:
                activations_dict[layer] = activations_dict[layer][:, select_indices, :]
            if ActivationSelectionStrategy.is_match(select_strategy, ActivationSelectionStrategy.FLATTEN):
                activations_dict[layer] = activations_dict[layer].flatten(0, 1)
                if len(activations_dict[layer].shape) != 2:
                    raise RuntimeError(
                        f"Invalid output for layer '{layer}'. Expected a torch.Tensor activation with shape"
                        f"(batch_size x sequence_length, model_dim), got {activations[layer].shape}"
                    )
        return activations_dict
