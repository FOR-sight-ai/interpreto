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

from collections.abc import Callable
from enum import Enum
from math import ceil
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float
from nnsight.modeling.language import LanguageModel
from tqdm import tqdm
from transformers import AutoModel, T5ForConditionalGeneration
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import modeling_auto
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from interpreto.commons.granularity import Granularity, GranularityAggregationStrategy
from interpreto.model_wrapping.splitting_utils import get_layer_by_idx, sort_paths, validate_path, walk_modules
from interpreto.model_wrapping.transformers_classes import (
    get_supported_hf_transformer_autoclasses,
    get_supported_hf_transformer_generation_autoclasses,
    get_supported_hf_transformer_generation_classes,
)
from interpreto.typing import ConceptsActivations, LatentActivations


class InitializationError(ValueError):
    """Raised to signal a problem with model initialization."""


class ActivationGranularity(Enum):
    """Activation selection strategies for :meth:`ModelWithSplitPoints.get_activations`."""

    ALL = "all"  # removed for now as it conflicts with batching
    CLS_TOKEN = "cls_token"
    ALL_TOKENS = Granularity.ALL_TOKENS
    TOKEN = Granularity.TOKEN
    WORD = Granularity.WORD
    SENTENCE = Granularity.SENTENCE
    SAMPLE = "sample"


class ModelWithSplitPoints(LanguageModel):
    """Code: [:octicons-mark-github-24: model_wrapping/model_with_split_points.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/model_wrapping/model_with_split_points.py)

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

    Examples:
        Load the model from its repository id, split it at the first layer,
        and get the raw activations for the first layer.
        >>> from datasets import load_dataset
        >>> from interpreto import ModelWithSplitPoints
        >>> # load and split the model
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     "bert-base-uncased",
        ...     split_points="bert.encoder.layer.1.output",
        ...     model_autoclass=AutoModelForMaskedLM,
        ...     batch_size=64,
        ...     device_map="auto",
        ... )
        >>> # get activations
        >>> dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")["train"]["text"]
        >>> activations = model_with_split_points.get_activations(
        ...     dataset,
        ...     activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN,
        ... )

        Load the model then pass it the `ModelWithSplitPoint`, split it at the first layer,
        get the word activations for the first layer, skip special tokens, and aggregate tokens activations by mean into words.
        >>> from transformers import AutoModelCausalLM, AutoTokenizer
        >>> from datasets import load_dataset
        >>> from interpreto import ModelWithSplitPoints
        >>> # load the model
        >>> model = AutoModelCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> # wrap and split the model
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     split_points="transformer.h.1.mlp"],,
        ...     model_autoclass=AutoModelForMaskedLM,
        ...     batch_size=16,
        ...     device_map="auto",
        ... )
        >>> # get activations
        >>> dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")["train"]["text"]
        >>> activations = model_with_split_points.get_activations(
        ...     dataset,
        ...     activation_granularity=ModelWithSplitPoints.activation_granularities.WORD,
        ...     aggregation_strategy=ModelWithSplitPoints.aggregation_strategies.MEAN,
        ... )
    """

    _example_input = "hello"
    activation_granularities = ActivationGranularity
    aggregation_strategies = GranularityAggregationStrategy

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_points: str | int | list[str] | list[int] | tuple[str] | tuple[int],
        *args: tuple[Any],
        model_autoclass: str | type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        output_tuple_index: int | None = None,
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
            batch_size (int): Batch size for the model.
            device_map (torch.device | str | None): Device map for the model. Directly passed to the model.
            output_tuple_index (int | None): If the output at the split point is a tuple, this is the index of the hidden state.
                If `None`, an element with 3 dimensions is searched for. If not found, an error is raised. If several elements are found,
                an error is raised.
        """
        self.model_autoclass = model_autoclass
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
                    self.model_autoclass = getattr(modeling_auto, model_autoclass)
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
                self.model_autoclass = model_autoclass

        # Handles model loading through LanguageModel._load
        super().__init__(
            model_or_repo_id,
            *args,
            config=config,
            tokenizer=tokenizer,  # type: ignore
            automodel=self.model_autoclass,  # type: ignore
            device_map=device_map,
            **kwargs,
        )
        self._model_paths = list(walk_modules(self._model))
        self.split_points = split_points
        self._model: PreTrainedModel
        if self.repo_id is None:
            self.repo_id = self._model.config.name_or_path
        # self.generator: Envoy | None
        if self._model.__class__.__name__ not in get_supported_hf_transformer_generation_classes():
            self.generator = None  # type: ignore
        self.batch_size = batch_size

        if not isinstance(model_or_repo_id, str) and device_map is not None:
            if device_map == "auto":
                raise ValueError(
                    "'auto' device_map is only supported when loading a model from a repository id. "
                    "Please specify a device_map, e.g. 'cuda' or 'cpu'."
                )
            self.to(device_map)  # type: ignore

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set. When providing a model instance, the tokenizer must be set.")
        self.output_tuple_index = output_tuple_index

    @property
    def split_points(self) -> list[str]:
        return self._split_points

    @split_points.setter
    def split_points(self, split_points: str | int | list[str] | list[int] | tuple[str] | tuple[int]) -> None:
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

    @staticmethod
    def pad_and_concat(
        tensor_list: list[Float[torch.Tensor, "n_i l_i d"]], pad_side: str, pad_value: float
    ) -> Float[torch.Tensor, "sum(n_i) max_l d"]:
        """
        Concatenates a list of 3D tensors along dim=0 after padding their second dimension to the same length.

        Args:
            tensor_list (List[Tensor]): List of tensors with shape (n_i, l_i, d)
            pad_side (str): 'left' or 'right' — side on which to apply padding along dim=1
            pad_value (float): Value to use for padding

        Returns:
            Tensor: Tensor of shape (sum(n_i), max_l, d)
        """
        if pad_side not in ("left", "right"):
            raise ValueError("pad_side must be either 'left' or 'right'")

        max_l = max(t.shape[1] for t in tensor_list)
        padded = []

        for t in tensor_list:
            n, l, d = t.shape
            pad_len = max_l - l

            if pad_len == 0:
                padded_tensor = t
            else:
                if pad_side == "right":
                    pad = (0, 0, 0, pad_len)  # pad dim=1 on the right
                else:  # pad_side == 'left'
                    pad = (0, 0, pad_len, 0)  # pad dim=1 on the left
                padded_tensor = F.pad(t, pad, value=pad_value)

            padded.append(padded_tensor)

        return torch.cat(padded, dim=0)

    # @jaxtyped(typechecker=beartype)
    def _apply_selection_strategy(
        self,
        inputs: BatchEncoding | torch.Tensor,
        activations: Float[torch.Tensor, "n l d"],
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityGranularityAggregationStrategy | None,
    ) -> tuple[Float[torch.Tensor, "n l d"] | Float[torch.Tensor, "ng d"], list[list[list[int]]] | None]:
        """Apply selection strategy to activations.

        Args:
            inputs (BatchEncoding | torch.Tensor): Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.
            activations (InterventionProxy): Activations to apply selection strategy to.
            activation_granularity (ActivationGranularity): Selection strategy to apply. see :meth:`ModelWithSplitPoints.get_activations`.
            aggregation_strategy (GranularityGranularityAggregationStrategy | None): Aggregation strategy to apply. see :meth:`ModelWithSplitPoints.get_activations`.

        Returns:
            torch.Tensor: The aggregated activations. (n, l, d) or (ng, d)
            list[list[list[int]]]: The indices of the granularity level, might be None.
        """
        # Apply selection rule
        match activation_granularity:
            case ActivationGranularity.ALL:
                return activations, None
            case ActivationGranularity.CLS_TOKEN:
                if isinstance(inputs, torch.Tensor):
                    if inputs[0, 0] != self.tokenizer.cls_token_id:
                        raise ValueError(
                            "The first token of the input tensor is not the CLS token. "
                            "Please provide a tensor with the CLS token as the first token."
                            "This may happen if you asking for a ``CLS_TOKEN`` granularity while not doing classification."
                        )

                if isinstance(inputs, BatchEncoding):
                    if inputs["input_ids"][0, 0] != self.tokenizer.cls_token_id:  # type: ignore
                        raise ValueError(
                            "The first token of the input tensor is not the CLS token. "
                            "Please provide a tensor with the CLS token as the first token."
                            "This may happen if you asking for a ``CLS_TOKEN`` granularity while not doing classification."
                        )
                return activations[:, 0, :], None
            case ActivationGranularity.ALL_TOKENS:
                return activations.flatten(0, 1), None
            case ActivationGranularity.TOKEN | ActivationGranularity.SAMPLE:
                if not isinstance(inputs, BatchEncoding):
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is TOKEN."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                        + f"Got: {type(inputs)}"
                    )
                if aggregation_strategy is None and activation_granularity == ActivationGranularity.SAMPLE:
                    raise ValueError("aggregation_strategy cannot be None when activation_granularity is SAMPLE.")

                # extract indices of activations to keep from inputs
                indices_list = Granularity.get_indices(inputs, Granularity.TOKEN, self.tokenizer)  # type: ignore

                # select activations based on indices
                activation_list: list[Float[torch.Tensor, "g d"]] = []

                # iterate over samples
                for i, indices in enumerate(indices_list):
                    indices_tensor = torch.tensor(indices).squeeze(1)
                    activation_list.append(activations[i, indices_tensor])

                    # aggregate activations for SAMPLE strategy
                    if activation_granularity == ActivationGranularity.SAMPLE:
                        activation_list[-1] = GranularityAggregationStrategy.aggregate(
                            activation_list[-1],
                            strategy=aggregation_strategy,  # type: ignore
                            dim=-2,
                        )

                # concat all activations
                flatten_activations: Float[torch.Tensor, "ng d"] = torch.concat(activation_list, dim=0)
                return flatten_activations, indices_list
            case ActivationGranularity.WORD | ActivationGranularity.SENTENCE:
                if not isinstance(inputs, BatchEncoding):
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is WORD or SENTENCE."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                        + f"Got: {type(inputs)}"
                    )
                if aggregation_strategy is None:
                    raise ValueError(
                        "aggregation_strategy cannot be None when activation_granularity is WORD or SENTENCE."
                    )

                # extract indices of activations to keep from inputs
                # activation_granularity and granularities correspond
                granularity: Granularity = activation_granularity.value  # type: ignore
                indices_list = Granularity.get_indices(inputs, granularity, self.tokenizer)

                # select activations based on indices
                activation_list: list[Float[torch.Tensor, "g d"]] = []

                # iterate over samples
                for i, indices in enumerate(indices_list):
                    # iterate over activations
                    for index in indices:
                        word_activations = activations[i, index]
                        aggregated_activations = GranularityAggregationStrategy.aggregate(
                            word_activations, strategy=aggregation_strategy, dim=-2
                        )
                        activation_list.append(aggregated_activations)

                # concat all activations
                flatten_activations: Float[torch.Tensor, "ng d"] = torch.concat(activation_list, dim=0)
                return flatten_activations, indices_list
            case _:
                raise ValueError(f"Invalid activation selection strategy: {activation_granularity}")

    def _reintegrate_selected_activations(
        self,
        initial_activations: Float[torch.Tensor, "n l d"],
        new_activations: Float[torch.Tensor, "n l d"] | Float[torch.Tensor, "ng d"],
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy | None,
        granularity_indices: list[list[list[int]]] | None = None,
    ):
        """
        Reintegrates the selected activations into the initial activations.

        Args:
            initial_activations (Float[torch.Tensor, "n l d"]): The initial activations tensor.
            new_activations (Float[torch.Tensor, "n l d"] | Float[torch.Tensor, "ng d"]): The new activations tensor.
            granularity_indices (list[list[list[int]]]): The indices of the granularity level.
            activation_granularity (ActivationGranularity): The granularity level.
            aggregation_strategy (GranularityAggregationStrategy | None): The aggregation strategy to use.

        Returns:
            Float[torch.Tensor, "n l d"]: The reintegrated activations tensor.
        """
        match activation_granularity:
            case ActivationGranularity.ALL:
                return new_activations
            case ActivationGranularity.CLS_TOKEN:
                initial_activations[:, 0, :] = new_activations
                return initial_activations
            case ActivationGranularity.ALL_TOKENS:
                return new_activations.view(initial_activations.shape)
            case ActivationGranularity.TOKEN:
                if granularity_indices is None:
                    raise ValueError("granularity_indices cannot be None when activation_granularity is TOKEN.")
                # iterate over samples
                current_index = 0
                for i, indices in enumerate(granularity_indices):
                    indices_tensor = torch.tensor(indices).squeeze(1)
                    initial_activations[i, indices_tensor] = new_activations[
                        current_index : current_index + len(indices)
                    ]
                    current_index += len(indices)

                return initial_activations
            case ActivationGranularity.WORD | ActivationGranularity.SENTENCE:
                if granularity_indices is None:
                    raise ValueError(
                        "granularity_indices cannot be None when activation_granularity is WORD or SENTENCE."
                    )
                if aggregation_strategy is None:
                    raise ValueError(
                        "aggregation_strategy cannot be None when activation_granularity is WORD or SENTENCE."
                    )
                # iterate over samples
                current_index = 0
                for i, indices in enumerate(granularity_indices):
                    # iterate over activations
                    for index in indices:
                        # extract the activations for the current word/sentence
                        aggregated_activations = new_activations[current_index : current_index + 1]

                        # repeat the activations to match the length of the word/sentence
                        unfolded_activations = GranularityAggregationStrategy.unfold(
                            aggregated_activations, aggregation_strategy, len(index)
                        )
                        torch_index = torch.tensor(index).to(initial_activations.device)
                        initial_activations[i, torch_index] = unfolded_activations.to(initial_activations.device)
                        current_index += 1
                return initial_activations
            case ActivationGranularity.SAMPLE:
                raise ValueError(
                    "Activations aggregated at the sample level cannot be reintegrated. "
                    "Please choose another granularity level, such as ALL_TOKENS, TOKEN, WORD, or SENTENCE."
                )
            case _:
                raise ValueError(f"Invalid activation selection strategy: {activation_granularity}")

    def _manage_output_tuple(self, activations: torch.Tensor | tuple[torch.Tensor], split_point: str) -> torch.Tensor:
        """Handles the case in which the model has a tuple of outputs, and we need to know which element is the hidden state.

        Args:
            activations (torch.Tensor | tuple[torch.Tensor]): The activations to manage.
            split_point (str): The split point for interpretable error messages.

        Returns:
            torch.Tensor: The managed activations.
            int | None: The index of the hidden state in the tuple.

        Raises:
            TypeError: If the activations are not a `torch.tensor` or a valid tuple.
            RuntimeError: If the activations are a tuple, but we were not able to determine which element is the hidden state.
        """
        if isinstance(activations, torch.Tensor):
            return activations

        if not isinstance(activations, tuple):
            raise TypeError(
                f"Failed to manipulate activations for split point '{split_point}'. "
                f"Wrong type of activations. Expected torch.Tensor or tuple[torch.Tensor], got {type(activations)}: {activations}"
            )

        if self.output_tuple_index is not None:
            return activations[self.output_tuple_index]

        for i, candidate in enumerate(activations):
            if candidate.dim() == 3:
                self.output_tuple_index = i
                return candidate

        raise RuntimeError(
            f"Failed to manipulate activations for split point '{split_point}'. "
            "Activations are tuples, which are not supported. "
            "Then the split point is not a valid path in the loaded model, because it has several outputs. "
            "Often, adding '.mlp' to the split point path solves the issue."
        )

    def get_activations(
        self,
        inputs: list[str] | torch.Tensor | BatchEncoding,
        activation_granularity: ActivationGranularity = ActivationGranularity.ALL_TOKENS,
        aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        pad_side: str = "left",
        tqdm_bar: bool = False,
        **kwargs,
    ) -> dict[str, LatentActivations]:
        """Get intermediate activations for all model split points

        Args:
            inputs list[str] | torch.Tensor | BatchEncoding: Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.
            activation_granularity (ActivationGranularity): Selection strategy for activations.

                Options are:
                - ``ModelWithSplitPoints.activation_granularities.ALL``:
                    the activations are returned as is ``(batch, seq_len, d_model)``.
                    They are padded manually so that each batch of activations can be concatenated.

                - ``ModelWithSplitPoints.activation_granularities.CLS_TOKEN``:
                    only the first token (e.g. ``[CLS]``) activation is returned ``(batch, d_model)``.

                - ``ModelWithSplitPoints.activation_granularities.ALL_TOKENS``:
                    every token activation is treated as a separate element ``(batch x seq_len, d_model)``.

                - ``ModelWithSplitPoints.activation_granularities.TOKEN``: remove special tokens.

                - ``ModelWithSplitPoints.activation_granularities.WORD``:
                    aggregate by words following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.WORD`.

                - ``ModelWithSplitPoints.activation_granularities.SENTENCE``:
                    aggregate by sentences following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.SENTENCE`.
                    Requires `spacy` to be installed.

                - ``ModelWithSplitPoints.activation_granularities.SAMPLE``:
                    activations are aggregated on the whole sample.

            aggregation_strategy: Strategy to aggregate token activations into larger inputs granularities.
                Applied for `WORD`, `SENTENCE` and `SAMPLE` activation strategies.
                Token activations of shape  n * (l, d) are aggregated on the sequence length dimension.
                The concatenated into (ng, d) tensors.
                Existing strategies are:

                - ``ModelWithSplitPoints.aggregation_strategies.SUM``:
                    Tokens activations are summed along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MEAN``:
                    Tokens activations are averaged along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MAX``:
                    The maximum of the token activations along the sequence length dimension is selected.

                - ``ModelWithSplitPoints.aggregation_strategies.SIGNED_MAX``:
                    The maximum of the absolute value of the activations multiplied by its initial sign.
                    signed_max([[-1, 0, 1, 2], [-3, 1, -2, 0]]) = [-3, 1, -2, 2]

            pad_side (str): 'left' or 'right' — side on which to apply padding along dim=1 only for ALL strategy.
            tqdm_bar (bool): Whether to display a progress bar.
            return_predictions (bool): Whether to return the model predictions along with the activations.
            **kwargs: Additional keyword arguments passed to the model forward pass.

        Returns:
            (dict[str, LatentActivations]) Dictionary having one key, value pair for each split point defined for the model. Keys correspond to split
                names in `self.split_points`, while values correspond to the extracted activations for the split point
                for the given `inputs`.
        """
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        if not self.split_points:
            raise RuntimeError(
                "No split points are currently defined for the model. "
                "Please set split points before calling get_activations."
            )

        # batch inputs
        if isinstance(inputs, BatchEncoding):
            batch_generator = []
            for i in range(0, len(inputs), self.batch_size):
                end_idx = min(i + self.batch_size, len(inputs))
                batch_generator.append({key: value[i:end_idx] for key, value in inputs.items()})
        else:
            batch_generator = (
                inputs[i : min(i + self.batch_size, len(inputs))] for i in range(0, len(inputs), self.batch_size)
            )

        activations: dict = {}
        for split_point in self.split_points + ["predictions"]:
            activations[split_point] = []

        # iterate over batch of inputs
        with torch.no_grad():  # TODO: find a way to add this back, to merge gradient and forward
            with self.session():
                for batch_inputs in tqdm(
                    batch_generator,
                    desc="Computing activations",
                    unit="batch",
                    total=ceil(len(inputs) / self.batch_size),
                    disable=not tqdm_bar,
                ):
                    # tokenize text inputs
                    if isinstance(batch_inputs, list):
                        if activation_granularity == ActivationGranularity.CLS_TOKEN:
                            self.tokenizer.padding_side = "right"
                        tokenized_inputs = self.tokenizer(
                            batch_inputs,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            return_offsets_mapping=True,
                        )
                        if isinstance(self.args[0], T5ForConditionalGeneration):
                            # TODO: find a way for this not to be necessary
                            tokenized_inputs["decoder_input_ids"] = tokenized_inputs["input_ids"]
                    else:
                        tokenized_inputs = batch_inputs

                    # extract offset mapping not supported by forward but necessary for sentence selection strategy
                    if isinstance(tokenized_inputs, (BatchEncoding, dict)):  # noqa: UP038
                        offset_mapping = tokenized_inputs.pop("offset_mapping", None)
                    else:
                        offset_mapping = None

                    # call model forward pass and save split point outputs
                    with self.trace(tokenized_inputs, **kwargs) as tracer:
                        batch_activations = tracer.cache(modules=[self.get(sp) for sp in self.split_points])
                        activations["predictions"].append(self.output.logits.argmax(dim=-1).cpu().save())
                    torch.cuda.empty_cache()

                    # put back offset mapping
                    if offset_mapping is not None:
                        tokenized_inputs["offset_mapping"] = offset_mapping  # type: ignore

                    for sp in self.split_points:
                        sp_module = batch_activations["model." + sp]
                        output_name = "nns_output" if hasattr(sp_module, "nns_output") else "output"
                        batch_outputs = getattr(sp_module, output_name)
                        batch_sp_activations = self._manage_output_tuple(batch_outputs, sp)

                        granular_activations, indices_list = self._apply_selection_strategy(
                            inputs=tokenized_inputs,  # type: ignore
                            activations=batch_sp_activations.cpu(),  # type: ignore
                            activation_granularity=activation_granularity,
                            aggregation_strategy=aggregation_strategy,
                        )

                        activations[sp].append(granular_activations)

                    # adapt predictions ot match the granularity indices
                    if indices_list is None:
                        if activation_granularity == ActivationGranularity.ALL_TOKENS:
                            activations["predictions"][-1] = (
                                activations["predictions"][-1].repeat(1, batch_sp_activations.shape[1]).flatten(0, 1)
                            )
                    else:
                        predictions = []
                        for i, indices in enumerate(indices_list):
                            predictions.append(activations["predictions"][-1][i].repeat(len(indices)))

                        activations["predictions"][-1] = torch.cat(predictions, dim=0)

        # concat activation batches
        for split_point in self.split_points:
            if activation_granularity == ActivationGranularity.ALL:
                # three dimensional tensor (n, l, d)
                activations[split_point] = ModelWithSplitPoints.pad_and_concat(activations[split_point], pad_side, 0.0)
            else:
                # two dimensional tensor (n*g, d)
                activations[split_point] = torch.cat(activations[split_point], dim=0)
        activations["predictions"] = torch.cat(activations["predictions"], dim=0)

        # Validate that activations have the expected type
        for layer, act in activations.items():
            if not isinstance(act, torch.Tensor):
                raise RuntimeError(
                    f"Invalid output for layer '{layer}'. Expected torch.Tensor activation, got {type(act)}: {act}"
                )
        return activations  # type: ignore

    def get_concepts_output_gradients(
        self,
        inputs: list[str] | torch.Tensor | BatchEncoding,
        encode_activations: Callable[[LatentActivations], ConceptsActivations],
        decode_activations: Callable[[ConceptsActivations], LatentActivations],
        targets: list[int] | None = None,
        split_point: str | None = None,
        activation_granularity: ActivationGranularity = ActivationGranularity.ALL_TOKENS,
        aggregation_strategy: GranularityAggregationStrategy | None = GranularityAggregationStrategy.MEAN,
        concepts_x_gradients: bool = False,
        tqdm_bar: bool = False,
        **kwargs,
    ) -> list[Float[torch.Tensor, "t g c"]]:
        """Get intermediate activations for all model split points

        Args:
            inputs list[str] | torch.Tensor | BatchEncoding: Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.
            activation_granularity (ActivationGranularity): Selection strategy for activations.

                Options are:
                - ``ModelWithSplitPoints.activation_granularities.ALL``:
                    the activations are returned as is ``(batch, seq_len, d_model)``.
                    They are padded manually so that each batch of activations can be concatenated.

                - ``ModelWithSplitPoints.activation_granularities.CLS_TOKEN``:
                    only the first token (e.g. ``[CLS]``) activation is returned ``(batch, d_model)``.

                - ``ModelWithSplitPoints.activation_granularities.ALL_TOKENS``:
                    every token activation is treated as a separate element ``(batch x seq_len, d_model)``.

                - ``ModelWithSplitPoints.activation_granularities.TOKEN``: remove special tokens.

                - ``ModelWithSplitPoints.activation_granularities.WORD``:
                    aggregate by words following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.WORD`.

                - ``ModelWithSplitPoints.activation_granularities.SENTENCE``:
                    aggregate by sentences following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.SENTENCE`.
                    Requires `spacy` to be installed.

                - ``ModelWithSplitPoints.activation_granularities.SAMPLE``:
                    activations are aggregated on the whole sample.

            aggregation_strategy: Strategy to aggregate token activations into larger inputs granularities.
                Applied for `WORD`, `SENTENCE` and `SAMPLE` activation strategies.
                Token activations of shape  n * (l, d) are aggregated on the sequence length dimension.
                The concatenated into (ng, d) tensors.
                Existing strategies are:

                - ``ModelWithSplitPoints.aggregation_strategies.SUM``:
                    Tokens activations are summed along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MEAN``:
                    Tokens activations are averaged along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MAX``:
                    The maximum of the token activations along the sequence length dimension is selected.

                - ``ModelWithSplitPoints.aggregation_strategies.SIGNED_MAX``:
                    The maximum of the absolute value of the activations multiplied by its initial sign.
                    signed_max([[-1, 0, 1, 2], [-3, 1, -2, 0]]) = [-3, 1, -2, 2]

            tqdm_bar (bool): Whether to display a progress bar.
            **kwargs: Additional keyword arguments passed to the model forward pass.

        Returns:
            gradients (list[torch.Tensor]): The gradients of the model output with respect to the concept activations.
            List length: correspond to the number of inputs.
                Tensor shape: (t, g, c) with t the target dimension, g the number of granularity elements in one input, and c the number of
                concepts.
        """
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.targets = targets
        if split_point is not None:
            local_split_point: str = split_point
        elif not self.split_points:
            raise ValueError(
                "The activations cannot correspond to `model_with_split_points` model. "
                "The `model_with_split_points` model do not have `split_point` defined. "
            )
        elif len(self.split_points) > 1:
            raise ValueError("Cannot determine the split point with multiple `model_with_split_points` split points. ")
        else:
            local_split_point: str = self.split_points[0]

        # batch inputs
        grad_batch_size = 1  # TODO: find a way to batch calls
        if isinstance(inputs, BatchEncoding):
            batch_generator = []
            for i in range(0, len(inputs), grad_batch_size):
                end_idx = min(i + grad_batch_size, len(inputs))
                batch_generator.append({key: value[i:end_idx] for key, value in inputs.items()})
        else:
            batch_generator = (
                inputs[i : min(i + grad_batch_size, len(inputs))] for i in range(0, len(inputs), grad_batch_size)
            )

        gradients_list: list[Float[torch.Tensor, "ng c"]] = []
        # iterate over batch of inputs
        for batch_inputs in tqdm(
            batch_generator,
            desc="Computing gradients",
            unit="sample",
            total=len(inputs),
            disable=not tqdm_bar,
        ):
            # tokenize text inputs
            if isinstance(batch_inputs, list):
                if activation_granularity == ActivationGranularity.CLS_TOKEN:
                    self.tokenizer.padding_side = "right"
                tokenized_inputs = self.tokenizer(
                    batch_inputs, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
                )
                if isinstance(self.args[0], T5ForConditionalGeneration):
                    # TODO: find a way for this not to be necessary
                    tokenized_inputs["decoder_input_ids"] = tokenized_inputs["input_ids"]
            else:
                tokenized_inputs = batch_inputs

            # extract offset mapping not supported by forward but necessary for sentence selection strategy
            if isinstance(tokenized_inputs, (BatchEncoding, dict)):  # noqa: UP038
                offset_mapping = tokenized_inputs.pop("offset_mapping", None)
            else:
                offset_mapping = None

            # call model forward pass and gradients on concept activations
            with self.trace(tokenized_inputs, **kwargs):
                curr_module = self.get(local_split_point)
                # Handle case in which module has .output attribute, and .nns_output gets overridden instead
                module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"

                # get activations
                layer_outputs = getattr(curr_module, module_out_name)
                raw_activations: Float[torch.Tensor, "1 l d"] = self._manage_output_tuple(
                    layer_outputs, local_split_point
                )

                if offset_mapping is not None:
                    tokenized_inputs["offset_mapping"] = offset_mapping  # type: ignore

                # apply selection strategy
                selected_activations: Float[torch.Tensor, "g, d"]
                selected_activations, granularity_indices = self._apply_selection_strategy(
                    inputs=tokenized_inputs,  # type: ignore
                    activations=raw_activations,  # use the last batch of activations
                    activation_granularity=activation_granularity,
                    aggregation_strategy=aggregation_strategy,
                )

                # encode activations into concepts
                concept_activations: Float[torch.Tensor, "g, c"] = encode_activations(selected_activations)

                # decode concepts back into activations
                decoded_activations: Float[torch.Tensor, "g, d"] = decode_activations(concept_activations)

                # reintegrate decoded activations into the original activations
                reconstructed_activations: Float[torch.Tensor, "1, l, d"] = self._reintegrate_selected_activations(
                    initial_activations=raw_activations,
                    new_activations=decoded_activations,
                    granularity_indices=granularity_indices,
                    activation_granularity=activation_granularity,
                    aggregation_strategy=aggregation_strategy,
                )

                # reintegrate the reconstructed activations into the original layer outputs
                if isinstance(layer_outputs, tuple):
                    layer_outputs = list(layer_outputs)
                    layer_outputs[self.output_tuple_index] = reconstructed_activations
                else:
                    layer_outputs = reconstructed_activations

                # assign the new outputs to the module output
                if hasattr(curr_module, "nns_output"):
                    curr_module.nns_output = layer_outputs
                else:
                    curr_module.output = layer_outputs

                # get logits
                all_logits = self.output.logits

                if len(all_logits.shape) == 3:  # generation (n, l, v)
                    logits, _ = all_logits.max(dim=-1)  # (n, l)
                else:  # classification (n, nb_classes)
                    logits = all_logits

                # TODO: test if gradient of sum is equivalent to individual gradients
                # It would allow batching the gradient computation

                # compute gradients for each target
                target_gradient = []
                if self.targets is None:
                    targets = range(logits.shape[1])
                else:
                    targets = self.targets

                for t in targets:
                    with logits[:, t].backward(retain_graph=True):
                        # compute the gradient of the concept activations
                        concept_activations_grad: Float[torch.Tensor, "g c"] = concept_activations.grad

                        # for gradient x concepts, multiply by concepts
                        if concepts_x_gradients:
                            concept_activations_grad *= concept_activations
                    target_gradient.append(concept_activations_grad)

                target_gradient: Float[torch.Tensor, "t g d"] = (
                    torch.stack(target_gradient, dim=0).detach().cpu().save()
                )

            gradients_list.append(target_gradient)
            torch.cuda.empty_cache()

        return gradients_list

    def get_split_activations(
        self, activations: dict[str, LatentActivations], split_point: str | None = None
    ) -> LatentActivations:
        """
        Extract activations for the specified split point.
        Verify that the given activations are valid for the `model_with_split_points` and `split_point`.
        Cases in which the activations are not valid include:

        * Activations are not a valid dictionary.
        * Specified split point does not exist in the activations.

        Args:
            activations (dict[str, LatentActivations]): A dictionary with model paths as keys and the corresponding
                tensors as values.
            split_point (str | None): The split point to extract activations from.
                If None, the `split_point` of the explainer is used.

        Returns:
            (LatentActivations): The activations for the explainer split point.

        Raises:
            TypeError: If the activations are not a valid dictionary.
            ValueError: If the specified split point is not found in the activations.
        """
        if split_point is not None:
            local_split_point: str = split_point
        elif not self.split_points:
            raise ValueError(
                "The activations cannot correspond to `model_with_split_points` model. "
                "The `model_with_split_points` model do not have `split_point` defined. "
            )
        elif len(self.split_points) > 1:
            raise ValueError("Cannot determine the split point with multiple `model_with_split_points` split points. ")
        else:
            local_split_point: str = self.split_points[0]

        if not isinstance(activations, dict) or not all(isinstance(act, torch.Tensor) for act in activations.values()):
            raise TypeError(
                "Invalid activations for the concept explainer. "
                "Activations should be a dictionary of model paths and torch.Tensor activations. "
                f"Got: '{type(activations)}'"
            )
        activations_split_points: list[str] = list(activations.keys())  # type: ignore
        if local_split_point not in activations_split_points:
            raise ValueError(
                f"Fitted split point '{local_split_point}' not found in activations.\n"
                f"Available split_points: {', '.join(activations_split_points)}."
            )

        return activations[local_split_point]  # type: ignore

    def get_latent_shape(
        self,
        inputs: str | list[str] | BatchEncoding | None = None,
    ) -> dict[str, torch.Size]:
        """Get the shape of the latent activations at the specified split point."""
        sizes = {}
        with self.scan(self._example_input if inputs is None else inputs):
            for split_point in self.split_points:
                curr_module = self.get(split_point)
                module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"
                module = getattr(curr_module, module_out_name)
                if isinstance(module, tuple):
                    for candidate in module:
                        if candidate.dim() == 3:
                            module = candidate
                            break
                sizes[split_point] = module.shape
        return sizes
