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

"""
Base class for model splitters.

``BaseSplitter`` is the abstract parent of ``ModelWithSplitPoints``,
``SplitterForClassification``, and ``SplitterForGeneration``.
It encapsulates the common initialization logic (NNsight wrapping, tokenizer
validation, split point resolution, padding helpers) and defines the abstract
interface that concept explainers rely on.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch
from nnsight.modeling.language import LanguageModel
from transformers import AutoModel, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from interpreto.commons.granularity import GranularityAggregationStrategy
from interpreto.concepts.splitters.splitting_utils import (
    get_layer_by_idx,
    validate_path,
    walk_modules,
)

# Prevents:
# UserWarning: Module ... of type ... has pre-defined a `output` attribute.
# nnsight access for `output` will be mounted at `.nns_output` instead of `.output` for this module only.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="nnsight.intervention.envoy",
    message=r".*has pre-defined a `output` attribute.*",
)


class InitializationError(ValueError):
    """Raised to signal a problem with model initialization."""


class BaseSplitter(LanguageModel, ABC):
    """Abstract base class for all Interpreto model splitters.

    Provides:
    - Shared initialization (NNsight model loading, input validation, tokenizer management)
    - Split point property with validation
    - Helpers for handling output tuples and padding tensors
    - Abstract interface expected by concept explainers

    Subclasses must implement ``get_activations``, ``_get_concept_output_gradients``,
    and ``get_latent_shape``.

    Arguments:
        model_or_repo_id (str | PreTrainedModel): One of:

            * A ``str`` corresponding to the ID of the model that should be loaded from the HF Hub.
            * A ``str`` corresponding to the local path of a folder containing a compatible checkpoint.
            * A preloaded ``transformers.PreTrainedModel`` object.

        split_point (str | int): The split location inside the model.

        automodel (type[AutoModel] | None): Hugging Face AutoClass for loading the model.
            Required when ``model_or_repo_id`` is a string.

        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None): Custom tokenizer.
            Required when providing a model instance.

        config (PretrainedConfig | None): Custom configuration for the loaded model.

        batch_size (int): Batch size for batched operations.

        device_map (torch.device | str | None): Device on which to load the model.

        output_tuple_index (int | None): If the output at the split point is a tuple,
            index of the hidden state element.  If None, a 3D tensor is searched for.
    """

    # Class-level enums exposed for the concept explainer interface
    aggregation_strategies = GranularityAggregationStrategy

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_point: str | int | None,
        *args: tuple[Any],
        automodel: type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        output_tuple_index: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize a BaseSplitter.

        Raises:
            InitializationError: If the model cannot be loaded due to a missing ``tokenizer`` or ``automodel``.
            ValueError: If ``device_map`` is set to ``'auto'`` with a pre-loaded model.
            TypeError: If ``model_or_repo_id`` is not a ``str`` or a ``PreTrainedModel``.
        """
        # ------------------------------------------------------------------
        # Input validation
        if isinstance(model_or_repo_id, PreTrainedModel):
            if tokenizer is None:
                raise InitializationError(
                    "Tokenizer is not set. When providing a model instance, the tokenizer must be set."
                )
        elif isinstance(model_or_repo_id, str):
            if automodel is None:
                raise InitializationError(
                    "Model autoclass not found.\n"
                    "The model class can be omitted if a pre-loaded model is passed to `model_or_repo_id` "
                    "param.\nIf an HF Hub ID is used, the corresponding autoclass must be specified in `automodel`.\n"
                    "Example: BaseSplitter('bert-base-uncased', automodel=AutoModelForMaskedLM, ...)"
                )
        else:
            raise TypeError(
                f"Invalid model_or_repo_id type: {type(model_or_repo_id)}. "
                "Expected `str` or `transformers.PreTrainedModel`."
            )

        # ------------------------------------------------------------------
        # Model loading through nnsight.LanguageModel
        super().__init__(
            model_or_repo_id,
            *args,
            config=config,
            tokenizer=tokenizer,  # type: ignore (under specification from NNsight)
            automodel=automodel,  # type: ignore (under specification from NNsight)
            device_map=device_map,
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Split point setup
        self._model_paths = list(walk_modules(self._model))
        self.split_point = split_point  # uses the property setter (overridable by subclasses)
        if not hasattr(self, "_split_point") or self._split_point is None:
            raise ValueError(
                "split_point was not resolved during initialization. "
                "Either pass a valid split_point or ensure the subclass setter resolves it."
            )
        self._model: PreTrainedModel  # narrow the NNsight type

        if self.repo_id is None:
            self.repo_id = self._model.config.name_or_path  # type: ignore (under specification from NNsight)

        self.batch_size = batch_size

        # ------------------------------------------------------------------
        # Device handling for pre-loaded models (nnsight ignores device_map in this case)
        if not isinstance(model_or_repo_id, str):
            if device_map is not None:
                if device_map == "auto":
                    raise ValueError(
                        "'auto' device_map is only supported when loading a generation model from a repository id. "
                        "Please specify a device_map, e.g. 'cuda' or 'cpu'."
                    )
                self.to(device_map)  # type: ignore (under specification from NNsight)

        # ------------------------------------------------------------------
        # Final validation
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set. When providing a model instance, the tokenizer must be set.")
        self.output_tuple_index = output_tuple_index

    # ======================================================================
    # Split point property
    # ======================================================================

    @property
    def split_point(self) -> str:
        """The split point of the model."""
        return self._split_point

    @split_point.setter
    def split_point(self, split_point: str | int | None) -> None:
        """Split point setter with validation.

        Args:
            split_point (str | int | None): The split location inside the model.
                Either a ``str`` path or an ``int`` layer index.
                Subclasses may accept ``None`` and resolve automatically;
                this base implementation requires a non-None value.
        """
        if split_point is None:
            raise ValueError(
                "split_point cannot be None. Provide a valid split point path (str) or layer index (int)."
            )
        if isinstance(split_point, int):
            str_split = get_layer_by_idx(split_point, model_paths=self._model_paths)
        else:
            str_split = split_point

        validate_path(self._model, str_split)
        self._split_point: str = str_split

    # ======================================================================
    # Shared helpers
    # ======================================================================

    def _manage_output_tuple(self, activations: torch.Tensor | tuple[torch.Tensor], split_point: str) -> torch.Tensor:
        """Extract the (n, l, d) hidden state from a possibly-tuple output at a split point.

        If the output is a tuple of tensors, finds the 3D tensor in it (or uses
        ``output_tuple_index`` if set).

        Args:
            activations: The raw output at the split point.
            split_point: The split point path (for error messages).

        Returns:
            The 3D activations tensor of shape (n, l, d).

        Raises:
            ValueError: If activations are not a 3D tensor.
            TypeError: If the activations are neither a tensor nor a tuple.
            RuntimeError: If the hidden state cannot be identified in a tuple.
        """
        if isinstance(activations, torch.Tensor):
            if activations.dim() != 3:
                raise ValueError(
                    f"Invalid activations for split point '{split_point}'. "
                    f"Expected a 3D tensor of shape (n, l, d), "
                    f"got a tensor of shape {activations.shape}. "
                    "It is recommended to look for another split point."
                )
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
                self.output_tuple_index: int | None = i
                return candidate

        raise RuntimeError(
            f"Failed to manipulate activations for split point '{split_point}'. "
            "Activations are tuples, and no tensor with three dimensions was found. "
            f"Found tensors of shape: {(t.shape for t in activations)}. "
            "It is recommended to look for another split point."
        )

    # ======================================================================
    # Interface (contract for concept explainers)
    # ======================================================================
    # get_activations and _get_concept_output_gradients intentionally use
    # permissive signatures (*args, **kwargs) because each subclass defines
    # task-specific required parameters that are not shared across siblings.
    # Using @abstractmethod with a strict signature would force LSP-violating
    # overrides.  The methods still raise NotImplementedError to enforce
    # implementation at runtime, and get_latent_shape remains truly abstract
    # (its signature is uniform across all subclasses).

    def get_activations(self, *args: Any, **kwargs: Any) -> Any:
        """Extract intermediate activations at the split point.

        Subclasses define the exact signature and return types appropriate for
        their task (classification vs generation vs full granularity).

        Raises:
            NotImplementedError: Always — subclasses must override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_activations")

    def _get_concept_output_gradients(self, *args: Any, **kwargs: Any) -> Any:
        """Compute gradients of model outputs w.r.t. concept activations.

        Subclasses define the full signature with task-specific parameters.

        Raises:
            NotImplementedError: Always — subclasses must override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _get_concept_output_gradients")

    @abstractmethod
    def get_latent_shape(self) -> torch.Size:
        """Return the shape of the latent activations at the split point."""
        ...
