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
validation, split point resolution) and defines the abstract
interface that concept explainers rely on.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch
from nnsight import Envoy, TransformersModel
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

# Prevents:
# UserWarning: Module ... has a submodule named 'output', which nnsight already serves ...
# nnsight access for `output` will be mounted at `.E_output` instead of `.output` for this module only.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="nnsight.intervention.envoy",
    message=r".*submodule named 'output'.*",
)


class InitializationError(ValueError):
    """Raised to signal a problem with model initialization."""


class BaseSplitter(TransformersModel, ABC):
    """Abstract base class for all Interpreto model splitters.

    Provides:
    - Shared initialization (NNsight model loading and tokenizer management)
    - Split point property with validation
    - Helpers for handling output tuples
    - Abstract interface expected by concept explainers

    Subclasses must implement ``get_activations``, ``_get_concept_output_gradients``,
    and ``get_latent_shape``.

    Arguments:
        model_or_repo_id (str | PreTrainedModel): One of:

            * A ``str`` corresponding to the ID of the model that should be loaded from the HF Hub.
            * A ``str`` corresponding to the local path of a folder containing a compatible checkpoint.
            * A preloaded ``transformers.PreTrainedModel`` object.

        split_point (str | int | None): The split location inside the model.
            Subclasses may accept ``None`` and resolve it automatically.

        task (str): Hugging Face pipeline task selecting the model architecture.

        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None): Custom tokenizer.
            If None, NNsight resolves it automatically when possible.

        batch_size (int): Batch size for batched operations.

        device_map (torch.device | str | None): Device on which to load the model.
    """

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_point: str | int | None,
        *,
        task: str | None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        **kwargs,
    ) -> None:
        """Initialize a BaseSplitter.

        Raises:
            InitializationError: If the tokenizer cannot be resolved.
            ValueError: If ``device_map`` is set to ``'auto'`` with a pre-loaded model.
        """
        # ------------------------------------------------------------------
        # Model loading through nnsight.TransformersModel
        super().__init__(
            model_or_repo_id,
            task=task,
            tokenizer=tokenizer,
            device_map=device_map,
            **kwargs,
        )

        self.batch_size = batch_size

        # ------------------------------------------------------------------
        # Device handling for pre-loaded models (nnsight ignores device_map in this case)
        if not isinstance(model_or_repo_id, str) and device_map is not None:
            if device_map == "auto":
                raise ValueError("'auto' device_map is only supported when loading from a repository id.")
            self.to(device_map)

        # ------------------------------------------------------------------
        # Final validation
        if self.tokenizer is None:
            raise InitializationError("Tokenizer could not be resolved automatically.")

        self.split_point = split_point  # type: ignore[assignment]

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
        prefix = f"{self.path}."
        modules = [
            (path.removeprefix(prefix), module) for path, module in self.named_modules() if path.startswith(prefix)
        ]

        if isinstance(split_point, int):
            matches = [(path, module) for path, module in modules if path.endswith(f".{split_point}")]
            if len(matches) != 1:
                raise ValueError(f"Layer {split_point} matched {[path for path, _ in matches]}")
            path, module = matches[0]
        else:
            try:
                path, module = next((path, module) for path, module in modules if path == split_point)
            except StopIteration:
                raise ValueError(f"The provided split point '{split_point}' is not valid.") from None

        self._split_point = path
        self._split_module = module

    @property
    def split_module(self) -> Envoy:
        """The NNsight module at the split point."""
        return self._split_module

    # ======================================================================
    # Shared helpers
    # ======================================================================

    def _extract_hidden_state(
        self, activations: torch.Tensor | tuple, split_point: str
    ) -> tuple[torch.Tensor, int | None]:
        """Extract the (n, l, d) hidden state from a possibly-tuple output at a split point.

        If the output is a tuple, finds the 3D tensor in it.

        Args:
            activations: The raw output at the split point.
            split_point: The split point path (for error messages).

        Returns:
            The 3D activations tensor and its index in the output tuple. The
            index is None when the output is directly a tensor.

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
            return activations, None

        if not isinstance(activations, tuple):
            raise TypeError(
                f"Failed to manipulate activations for split point '{split_point}'. "
                f"Wrong type of activations. Expected torch.Tensor or tuple[torch.Tensor], got {type(activations)}: {activations}"
            )

        for i, candidate in enumerate(activations):
            if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
                return candidate, i

        raise RuntimeError(
            f"Failed to manipulate activations for split point '{split_point}'. "
            "Activations are tuples, and no tensor with three dimensions was found. "
            f"Found members: {[type(member).__name__ for member in activations]}. "
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

    @abstractmethod
    def inputs_to_activations(self, *args: Any, **kwargs: Any) -> Any:
        """Unbatched activation extraction from raw inputs.

        Subclasses define the exact signature and return types appropriate for
        their task (classification vs generation vs full granularity).

        Raises:
            NotImplementedError: Always — subclasses must override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement inputs_to_activations")

    @abstractmethod
    def get_activations(self, *args: Any, **kwargs: Any) -> Any:
        """Extract intermediate activations at the split point.

        Subclasses define the exact signature and return types appropriate for
        their task (classification vs generation vs full granularity).

        Raises:
            NotImplementedError: Always — subclasses must override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_activations")

    @abstractmethod
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
