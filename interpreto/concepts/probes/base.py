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
Base class for torch-based probe models.

This module defines [Probe][interpreto.concepts.probes.Probe], the abstract base
for all probe models in the package. It provides:

- A `fitted` flag persisted as a buffer (survives `state_dict` round-trips).
- A [load_state_dict][interpreto.concepts.probes.Probe.load_state_dict] override that
  handles dynamically-sized buffers and parameters created during
  [fit][interpreto.concepts.probes.Probe.fit]. Thus allowing saving and loading probes.
- The [assert_fitted][interpreto.concepts.probes.base.assert_fitted] decorator to guard methods
  that require a fitted model.

All concrete probes (centroid and linear) inherit or reference this base.


This module also provides [ProbeExplainer][interpreto.concepts.probes.base.ProbeExplainer],
which integrates any pre-instantiated Probe into the concept explainer pipeline.
The probe must be provided already instantiated (and optionally already fitted).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import wraps
from typing import Any

import torch
from jaxtyping import Float
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from interpreto.concepts.base import ConceptEncoderExplainer, check_fitted
from interpreto.concepts.splitters.base_splitter import BaseSplitter
from interpreto.typing import ConceptsActivations, LatentActivations

# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def assert_fitted(fn):
    """Decorator that raises `RuntimeError` if the probe is not fitted."""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        return fn(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Base Probe Model
# ---------------------------------------------------------------------------


class Probe(nn.Module, ABC):
    """
    Abstract base for all torch-based probe models.

    Satisfies `ConceptModelProtocol` structurally
    (no inheritance from `Protocol` needed at runtime).

    Subclasses must implement:
        - [fit][interpreto.concepts.probes.Probe.fit] — learn probe parameters from activations and labels.
        - [encode][interpreto.concepts.probes.Probe.encode] — map activations to concept scores.

    Attributes:
        nb_concepts (int): Number of concepts the probe was fitted on. Set by subclasses.
        fitted (bool): Whether the probe has been fitted (backed by a persistent buffer).
    """

    nb_concepts: int

    def __init__(self):
        super().__init__()
        self.register_buffer("_fitted_flag", torch.tensor(False, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Fitted state management
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """Whether the probe has been fitted or loaded from a state dict."""
        return bool(self._fitted_flag.item())  # type: ignore

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted_flag.fill_(bool(value))  # type: ignore

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        """Fit the probe on activations and multi-label targets.

        Args:
            x: Latent activations.
            y: Binary multi-label targets.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        """Encode activations into concept scores.

        Args:
            x: Latent activations.

        Returns:
            Concept scores. Higher values indicate stronger alignment with the concept.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State dict handling for dynamic shapes
    # ------------------------------------------------------------------

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        """Load a state dict, handling dynamically-sized buffers/parameters.

        Probes register buffers with shape `(0,)` at `__init__` time. After
        [fit][interpreto.concepts.probes.Probe.fit], these become their real shapes
        (e.g. `(c, d)`). When loading a fitted state dict into a fresh probe, the standard
        `nn.Module.load_state_dict` would raise a size-mismatch error.

        This override pre-allocates buffers to the correct shape before
        delegating to the parent implementation.
        """
        for key, value in state_dict.items():
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                submodule = self.get_submodule(parts[0])
                attr = parts[1]
            else:
                submodule = self
                attr = parts[0]

            # Resize existing buffer if shape doesn't match
            if attr in submodule._buffers and submodule._buffers[attr] is not None:
                if submodule._buffers[attr].shape != value.shape:  # type: ignore
                    submodule._buffers[attr] = torch.empty_like(value)
            # Register buffer for keys that don't exist yet
            elif attr not in submodule._buffers and attr not in submodule._parameters:
                submodule.register_buffer(attr, torch.empty_like(value))

        return super().load_state_dict(state_dict, strict=strict, assign=assign)


# ---------------------------------------------------------------------------
# Base Probe Explainer
# ---------------------------------------------------------------------------


class ProbeExplainer(ConceptEncoderExplainer[Probe]):
    """Concept explainer backed by a `Probe`.

    Integrates any pre-instantiated torch probe into the concept explainer
    pipeline, connecting it to a
    `BaseSplitter`
    for activation extraction.

    The probe is provided already instantiated (unfitted or pre-fitted).
    Calling [fit][interpreto.concepts.probes.base.ProbeExplainer.fit]
    delegates to the probe's own `fit` method.

    Args:
        splitter (interpreto.concepts.splitters.BaseSplitter): Wrapped transformer model.
        concept_model (interpreto.concepts.probes.Probe): An instantiated torch probe.
        split_point (str | None): Layer name to extract activations from.

    Example::

        from interpreto.concepts import LinearRegressionProbe, ProbeExplainer

        probe = LinearRegressionProbe()
        explainer = ProbeExplainer(splitter, probe)
        explainer.fit(activations, labels)
        concepts = explainer.activations_to_concepts(activations)
    """

    def __init__(
        self,
        splitter: BaseSplitter,
        concept_model: Probe,
    ):
        if not isinstance(concept_model, Probe):
            raise TypeError(f"concept_model must be a Probe instance, got {type(concept_model).__name__}.")
        super().__init__(
            splitter=splitter,
            concept_model=concept_model,
        )

    @property
    def concept_model(self) -> Probe:
        """The underlying torch probe model."""
        return self._concept_model  # type: ignore

    @property
    def is_fitted(self) -> bool:
        """Delegates to the probe's `fitted` flag."""
        return self.concept_model.fitted

    def fit(
        self,
        activations: LatentActivations,
        labels: Float[torch.Tensor, "n c"],
    ):
        """Fit the probe on activations and multi-label targets.

        Args:
            activations: Latent activations (2D tensor).
            labels: Binary multi-label targets of shape `(n, c)`.
                This should be a matrix can be an extended vector `(n, 1)` for a single concept.
                However, we allow and recommend to train several probes simultaneously.
        """
        if len(activations.shape) != 2:
            raise ValueError(f"Expected activations to be a 2D array, (n, d), got shape {activations.shape}")
        if activations.shape[0] != labels.shape[0]:
            raise ValueError(
                "Activations and labels must have the same number of samples, "
                f"got {activations.shape[0]} and {labels.shape[0]}."
            )

        self.concept_model.fit(activations, labels)

    @check_fitted
    def activations_to_concepts(self, activations: LatentActivations) -> ConceptsActivations:
        """Encode activations into concept scores using the fitted probe.

        Args:
            activations: Latent activations of shape `(n, d)`.

        Returns:
            Concept scores of shape `(n, c)`.
        """
        # Use the _fitted_flag buffer (always present) to infer the probe's device.
        probe_device = self.concept_model._fitted_flag.device  # type: ignore
        if activations.device != probe_device:
            activations = activations.to(probe_device)  # type: ignore
        return self.concept_model.encode(activations)
