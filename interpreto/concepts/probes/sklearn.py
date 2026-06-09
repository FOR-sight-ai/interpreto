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
Scikit-learn backed probe model and explainer.

Provides [SklearnProbe][interpreto.concepts.probes.sklearn.SklearnProbe], a thin wrapper around
any sklearn classifier that exposes a `decision_function`, and
[SklearnProbeExplainer][interpreto.concepts.probes.sklearn.SklearnProbeExplainer] which
integrates it with the concept explainer pipeline.

Note:
    `SklearnProbe` does **not** inherit from `nn.Module` and therefore
    does not support `state_dict` serialization. Use pickle or joblib for
    persistence. It is limited to single-concept (binary) classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from sklearn.svm import SVC

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.concepts.probes.base import assert_fitted
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import ConceptsActivations, LatentActivations


class SklearnProbe:
    """Probe wrapping a scikit-learn classifier with `decision_function`.

    Satisfies [ConceptModelProtocol][interpreto.typing.ConceptModelProtocol] structurally.
    Currently limited to a single binary concept.

    Args:
        sklearn_class (Any): Scikit-learn estimator class (e.g. `SVC`).
        sklearn_kwargs (dict[str, Any]): Keyword arguments forwarded to the
            estimator constructor.
    """

    nb_concepts = 1  # TODO: have something working for more concepts

    def __init__(self, sklearn_class: Any, sklearn_kwargs: dict[str, Any]):
        self.model = sklearn_class(**sklearn_kwargs)
        self.fitted = False

    def fit(self, X: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n"]):
        """Fit the concept model."""
        np_X, np_y = np.array(X), np.array(y)
        self.model.fit(np_X, np_y)
        self.fitted = True

    @assert_fitted
    def encode(self, X: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n 1"]:
        """Encode the given activations using the concept model."""
        np_X = np.array(X)
        np_y = self.model.decision_function(np_X)
        return torch.from_numpy(np_y).unsqueeze(1)


class SklearnProbeExplainer(ConceptEncoderExplainer[SklearnProbe]):
    """Concept explainer using a scikit-learn probe.

    Integrates [SklearnProbe][interpreto.concepts.probes.sklearn.SklearnProbe] into the concept
    explainer pipeline, connecting it to a
    [ModelWithSplitPoints][interpreto.model_wrapping.model_with_split_points.ModelWithSplitPoints]
    for activation extraction.

    Args:
        model_with_split_points (ModelWithSplitPoints): Wrapped transformer model.
        split_point (str | None): Layer name to extract activations from.
        sklearn_class (Any): Scikit-learn estimator class (default: `SVC`).
        sklearn_kwargs (dict[str, Any]): Arguments forwarded to the sklearn estimator.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        split_point: str | None = None,
        sklearn_class: Any = SVC,
        sklearn_kwargs: dict[str, Any] = {},
    ):
        self.concept_model: SklearnProbe
        concept_model = SklearnProbe(sklearn_class, sklearn_kwargs)
        super().__init__(
            model_with_split_points=model_with_split_points,
            concept_model=concept_model,
            split_point=split_point,
        )

    @property
    def is_fitted(self) -> bool:
        """Delegates to the probe's `fitted` flag."""
        return self.concept_model.fitted

    def fit(
        self,
        activations: LatentActivations | dict[str, LatentActivations],
        labels: torch.Tensor,
    ):
        """Fit the concept model."""
        split_activations = self._sanitize_activations(activations)

        if len(split_activations.shape) != 2:
            raise ValueError(f"Expected activations to be a 2D array, (n, d), got shape {split_activations.shape}")
        if split_activations.shape[0] != labels.shape[0]:
            raise ValueError(
                "Expected activations and labels to have the same number of rows, "
                f"got {split_activations.shape[0]} and {labels.shape[0]}"
            )

        self.concept_model.fit(split_activations, labels)

    def encode_activations(self, activations: LatentActivations) -> ConceptsActivations:
        return self.concept_model.encode(activations)
