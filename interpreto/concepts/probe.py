from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.svm import SVC

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import LatentActivations, ConceptsActivations


class SklearnProbe:
    """Follows the ConceptModelProtocol."""

    nb_concepts = 1

    def __init__(self, sklearn_class: Any, sklearn_kwargs: dict[str, Any]):
        self.model = sklearn_class(**sklearn_kwargs)
        self.fitted = False

    def fit(self, x, y):
        """Fit the concept model."""
        self.model.fit(x, y)
        self.fitted = True

    def encode(self, x):
        """Encode the given activations using the concept model."""
        return self.model.predict(x)


class ProbeExplainer(ConceptEncoderExplainer[SklearnProbe]):
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

    def fit(
        self,
        activations: LatentActivations | dict[str, LatentActivations],
        labels: np.ndarray,
    ):
        """Fit the concept model."""
        split_activations = self._sanitize_activations(activations)
        np_activations = np.array(split_activations)

        if len(np_activations.shape) != 2:
            raise ValueError(
                "Expected activations to be a 2D array, (n, d), got shape "
                f"{np_activations.shape}"
            )
        if np_activations.shape[0] != labels.shape[0]:
            raise ValueError(
                "Expected activations and labels to have the same number of rows, "
                f"got {np_activations.shape[0]} and {labels.shape[0]}"
            )

        self.concept_model.fit(np_activations, labels)

    # TODO: check fitted
    def encode_activations(self, activations: LatentActivations) -> ConceptsActivations:
        assert self.concept_model.fitted
        np_activations = np.array(activations)
        np_probed = self.concept_model.encode(np_activations)
        return torch.from_numpy(np_probed).unsqueeze(1)
