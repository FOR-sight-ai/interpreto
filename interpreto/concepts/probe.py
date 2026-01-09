from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.svm import SVC
from torch import nn

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import ConceptsActivations, LatentActivations


class SklearnProbe:
    """Follows the ConceptModelProtocol."""

    nb_concepts = 1

    def __init__(self, sklearn_class: Any, sklearn_kwargs: dict[str, Any]):
        self.model = sklearn_class(**sklearn_kwargs)
        self.fitted = False

    def fit(self, x, y):
        """Fit the concept model."""
        x, y = np.array(x), np.array(y)
        self.model.fit(x, y)
        self.fitted = True

    def encode(self, x):
        """Encode the given activations using the concept model."""
        return self.model.decision_function(x)


class LinearRegressionProbe(nn.Module):
    """
    Linear regression probe (closed-form solution) with intercept.

    Supports:
        y : (n, c)
    """

    def __init__(self):
        super().__init__()
        self.weight = None  # nn.Parameter, shape (d, c)
        self.bias = None  # nn.Parameter, shape (c,)
        self.fitted = False

    def fit(self, X, y):
        """
        X : (n, d)
        y : (n, c)
        """
        n, d = X.shape

        # Design matrix with bias
        ones = torch.ones(n, 1, dtype=X.dtype, device=X.device)
        X_design = torch.cat([ones, X], dim=1)  # (n, 1 + d)

        # Closed-form OLS: beta = (X^T X)^(-1) X^T y
        XT = X_design.T
        beta = torch.linalg.pinv(XT @ X_design) @ XT @ y  # (d+1, c)

        # Extract parameters
        with torch.no_grad():
            b = beta[0]  # (c,)
            w = beta[1:]  # (d, c)

        self.weight = nn.Parameter(w.clone())
        self.bias = nn.Parameter(b.clone())

        self.fitted = True

    def encode(self, X):
        """
        X : (n, d)
        Returns :
            (n, c)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

        scores = X @ self.weight + self.bias  # (n, c)
        return scores


class LogisticRegressionProbe(nn.Module):
    """
    (Multi-label) logistic regression probe with intercept.

    Each output column is an independent binary classifier.
    """

    def __init__(self, lr: float = 1e-2, max_iter: int = 1000, l2: float = 0.0):
        super().__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2

        self.weight = None  # nn.Parameter, shape (d, c)
        self.bias = None  # nn.Parameter, shape (c,)
        self.fitted = False

    def fit(self, X, y):
        """
        X : (n, d)
        y : (n, c) with values in {0, 1}
        """
        n, d = X.shape

        y = y.float()  # (n, c)
        c = y.size(1)

        if self.weight is None or self.bias is None:
            self.weight = nn.Parameter(
                torch.zeros(d, c, dtype=X.dtype, device=X.device)
            )
            self.bias = nn.Parameter(torch.zeros(c, dtype=X.dtype, device=X.device))

        optimizer = torch.optim.Adam([self.weight, self.bias], lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            logits = X @ self.weight + self.bias  # (n, c)
            loss = loss_fn(logits, y)

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * (self.weight**2).sum()

            loss.backward()
            optimizer.step()

        self.fitted = True

    def encode(self, X):
        """
        X : (n, d)
        Returns :
            (n, c)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

        return X @ self.weight + self.bias


class LinearSVMProbe(nn.Module):
    """
    Linear SVM-style probe (soft-margin) with intercept.

    Multi-label: each output column is an independent classifier.
    """

    def __init__(self, lr: float = 1e-2, max_iter: int = 1000, l2: float = 0.0):
        super().__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2

        self.weight = None  # nn.Parameter, shape (d, c)
        self.bias = None  # nn.Parameter, shape (c,)

        self.fitted = False

    def fit(self, X, y):
        """
        X : (n, d)
        y : (n, c) in {0,1} (mapped to {-1,1})
        """
        n, d = X.shape

        y = y.float()  # (n, c)
        c = y.size(1)

        # Map {0,1} -> {-1,1}
        y = 2 * y - 1

        if self.weight is None or self.bias is None:
            self.weight = nn.Parameter(
                torch.zeros(d, c, dtype=X.dtype, device=X.device)
            )
            self.bias = nn.Parameter(torch.zeros(c, dtype=X.dtype, device=X.device))

        optimizer = torch.optim.Adam([self.weight, self.bias], lr=self.lr)

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            logits = X @ self.weight + self.bias  # (n, c)

            margins = 1.0 - y * logits
            hinge_loss = torch.clamp(margins, min=0.0).mean()

            loss = hinge_loss
            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * (self.weight**2).sum()

            loss.backward()
            optimizer.step()

        self.fitted = True

    def encode(self, X):
        """
        X : (n, d)
        Returns :
            (n, c)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

        return X @ self.weight + self.bias


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

        if len(split_activations.shape) != 2:
            raise ValueError(
                f"Expected activations to be a 2D array, (n, d), got shape {split_activations.shape}"
            )
        if split_activations.shape[0] != labels.shape[0]:
            raise ValueError(
                "Expected activations and labels to have the same number of rows, "
                f"got {split_activations.shape[0]} and {labels.shape[0]}"
            )

        self.concept_model.fit(split_activations, labels)

    # TODO: check fitted
    def encode_activations(self, activations: LatentActivations) -> ConceptsActivations:
        assert self.concept_model.fitted
        np_activations = np.array(activations)
        np_probed = self.concept_model.encode(np_activations)
        return torch.from_numpy(np_probed).unsqueeze(1)
