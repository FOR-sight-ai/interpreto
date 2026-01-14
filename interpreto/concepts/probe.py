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


class MeansDiffProbe(nn.Module):
    """
    MeansDiff probe (multi-label, multi-output).

    For each concept j:
        w_j = mean(X | y_j=1) - mean(X | y_j=0)

    Produces:
        weight: (d, c)
        bias:   (c,)
        encode(X) = X @ weight + bias

    bias modes:
        - "zero":     b = 0
        - "midpoint": nearest-centroid midpoint bias
        - "bce":      choose b_j to minimize binary cross-entropy on logits for class j
                      with fixed w_j (1D convex optimization per class via Newton)
    """

    def __init__(
        self,
        bias: str = "zero",  # no impact on the direction itself
        eps: float = 1e-8,
        bce_newton_iters: int = 50,
        bce_newton_tol: float = 1e-8,
    ):
        super().__init__()
        if bias not in {"zero", "midpoint", "bce"}:
            raise ValueError("bias must be one of {'zero', 'midpoint', 'bce'}")
        self.bias_mode = bias
        self.eps = eps
        self.bce_newton_iters = int(bce_newton_iters)
        self.bce_newton_tol = float(bce_newton_tol)

        self.weight = None  # nn.Parameter, shape (d, c)
        self.bias = None  # nn.Parameter, shape (c,)
        self.fitted = False

    @torch.no_grad()
    def _bce_optimal_bias(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Find per-class bias b (c,) minimizing BCEWithLogitsLoss(scores + b, y)
        with scores fixed. Uses Newton iterations on b (convex in b).

        scores: (n, c)
        y:      (n, c) in {0,1}
        """
        # Good initialization: logit of prevalence (works even if scores ~ 0)
        p = y.mean(dim=0).clamp(self.eps, 1.0 - self.eps)  # (c,)
        b = torch.log(p / (1.0 - p))  # (c,)

        for _ in range(self.bce_newton_iters):
            logits = scores + b  # broadcast: (n, c)
            p_hat = torch.sigmoid(logits)  # (n, c)

            # Gradient and Hessian of mean BCE wrt b:
            # g = mean(p_hat - y)
            # h = mean(p_hat * (1 - p_hat))
            g = (p_hat - y).mean(dim=0)  # (c,)
            h = (p_hat * (1.0 - p_hat)).mean(dim=0).clamp_min(self.eps)  # (c,)

            step = g / h
            b_next = b - step

            if step.abs().max().item() < self.bce_newton_tol:
                b = b_next
                break
            b = b_next

        return b

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X : (n, d)
        y : (n, c) with values in {0, 1}
        """
        n, d = X.shape

        y = y.to(dtype=X.dtype)

        # Counts
        n1 = y.sum(dim=0)  # (c,)
        n0 = n - n1  # (c,)

        # Sums
        s1 = y.t() @ X  # (c, d)
        sumX = X.sum(dim=0)  # (d,)
        s0 = (n * sumX.unsqueeze(0)) - s1  # (c, d)

        # Means (avoid division by 0)
        mu1 = s1 / (n1.unsqueeze(1).clamp_min(self.eps))  # (c, d)
        mu0 = s0 / (n0.unsqueeze(1).clamp_min(self.eps))  # (c, d)

        w = (mu1 - mu0).t()  # (d, c)

        if self.bias_mode == "zero":
            b = torch.zeros(y.size(1), dtype=X.dtype, device=X.device)  # (c,)
        elif self.bias_mode == "midpoint":
            # midpoint / nearest-centroid bias
            mu1_sq = (mu1 * mu1).sum(dim=1)  # (c,)
            mu0_sq = (mu0 * mu0).sum(dim=1)  # (c,)
            b = -0.5 * (mu1_sq - mu0_sq)  # (c,)

        else:  # "bce"
            # scores = X @ w are fixed; find b that minimizes BCE per column
            with torch.no_grad():
                scores = X @ w  # (n, c)
                b = self._bce_optimal_bias(scores=scores, y=y)  # (c,)

        # If a class has no positives or no negatives, w/b are ill-defined.
        # Set them to 0 to avoid inf/nan, but keep shapes consistent.
        invalid = (n1 < 1) | (n0 < 1)  # (c,)
        if invalid.any():
            w = w.clone()
            b = b.clone()
            w[:, invalid] = 0.0
            b[invalid] = 0.0

        self.weight = nn.Parameter(w.clone())
        self.bias = nn.Parameter(b.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (n, d)
        Returns : (n, c)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        return X @ self.weight + self.bias  # type: ignore


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
