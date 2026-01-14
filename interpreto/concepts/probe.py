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


class _CentroidBaseProbe(nn.Module):
    """
    Shared base for centroid-based probes (multi-label, multi-output).

    Stores:
        - centroids: (c, d) as nn.Parameter
        - optional global z-score buffers: x_mean (d,), x_std (d,)
    """

    def __init__(self, normalization: str = "none", eps: float = 1e-8):
        super().__init__()
        if normalization not in {"none", "zscore"}:
            raise ValueError("normalization must be one of {'none','zscore'}")
        self.normalization = normalization
        self.eps = float(eps)

        self.centroids = None  # nn.Parameter, (c, d)
        self.fitted = False

        self.register_buffer("x_mean", torch.empty(0))  # (d,)
        self.register_buffer("x_std", torch.empty(0))  # (d,)

    def _fit_norm_stats(self, X: torch.Tensor):
        if self.normalization == "zscore":
            self.x_mean = X.mean(dim=0).detach()
            self.std = X.std(dim=0, unbiased=False).clamp_min(self.eps).detach()
        else:
            self.x_mean = torch.empty(0, device=X.device, dtype=X.dtype)
            self.x_std = torch.empty(0, device=X.device, dtype=X.dtype)

    def _apply_norm(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalization == "zscore":
            if self.x_mean.numel() == 0 or self.x_std.numel() == 0:
                raise RuntimeError("z-score stats not initialized. Call fit first.")
            return (X - self.x_mean) / self.x_std
        return X

    def _fit_centroids(
        self, Xn: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Xn: (n,d) normalized
        y:  (n,c) float
        Returns:
            centroids: (c,d)
            n_pos:     (c,)
        """
        n_pos = y.sum(dim=0)  # (c,)
        sum_pos = y.t() @ Xn  # (c,d)
        centroids = sum_pos / n_pos.unsqueeze(1).clamp_min(self.eps)

        invalid = n_pos < 1
        if invalid.any():
            centroids = centroids.clone()
            centroids[invalid] = 0.0

        return centroids, n_pos


class CentroidDotProbe(_CentroidBaseProbe):
    """
    score = x · centroid
    """

    def __init__(self, normalization: str = "none", eps: float = 1e-8):
        super().__init__(normalization=normalization, eps=eps)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        y = y.to(dtype=X.dtype)
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)

        centroids, _ = self._fit_centroids(Xn, y)
        self.centroids = nn.Parameter(centroids.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        Xn = self._apply_norm(X)
        C: torch.Tensor = self.centroids  # type: ignore
        return Xn @ C.t()


class CentroidCosineProbe(_CentroidBaseProbe):
    """
    score = cosine(x, centroid)

    Normalization behavior:
        - If normalization="zscore": z-score first, then unit-normalize in encode and fit.
        - If normalization="none": unit-normalize in encode and fit.
    """

    def __init__(self, normalization: str = "none", eps: float = 1e-8):
        super().__init__(normalization=normalization, eps=eps)

    def _unit(self, X: torch.Tensor) -> torch.Tensor:
        return X / X.norm(dim=1, keepdim=True).clamp_min(self.eps)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        y = y.to(dtype=X.dtype)
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)
        Xn = self._unit(Xn)

        centroids, _ = self._fit_centroids(Xn, y)
        centroids = self._unit(centroids)  # unit-normalize centroids row-wise
        self.centroids = nn.Parameter(centroids.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        Xn = self._apply_norm(X)
        Xn = self._unit(Xn)
        C: torch.Tensor = self.centroids  # type: ignore  # already unit
        return Xn @ C.t()


class CentroidSqL2Probe(_CentroidBaseProbe):
    """
    score = -||x - centroid||^2   (computed efficiently)

    Recommended normalization: "zscore" (optional).
    """

    def __init__(self, normalization: str = "none", eps: float = 1e-8):
        super().__init__(normalization=normalization, eps=eps)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        y = y.to(dtype=X.dtype)
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)

        centroids, _ = self._fit_centroids(Xn, y)
        self.centroids = nn.Parameter(centroids.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        Xn = self._apply_norm(X)
        C: torch.Tensor = self.centroids  # type: ignore  # (c,d)

        x2 = (Xn * Xn).sum(dim=1, keepdim=True)  # (n,1)
        c2 = (C * C).sum(dim=1, keepdim=True).t()  # (1,c)
        dots = Xn @ C.t()  # (n,c)
        dist2 = x2 + c2 - 2.0 * dots
        return -dist2


class CentroidMahalanobisCommonVarProbe(_CentroidBaseProbe):
    """
    score = -(x-c)^T diag(inv_var_common) (x-c)
    where inv_var_common is a single (d,) diagonal precision estimated from all samples
    in the normalized space.
    """

    def __init__(self, normalization: str = "zscore", eps: float = 1e-8):
        super().__init__(normalization=normalization, eps=eps)
        self.register_buffer("inv_var_common", torch.empty(0))  # (d,)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        y = y.to(dtype=X.dtype)
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)

        centroids, _ = self._fit_centroids(Xn, y)

        var = Xn.var(dim=0, unbiased=False).clamp_min(self.eps)  # (d,)
        self.inv_var_common = (1.0 / var).detach()

        self.centroids = nn.Parameter(centroids.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        if self.inv_var_common.numel() == 0:
            raise RuntimeError("inv_var_common not initialized. Call fit first.")

        Xn = self._apply_norm(X)
        C: torch.Tensor = self.centroids  # type: ignore

        inv = self.inv_var_common.to(dtype=Xn.dtype, device=Xn.device)  # (d,)
        diff = Xn.unsqueeze(1) - C.unsqueeze(0)  # (n,c,d)
        dist2 = (diff * diff * inv.view(1, 1, -1)).sum(dim=2)  # (n,c)
        return -dist2


class CentroidMahalanobisClasswiseVarProbe(_CentroidBaseProbe):
    """
    score = -(x-c)^T diag(inv_var_j) (x-c)
    where inv_var_j is per-class diagonal precision (c,d) estimated from positives only
    in the normalized space.
    """

    def __init__(self, normalization: str = "zscore", eps: float = 1e-8):
        super().__init__(normalization=normalization, eps=eps)
        self.register_buffer("inv_var_classwise", torch.empty(0))  # (c,d)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        y = y.to(dtype=X.dtype)
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)

        centroids, n_pos = self._fit_centroids(Xn, y)
        invalid = n_pos < 1

        # E[x^2|pos] - (E[x|pos])^2
        sum_pos_sq = y.t() @ (Xn * Xn)  # (c,d)
        ex2 = sum_pos_sq / n_pos.unsqueeze(1).clamp_min(self.eps)
        var_pos = (ex2 - centroids * centroids).clamp_min(self.eps)
        inv_var = (1.0 / var_pos).clamp_max(1.0 / self.eps)

        if invalid.any():
            inv_var = inv_var.clone()
            inv_var[invalid] = 0.0

        self.inv_var_classwise = inv_var.detach()

        self.centroids = nn.Parameter(centroids.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        if self.inv_var_classwise.numel() == 0:
            raise RuntimeError("inv_var_classwise not initialized. Call fit first.")

        Xn = self._apply_norm(X)
        C: torch.Tensor = self.centroids  # type: ignore
        inv = self.inv_var_classwise.to(dtype=Xn.dtype, device=Xn.device)  # (c,d)

        diff = Xn.unsqueeze(1) - C.unsqueeze(0)  # (n,c,d)
        dist2 = (diff * diff * inv.unsqueeze(0)).sum(dim=2)  # (n,c)
        return -dist2


class EllipsoidalBoundaryProbe(nn.Module):
    """
    Multi-label ellipsoidal boundary probe (one ellipsoid per concept).

    For each concept j, fit a Gaussian-like ellipsoid on positives:
        mu_j  = mean(X | y_j=1)
        var_j = var(X | y_j=1)   (diagonal; with shrinkage for stability)

    Encode returns a score (n, c):
        score_ij = - sum_k ((x_ik - mu_jk)^2 / var_jk)
    Larger score => closer to the positive ellipsoid center.
    """

    def __init__(self, eps: float = 1e-8, var_shrink: float = 0.0):
        """
        eps: numerical stability for division/clamping.
        var_shrink: shrinkage toward global variance:
            var_j <- (1-var_shrink)*var_pos_j + var_shrink*var_global
        """
        super().__init__()
        self.eps = float(eps)
        self.var_shrink = float(var_shrink)

        self.mu = None  # nn.Parameter, shape (c, d)
        self.inv_var = None  # nn.Parameter, shape (c, d)  (diagonal precision)
        self.fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X : (n, d)
        y : (n, c) in {0,1}
        """
        n, d = X.shape
        y = y.to(dtype=X.dtype)  # (n, c)
        c = y.size(1)

        n_pos = y.sum(dim=0)  # (c,)

        # Positive sums for mean: (c,d)
        sum_pos = y.t() @ X
        mu = sum_pos / n_pos.unsqueeze(1).clamp_min(self.eps)  # (c, d)

        # Compute diag var on positives:
        # E[x^2|pos] - (E[x|pos])^2
        sum_pos_sq = y.t() @ (X * X)  # (c, d)
        ex2 = sum_pos_sq / n_pos.unsqueeze(1).clamp_min(self.eps)  # (c, d)
        var_pos = (ex2 - mu * mu).clamp_min(self.eps)  # (c, d)

        if self.var_shrink > 0.0:
            var_global = X.var(dim=0, unbiased=False).clamp_min(self.eps)  # (d,)
            var_pos = (
                1.0 - self.var_shrink
            ) * var_pos + self.var_shrink * var_global.unsqueeze(0)

        inv_var = (1.0 / var_pos).clamp_max(1.0 / self.eps)  # (c, d)

        self.mu = nn.Parameter(mu.clone())
        self.inv_var = nn.Parameter(inv_var.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (n, d)
        Returns : (n, c) scores (higher => closer to positive ellipsoid)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

        mu: torch.Tensor = self.mu  # type: ignore
        inv_var: torch.Tensor = self.inv_var  # type: ignore

        # diff: (n, c, d)
        diff = X.unsqueeze(1) - mu.unsqueeze(0)
        # mahal_diag: (n, c)
        mahal_diag = (diff * diff * inv_var.unsqueeze(0)).sum(dim=2)
        return -mahal_diag


class SVDDProbe(nn.Module):
    """
    Multi-label SVDD (Support Vector Data Description), one hypersphere per concept.

    For each concept j, fit a center a_j and radius r_j on positives by minimizing:
        L_j = r_j^2 + C * mean_pos( relu(||x - a_j||^2 - r_j^2) ) + 0.5*l2*||a_j||^2

    Encode returns a margin score (n, c):
        score_ij = r_j^2 - ||x_i - a_j||^2
    Larger score => more inside the sphere (positive).

    Matches the probe API in probe.py (fit/encode, self.fitted). :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        lr: float = 5e-2,
        max_iter: int = 2000,
        C: float = 1.0,
        l2: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.C = float(C)
        self.l2 = float(l2)
        self.eps = float(eps)

        self.center = None  # nn.Parameter, shape (c, d)
        self._log_radius = (
            None  # nn.Parameter, shape (c,)   (radius = softplus(log)+eps)
        )
        self.fitted = False

    def _radius(self) -> torch.Tensor:
        log_r: torch.Tensor = self._log_radius  # type: ignore
        return torch.nn.functional.softplus(log_r) + self.eps

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X : (n, d)
        y : (n, c) in {0,1}
        """
        n, d = X.shape
        y = y.to(dtype=X.dtype)  # (n, c)
        c = y.size(1)

        # Initialize center at positive mean (fallback 0 if no positives)
        n_pos = y.sum(dim=0)  # (c,)
        sum_pos = y.t() @ X  # (c, d)
        mu_pos = sum_pos / n_pos.unsqueeze(1).clamp_min(self.eps)  # (c, d)

        if self.center is None or self._log_radius is None:
            self.center = nn.Parameter(mu_pos.clone())
            with torch.no_grad():
                # dist2 for init (n,c)
                diff = X.unsqueeze(1) - mu_pos.unsqueeze(0)
                dist2 = (diff * diff).sum(dim=2)
                # masked mean dist2 as a rough scale
                denom = n_pos.clamp_min(1.0).unsqueeze(0)
                mean_dist2 = (dist2 * y).sum(dim=0) / denom.squeeze(0)
                r0 = torch.sqrt(mean_dist2.clamp_min(self.eps))
                # inverse softplus approx: softplus(z)=r -> z ~ log(exp(r)-1)
                log_r0 = torch.log(torch.expm1(r0.clamp_min(self.eps)))
            self._log_radius = nn.Parameter(log_r0.clone())

        optimizer = torch.optim.Adam([self.center, self._log_radius], lr=self.lr)

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            r = self._radius()  # (c,)
            r2 = r * r  # (c,)

            # dist2: (n, c)
            diff = X.unsqueeze(1) - self.center.unsqueeze(0)  # (n,c,d)
            dist2 = (diff * diff).sum(dim=2)  # (n,c)

            # hinge on positives only: relu(dist2 - r2)
            hinge = torch.relu(dist2 - r2.unsqueeze(0))  # (n,c)
            # mean over positives per concept
            denom = n_pos.clamp_min(1.0)  # (c,)
            hinge_mean = (hinge * y).sum(dim=0) / denom  # (c,)

            loss = r2 + self.C * hinge_mean  # (c,)
            loss = loss.mean()

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * (self.center * self.center).sum()

            loss.backward()
            optimizer.step()

        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (n, d)
        Returns : (n, c) margins (higher => more inside SVDD sphere)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

        r = self._radius()  # (c,)
        r2 = r * r

        diff = X.unsqueeze(1) - self.center.unsqueeze(0)  # (n,c,d)  # type: ignore
        dist2 = (diff * diff).sum(dim=2)  # (n,c)
        return r2.unsqueeze(0) - dist2


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
