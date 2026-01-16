from __future__ import annotations

from abc import ABC, abstractmethod
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


class LinearProbeBase(nn.Module, ABC):
    """
    Base class for linear probes with intercept.

    Stores:
        weight: (d, c)
        bias:   (c,)

    encode(X) returns logits/scores: (n, c)
    """

    def __init__(self):
        super().__init__()
        self.weight = None  # nn.Parameter, (d, c)
        self.bias = None  # nn.Parameter, (c,)
        self.fitted = False

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

    def _init_params(self, d: int, c: int, *, dtype: torch.dtype, device: torch.device):
        self.weight = nn.Parameter(torch.zeros(d, c, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(c, dtype=dtype, device=device))

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return X @ self.weight + self.bias  # type: ignore


class LinearRegressionProbe(LinearProbeBase):
    """
    Multi-output linear regression probe with intercept.

    - If l2 == 0.0: uses OLS closed form via pseudo-inverse.
    - If l2  > 0.0: uses ridge regression closed form on augmented design,
      without penalizing the intercept term.
    """

    def __init__(self, l2: float = 0.0):
        super().__init__()
        self.l2 = float(l2)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Expected X and y to be 2D, got {X.shape=} {y.shape=}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n, d = X.shape
        c = y.shape[1]

        dtype = X.dtype
        device = X.device

        y = y.to(dtype=dtype, device=device)

        ones = torch.ones(n, 1, dtype=dtype, device=device)
        X_aug = torch.cat([X, ones], dim=1)  # (n, d+1)

        if self.l2 == 0.0:
            # OLS via pseudo-inverse for stability
            X_pinv = torch.linalg.pinv(X_aug)
            W_aug = X_pinv @ y  # (d+1, c)
        else:
            # Ridge closed form: (X^T X + l2*I)^{-1} X^T y
            # Do not penalize the intercept (last column of X_aug)
            XtX = X_aug.T @ X_aug  # (d+1, d+1)
            Xty = X_aug.T @ y  # (d+1, c)

            reg = torch.ones(d + 1, dtype=dtype, device=device)
            reg[-1] = 0.0  # no penalty on bias
            A = XtX + self.l2 * torch.diag(reg)
            W_aug = torch.linalg.solve(A, Xty)  # (d+1, c)

        self.weight = nn.Parameter(W_aug[:d, :].clone())  # (d, c)
        self.bias = nn.Parameter(W_aug[d, :].clone())  # (c,)
        self.fitted = True


class MeansDiffProbe(LinearProbeBase):
    """
    MeansDiff probe (multi-label, multi-output).

    For each concept j:
        w_j = mean(X | y_j=1) - mean(X | y_j=0)

    bias modes:
        - "zero":       b = 0
        - "prevalence": b_j = logit(mean(y_j))
        - "bce":        choose b_j to minimize BCE on logits with fixed w_j (Newton)
    """

    def __init__(
        self,
        bias: str = "zero",
        eps: float = 1e-8,
        bce_newton_iters: int = 50,
        bce_newton_tol: float = 1e-8,
    ):
        super().__init__()
        if bias not in {"zero", "prevalence", "bce"}:
            raise ValueError("bias must be one of {'zero','prevalence','bce'}")

        self.bias_mode = bias
        self.eps = float(eps)
        self.bce_newton_iters = int(bce_newton_iters)
        self.bce_newton_tol = float(bce_newton_tol)

    @torch.no_grad()
    def _bce_optimal_bias(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = y.mean(dim=0).clamp(self.eps, 1.0 - self.eps)  # (c,)
        b = torch.log(p / (1.0 - p))  # (c,)

        for _ in range(self.bce_newton_iters):
            logits = scores + b
            p_hat = torch.sigmoid(logits)

            g = (p_hat - y).mean(dim=0)
            h = (p_hat * (1.0 - p_hat)).mean(dim=0).clamp_min(self.eps)

            step = g / h
            b_next = b - step

            if step.abs().max().item() < self.bce_newton_tol:
                b = b_next
                break
            b = b_next

        return b

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Expected X and y to be 2D, got {X.shape=} {y.shape=}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n, d = X.shape
        y = y.to(dtype=X.dtype)
        c = y.shape[1]

        n1 = y.sum(dim=0)  # (c,)
        n0 = n - n1

        s1 = y.t() @ X
        sumX = X.sum(dim=0)
        s0 = (n * sumX.unsqueeze(0)) - s1

        mu1 = s1 / n1.unsqueeze(1).clamp_min(self.eps)
        mu0 = s0 / n0.unsqueeze(1).clamp_min(self.eps)

        w = (mu1 - mu0).t()  # (d, c)

        if self.bias_mode == "zero":
            b = torch.zeros(c, dtype=X.dtype, device=X.device)

        elif self.bias_mode == "prevalence":
            p = y.mean(dim=0).clamp(self.eps, 1.0 - self.eps)
            b = torch.log(p / (1.0 - p))

        else:  # "bce"
            scores = X @ w
            b = self._bce_optimal_bias(scores=scores, y=y)

        self.weight = nn.Parameter(w.clone())
        self.bias = nn.Parameter(b.clone())
        self.fitted = True


class _GDLinearProbe(LinearProbeBase):
    """
    Gradient-descent linear probe skeleton.

    Optional init:
        - init="zeros": standard zero init
        - means_diff_init: wether to initialize weight with MeansDiffProbe direction.
            init_bias:
                - "zero":      b = 0
                - "prevalence": b = logit(prevalence) (logistic-friendly)
    """

    def __init__(
        self,
        lr: float = 1e-2,
        max_iter: int = 20,
        l2: float = 0.0,
        *,
        means_diff_init: bool = False,
        init_bias: str = "zero",  # "zero" | "prevalence"
        init_eps: float = 1e-8,
    ):
        super().__init__()
        if init_bias not in {"zero", "prevalence"}:
            raise ValueError("init_bias must be one of {'zero','prevalence'}")

        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)

        self.means_diff_init = means_diff_init
        self.init_bias = init_bias
        self.init_eps = float(init_eps)

    @abstractmethod
    def _prepare_targets(self, y: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _init_from_means_diff(self, X: torch.Tensor, y01: torch.Tensor):
        md = MeansDiffProbe(bias=self.init_bias, eps=self.init_eps).to(device=X.device)

        # MeansDiff expects float {0,1}
        md.fit(X, y01.to(dtype=X.dtype))

        # Initialize parameters from MeansDiff without tracking gradients
        # to avoid in-place ops on leaf Variables that require grad.
        with torch.no_grad():
            self.weight.copy_(md.weight.detach())  # type: ignore
            self.bias.copy_(md.bias.detach())  # type: ignore

        del md

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Expected X and y to be 2D, got {X.shape=} {y.shape=}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n, d = X.shape
        c = y.shape[1]

        # initialize parameters
        self._init_params(d, c, dtype=X.dtype, device=X.device)
        if self.means_diff_init:
            # Ensure {0,1} for init
            y01 = (y.to(dtype=X.dtype) > 0.5).to(dtype=X.dtype)
            self._init_from_means_diff(X, y01)

        y_prepared = self._prepare_targets(y, dtype=X.dtype)

        optimizer = torch.optim.Adam([self.weight, self.bias], lr=self.lr)  # type: ignore

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            logits = X @ self.weight + self.bias  # type: ignore
            loss = self._loss(logits, y_prepared)

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * (self.weight**2).sum()  # type: ignore

            loss.backward()
            optimizer.step()
        self.fitted = True


class LogisticRegressionProbe(_GDLinearProbe):
    def __init__(
        self,
        lr: float = 1e-2,
        max_iter: int = 20,
        l2: float = 0.0,
        *,
        means_diff_init: bool = False,
        init_bias: str = "prevalence",
        init_eps: float = 1e-8,
    ):
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            l2=l2,
            means_diff_init=means_diff_init,
            init_bias=init_bias,
            init_eps=init_eps,
        )
        self._loss_fn = nn.BCEWithLogitsLoss()

    def _prepare_targets(self, y: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return y.to(dtype=dtype)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(logits, y)


class LinearSVMProbe(_GDLinearProbe):
    def __init__(
        self,
        lr: float = 1e-2,
        max_iter: int = 20,
        l2: float = 0.0,
        *,
        means_diff_init: bool = False,
        init_bias: str = "prevalence",
        init_eps: float = 1e-8,
    ):
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            l2=l2,
            means_diff_init=means_diff_init,
            init_bias=init_bias,
            init_eps=init_eps,
        )

    def _prepare_targets(self, y: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        y = y.to(dtype=dtype)
        return 2.0 * y - 1.0

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        margins = 1.0 - y * logits
        return torch.clamp(margins, min=0.0).mean()


class _CentroidBaseProbe(nn.Module):
    """
    Shared base for centroid-based probes (multi-label, multi-output).

    - Provides dataset-level normalization utilities shared with Gaussian probes:
        normalization in {"none", "zscore", "standardization", "whitening", "lowrank_whitening",
        "diagonal_whitening", "diag_whitening"}.
        Aliases map to: standardization/diagonal_whitening/diag_whitening -> zscore.

    Stores:
        - centroids: (c, d_norm) as nn.Parameter
        - normalization buffers:
            x_mean (d,), x_std (d,) for z-score
            whiten_V (d, r), whiten_inv_s (r,) for whitening
    """

    def __init__(
        self,
        normalization: str = "none",
        eps: float = 1e-8,
        lowrank_rank: int | None = None,
    ):
        super().__init__()
        # Normalize aliases
        alias = {
            "standardization": "zscore",
            "diagonal_whitening": "zscore",
            "diag_whitening": "zscore",
        }
        normalization = alias.get(normalization, normalization)
        if normalization not in {"none", "zscore", "whitening", "lowrank_whitening"}:
            raise ValueError("normalization must be one of {'none','zscore','whitening','lowrank_whitening'}")
        if normalization == "lowrank_whitening" and (lowrank_rank is None or lowrank_rank <= 0):
            raise ValueError("lowrank_rank must be a positive int for lowrank_whitening")

        self.normalization = normalization
        self.lowrank_rank = None if lowrank_rank is None else int(lowrank_rank)
        self.eps = float(eps)

        self.centroids = None  # nn.Parameter, (c, d_norm)
        self.fitted = False

        # Normalization buffers
        self.register_buffer("x_mean", torch.empty(0))  # (d,)
        self.register_buffer("x_std", torch.empty(0))  # (d,)
        self.register_buffer("whiten_V", torch.empty(0))  # (d, r)
        self.register_buffer("whiten_inv_s", torch.empty(0))  # (r,)

    def _fit_norm_stats(self, X: torch.Tensor):
        if self.normalization == "zscore":
            self.x_mean = X.mean(dim=0).detach()
            self.x_std = X.std(dim=0, unbiased=False).clamp_min(self.eps).detach()
            # clear whitening buffers
            self.whiten_V = torch.empty(0, device=X.device, dtype=X.dtype)
            self.whiten_inv_s = torch.empty(0, device=X.device, dtype=X.dtype)
        elif self.normalization in {"whitening", "lowrank_whitening"}:
            n, d = X.shape
            mean = X.mean(dim=0)
            self.x_mean = mean.detach()
            # Center
            Xc = X - mean.unsqueeze(0)
            # Economic SVD: Xc = U S V^T
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            # whitening scale  ~ sqrt(n) / S
            inv_s = (torch.sqrt(torch.tensor(float(n), device=X.device, dtype=X.dtype)) / S).clamp_max(1.0 / self.eps)
            if self.normalization == "whitening":
                V = Vh.transpose(0, 1)  # (d, r)
            else:
                r = min(self.lowrank_rank or 0, Vh.shape[0])
                V = Vh[:r, :].transpose(0, 1)  # (d, r)
                inv_s = inv_s[:r]
            self.whiten_V = V.detach().clone()
            self.whiten_inv_s = inv_s.detach().clone()
            # clear zscore std
            self.x_std = torch.empty(0, device=X.device, dtype=X.dtype)
        else:  # "none"
            # clear all buffers
            self.x_mean = torch.empty(0, device=X.device, dtype=X.dtype)
            self.x_std = torch.empty(0, device=X.device, dtype=X.dtype)
            self.whiten_V = torch.empty(0, device=X.device, dtype=X.dtype)
            self.whiten_inv_s = torch.empty(0, device=X.device, dtype=X.dtype)

    def _apply_norm(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalization == "zscore":
            if self.x_mean.numel() == 0 or self.x_std.numel() == 0:
                raise RuntimeError("z-score stats not initialized. Call fit first.")
            return (X - self.x_mean) / self.x_std
        elif self.normalization in {"whitening", "lowrank_whitening"}:
            if self.x_mean.numel() == 0 or self.whiten_V.numel() == 0 or self.whiten_inv_s.numel() == 0:
                raise RuntimeError("whitening stats not initialized. Call fit first.")
            Xc = X - self.x_mean
            Z = Xc @ self.whiten_V
            return Z * self.whiten_inv_s.unsqueeze(0)
        else:
            return X

    def _fit_centroids(self, Xn: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        # E[x^2|pos] - (E[x|pos])^2
        sum_pos_sq = y.t() @ (Xn * Xn)  # (c,d)
        ex2 = sum_pos_sq / n_pos.unsqueeze(1).clamp_min(self.eps)
        var_pos = (ex2 - centroids * centroids).clamp_min(self.eps)
        inv_var = (1.0 / var_pos).clamp_max(1.0 / self.eps)

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


class EllipsoidalBoundaryProbe(_CentroidBaseProbe):
    """
    Multi-label ellipsoidal boundary probe (one ellipsoid per concept).

    For each concept j, fit a Gaussian-like ellipsoid on positives:
        mu_j  = mean(X | y_j=1)
        var_j = var(X | y_j=1)   (diagonal; with shrinkage for stability)

    Encode returns a score (n, c):
        score_ij = - sum_k ((x_ik - mu_jk)^2 / var_jk)
    Larger score => closer to the positive ellipsoid center.
    """

    def __init__(
        self,
        normalization: str = "none",
        eps: float = 1e-8,
        var_shrink: float = 0.0,
        lowrank_rank: int | None = None,
    ):
        """
        eps: numerical stability for division/clamping.
        var_shrink: shrinkage toward global variance:
            var_j <- (1-var_shrink)*var_pos_j + var_shrink*var_global
        """
        super().__init__(normalization=normalization, eps=eps, lowrank_rank=lowrank_rank)
        self.var_shrink = float(var_shrink)

        # Use base centroids container for centers
        # self.centroids: nn.Parameter, shape (c, d)
        self.inv_var = None  # nn.Parameter, shape (c, d)  (diagonal precision)
        self.fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X : (n, d)
        y : (n, c) in {0,1}
        """
        # Normalize features if requested
        y = y.to(dtype=X.dtype)
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)

        n, d = Xn.shape
        c = y.size(1)

        n_pos = y.sum(dim=0)  # (c,)

        # Positive means for centers using shared helper
        mu, _ = self._fit_centroids(Xn, y)  # (c, d)

        # Compute diag var on positives in normalized space:
        # E[x^2|pos] - (E[x|pos])^2
        sum_pos_sq = y.t() @ (Xn * Xn)  # (c, d)
        ex2 = sum_pos_sq / n_pos.unsqueeze(1).clamp_min(self.eps)  # (c, d)
        var_pos = (ex2 - mu * mu).clamp_min(self.eps)  # (c, d)

        if self.var_shrink > 0.0:
            var_global = Xn.var(dim=0, unbiased=False).clamp_min(self.eps)  # (d,)
            var_pos = (1.0 - self.var_shrink) * var_pos + self.var_shrink * var_global.unsqueeze(0)

        inv_var = (1.0 / var_pos).clamp_max(1.0 / self.eps)  # (c, d)

        # Store centers in base centroids parameter for consistency
        self.centroids = nn.Parameter(mu.clone())
        self.inv_var = nn.Parameter(inv_var.clone())
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (n, d)
        Returns : (n, c) scores (higher => closer to positive ellipsoid)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

        Xn = self._apply_norm(X)
        mu: torch.Tensor = self.centroids  # type: ignore
        inv_var: torch.Tensor = self.inv_var  # type: ignore

        # diff: (n, c, d)
        diff = Xn.unsqueeze(1) - mu.unsqueeze(0)
        # mahalanobis_diag: (n, c)
        mahalanobis_diag = (diff * diff * inv_var.unsqueeze(0)).sum(dim=2)
        return -mahalanobis_diag


class SVDDProbe(_CentroidBaseProbe):
    """
    Multi-label SVDD (Support Vector Data Description), one hyper-sphere per concept.

    For each concept j, fit a center a_j and radius r_j on positives by minimizing:
        L_j = r_j^2 + C * mean_pos( relu(||x - a_j||^2 - r_j^2) ) + 0.5*l2*||a_j||^2

    Encode returns a margin score (n, c):
        score_ij = r_j^2 - ||x_i - a_j||^2
    Larger score => more inside the sphere (positive).
    """

    def __init__(
        self,
        lr: float = 5e-2,
        max_iter: int = 2000,
        C: float = 1.0,
        l2: float = 0.0,
        normalization: str = "none",
        eps: float = 1e-8,
        lowrank_rank: int | None = None,
    ):
        super().__init__(normalization=normalization, eps=eps, lowrank_rank=lowrank_rank)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.C = float(C)
        self.l2 = float(l2)

        # Use base centroids container for centers
        # self.centroids: nn.Parameter, shape (c, d)
        self._log_radius = None  # nn.Parameter, shape (c,)   (radius = softplus(log)+eps)
        self.fitted = False

    def _radius(self) -> torch.Tensor:
        log_r: torch.Tensor = self._log_radius  # type: ignore
        return torch.nn.functional.softplus(log_r) + self.eps

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X : (n, d)
        y : (n, c) in {0,1}
        """
        # Work in normalized space
        y = y.to(dtype=X.dtype)
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)
        n, d = Xn.shape
        c = y.size(1)

        # Initialize center at positive mean (fallback 0 if no positives)
        n_pos = y.sum(dim=0)  # (c,)
        mu_pos, _ = self._fit_centroids(Xn, y)  # (c, d)

        if self.centroids is None or self._log_radius is None:
            # Initialize learnable centers at class-positive means
            self.centroids = nn.Parameter(mu_pos.clone())
            # init radius from median pos distance (approx), else 1.0
            with torch.no_grad():
                # dist2 for init (n,c)
                diff = Xn.unsqueeze(1) - mu_pos.unsqueeze(0)
                dist2 = (diff * diff).sum(dim=2)
                # masked mean dist2 as a rough scale
                denom = n_pos.clamp_min(1.0).unsqueeze(0)
                mean_dist2 = (dist2 * y).sum(dim=0) / denom.squeeze(0)
                r0 = torch.sqrt(mean_dist2.clamp_min(self.eps))
                # inverse softplus approx: softplus(z)=r -> z ~ log(exp(r)-1)
                log_r0 = torch.log(torch.expm1(r0.clamp_min(self.eps)))
            self._log_radius = nn.Parameter(log_r0.clone())

        optimizer = torch.optim.Adam([self.centroids, self._log_radius], lr=self.lr)  # type: ignore

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            r = self._radius()  # (c,)
            r2 = r * r  # (c,)

            # dist2: (n, c)
            diff = Xn.unsqueeze(1) - self.centroids.unsqueeze(0)  # (n,c,d)  # type: ignore
            dist2 = (diff * diff).sum(dim=2)  # (n,c)

            # hinge on positives only: relu(dist2 - r2)
            hinge = torch.relu(dist2 - r2.unsqueeze(0))  # (n,c)
            # mean over positives per concept
            denom = n_pos.clamp_min(1.0)  # (c,)
            hinge_mean = (hinge * y).sum(dim=0) / denom  # (c,)

            loss = r2 + self.C * hinge_mean  # (c,)
            loss = loss.mean()

            if self.l2 > 0.0:
                Ctr: torch.Tensor = self.centroids  # type: ignore
                loss = loss + 0.5 * self.l2 * (Ctr * Ctr).sum()

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

        Xn = self._apply_norm(X)
        diff = Xn.unsqueeze(1) - self.centroids.unsqueeze(0)  # (n,c,d)  # type: ignore
        dist2 = (diff * diff).sum(dim=2)  # (n,c)
        return r2.unsqueeze(0) - dist2


class GaussianLikelihoodProbe(_CentroidBaseProbe):
    """
    Multi-label Gaussian likelihood (QDA-style, diagonal covariance).

    - Normalization learned on the entire dataset (negatives approximated as N(0, I)).
      normalization in {"standardization", "whitening", "lowrank_whitening"}.
    - For each concept j, fit positive-class diagonal Gaussian in normalized space:
        mu_j, var_j (diagonal) from positives only.
    - Score is proportional to the positive log-likelihood (constant terms dropped):
        s_ij = -0.5 * [ (x-\mu_j)^T diag(1/var_j) (x-\mu_j) + log_det(var_j) ]
      This approximates the log-likelihood ratio vs N(0, I), up to a sample-dependent
      constant independent of the concept.
    """

    def __init__(
        self,
        normalization: str = "standardization",
        lowrank_rank: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(normalization=normalization, eps=eps, lowrank_rank=lowrank_rank)

        # Positive class stats (in normalized space)
        self.mu = None  # nn.Parameter, (c, d_norm)
        self.inv_var = None  # nn.Parameter, (c, d_norm)
        self.register_buffer("log_det", torch.empty(0))  # (c,)

        self.fitted = False

    # normalization utilities are inherited from _CentroidBaseProbe

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: (n, d) features
        y: (n, c) in {0,1}, multi-label
        """

        # Normalization on the whole dataset
        self._fit_norm_stats(X)
        Xn = self._apply_norm(X)

        y = y.to(dtype=X.dtype)
        n, d_norm = Xn.shape
        c = y.shape[1]

        n_pos = y.sum(dim=0)  # (c,)
        # Positive means (reuse centroid helper)
        mu, _ = self._fit_centroids(Xn, y)  # (c, d_norm)

        sum_pos_sq = y.t() @ (Xn * Xn)  # (c, d_norm)
        ex2 = sum_pos_sq / n_pos.unsqueeze(1).clamp_min(self.eps)
        var_pos = (ex2 - mu * mu).clamp_min(self.eps)  # (c, d_norm)

        inv_var = (1.0 / var_pos).clamp_max(1.0 / self.eps)

        log_det = torch.log(var_pos.clamp_min(self.eps)).sum(dim=1)  # (c,)

        self.mu = nn.Parameter(mu.clone())  # (c, d_norm)
        self.inv_var = nn.Parameter(inv_var.clone())  # (c, d_norm)
        self.log_det = log_det.detach().clone()
        self.fitted = True

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")

        Xn = self._apply_norm(X)
        mu: torch.Tensor = self.mu  # type: ignore
        inv_var: torch.Tensor = self.inv_var  # type: ignore
        log_det = self.log_det.to(dtype=Xn.dtype, device=Xn.device)

        # Pre-compute terms
        X2 = Xn * Xn  # (n, d)
        x2_inv = X2 @ inv_var.t()  # (n, c)
        q = mu * inv_var  # (c, d)
        x_q = Xn @ q.t()  # (n, c)
        mu2_inv = (mu * mu * inv_var).sum(dim=1).unsqueeze(0)  # (1, c)

        # score = -0.5 * [ x^T P x - 2 mu^T P x + mu^T P mu + log_det ]
        scores = -0.5 * (x2_inv - 2.0 * x_q + mu2_inv + log_det.unsqueeze(0))
        return scores


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
            raise ValueError(f"Expected activations to be a 2D array, (n, d), got shape {split_activations.shape}")
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
