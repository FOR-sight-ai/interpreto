from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from interpreto.concepts.probes.linear_probe_models import ProbeModelInterface, assert_fitted
from interpreto.concepts.probes.bias_calibrators import BiasCalibrator
from interpreto.concepts.probes.normalizations import NormalizationBase, Standardization


class _BaseCentroidProbe(ProbeModelInterface):
    """
    Shared base for centroid-based probes (multi-label, multi-output).

    Expects a normalization module with:
      - fit(X) -> self
      - forward(X) -> normalized X
    """

    def __init__(
        self,
        normalization: NormalizationBase | None = None,
        bias_calibrator: BiasCalibrator | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)
        self.normalization = normalization
        self.bias_calibrator = bias_calibrator
        self.register_buffer("centroids", torch.empty(0))  # (c, d)
        self.register_buffer("bias", torch.empty(0))  # (c,)
        self.fitted = False

    @torch.no_grad()
    def _fit_centroids(self, Xn: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_pos = y.sum(dim=0)  # (c,)
        sum_pos = y.transpose(0, 1) @ Xn  # (c,d)
        self.centroids = sum_pos / n_pos.unsqueeze(1).clamp_min(self.eps)
        return n_pos

    @abstractmethod
    def _compute_scores(self, Xn: torch.Tensor) -> torch.Tensor:
        """Compute raw scores from normalized input (without bias)."""
        raise NotImplementedError

    def _calibrate_bias(self, Xn: torch.Tensor, y: torch.Tensor):
        """Calibrate bias using the bias_calibrator function or set to zero."""
        c = y.shape[1]
        if self.bias_calibrator is not None:
            scores = self._compute_scores(Xn)
            self.bias = self.bias_calibrator(scores, y)
        else:
            self.bias = torch.zeros(c, dtype=Xn.dtype, device=Xn.device)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        Xn = self.normalization.fit_transform(X) if self.normalization else X
        self._fit_centroids(Xn, y)
        self._calibrate_bias(Xn, y)
        self.fitted = True

    @assert_fitted
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        Xn = self.normalization(X) if self.normalization else X
        scores = self._compute_scores(Xn)
        return scores + self.bias


class DotProductCentroidProbe(_BaseCentroidProbe):
    """
    score = x · centroid + bias
    """

    def _compute_scores(self, Xn: torch.Tensor) -> torch.Tensor:
        return Xn @ self.centroids.t()


class CosineCentroidProbe(_BaseCentroidProbe):
    """
    score = cosine(x, centroid) + bias
    """

    def _unit(self, X: torch.Tensor) -> torch.Tensor:
        return X / X.norm(dim=1, keepdim=True).clamp_min(self.eps)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        Xn = self.normalization.fit_transform(X) if self.normalization else X
        self._fit_centroids(Xn, y)
        self.centroids = self._unit(self.centroids)
        self._calibrate_bias(Xn, y)
        self.fitted = True

    def _compute_scores(self, Xn: torch.Tensor) -> torch.Tensor:
        Xn = self._unit(Xn)
        return Xn @ self.centroids.t()


class SqL2CentroidProbe(_BaseCentroidProbe):
    """
    score = -||x - centroid||^2 + bias  (computed efficiently)
          = - [(x·x) + (c·c) - 2(x·c)] + bias

    Recommended normalization: Standardization (optional).
    """

    def _compute_scores(self, Xn: torch.Tensor) -> torch.Tensor:
        C = self.centroids  # (c, d)

        x2 = (Xn * Xn).sum(dim=1, keepdim=True)  # (n, 1)
        c2 = (C * C).sum(dim=1, keepdim=True).t()  # (1, c)
        dots = Xn @ C.t()  # (n, c)
        dist2 = x2 + c2 - 2.0 * dots
        return -dist2


class DiagonalMahalanobisCentroidProbe(_BaseCentroidProbe):
    """
    score = -(x-c)^T diag(1 / var) (x-c) + bias

    depending on shrinkage values, var can be:
        - shrinkage = 0: a class-wise diagonal variance (c,d) estimated from positives only
        - 0 < shrinkage < 1: a convex combination of class-wise and common variance, a regularized version
        - shrinkage = 1: a common diagonal variance (d,) estimated from all samples
    depending on the shrinkage strategy.
    """

    def __init__(
        self,
        normalization: NormalizationBase | None = None,
        bias_calibrator: BiasCalibrator | None = None,
        eps: float = 1e-8,
        shrinkage: float = 1.0,
    ):
        normalization = normalization or Standardization()  # default normalization
        super().__init__(normalization=normalization, bias_calibrator=bias_calibrator, eps=eps)
        assert 0.0 <= shrinkage <= 1.0, "shrinkage must be in [0.0, 1.0]"
        self.shrinkage = float(shrinkage)
        self.register_buffer("inv_var", torch.empty(0))  # (c, d) or (1, d)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # copied base init to prevent recomputing Xn
        Xn = self.normalization.fit_transform(X) if self.normalization else X
        n_pos = self._fit_centroids(Xn, y)

        # var = (1-shrinkage)*classwise_var + shrinkage*global_var
        var = torch.zeros_like(self.centroids)  # (c, d)

        # global diagonal variance
        if self.shrinkage > 0.0:
            global_var = Xn.var(dim=0, unbiased=False)  # (d,)
            var += self.shrinkage * global_var.unsqueeze(0)

        # class-wise variance
        if self.shrinkage < 1.0:
            # class-wise diagonal variance on positives
            # E[x^2|pos] - E[x|pos]^2
            sum_pos_sq = y.t() @ (Xn * Xn)  # (c, d)
            ex2 = sum_pos_sq / n_pos.unsqueeze(1)
            classwise_var = ex2 - self.centroids * self.centroids  # (c, d)

            var += (1.0 - self.shrinkage) * classwise_var

        # compute inverse common variance
        self.inv_var = (1.0 / var.clamp_min(self.eps)).detach()  # (c, d)

        self._calibrate_bias(Xn, y)
        self.fitted = True

    def _compute_scores(self, Xn: torch.Tensor) -> torch.Tensor:
        # - (x - c)² * inv_var
        diff = Xn.unsqueeze(1) - self.centroids.unsqueeze(0)  # (n, c, d)
        dist2 = (diff * diff * self.inv_var.unsqueeze(0)).sum(dim=2)  # (n, c)
        return -dist2


class SVDDCentroidProbe(_BaseCentroidProbe):
    """
    Multi-label SVDD (Support Vector Data Description), one hyper-sphere per concept.

    For each concept j, fit a center a_j and radius r_j on positives by minimizing:
        L_j = r_j^2 + C * mean_pos( relu(||x - a_j||^2 - r_j^2) ) + 0.5*l2*||a_j||^2

    Encode returns a margin score (n, c):
        score_ij = r_j^2 - ||x_i - a_j||^2
    Larger score => more inside the sphere (positive).

    loss: hinge on positives only: r2 + C * relu(dist2 - r2)
    """

    def __init__(
        self,
        lr: float = 5e-2,
        max_iter: int = 2000,
        C: float = 1.0,
        l2: float = 0.0,
        normalization: NormalizationBase | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(normalization=normalization, eps=eps)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.C = float(C)
        self.l2 = float(l2)

        # Use base centroids container for centers
        # self.centroids: nn.Parameter, shape (c, d)
        self._log_radius = None  # nn.Parameter, shape (c,)   (radius = softplus(log)+eps)
        self.fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X : (n, d)
        y : (n, c) in {0,1}
        """
        # Initialize centroids from positive means
        Xn = self.normalization.fit_transform(X) if self.normalization else X if self.normalization else X
        n_pos = self._fit_centroids(Xn, y)  # (c,)
        mu_pos = self.centroids  # (c, d)
        del self.centroids  # pass from buffer to parameter
        self.centroids = nn.Parameter(mu_pos)

        # Initialize radius from median pos distance (approx), else 1.0
        with torch.no_grad():
            # dist2 for init (n,c)
            diff = Xn.unsqueeze(1) - mu_pos.unsqueeze(0)  # (n,c,d)
            dist2 = (diff * diff).sum(dim=2)  # (n,c)
            # masked mean dist2 as a rough scale
            mean_dist2 = (dist2 * y).sum(dim=0) / n_pos.clamp_min(1.0)  # (c,)
            r0 = torch.sqrt(mean_dist2.clamp_min(self.eps))  # (c,)
            # inverse softplus approx: softplus(z)=r -> z ~ log(exp(r)-1)
            log_r0 = torch.log(torch.expm1(r0))
        self._log_radius = nn.Parameter(log_r0.clone())

        optimizer = torch.optim.Adam([self.centroids, self._log_radius], lr=self.lr)

        # train center + radius with Hinge loss
        for _ in range(self.max_iter):
            optimizer.zero_grad()

            r = F.softplus(self._log_radius)  # (c,)
            r2 = r * r  # (c,)

            # dist2: (n, c)
            diff = Xn.unsqueeze(1) - self.centroids.unsqueeze(0)  # (n,c,d)
            dist2 = (diff * diff).sum(dim=2)  # (n,c)

            # hinge on positives only: relu(dist2 - r2)
            hinge = torch.relu(dist2 - r2.unsqueeze(0))  # (n,c)
            # mean over positives per concept
            hinge_mean = (hinge * y).sum(dim=0) / n_pos.clamp_min(1.0)  # (c,)

            loss = r2 + self.C * hinge_mean  # (c,)
            loss = loss.mean()

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * (self.centroids * self.centroids).sum()

            loss.backward()
            optimizer.step()

        self.fitted = True

    @assert_fitted
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (n, d)
        Returns : (n, c) margins (higher => more inside SVDD sphere)
        """
        Xn = self.normalization(X) if self.normalization else X
        r = F.softplus(self._log_radius)  # (c,)  # type: ignore
        r2 = r * r

        diff = Xn.unsqueeze(1) - self.centroids.unsqueeze(0)  # (n,c,d)  # type: ignore
        dist2 = (diff * diff).sum(dim=2)  # (n,c)
        return r2.unsqueeze(0) - dist2
