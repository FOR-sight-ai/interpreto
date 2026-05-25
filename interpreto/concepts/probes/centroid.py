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
Centroid-based probe models.

Each probe in this module assigns concept scores based on distances between
activation vectors and learned centroid vectors (one per concept). The general
pipeline is:

1. (Optional) Normalize inputs via a [NormalizationBase][interpreto.concepts.probes.normalizations.NormalizationBase] layer.
2. Compute one centroid per concept from positive samples.
3. Score new inputs by a distance metric to each centroid.
4. Add a calibrated bias (see [bias_calibrators][interpreto.concepts.probes.bias_calibrators]).

All probes store their state in `nn.Module` buffers/parameters and support
`state_dict` serialization via [load_state_dict][interpreto.concepts.probes.base.Probe.load_state_dict].
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn

from interpreto.concepts.probes.base import Probe
from interpreto.concepts.probes.bias_calibrators import BiasCalibrator
from interpreto.concepts.probes.normalizations import NormalizationBase, Standardization


class BaseCentroidProbe(Probe):
    """Shared base for centroid-based probes (multi-label, multi-output).

    Subclasses only need to implement [_compute_scores][interpreto.concepts.probes.centroid.BaseCentroidProbe._compute_scores]
    which maps normalized inputs to raw (unbiased) concept scores.

    Args:
        normalization (NormalizationBase | None): Optional input normalization layer
            (fitted jointly during [fit][interpreto.concepts.probes.centroid.BaseCentroidProbe.fit]).
        bias_calibrator (BiasCalibrator | None): Optional post-hoc bias calibration
            function (see [bias_calibrators][interpreto.concepts.probes.bias_calibrators]).
            If `None`, bias is set to zero.
        eps (float): Numerical stability floor for centroid computation.

    Attributes:
        centroids (torch.Tensor): Centroid per concept, shape `(c, d)`.
        bias (torch.Tensor): Additive per-concept bias, shape `(c,)`.
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
        self.register_buffer("centroids", torch.empty(0))  # (c, d) after fit
        self.register_buffer("bias", torch.empty(0))  # (c,) after fit

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def _fit_centroids(
        self, xn: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]
    ) -> Float[torch.Tensor, "c"]:
        """Compute centroids as the mean of positive samples per concept.

        Args:
            xn: Normalized activations.
            y: Binary multi-label targets.

        Returns:
            Number of positive samples per concept.
        """
        n_pos: Float[torch.Tensor, "c"] = y.sum(dim=0)
        sum_pos: Float[torch.Tensor, "c d"] = y.transpose(0, 1) @ xn
        self.centroids = sum_pos / n_pos.unsqueeze(1).clamp_min(self.eps)
        return n_pos

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def _compute_scores(self, xn: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        """Compute raw scores from normalized input (without bias).

        Args:
            xn: Normalized activations.

        Returns:
            Raw concept scores before bias.
        """
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def _calibrate_bias(self, xn: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        """Calibrate bias using the bias_calibrator function or set to zero."""
        c = y.shape[1]
        if self.bias_calibrator is not None:
            scores = self._compute_scores(xn)
            self.bias = self.bias_calibrator(scores, y)
        else:
            self.bias = torch.zeros(c, dtype=xn.dtype, device=xn.device)

    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        """Fit centroids and bias from activations and labels.

        Args:
            x: Raw activations.
            y: Binary multi-label targets.
        """
        xn = self.normalization.fit_transform(x) if self.normalization else x
        self._fit_centroids(xn, y)
        self._calibrate_bias(xn, y)
        self.fitted = True

    @jaxtyped(typechecker=beartype)
    def encode(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        """Encode activations into concept scores.

        Args:
            x: Raw activations.

        Returns:
            Concept scores `(raw_score + bias)`.
        """
        xn = self.normalization(x) if self.normalization else x
        scores: Float[torch.Tensor, "n c"] = self._compute_scores(xn)
        return scores + self.bias


class DotProductCentroidProbe(BaseCentroidProbe):
    """Centroid probe using dot-product similarity.

    `score_ij = x_i · centroid_j + bias_j`

    Args:
        normalization (NormalizationBase | None): Optional input normalization layer.
        bias_calibrator (BiasCalibrator | None): Optional post-hoc bias calibration function.
        eps (float): Numerical stability floor.
    """

    @jaxtyped(typechecker=beartype)
    def _compute_scores(self, xn: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        return xn @ self.centroids.t()


class CosineCentroidProbe(BaseCentroidProbe):
    """Centroid probe using cosine similarity.

    `score_ij = cosine(x_i, centroid_j) + bias_j`

    Centroids are L2-normalized after fitting, and inputs are normalized
    before scoring.

    Args:
        normalization (NormalizationBase | None): Optional input normalization layer.
        bias_calibrator (BiasCalibrator | None): Optional post-hoc bias calibration function.
        eps (float): Numerical stability floor.
    """

    @jaxtyped(typechecker=beartype)
    def _unit(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        """L2-normalize along the feature dimension."""
        return x / x.norm(dim=1, keepdim=True).clamp_min(self.eps)

    @jaxtyped(typechecker=beartype)
    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        xn = self.normalization.fit_transform(x) if self.normalization else x
        self._fit_centroids(xn, y)
        self.centroids = self._unit(self.centroids)
        self._calibrate_bias(xn, y)
        self.fitted = True

    @jaxtyped(typechecker=beartype)
    def _compute_scores(self, xn: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        xn = self._unit(xn)
        return xn @ self.centroids.t()


class SqL2CentroidProbe(BaseCentroidProbe):
    """Centroid probe using negative squared Euclidean distance.

    `score_ij = -||x_i - centroid_j||² + bias_j`

    Computed efficiently via the expansion:
    `dist² = (x·x) + (c·c) - 2(x·c)`

    Recommended normalization: [Standardization][interpreto.concepts.probes.normalizations.Standardization].

    Args:
        normalization (NormalizationBase | None): Optional input normalization layer.
        bias_calibrator (BiasCalibrator | None): Optional post-hoc bias calibration function.
        eps (float): Numerical stability floor.
    """

    @jaxtyped(typechecker=beartype)
    def _compute_scores(self, xn: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        C: Float[torch.Tensor, "c d"] = self.centroids

        x2: Float[torch.Tensor, "n 1"] = (xn * xn).sum(dim=1, keepdim=True)
        c2: Float[torch.Tensor, "1 c"] = (C * C).sum(dim=1, keepdim=True).t()
        dots: Float[torch.Tensor, "n c"] = xn @ C.t()
        dist2: Float[torch.Tensor, "n c"] = x2 + c2 - 2.0 * dots
        return -dist2


class DiagonalMahalanobisCentroidProbe(BaseCentroidProbe):
    """Centroid probe using diagonal Mahalanobis distance.

    `score_ij = -(x_i - c_j)^T diag(1/var) (x_i - c_j) + bias_j`

    The variance matrix is controlled by the `shrinkage` parameter:

    - `shrinkage = 0`: class-wise diagonal variance `(c, d)` from positives only.
    - `0 < shrinkage < 1`: convex combination of class-wise and pooled variance.
    - `shrinkage = 1`: pooled diagonal variance `(d,)` from all samples.

    Default normalization is [Standardization][interpreto.concepts.probes.normalizations.Standardization]
    if none is provided.

    Args:
        normalization (NormalizationBase | None): Input normalization (defaults to Standardization).
        bias_calibrator (BiasCalibrator | None): Post-hoc bias calibration function.
        eps (float): Numerical stability floor.
        shrinkage (float): Shrinkage coefficient in [0, 1] for the variance estimate.
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
        self.register_buffer("inv_var", torch.empty(0))  # (c, d) or (1, d) after fit

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        # copied base init to prevent recomputing xn
        xn = self.normalization.fit_transform(x) if self.normalization else x
        n_pos = self._fit_centroids(xn, y)

        # var = (1-shrinkage)*classwise_var + shrinkage*global_var
        var: Float[torch.Tensor, "c d"] = torch.zeros_like(self.centroids)

        # global diagonal variance
        if self.shrinkage > 0.0:
            global_var: Float[torch.Tensor, "d"] = xn.var(dim=0, unbiased=False)
            var += self.shrinkage * global_var.unsqueeze(0)

        # class-wise variance
        if self.shrinkage < 1.0:
            # class-wise diagonal variance on positives: E[x^2|pos] - E[x|pos]^2
            sum_pos_sq: Float[torch.Tensor, "c d"] = y.t() @ (xn * xn)
            ex2: Float[torch.Tensor, "c d"] = sum_pos_sq / n_pos.unsqueeze(1)
            classwise_var: Float[torch.Tensor, "c d"] = ex2 - self.centroids * self.centroids

            var += (1.0 - self.shrinkage) * classwise_var

        # compute inverse common variance
        self.inv_var: Float[torch.Tensor, "c d"] = (1.0 / var.clamp_min(self.eps)).detach()

        self._calibrate_bias(xn, y)
        self.fitted = True

    @jaxtyped(typechecker=beartype)
    def _compute_scores(self, xn: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        # -(x - c)² * inv_var, summed over d
        diff: Float[torch.Tensor, "n c d"] = xn.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist2: Float[torch.Tensor, "n c"] = (diff * diff * self.inv_var.unsqueeze(0)).sum(dim=2)
        return -dist2


class SVDDCentroidProbe(BaseCentroidProbe):
    """Multi-label SVDD (Support Vector Data Description) probe.

    Fits one hyper-sphere per concept. For each concept *j*, jointly optimizes
    a center `a_j` and squared radius `r²_j` on positive samples by
    minimizing::

        L_j = r²_j + C * mean_pos( relu(||x - a_j||² - r²_j) ) + 0.5*l2*||a_j||²

    Encoding returns a margin score::

        score_ij = r²_j - ||x_i - a_j||²

    Positive scores indicate the sample is inside the concept sphere.

    Args:
        lr (float): Adam learning rate for center/radius optimization.
        max_iter (int): Number of optimization steps.
        C (float): Hinge loss penalty weight on outliers.
        l2 (float): L2 regularization on centroids.
        normalization (NormalizationBase | None): Optional input normalization.
        eps (float): Numerical stability floor.
    """

    def __init__(
        self,
        lr: float = 5e-2,
        max_iter: int = 20,
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

    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        """Fit SVDD centers and radii via Adam optimization.

        Args:
            x: Raw activations.
            y: Binary multi-label targets.
        """
        # Initialize centroids from positive means
        xn = self.normalization.fit_transform(x) if self.normalization else x
        n_pos: Float[torch.Tensor, "c"] = self._fit_centroids(xn, y)
        mu_pos: Float[torch.Tensor, "c d"] = self.centroids
        del self.centroids  # pass from buffer to parameter
        self.centroids = nn.Parameter(mu_pos)

        # Initialize r2 from mean positive distance squared
        with torch.no_grad():
            diff: Float[torch.Tensor, "n c d"] = xn.unsqueeze(1) - mu_pos.unsqueeze(0)
            dist2: Float[torch.Tensor, "n c"] = (diff * diff).sum(dim=2)
            mean_dist2: Float[torch.Tensor, "c"] = (dist2 * y).sum(dim=0) / n_pos.clamp_min(1.0)
            # inverse softplus: softplus(z)=r2 -> z = log(exp(r2)-1)
            log_r2_init: Float[torch.Tensor, "c"] = torch.log(torch.expm1(mean_dist2.clamp_min(self.eps)))
        _log_r2 = nn.Parameter(log_r2_init.clone())

        optimizer = torch.optim.Adam([self.centroids, _log_r2], lr=self.lr)

        # train center + radius with Hinge loss
        for _ in range(self.max_iter):
            optimizer.zero_grad()

            r2: Float[torch.Tensor, "c"] = F.softplus(_log_r2)

            diff: Float[torch.Tensor, "n c d"] = xn.unsqueeze(1) - self.centroids.unsqueeze(0)
            dist2: Float[torch.Tensor, "n c"] = (diff * diff).sum(dim=2)

            # hinge on positives only: relu(dist2 - r2)
            hinge: Float[torch.Tensor, "n c"] = torch.relu(dist2 - r2.unsqueeze(0))
            # mean over positives per concept
            hinge_mean: Float[torch.Tensor, "c"] = (hinge * y).sum(dim=0) / n_pos.clamp_min(1.0)

            loss: Float[torch.Tensor, "c"] = r2 + self.C * hinge_mean
            loss = loss.mean()

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * (self.centroids * self.centroids).sum()

            loss.backward()
            optimizer.step()

        self.bias: Float[torch.Tensor, "c"] = F.softplus(_log_r2).detach()
        self.fitted = True

    @jaxtyped(typechecker=beartype)
    def _compute_scores(self, xn: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        """Compute negative squared distances to centroids.

        Returns `-||x - centroid||²` so that `encode = score + r²` gives
        positive values for points inside the sphere.
        """
        diff: Float[torch.Tensor, "n c d"] = xn.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist2: Float[torch.Tensor, "n c"] = (diff * diff).sum(dim=2)
        return -dist2
