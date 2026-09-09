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
Linear probe models for concept-based interpretability.

Each probe learns a linear mapping from latent activations to concept scores::

    score = x @ weight + bias    # (n, c)

Available probes:
    - [LinearRegressionProbe][interpreto.concepts.probes.linear.LinearRegressionProbe] — closed-form OLS/ridge regression.
    - [MeansDiffProbe][interpreto.concepts.probes.linear.MeansDiffProbe] — weight = difference of class means.
    - [LogisticRegressionProbe][interpreto.concepts.probes.linear.LogisticRegressionProbe] — gradient-descent logistic regression.
    - [LinearSVMProbe][interpreto.concepts.probes.linear.LinearSVMProbe] — gradient-descent linear SVM (hinge loss).

All probes support optional input normalization and bias calibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn

from interpreto.concepts.probes.base import Probe
from interpreto.concepts.probes.bias_calibrators import BiasCalibrator, prevalence_bias
from interpreto.concepts.probes.normalizations import NormalizationBase


class BaseLinearProbe(Probe, ABC):
    """Code: [:octicons-mark-github-24: `concepts/probes/linear.py`](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/probes/linear.py)
    Abstract base class for linear concept probes.

    Linear concept probes score activations by an affine map from latent activations to concept logits. This follows
    the general idea of Concept Activation Vectors, where linear directions in activation space are used to represent
    user-defined concepts[^1].

    Stores weight `(d, c)` and bias `(c,)` as buffers at init (empty),
    promoted to `nn.Parameter` during [fit][interpreto.concepts.probes.linear.BaseLinearProbe.fit].

    [^1]:
        Kim, B. et al., [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://proceedings.mlr.press/v80/kim18d.html).
        Proceedings of the 35th International Conference on Machine Learning, 2018.

    Args:
        normalization (NormalizationBase | None): Optional input normalization
            fitted jointly during [fit][interpreto.concepts.probes.linear.BaseLinearProbe.fit].

    Attributes:
        weight (torch.Tensor): Linear projection, shape `(d, c)` after fit.
        bias (torch.Tensor): Per-concept intercept, shape `(c,)` after fit.
    """

    def __init__(self, normalization: NormalizationBase | None = None):
        super().__init__()
        self.normalization = normalization
        self.register_buffer("weight", torch.empty(0))  # (d, c) after fit
        self.register_buffer("bias", torch.empty(0))  # (c,) after fit

    def _init_params(self, d: int, c: int, *, dtype: torch.dtype, device: torch.device):
        """Initialize weight and bias as zero-filled parameters."""
        self.weight = nn.Parameter(torch.zeros(d, c, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(c, dtype=dtype, device=device))

    def encode(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n c"]:
        """Encode activations into concept scores via linear projection.

        Args:
            x: Raw activations.

        Returns:
            Concept scores (logits).
        """
        xn: Float[torch.Tensor, "n d"] = self.normalization(x) if self.normalization else x
        return xn @ self.weight + self.bias  # type: ignore


class LinearRegressionProbe(BaseLinearProbe):
    """Code: [:octicons-mark-github-24: `concepts/probes/linear.py`](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/probes/linear.py)

    Multi-output linear regression probe with intercept.

    This probe fits concept scores using ordinary least squares or ridge regression, with an optional unpenalized
    intercept term[^1].

    Fits the linear model in closed form:

    - `l2 == 0`: OLS via pseudo-inverse.
    - `l2 > 0`: Ridge regression (intercept is not penalized).

    [^1]:
        Hastie, T., Tibshirani, R., Friedman, J., [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7).
        Springer, 2nd edition, 2009.

    Args:
        l2 (float): L2 regularization strength (0 for OLS).
        bias_calibrator (BiasCalibrator | None): If provided, overrides the
            regression intercept with a calibrated bias.
        normalization (NormalizationBase | None): Optional input normalization.
    """

    def __init__(
        self,
        l2: float = 0.0,
        bias_calibrator: BiasCalibrator | None = None,
        normalization: NormalizationBase | None = None,
    ):
        super().__init__(normalization=normalization)
        self.l2 = float(l2)
        self.bias_calibrator = bias_calibrator

    @torch.no_grad()
    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        xn: Float[torch.Tensor, "n d"] = self.normalization.fit_transform(x) if self.normalization else x
        n, d = xn.shape

        ones = torch.ones(n, 1, dtype=xn.dtype, device=xn.device)
        x_aug: Float[torch.Tensor, "n d_plus_1"] = torch.cat([xn, ones], dim=1)

        if self.l2 == 0.0:
            # OLS via pseudo-inverse for stability
            x_pinv = torch.linalg.pinv(x_aug)
            W_aug: Float[torch.Tensor, "d_plus_1 c"] = x_pinv @ y
        else:
            # Ridge closed form: (x^T x + l2*I)^{-1} x^T y
            # Do not penalize the intercept (last column of x_aug)
            xtx: Float[torch.Tensor, "d_plus_1 d_plus_1"] = x_aug.T @ x_aug
            xty: Float[torch.Tensor, "d_plus_1 c"] = x_aug.T @ y

            reg = torch.ones(d + 1, dtype=xn.dtype, device=xn.device)
            reg[-1] = 0.0  # no penalty on bias
            A = xtx + self.l2 * torch.diag(reg)
            W_aug: Float[torch.Tensor, "d_plus_1 c"] = torch.linalg.solve(A, xty)

        weight: Float[torch.Tensor, "d c"] = W_aug[:d, :].clone()

        if self.bias_calibrator is None:
            bias: Float[torch.Tensor, "c"] = W_aug[d, :].clone()
        else:
            scores: Float[torch.Tensor, "n c"] = xn @ weight
            bias: Float[torch.Tensor, "c"] = self.bias_calibrator(scores, y)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.fitted = True


class MeansDiffProbe(BaseLinearProbe):
    """Code: [:octicons-mark-github-24: `concepts/probes/linear.py`](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/probes/linear.py)

    Means-difference probe (multi-label, multi-output).

    For each concept *j*, the weight vector is the difference between the
    mean activation of positive and negative samples::

        $$w_j = mean(x | y_j=1) - mean(x | y_j=0)$$

    This is equivalent to Fisher’s Linear Discriminant with shared identity
    covariance assumption[^1][^2].

    [^1]:
        Fisher, R. A., [The Use of Multiple Measurements in Taxonomic Problems](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x).
        Annals of Eugenics, 7(2), 1936, pp. 179-188.
    [^2]:
        Hastie, T., Tibshirani, R., Friedman, J., [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7).
        Springer, 2nd edition, 2009.

    Args:
        bias_calibrator (BiasCalibrator | None): Post-hoc bias calibration.
            If `None`, bias is zero.
        normalization (NormalizationBase | None): Optional input normalization.
        eps (float): Floor for sample count denominators.
    """

    def __init__(
        self,
        bias_calibrator: BiasCalibrator | None = None,
        normalization: NormalizationBase | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(normalization=normalization)
        self.bias_calibrator = bias_calibrator
        self.eps = float(eps)

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        xn: Float[torch.Tensor, "n d"] = self.normalization.fit_transform(x) if self.normalization else x
        n = xn.shape[0]
        c = y.shape[1]

        # number of positive and negative samples per class
        n1: Float[torch.Tensor, "c"] = y.sum(dim=0)
        n0: Float[torch.Tensor, "c"] = n - n1

        s1: Float[torch.Tensor, "c d"] = y.t() @ xn
        sumx: Float[torch.Tensor, "d"] = xn.sum(dim=0)
        s0: Float[torch.Tensor, "c d"] = sumx.unsqueeze(0) - s1

        mu1: Float[torch.Tensor, "c d"] = s1 / n1.unsqueeze(1).clamp_min(self.eps)
        mu0: Float[torch.Tensor, "c d"] = s0 / n0.unsqueeze(1).clamp_min(self.eps)

        w: Float[torch.Tensor, "d c"] = (mu1 - mu0).t()

        if self.bias_calibrator is None:
            bias: Float[torch.Tensor, "c"] = torch.zeros(c, dtype=xn.dtype, device=xn.device)
        else:
            scores: Float[torch.Tensor, "n c"] = xn @ w
            bias: Float[torch.Tensor, "c"] = self.bias_calibrator(scores, y)

        self.weight = nn.Parameter(w.clone())
        self.bias = nn.Parameter(bias)
        self.fitted = True


class _GDLinearProbe(BaseLinearProbe):
    """Code: [:octicons-mark-github-24: `concepts/probes/linear.py`](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/probes/linear.py)

    Gradient-descent linear probe skeleton (private base).

    Trains weight and bias via Adam on a configurable loss function.
    Optionally initializes from [MeansDiffProbe][interpreto.concepts.probes.linear.MeansDiffProbe]
    for faster convergence.

    Args:
        lr (float): Adam learning rate.
        max_iter (int): Number of optimization steps.
        l2 (float): L2 regularization on weight (bias not penalized).
        init_from_means_diff (bool): If `True`, initialize weight/bias from
            [MeansDiffProbe][interpreto.concepts.probes.linear.MeansDiffProbe].
        init_bias_calibrator (BiasCalibrator | None): Bias calibrator used for
            MeansDiff initialization.
        normalization (NormalizationBase | None): Optional input normalization.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        max_iter: int = 20,
        l2: float = 0.0,
        init_from_means_diff: bool = True,
        init_bias_calibrator: BiasCalibrator | None = prevalence_bias,
        normalization: NormalizationBase | None = None,
    ):
        super().__init__(normalization=normalization)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.init_from_means_diff = init_from_means_diff
        self.init_bias_calibrator = init_bias_calibrator

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def _prepare_targets(self, y: Float[torch.Tensor, "n c"]) -> Float[torch.Tensor, "n c"]:
        raise NotImplementedError

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def _loss(self, logits: Float[torch.Tensor, "n c"], y: Float[torch.Tensor, "n c"]) -> Float[torch.Tensor, ""]:
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def _init_from_means_diff(self, xn: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        """Initialize from MeansDiff on already-normalized data."""
        md = MeansDiffProbe(bias_calibrator=self.init_bias_calibrator, eps=1e-8)
        md.to(device=xn.device)
        md.fit(xn, y)  # fit on normalized data directly (no normalization in md)

        # Initialize parameters from MeansDiff without tracking gradients
        with torch.no_grad():
            self.weight.copy_(md.weight.detach())  # type: ignore
            self.bias.copy_(md.bias.detach())  # type: ignore

        del md

    @jaxtyped(typechecker=beartype)
    def fit(self, x: Float[torch.Tensor, "n d"], y: Float[torch.Tensor, "n c"]):
        xn = self.normalization.fit_transform(x) if self.normalization else x
        n, d = xn.shape
        c = y.shape[1]

        # initialize parameters
        self._init_params(d, c, dtype=xn.dtype, device=xn.device)
        if self.init_from_means_diff:
            self._init_from_means_diff(xn, y)

        y_prepared = self._prepare_targets(y)

        optimizer = torch.optim.Adam([self.weight, self.bias], lr=self.lr)  # type: ignore

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            logits: Float[torch.Tensor, n, c] = xn @ self.weight + self.bias
            loss = self._loss(logits, y_prepared)

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * (self.weight**2).sum()  # type: ignore

            loss.backward()
            optimizer.step()
        self.fitted = True


class LogisticRegressionProbe(_GDLinearProbe):
    """Code: [:octicons-mark-github-24: `concepts/probes/linear.py`](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/probes/linear.py)

    Multi-label logistic regression probe (BCE loss, Adam optimizer).

    Minimizes binary cross-entropy with logits, optionally with L2 weight
    regularization. Initialized from [MeansDiffProbe][interpreto.concepts.probes.linear.MeansDiffProbe]
    by default.

    Args:
        lr (float): Adam learning rate.
        max_iter (int): Number of optimization steps.
        l2 (float): L2 regularization on weight (bias not penalized).
        init_from_means_diff (bool): If `True`, initialize weight/bias from MeansDiffProbe.
        init_bias_calibrator (BiasCalibrator | None): Bias calibrator for MeansDiff initialization.
        normalization (NormalizationBase | None): Optional input normalization.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        max_iter: int = 20,
        l2: float = 0.0,
        init_from_means_diff: bool = True,
        init_bias_calibrator: BiasCalibrator | None = prevalence_bias,
        normalization: NormalizationBase | None = None,
    ):
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            l2=l2,
            init_from_means_diff=init_from_means_diff,
            init_bias_calibrator=init_bias_calibrator,
            normalization=normalization,
        )
        self._loss_fn = nn.BCEWithLogitsLoss()

    def _prepare_targets(self, y: Float[torch.Tensor, "n c"]) -> Float[torch.Tensor, "n c"]:
        return y

    def _loss(self, logits: Float[torch.Tensor, "n c"], y: Float[torch.Tensor, "n c"]) -> Float[torch.Tensor, ""]:
        return self._loss_fn(logits, y)


class LinearSVMProbe(_GDLinearProbe):
    """Multi-label linear SVM probe (hinge loss, Adam optimizer).

    This is the linear model used in CAV[^1].

    Targets are mapped to {-1, +1} and the loss is the mean of
    `max(0, 1 - y * logits)`. Optionally with L2 weight regularization.
    Initialized from [MeansDiffProbe][interpreto.concepts.probes.linear.MeansDiffProbe] by default.

    [^1]:
        Kim, B. et al., [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://proceedings.mlr.press/v80/kim18d.html).
        Proceedings of the 35th International Conference on Machine Learning, 2018.

    Args:
        lr (float): Adam learning rate.
        max_iter (int): Number of optimization steps.
        l2 (float): L2 regularization on weight (bias not penalized).
        init_from_means_diff (bool): If `True`, initialize weight/bias from MeansDiffProbe.
        init_bias_calibrator (BiasCalibrator | None): Bias calibrator for MeansDiff initialization.
        normalization (NormalizationBase | None): Optional input normalization.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        max_iter: int = 20,
        l2: float = 0.0,
        init_from_means_diff: bool = True,
        init_bias_calibrator: BiasCalibrator | None = prevalence_bias,
        normalization: NormalizationBase | None = None,
    ):
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            l2=l2,
            init_from_means_diff=init_from_means_diff,
            init_bias_calibrator=init_bias_calibrator,
            normalization=normalization,
        )

    def _prepare_targets(self, y: Float[torch.Tensor, "n c"]) -> Float[torch.Tensor, "n c"]:
        return 2.0 * y - 1.0

    def _loss(self, logits: Float[torch.Tensor, "n c"], y: Float[torch.Tensor, "n c"]) -> Float[torch.Tensor, ""]:
        margins = 1.0 - y * logits
        return torch.clamp(margins, min=0.0).mean()
