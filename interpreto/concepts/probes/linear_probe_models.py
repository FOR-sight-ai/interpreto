from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import torch
from torch import nn

from interpreto.concepts.probes.bias_calibrators import BiasCalibrator, prevalence_bias
from interpreto.concepts.probes.normalizations import NormalizationBase


def assert_fitted(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        return fn(self, *args, **kwargs)

    return wrapper


class ProbeModelInterface(nn.Module, ABC):
    """
    Base class for probe models.
    """

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    @assert_fitted
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearProbeBase(ProbeModelInterface, ABC):
    """
    Base class for linear probes with intercept.

    Stores:
        weight: (d, c)
        bias:   (c,)

    encode(X) returns logits/scores: (n, c)
    """

    def __init__(self, normalization: NormalizationBase | None = None):
        super().__init__()
        self.normalization = normalization
        self.weight = None  # nn.Parameter, (d, c)
        self.bias = None  # nn.Parameter, (c,)
        self.fitted = False

    def _init_params(self, d: int, c: int, *, dtype: torch.dtype, device: torch.device):
        self.weight = nn.Parameter(torch.zeros(d, c, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(c, dtype=dtype, device=device))

    @assert_fitted
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        Xn = self.normalization(X) if self.normalization else X
        return Xn @ self.weight + self.bias  # type: ignore


class LinearRegressionProbe(LinearProbeBase):
    """
    Multi-output linear regression probe with intercept.

    - If l2 == 0.0: uses OLS closed form via pseudo-inverse.
    - If l2  > 0.0: uses ridge regression closed form on augmented design,
      without penalizing the intercept term.
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
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        Xn = self.normalization.fit_transform(X) if self.normalization else X
        n, d = Xn.shape

        ones = torch.ones(n, 1, dtype=Xn.dtype, device=Xn.device)
        X_aug = torch.cat([Xn, ones], dim=1)  # (n, d+1)

        if self.l2 == 0.0:
            # OLS via pseudo-inverse for stability
            X_pinv = torch.linalg.pinv(X_aug)
            W_aug = X_pinv @ y  # (d+1, c)
        else:
            # Ridge closed form: (X^T X + l2*I)^{-1} X^T y
            # Do not penalize the intercept (last column of X_aug)
            XtX = X_aug.T @ X_aug  # (d+1, d+1)
            Xty = X_aug.T @ y  # (d+1, c)

            reg = torch.ones(d + 1, dtype=Xn.dtype, device=Xn.device)
            reg[-1] = 0.0  # no penalty on bias
            A = XtX + self.l2 * torch.diag(reg)
            W_aug = torch.linalg.solve(A, Xty)  # (d+1, c)

        weight = W_aug[:d, :].clone()

        if self.bias_calibrator is None:
            bias = W_aug[d, :].clone()
        else:
            scores = Xn @ weight
            bias = self.bias_calibrator(scores, y)

        self.weight = nn.Parameter(weight)  # (d, c)
        self.bias = nn.Parameter(bias)  # (c,)
        self.fitted = True


class MeansDiffProbe(LinearProbeBase):
    """
    MeansDiff probe (multi-label, multi-output).

    For each concept j:
        w_j = mean(X | y_j=1) - mean(X | y_j=0)
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

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        Xn = self.normalization.fit_transform(X) if self.normalization else X
        n = Xn.shape[0]
        c = y.shape[1]

        # number of positive and negative samples per class
        n1 = y.sum(dim=0)  # (c,)
        n0 = n - n1

        s1 = y.t() @ Xn  # (c, d)
        sumX = Xn.sum(dim=0)  # (d,)
        s0 = sumX.unsqueeze(0) - s1  # (c, d)

        mu1 = s1 / n1.unsqueeze(1).clamp_min(self.eps)
        mu0 = s0 / n0.unsqueeze(1).clamp_min(self.eps)

        w = (mu1 - mu0).t()  # (d, c)

        if self.bias_calibrator is None:
            bias = torch.zeros(c, dtype=Xn.dtype, device=Xn.device)
        else:
            scores = Xn @ w
            bias = self.bias_calibrator(scores, y)

        self.weight = nn.Parameter(w.clone())
        self.bias = nn.Parameter(bias)
        self.fitted = True


class _GDLinearProbe(LinearProbeBase):
    """
    Gradient-descent linear probe skeleton.

    Optional init:
        - init_from_means_diff: if True, initialize weight and bias from MeansDiff
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
    def _prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _init_from_means_diff(self, Xn: torch.Tensor, y: torch.Tensor):
        """Initialize from MeansDiff on already-normalized data."""
        md = MeansDiffProbe(bias_calibrator=self.init_bias_calibrator, eps=1e-8)
        md.to(device=Xn.device)
        md.fit(Xn, y)  # fit on normalized data directly (no normalization in md)

        # Initialize parameters from MeansDiff without tracking gradients
        with torch.no_grad():
            self.weight.copy_(md.weight.detach())  # type: ignore
            self.bias.copy_(md.bias.detach())  # type: ignore

        del md

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        Xn = self.normalization.fit_transform(X) if self.normalization else X
        n, d = Xn.shape
        c = y.shape[1]

        # initialize parameters
        self._init_params(d, c, dtype=Xn.dtype, device=Xn.device)
        if self.init_from_means_diff:
            self._init_from_means_diff(Xn, y)

        y_prepared = self._prepare_targets(y)

        optimizer = torch.optim.Adam([self.weight, self.bias], lr=self.lr)  # type: ignore

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            logits = Xn @ self.weight + self.bias  # type: ignore
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

    def _prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(logits, y)


class LinearSVMProbe(_GDLinearProbe):
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

    def _prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * y - 1.0

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        margins = 1.0 - y * logits
        return torch.clamp(margins, min=0.0).mean()
