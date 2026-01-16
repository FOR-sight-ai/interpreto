from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import torch
from torch import nn

from interpreto.concepts.probes.bias_calibrators import BiasCalibratorBase, PrevalenceBias


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

    def __init__(self):
        super().__init__()
        self.weight = None  # nn.Parameter, (d, c)
        self.bias = None  # nn.Parameter, (c,)
        self.fitted = False

    def _init_params(self, d: int, c: int, *, dtype: torch.dtype, device: torch.device):
        self.weight = nn.Parameter(torch.zeros(d, c, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(c, dtype=dtype, device=device))

    @assert_fitted
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.weight + self.bias  # type: ignore


class LinearRegressionProbe(LinearProbeBase):
    """
    Multi-output linear regression probe with intercept.

    - If l2 == 0.0: uses OLS closed form via pseudo-inverse.
    - If l2  > 0.0: uses ridge regression closed form on augmented design,
      without penalizing the intercept term.
    """

    def __init__(self, l2: float = 0.0, bias_calibrator: BiasCalibratorBase | None = None):
        super().__init__()
        self.l2 = float(l2)
        self.bias_calibrator = bias_calibrator

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
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

            reg = torch.ones(d + 1, dtype=X.dtype, device=X.device)
            reg[-1] = 0.0  # no penalty on bias
            A = XtX + self.l2 * torch.diag(reg)
            W_aug = torch.linalg.solve(A, Xty)  # (d+1, c)

        if self.bias_calibrator is None:
            bias = W_aug[d, :].clone()
        else:
            self.bias_calibrator.fit(scores=X @ self.weight, y=y)
            bias = self.bias_calibrator.bias.clone()
            del self.bias_calibrator

        self.weight = nn.Parameter(W_aug[:d, :].clone())  # (d, c)
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
        bias_calibrator: BiasCalibratorBase | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        # number of positive and negative samples per class
        n1 = y.sum(dim=0)  # (c,)
        n0 = n - n1

        s1 = y.t() @ X
        sumX = X.sum(dim=0)
        s0 = (X.shape[0] * sumX.unsqueeze(0)) - s1

        mu1 = s1 / n1.unsqueeze(1).clamp_min(self.eps)
        mu0 = s0 / n0.unsqueeze(1).clamp_min(self.eps)

        w = (mu1 - mu0).t()  # (d, c)
        
        if self.bias_calibrator is None:
            bias = torch.zeros(c, dtype=X.dtype, device=X.device)
        else:
            self.bias_calibrator.fit(scores=X @ self.weight, y=y)
            bias = self.bias_calibrator.bias.clone()
            del self.bias_calibrator

        self.weight = nn.Parameter(w.clone())
        self.bias = nn.Parameter(bias)
        self.fitted = True


class _GDLinearProbe(LinearProbeBase):
    """
    Gradient-descent linear probe skeleton.

    Optional init:
        - init_bias:
            - None: no initialization
            - otherwise initialize with mean difference  with specified bias calibrator
    """

    def __init__(
        self,
        lr: float = 1e-2,
        max_iter: int = 20,
        l2: float = 0.0,
        init_bias: BiasCalibratorBase | None = None,
    ):
        super().__init__()
        if init_bias not in {"zero", "prevalence"}:
            raise ValueError("init_bias must be one of {'zero','prevalence'}")

        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.init_bias = init_bias

    @abstractmethod
    def _prepare_targets(self, y: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _init_from_means_diff(self, X: torch.Tensor, y: torch.Tensor):

        md = MeansDiffProbe(bias=self.init_bias).to(device=X.device)
        md.fit(X, y)

        # Initialize parameters from MeansDiff without tracking gradients
        # to avoid in-place ops on leaf Variables that require grad.
        with torch.no_grad():
            self.weight.copy_(md.weight.detach())  # type: ignore
            self.bias.copy_(md.bias.detach())  # type: ignore

        del md

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        n, d = X.shape
        c = y.shape[1]

        # initialize parameters
        self._init_params(d, c, dtype=X.dtype, device=X.device)
        if self.init_bias:  # init weight and bias with means diff
            # Ensure {0,1} for init
            self._init_from_means_diff(X, y)

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
        init_bias: BiasCalibratorBase | None = PrevalenceBias(),
    ):
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            l2=l2,
            init_bias=init_bias,
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
        init_bias: BiasCalibratorBase | None = PrevalenceBias(),
    ):
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            l2=l2,
            init_bias=init_bias,
        )

    def _prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * y - 1.0

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        margins = 1.0 - y * logits
        return torch.clamp(margins, min=0.0).mean()
