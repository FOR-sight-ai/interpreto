from __future__ import annotations

from abc import abstractmethod

from functools import wraps

import torch
import torch.nn as nn

def assert_fitted(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.fitted:
            raise RuntimeError("Model is not fitted or loaded (self.fitted is False).")
        return fn(self, *args, **kwargs)

    return wrapper

class NormalizationBase(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.fitted = False

    @abstractmethod
    def fit(self, X: torch.Tensor) -> "NormalizationBase":
        raise NotImplementedError

    @abstractmethod
    @assert_fitted
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @assert_fitted
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


class Standardization(NormalizationBase):
    """
    z = (x - mean) / std
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__(eps=eps)
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("std", torch.empty(0))

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "Standardization":
        self.mean = X.mean(dim=0).detach()
        self.std = X.std(dim=0, unbiased=False).clamp_min(self.eps).detach()
        return self

    @assert_fitted
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std


class Whitening(NormalizationBase):
    """
    Whitening via econ SVD of centered X:
        Xc = U S V^T
        z  = (X - mean) @ V_r * (sqrt(n)/S_r)

    rank:
      - None (default): full whitening (r = min(n, d))
      - int: low-rank whitening keeping top-r components
    """
    def __init__(self, rank: int | None = None, eps: float = 1e-8):
        super().__init__(eps=eps)
        self.rank = None if rank is None else int(rank)
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("V", torch.empty(0))       # (d, r)
        self.register_buffer("inv_s", torch.empty(0))   # (r,)

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "Whitening":
        n = X.shape[0]
        mean = X.mean(dim=0)
        Xc = X - mean.unsqueeze(0)

        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

        inv_s = (torch.sqrt(torch.tensor(float(n), device=X.device, dtype=X.dtype)) / S).clamp_max(1.0 / self.eps)
        V = Vh.transpose(0, 1)

        if self.rank is not None:
            r = min(self.rank, V.shape[1])
            V = V[:, :r]
            inv_s = inv_s[:r]

        self.mean = mean.detach()
        self.V = V.detach()
        self.inv_s = inv_s.detach()
        return self

    @assert_fitted
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        Z = (X - self.mean) @ self.V
        return Z * self.inv_s.unsqueeze(0)