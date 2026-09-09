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
Input normalization layers for probe models.

Normalization is applied to activations *before* the probe scoring step.
Each normalization layer is an `nn.Module` with a `fit` / `transform`
interface (similar to scikit-learn), and its learned statistics are persisted as
buffers for `state_dict` serialization.

Available normalizations:
    - [Standardization][interpreto.concepts.probes.normalizations.Standardization] — zero-mean, unit-variance per feature.
    - [Whitening][interpreto.concepts.probes.normalizations.Whitening] — SVD-based de-correlation (optionally low-rank).
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn

from interpreto.concepts.probes.base import assert_fitted


class NormalizationBase(nn.Module):
    """Abstract base for input normalization layers.

    Provides a `fit` / `transform` / `fit_transform` interface.
    Calling the instance (`__call__`) delegates to `transform`.

    Args:
        eps (float): Numerical stability floor for divisions.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_fitted_flag", torch.tensor(False, dtype=torch.bool))

    @property
    def fitted(self) -> bool:
        return bool(self._fitted_flag.item())  # type: ignore

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted_flag.fill_(bool(value))  # type: ignore

    @abstractmethod
    def fit(self, X: Float[torch.Tensor, "n d"]) -> "NormalizationBase":
        """Fit normalization statistics from data.

        Args:
            X: Training activations.

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    @assert_fitted
    def transform(self, X: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        """Transform activations using fitted statistics.

        Args:
            X: Activations to normalize.

        Returns:
            Normalized activations (or `(n, r)` for rank-reduced whitening).
        """
        raise NotImplementedError

    @assert_fitted
    def __call__(self, X: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        return self.transform(X)

    def fit_transform(self, X: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        """Convenience: fit then transform in one call."""
        self.fit(X)
        return self.transform(X)


class Standardization(NormalizationBase):
    """Code: [:octicons-mark-github-24: `concepts/probes/normalizations.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/probes/normalizations.py)

    Per-feature zero-mean, unit-variance normalization.

    `z = (x - mean) / std`

    Attributes:
        mean (torch.Tensor): Feature means from training data, shape `(d,)`.
        std (torch.Tensor): Feature standard deviations (clamped to eps), shape `(d,)`.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(eps=eps)
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("std", torch.empty(0))

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def fit(self, X: Float[torch.Tensor, "n d"]) -> "Standardization":
        self.mean = X.mean(dim=0).detach()
        self.std = X.std(dim=0, unbiased=False).clamp_min(self.eps).detach()
        self.fitted = True
        return self

    @assert_fitted
    @jaxtyped(typechecker=beartype)
    def transform(self, X: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n d"]:
        return (X - self.mean) / self.std


class Whitening(NormalizationBase):
    """Code: :octicons-mark-github-24: concepts/probes/normalizations.py

    SVD-based whitening normalization.

    Whitening projects centered activations onto singular-vector directions and rescales them by the inverse singular
    values, as in PCA whitening[^1].

        z = (X - mean) @ V_r * (sqrt(n) / S_r)

    This produces decorrelated, unit-variance features in the rotated space.

    [^1]:
        Murphy, K. P., [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/book0.html).
        MIT Press, 2012.
    Args:
        rank (int | None): If `None` (default), full whitening (r = min(n, d)).
            If int, low-rank whitening keeping the top-r singular components.
        eps (float): Numerical stability floor.

    Attributes:
        mean (torch.Tensor): Feature means, shape `(d,)`.
        V (torch.Tensor): Right singular vectors, shape `(d, r)`.
        inv_s (torch.Tensor): Scaling factors `sqrt(n) / s_i`, shape `(r,)`.
    """

    def __init__(self, rank: int | None = None, eps: float = 1e-8):
        super().__init__(eps=eps)
        self.rank = None if rank is None else int(rank)
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("V", torch.empty(0))  # (d, r)
        self.register_buffer("inv_s", torch.empty(0))  # (r,)

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def fit(self, X: Float[torch.Tensor, "n d"]) -> "Whitening":
        n = X.shape[0]
        mean: Float[torch.Tensor, "d"] = X.mean(dim=0)
        Xc: Float[torch.Tensor, "n d"] = X - mean.unsqueeze(0)

        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

        inv_s: Float[torch.Tensor, "r"] = (
            torch.sqrt(torch.tensor(float(n), device=X.device, dtype=X.dtype)) / S
        ).clamp_max(1.0 / self.eps)
        V: Float[torch.Tensor, "d r"] = Vh.transpose(0, 1)

        if self.rank is not None:
            r = min(self.rank, V.shape[1])
            V = V[:, :r]
            inv_s = inv_s[:r]

        self.mean = mean.detach()
        self.V = V.detach()
        self.inv_s = inv_s.detach()
        self.fitted = True
        return self

    @assert_fitted
    @jaxtyped(typechecker=beartype)
    def transform(self, X: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n r"]:
        Z: Float[torch.Tensor, "n r"] = (X - self.mean) @ self.V
        return Z * self.inv_s.unsqueeze(0)
