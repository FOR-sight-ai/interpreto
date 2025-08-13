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
Aggregations used at the end of an attribution method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


class Aggregator:
    """
    Abstract class for aggregation made at the end of attribution methods
    """

    def aggregate(self, results: torch.Tensor, mask) -> torch.Tensor:
        """
        Get results from multiple "Inference wrappers", aggregate results and gives an explanation
        """
        return results.squeeze(0)

    def __call__(self, results: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.aggregate(results, mask)


class TorchAggregator(Aggregator):
    """
    Basic aggregator using built-in torch methods to perform aggregation
    """

    _method: Callable[[torch.Tensor, int], torch.Tensor]

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        results: Float[Tensor, "p t l"],
        mask: torch.Tensor | None = None,
    ) -> Float[Tensor, "t l"]:
        return self._method(results, dim=0)  # type: ignore


class MeanAggregator(TorchAggregator):
    """
    Mean of attributions
    """

    _method = torch.mean


class SquaredMeanAggregator(TorchAggregator):
    """
    Mean of squares of attributions
    """

    @staticmethod
    def _method(x: torch.Tensor, dim: int = 0) -> torch.Tensor:  # type: ignore
        return torch.mean(torch.square(x), dim=dim)


class SumAggregator(TorchAggregator):
    """
    Sum of attributions
    """

    _method = torch.sum


class VarianceAggregator(TorchAggregator):
    """
    Variance of attributions
    """

    _method = torch.var


class MaskwiseMeanAggregator(Aggregator):
    """Average scores for each target weighted by their perturbation mask.

    This aggregator assumes that ``results`` contains one score per
    perturbation and target and that ``mask`` indicates which input elements were
    perturbed. The returned tensor has one importance score per target and input
    element.
    """

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        results: Float[Tensor, "p t"],
        mask: Float[Tensor, "p l"],
    ) -> Float[Tensor, "t l"]:
        return torch.einsum("pt,pl->tl", results, mask) / mask.sum(dim=0)


class OcclusionAggregator(Aggregator):
    """Aggregate occlusion scores using a reference prediction.

    The first row in ``results`` must correspond to the unperturbed prediction.
    Subsequent rows represent the prediction for each perturbation. The returned
    tensor contains the mean difference between the reference and each
    perturbation weighted by the corresponding mask.
    """

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        results: Float[Tensor, "p t"],
        mask: Float[Tensor, "p l"],
    ) -> Float[Tensor, "t l"]:
        # first prediction is reference, unmodified input
        scores = results[..., 0, :] - results[..., 1:, :]
        mask = mask[..., 1:, :]
        return torch.einsum("pt,pl->tl", scores, mask) / mask.sum(dim=0)


class TrapezoidalMeanAggregator(Aggregator):
    """
    Weighted mean using the trapezoidal rule.

    This aggregator performs a weighted average across the first axis of the input tensor,
    assigning a weight of 0.5 to the first and last elements, and 1.0 to all intermediate ones.

    This reflects trapezoidal numerical integration over uniformly spaced samples.
    """

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        results: Float[Tensor, "p t l"],
        mask: torch.Tensor | None = None,
    ) -> Float[Tensor, "t l"]:
        return torch.trapezoid(results, dim=0) / (results.shape[0] - 1)
