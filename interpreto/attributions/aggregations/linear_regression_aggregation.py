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
Aggregator for LIME and KernelSHAP
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn import functional as F

from interpreto.attributions.aggregations.base import Aggregator

DistancesFromMaskProtocol = Callable[[Float[Tensor, "p l"]], Float[Tensor, "p"]]
SimilarityKernelProtocol = Callable[[Float[Tensor, "p"], float], Float[Tensor, "p"]]


def hamming_distance(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hamming distance between a mask and the original input
    """
    return torch.sum(mask, dim=1).float()


def euclidean_distance(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distance between a mask and the original input
    """
    return torch.norm(mask, dim=1)


def cosine_distance(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine distance between a mask and the original input
    """
    return 1 - F.cosine_similarity(torch.ones_like(mask), mask, dim=1)


class DistancesFromMask(Enum):
    """
    Enumeration of available distance functions.
    """

    HAMMING = staticmethod(hamming_distance)
    EUCLIDEAN = staticmethod(euclidean_distance)
    COSINE = staticmethod(cosine_distance)


def exponential_kernel(distances: torch.Tensor, kernel_width: float) -> torch.Tensor:
    """
    Compute the exponential kernel for a given distance matrix
    """
    return torch.exp(-(distances**2) / (kernel_width**2))


def ones_kernel(distances: torch.Tensor, kernel_width: float) -> torch.Tensor:
    """
    Compute the ones kernel for a given distance matrix
    """
    return torch.ones_like(distances)


class Kernels(Enum):
    """
    Enumeration of available kernels.
    """

    EXPONENTIAL = staticmethod(exponential_kernel)
    ONES = staticmethod(ones_kernel)


def default_kernel_width_fn(mask: torch.Tensor) -> float:
    """
    Compute the default kernel width for a given mask
    """
    return (mask.shape[1] ** 0.5) * 0.75  # kernel parameter inspired by LIME


class LinearRegressionAggregator(Aggregator):
    """
    Aggregator for masked perturbations using a linear model
    """

    def __init__(
        self,
        distance_function: DistancesFromMaskProtocol | None = None,  # noqa: UP036
        similarity_kernel: SimilarityKernelProtocol = exponential_kernel,
        kernel_width: float | Callable | None = None,
    ):
        """
        Initialize the aggregator.

        Args:
            distance_function (DistancesFromMaskProtocol): distance function used to compute the similarity between perturbations and the original input from the mask.
            similarity_kernel (Callable): similarity kernel used to compute the similarity between perturbations and the original input
            kernel_width (float | Callable): kernel width used to compute the similarity between perturbations and the original input
        """
        self.distance_function = distance_function
        self.similarity_kernel = similarity_kernel
        self.kernel_width = kernel_width if kernel_width is not None else default_kernel_width_fn

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        results: Float[Tensor, "p t"],
        mask: Float[Tensor, "p l"],
    ) -> Float[Tensor, "t l"]:
        """
        Aggregate the results of the perturbations using a linear model.
        We use the closed form solution for the weighted least squares problem.

        Args:
            results (torch.Tensor): The results of the perturbations. Shape: (p, t).
                p the number of perturbations
                t the number of target classes/tokens
            mask (torch.Tensor): The mask used for the perturbations. Shape: (p, l).
                l the length of the sequence
        Returns:
            torch.Tensor: The aggregated results. Shape: (t, l).
        """
        # Simplify typing
        p, l = mask.shape
        t = results.shape[1]

        if results.shape[0] != p:
            raise ValueError(
                "The number of perturbations must match the number of results."
                + f"Got {results.shape} perturbations and {results.shape} results."
            )

        # Define the weights for the linear model
        if self.distance_function is not None:  # LIME
            # Compute distance between perturbations and original input using the mask
            distances: Float[Tensor, p] = self.distance_function(mask)

            # Compute the similarities between perturbations and original input using the distances
            if isinstance(self.kernel_width, Callable):
                kernel_width: float = self.kernel_width(mask)
            else:
                assert isinstance(self.kernel_width, float) or isinstance(self.kernel_width, int)
                kernel_width: float = self.kernel_width
            weights: Float[Tensor, p] = self.similarity_kernel(distances, kernel_width)
        else:  # Kernel SHAP
            weights: Float[Tensor, p] = self.similarity_kernel(mask[:, 0], 1.0)

        # We take the opposite of the mask
        # Otherwise the linear treat the perturbation of a feature as passing from 0 to 1 for this feature
        # We want exactly the opposite
        mask = 1 - mask

        # Compute closed form solution for the linear model
        # Add a constant column for the bias term
        X: Float[Tensor, p, l + 1] = torch.cat([torch.ones(results.shape[0], 1, device=mask.device), mask], dim=1)

        # formula from wikipedia: https://en.wikipedia.org/wiki/Weighted_least_squares
        # \theta =(X^T w X)^{-1} (X^T w y)
        XT_W: Float[Tensor, l + 1, p] = X.T * weights
        epsilon: Float[Tensor, l + 1, l + 1] = (
            torch.eye(XT_W.shape[0], device=XT_W.device) * 1e-6
        )  # To prevent inversion errors
        theta: Float[Tensor, l + 1, t] = torch.inverse(XT_W @ X + epsilon) @ (XT_W @ results)

        # exclude the bias term
        token_importance: Float[Tensor, t, l] = theta[1:].T

        return token_importance
