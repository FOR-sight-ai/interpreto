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

from __future__ import annotations

from enum import Enum

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from interpreto.attributions.aggregations.base import Aggregator


class SobolIndicesOrders(Enum):
    """
    Enumeration of available Sobol indices orders.
    """

    FIRST_ORDER = "first order"
    TOTAL_ORDER = "total order"


class SobolAggregator(Aggregator):
    """
    Aggregates Sobol indices from model outputs.
    """

    def __init__(
        self,
        n_input_perturbations: int,
        sobol_indices_order: SobolIndicesOrders = SobolIndicesOrders.FIRST_ORDER,
    ):
        """
        Initialize the aggregator.

        Args:
            n_input_perturbations (int): Number of token perturbations.
            sobol_indices_order (SobolIndicesOrders): Sobol indices order, either `FIRST_ORDER` or `TOTAL_ORDER`.
        """
        self.n_input_perturbations = n_input_perturbations
        self.sobol_indices_order = sobol_indices_order.value

    @jaxtyped(typechecker=beartype)
    def aggregate(self, results: Float[torch.Tensor, "p t"], mask: torch.Tensor) -> Float[torch.Tensor, "t l"]:  # noqa: UP037
        """
        Compute the Sobol indices from the model outputs perturbed inputs.

        Args:
            results (torch.Tensor): The model outputs on perturbed inputs. Shape: (p, t) with p = (l + 2) * k
                - l is the number of elements perturbed.
                - k is the number of perturbations per element.
                - t is the targets dimensions, i.e. the number of classes or number of generated tokens.
            mask (torch.Tensor): Ignored.

        Returns:
            token_importance (torch.Tensor): The Sobol attribution indices for each token. Shape: (l,)
        """
        # simplify typing
        k = self.n_input_perturbations
        l = (results.shape[0] // k) - 2
        t = results.shape[1]

        # Extract initial matrices
        fA: Float[torch.Tensor, k, t] = results[:k]
        fB: Float[torch.Tensor, k, t] = results[k : 2 * k]
        fC: Float[torch.Tensor, l, k, t] = results[2 * k :].view(l, k, t)

        var_fA: Float[torch.Tensor, 1, t] = torch.var(fA, dim=0, keepdim=True)

        token_importance: Float[torch.Tensor, l, t]
        if self.sobol_indices_order == SobolIndicesOrders.FIRST_ORDER.value:
            # (var(f(A)) - 0.5*mean(f(B) - f(C))^2) / var(f(A))
            fBfC_diff: Float[torch.Tensor, l, k, t] = fB.unsqueeze(0) - fC
            fBfC_mean_square_diff: Float[torch.Tensor, l, t] = torch.mean(fBfC_diff**2, dim=1) / 2
            token_importance: Float[torch.Tensor, l, t] = (var_fA - fBfC_mean_square_diff) / (var_fA + 1e-6)
        elif self.sobol_indices_order == SobolIndicesOrders.TOTAL_ORDER.value:
            # mean(f(A) - f(C))^2) / 2var(f(A))
            fAfC_diff: Float[torch.Tensor, l, k, t] = fA.unsqueeze(0) - fC
            fAfC_mean_square_diff: Float[torch.Tensor, l, t] = torch.mean(fAfC_diff**2, dim=1)
            token_importance: Float[torch.Tensor, l, t] = fAfC_mean_square_diff / (2 * var_fA + 1e-6)
        else:
            raise ValueError(f"Unknown Sobol indices order: {self.sobol_indices_order}")

        return token_importance.T
