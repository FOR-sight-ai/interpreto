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
Sobol perturbations for NLP
"""

from __future__ import annotations

from enum import Enum

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from scipy.stats import qmc

from interpreto.attributions.perturbations.base import MaskPerturbator


class SequenceSamplers(Enum):
    """
    Enumeration of available samplers for Sobol perturbations.
    """

    SOBOL = qmc.Sobol
    HALTON = qmc.Halton
    LatinHypercube = qmc.LatinHypercube


class SobolPerturbator(MaskPerturbator):
    """
    Perturbator producing Sobol (quasi-Monte-Carlo) masks for Sobol attribution.

    It is combined with a modality base at runtime by the Sobol method.
    """

    def __init__(
        self,
        *,
        n_token_perturbations: int = 16,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        is_binarized: bool = True,
        **kwargs,
    ):
        """
        Args:
            n_token_perturbations (int): Monte-Carlo samples per granularity unit.
            sampler (SequenceSamplers): Sobol sequence sampler, either `SOBOL`, `HALTON` or `LatinHypercube`.
            is_binarized (bool): whether the quasi-Monte-Carlo design is thresholded into a binary
                mask. Tokens are discrete so the text side requires it; images blend continuously.
        """
        # total p = (g + 2) * k is determined at mask time, not up front.
        super().__init__(n_perturbations=-1, **kwargs)
        self.n_token_perturbations = n_token_perturbations
        self.sampler_class = sampler.value
        self.is_binarized = is_binarized

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int, **kwargs) -> Float[torch.Tensor, "p {mask_dim}"]:
        """
        Generates a quasi-Monte-Carlo mask for each granularity unit in the sequence.

        Args:
            mask_dim (int): number of granularity units `g`.

        Returns:
            torch.Tensor: shape `((g + 2) * k, g)`.
        """
        l, k = mask_dim, self.n_token_perturbations
        p = (l + 2) * k

        # two independent random matrices A & B
        AB: Float[torch.Tensor, k, 2 * l] = torch.Tensor(self.sampler_class(2 * l).random(k))
        A: Float[torch.Tensor, k, l] = AB[:, :l]
        B: Float[torch.Tensor, k, l] = AB[:, l:]

        # C is a collection of C_i; C_i is A with its i-th column replaced by B[:, i]
        C: Float[torch.Tensor, l, k, l] = A.repeat(l, 1, 1)
        indices = torch.arange(l)
        C[indices, :, indices] = B.T

        masks: Float[torch.Tensor, p, l] = torch.concat([A, B, C.view(l * k, l)], dim=0)

        if self.is_binarized:
            return (masks < 0.5).float()
        return masks
