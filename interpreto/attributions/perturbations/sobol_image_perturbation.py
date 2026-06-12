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
Sobol perturbations for images. Image-side analog of `SobolTokenPerturbator`.
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from interpreto.attributions.perturbations.image_base import ImageMaskPerturbator
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers
from interpreto.commons.granularity import ImageGranularity


class SobolImagePerturbator(ImageMaskPerturbator):
    """
    Perturbator producing Sobol (quasi-Monte-Carlo) masks for Sobol attribution.
    """

    __slots__ = ("n_token_perturbations", "sampler_class")

    def __init__(
        self,
        granularity: ImageGranularity = ImageGranularity.PATCH,
        replace_value: float = 0.0,
        n_token_perturbations: int = 16,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        patch_size: int | None = None,
    ):
        """
        Args:
            granularity (ImageGranularity): unit over which masks are defined.
            replace_value (float): baseline written into masked positions.
            n_token_perturbations (int): Monte-Carlo samples per granularity unit.
            sampler (SequenceSamplers): `SOBOL`, `HALTON`, or `LatinHypercube`.
            patch_size (int): patch side length (reconciled by the explainer).
        """
        # total p = (g + 2) * k is determined at mask time, not up front.
        super().__init__(
            granularity=granularity,
            n_perturbations=-1,
            replace_value=replace_value,
            patch_size=patch_size,
        )
        self.n_token_perturbations = n_token_perturbations
        self.sampler_class = sampler.value

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int, **kwargs) -> Float[torch.Tensor, "p {mask_dim}"]:
        """
        Generates a binary mask for each token in the sequence.

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

        return (masks < 0.5).float()
