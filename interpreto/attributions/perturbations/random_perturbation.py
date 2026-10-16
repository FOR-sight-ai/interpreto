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
Random perturbation for token-wise masking, used in LIME
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from interpreto.attributions.perturbations.base import MaskPerturbator


class RandomMaskedPerturbator(MaskPerturbator):
    """
    Perturbator masking a random subset of granularity units, used by LIME.

    It is combined with a modality base at runtime by the Lime method.
    """

    def __init__(self, *, perturb_probability: float = 0.5, **kwargs):
        """
        Args:
            perturb_probability (float): probability that a unit is masked.
        """
        super().__init__(**kwargs)
        self.perturb_probability = perturb_probability

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int) -> Float[Tensor, "{self.n_perturbations} {mask_dim}"]:
        """
        Return a random perturbation mask of shape `(n_perturbations, g)`.

        Args:
            mask_dim (int): number of granularity units `g`.

        Returns:
            torch.Tensor: mask of shape `(p, g)`; `1` = masked, `0` = kept.
        """
        p, l = self.n_perturbations, mask_dim
        rands: Float[Tensor, "{p} {l}"] = torch.rand((p, l))
        masks: Float[Tensor, "{p} {l}"] = (rands < self.perturb_probability).float()
        return masks
