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
Single-unit occlusion for images. Image-side analog of `OcclusionPerturbator`.
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from interpreto.attributions.perturbations.image_base import ImageMaskPerturbator
from interpreto.commons.granularity import ImageGranularity


class OcclusionImagePerturbator(ImageMaskPerturbator):
    """
    Occlusion perturbator: one reference (nothing masked) plus one perturbation
    per granularity unit, each masking exactly that single unit.
    """

    __slots__ = ()

    def __init__(
        self,
        granularity: ImageGranularity = ImageGranularity.PATCH,
        replace_value: float = 0.0,
        patch_size: int | None = None,
    ) -> None:
        """
        Args:
            granularity (ImageGranularity): unit over which occlusion is applied.
            replace_value (float): baseline written into the occluded unit.
            patch_size (int): patch side length (reconciled by the explainer).
        """
        # n_perturbations is determined by g at mask time (l + 1), not up front.
        super().__init__(
            granularity=granularity,
            n_perturbations=-1,
            replace_value=replace_value,
            patch_size=patch_size,
        )

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int) -> Float[torch.Tensor, "p l"]:
        """
        Return single-unit occlusion masks.

        Args:
            mask_dim (int): number of granularity units `g`.

        Returns:
            torch.Tensor: shape `(g + 1, g)`. Row 0 is all-zeros (reference,
                nothing masked); the remaining `g` rows form the identity, each
                masking exactly one unit.
        """
        l = mask_dim
        p = l + 1
        mask: Float[torch.Tensor, "{p} {l}"] = torch.cat([torch.zeros(1, l), torch.eye(l)], dim=0)
        assert mask.shape[0] == p
        return mask
