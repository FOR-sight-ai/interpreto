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

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

# attributions/perturbations/occlusion_perturbation.py
from .base import MaskPerturbator


class OcclusionPerturbator(MaskPerturbator):  # change inheritance, might make the type checker unhappy
    """
    Modality-agnostic occlusion mask: one reference plus one perturbation per granularity unit.

    Carries no fields of its own. It is combined with a modality base at runtime by the
    `Occlusion` explainer.
    """

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int) -> Float[torch.Tensor, "p g"]:
        """Return a mask performing single-token occlusions.

        Args:
            mask_dim (int): Length of the granularity depedent input sequence.

        Returns:
            torch.Tensor: Tensor of shape ``(mask_dim + 1, mask_dim)`` where the
                first row is all zeros (reference) and the remaining rows are the
                identity matrix.
        """

        g = mask_dim
        p = g + 1
        mask: Float[torch.Tensor, "{p} {g}"] = torch.cat([torch.zeros(1, g), torch.eye(g)], dim=0)
        assert mask.shape[0] == p
        return mask
