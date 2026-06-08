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

from interpreto.attributions.perturbations.image_base import ImageTensorPerturbator


class GaussianNoiseImagePerturbator(ImageTensorPerturbator):
    """
    Image-side analog of `GaussianNoisePerturbator`.

    Adds independent Gaussian noise to every (channel, row, col) entry of
    `pixel_values`. Used by SmoothGrad-style methods on ViT.
    """

    __slots__ = ("n_perturbations", "std")

    def __init__(
        self,
        n_perturbations: int = 10,
        *,
        std: float = 0.1,
    ) -> None:
        """
        Args:
            n_perturbations: Number of noisy samples to generate.
            std: Standard deviation of the Gaussian noise applied per pixel-channel.
        """
        self.n_perturbations = n_perturbations
        self.std = std

    @jaxtyped(typechecker=beartype)
    def perturb_tensor(
        self, pixel_values: Float[torch.Tensor, "1 3 H W"]
    ) -> tuple[Float[torch.Tensor, "p 3 H W"], None]:
        """
        Args:
            pixel_values: Shape (1, 3, H, W).
        Returns:
            perturbed_embeds: Shape (p, 3, H, W), one independently noised copy per perturbation.
            mask: None — granularity is applied post-hoc by the aggregator.
        """
        perturbed_embeds: Float[torch.Tensor, "p 3 H W"] = pixel_values.repeat(self.n_perturbations, 1, 1, 1)
        perturbed_embeds += torch.randn_like(perturbed_embeds) * self.std
        return perturbed_embeds, None
