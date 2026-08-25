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

from .base_merged import TensorPerturbator


class GaussianNoisePerturbator(TensorPerturbator):
    """
    Modality-agnostic Gaussian noise Perturbator.

    It is combined with a modality base at runtime by the explainer, which is what decides
    whether the tensor is `(1, l, d)` embeddings or `(1, 3, H, W)` pixel values.
    """

    __slots__ = ("std",)

    def __init__(self, *, std: float = 0.1, **kwargs):
        """
        Args:
            std: standard deviation of the Gaussian noise, in the units of the tensor being
                perturbed — embedding units on the text side, normalized pixel units on the
                image side.
        """
        super().__init__(**kwargs)
        self.std = std

    @jaxtyped(typechecker=beartype)
    def perturb_tensor(self, inputs: Float[torch.Tensor, "1 *rest"]) -> tuple[Float[torch.Tensor, "p *rest"], None]:
        """
        Add independent Gaussian noise to every entry of `inputs`.

        Args:
            inputs: shape `(1, l, d)` for text embeddings, `(1, 3, H, W)` for pixel values.

        Returns:
            perturbed: shape `(p, *rest)`, one independently noised copy per perturbation.
            mask: None — granularity units are recovered from the gradients afterwards, so there
                is no mask to report.
        """
        # Repeat along the perturbation axis only, whatever the rank of the trailing dimensions.
        perturbed: Float[torch.Tensor, "p *rest"] = inputs.repeat(self.n_perturbations, *(1,) * (inputs.ndim - 1))
        perturbed += torch.randn_like(perturbed) * self.std
        return perturbed, None
