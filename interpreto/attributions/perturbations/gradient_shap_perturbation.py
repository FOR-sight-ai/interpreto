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
"""Perturbation for GradientSHAP."""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from interpreto.attributions.perturbations.linear_interpolation_perturbation import LinearInterpolationPerturbator


class GradientShapPerturbator(LinearInterpolationPerturbator):
    """
    Perturbator for GradientSHAP, introducing randomness both in interpolation coefficients (alphas)
    and in the baseline, to approximate the expectation over multiple noisy baselines and paths.
    """

    def __init__(self, *, std: float = 0.1, **kwargs):
        """
        Args:
            std (float, optional): Standard deviation of the Gaussian noise added to the baseline.
                Defaults to 0.1.
        """
        super().__init__(**kwargs)
        self.std = std

    @jaxtyped(typechecker=beartype)
    def _generate_baseline(
        self, processed_inputs: Float[torch.Tensor, "1 l d"] | Float[torch.Tensor, "1 3 H W"]
    ) -> Float[torch.Tensor, "p l d"] | Float[torch.Tensor, "p 3 H W"]:
        """
        Generates multiple noisy baselines for GradientSHAP.

        - Replicates the baseline for each interpolation step and batch element.
        - Adds Gaussian noise with standard deviation `std`.
        """
        baseline = self.adjust_baseline(self.baseline, processed_inputs)
        baseline = baseline.to(processed_inputs.device)

        baseline: Float[torch.Tensor, "p l d"] | Float[torch.Tensor, "p 3 H W"] = baseline.unsqueeze(0).repeat(
            self.n_perturbations, *(1,) * (processed_inputs.ndim - 1)
        )
        baseline += torch.randn_like(baseline) * self.std  # noise

        return baseline

    @jaxtyped(typechecker=beartype)
    def _generate_alphas(
        self, shape: torch.Size, device: torch.device
    ) -> Float[torch.Tensor, "p 1 1"] | Float[torch.Tensor, "p 1 1 1"]:
        """
        Generates random interpolation coefficients (alphas) for GradientSHAP.
        """
        return torch.rand(self.n_perturbations, *(1,) * (len(shape) - 1), device=device)
