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

from interpreto.attributions.perturbations.linear_interpolation_perturbation import (
    LinearInterpolationPerturbator,
)
from interpreto.typing import TensorBaseline


class GradientShapPerturbator(LinearInterpolationPerturbator):
    """
    Perturbator for GradientSHAP, introducing randomness both in interpolation coefficients (alphas)
    and in the baseline, to approximate the expectation over multiple noisy baselines and paths.
    """

    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        baseline: TensorBaseline = None,
        n_perturbations: int = 10,
        std: float = 0.1,
    ):
        """
        Initializes the GradientShapPerturbator.

        Args:
            inputs_embedder (torch.nn.Module, optional): Optional module to transform inputs into embeddings. Defaults to None.
            baseline (TensorBaseline, optional): The reference embedding (can be a tensor, float, int, or None). Defaults to None.
            n_perturbations (int, optional): Number of random samples for interpolation. Defaults to 10.
            std (float, optional): Standard deviation of the Gaussian noise added to the baseline. Defaults to 0.1.
        """
        super().__init__(inputs_embedder=inputs_embedder, baseline=baseline, n_perturbations=n_perturbations)
        self.std = std

    def _generate_baseline(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generates multiple noisy baselines for GradientSHAP.

        - Replicates the baseline for each interpolation step and batch element.
        - Adds Gaussian noise with standard deviation `std`.
        """
        baseline = self.adjust_baseline(self.baseline, embeddings)
        baseline = baseline.to(embeddings.device)

        baseline = baseline.unsqueeze(0).expand(1, *baseline.shape)  # (1, l, d)
        baseline = baseline.unsqueeze(0).repeat(self.n_perturbations, 1, 1, 1)  # (p, 1, l, d)
        baseline += torch.randn_like(baseline) * self.std  # noise

        return baseline

    def _generate_alphas(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Generates random interpolation coefficients (alphas) for GradientSHAP.
        """
        return torch.rand(self.n_perturbations, 1, 1, 1, device=device)
