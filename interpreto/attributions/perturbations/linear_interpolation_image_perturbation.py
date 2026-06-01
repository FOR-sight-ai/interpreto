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

"""Image-side linear-interpolation perturbation (Integrated Gradients)."""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from interpreto.attributions.perturbations.image_base import ImageEmbeddingsPerturbator
from interpreto.typing import TensorBaseline


class LinearInterpolationImagePerturbator(ImageEmbeddingsPerturbator):
    """
    Image-side analog of `LinearInterpolationPerturbator`.

    Interpolates linearly between a baseline and the input `pixel_values`
    `(1, 3, H, W)`. Used by image Integrated Gradients; base class for
    `GradientShapImagePerturbator`. Mirrors the text version exactly except
    that it operates in raw pixel space (no `inputs_embedder` indirection).
    """

    __slots__ = ("n_perturbations", "baseline")

    def __init__(
        self,
        baseline: TensorBaseline = None,
        n_perturbations: int = 10,
    ) -> None:
        """
        Args:
            baseline (TensorBaseline, optional): baseline value (torch.Tensor, int, float, or None).
            n_perturbations (int): number of interpolation steps between baseline and input.
        """
        assert isinstance(baseline, (torch.Tensor, int, float, type(None)))  # noqa: UP038
        self.n_perturbations = n_perturbations
        self.baseline = baseline

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def adjust_baseline(baseline: TensorBaseline, inputs: torch.Tensor) -> torch.Tensor:
        """
        Ensures the 'baseline' is correctly shaped relative to 'inputs'.

        - If baseline is None or zero, it is replaced with a small constant (1e-6) to avoid NaN gradients.
        - If baseline is a float/int, it is broadcast to `inputs.shape[1:]` (= (3, H, W)).
        - If baseline is a tensor, its shape must match `inputs.shape[1:]`.
        """
        input_shape = inputs.shape[1:]

        # When all values are zero, the gradients are always NaN; nudge to a small value.
        if baseline is None or (isinstance(baseline, int | float) and baseline in [0, 0.0]):
            baseline = 1e-6

        if isinstance(baseline, int | float):
            return torch.full(input_shape, baseline, dtype=inputs.dtype, device=inputs.device)
        if not isinstance(baseline, torch.Tensor):
            raise TypeError(f"Expected baseline to be a torch.Tensor, int, or float, but got {type(baseline)}.")
        if baseline.shape != input_shape:
            raise ValueError(f"Baseline shape {baseline.shape} does not match expected shape {input_shape}.")
        if baseline.dtype != inputs.dtype:
            raise ValueError(f"Baseline dtype {baseline.dtype} does not match expected dtype {inputs.dtype}.")
        return baseline

    def _generate_baseline(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Default: a single fixed baseline without noise, shape (1, 3, H, W)."""
        baseline = self.adjust_baseline(self.baseline, pixel_values)
        return baseline.to(pixel_values.device).unsqueeze(0)

    def _generate_alphas(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Default: evenly spaced interpolation coefficients in [0, 1], shape (p, 1, 1, 1)."""
        return torch.linspace(0, 1, self.n_perturbations, device=device).view(-1, *([1] * (len(shape) - 1)))

    @jaxtyped(typechecker=beartype)
    def perturb_embeds(
        self, pixel_values: Float[torch.Tensor, "1 3 H W"]
    ) -> tuple[Float[torch.Tensor, "p 3 H W"], None]:
        """
        Linearly interpolate between the baseline and the input pixels.

        Args:
            pixel_values: Shape (1, 3, H, W).
        Returns:
            perturbed_embeds: Shape (p, 3, H, W).
            mask: None — granularity is applied post-hoc by the aggregator.
        """
        baseline: Float[torch.Tensor, "_ 3 H W"] = self._generate_baseline(pixel_values)
        alphas: Float[torch.Tensor, "p 1 1 1"] = self._generate_alphas(pixel_values.shape, pixel_values.device)

        perturbed_embeds: Float[torch.Tensor, "p 3 H W"] = (1 - alphas) * pixel_values + alphas * baseline
        return perturbed_embeds, None
