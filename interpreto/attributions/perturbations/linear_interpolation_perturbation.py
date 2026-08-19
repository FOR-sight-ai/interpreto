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

from interpreto.attributions.perturbations.base import ImageTensorPerturbator, TextTensorPerturbator
from interpreto.typing import TensorBaseline


class LinearInterpolationPerturbator(TextTensorPerturbator):
    """
    Perturbation using linear interpolation between a reference point (baseline) and the input.
    This class can serve as a base for different interpolation-based perturbators.
    """

    def __init__(
        self,
        inputs_embedder: torch.nn.Module,
        baseline: TensorBaseline = None,
        n_perturbations: int = 10,
    ):
        """
        Initializes the LinearInterpolationPerturbation instance.

        Args:
            inputs_embedder (torch.nn.Module): Module to transform inputs into embeddings. Defaults to None.
            baseline (TensorBaseline, optional): The baseline value for the perturbation.
                It can be a torch.Tensor, int, float, or None. Defaults to None.
            n_perturbations (int, optional): Number of interpolation steps between baseline and input. Defaults to 10.

        Raises:
            AssertionError: If the baseline is not a torch.Tensor, int, float, or None.
        """
        assert isinstance(baseline, (torch.Tensor, int, float, type(None)))  # noqa: UP038
        super().__init__(inputs_embedder=inputs_embedder)
        self.n_perturbations = n_perturbations
        self.baseline = baseline

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def adjust_baseline(baseline: TensorBaseline, inputs: torch.Tensor) -> torch.Tensor:
        """
        Ensures the 'baseline' argument is correctly adjusted based on the shape of 'inputs' (PyTorch tensor).

        - If baseline is None, it is replaced with a tensor of zeros matching input.shape[1:].
        - If baseline is a float, it is broadcasted to input.shape[1:].
        - If baseline is a tensor, its shape must match input.shape[1:]; otherwise, an error is raised.

        Args:
            baseline: The baseline to adjust.
            inputs: The input to adjust the baseline for.

        Returns:
            The adjusted baseline.
        """
        # Shape: (batch_size, *input_shape)
        input_shape = inputs.shape[1:]

        # When all values are zero, the gradients are always NaN.
        # To avoid this, we set the baseline to a small value.
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

    def _generate_baseline(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generates the baseline tensor for interpolation.
        Default behavior: uses a fixed baseline without noise.
        """
        baseline = self.adjust_baseline(self.baseline, embeddings)
        return baseline.to(embeddings.device).unsqueeze(0)

    def _generate_alphas(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Generates interpolation coefficients (alphas).
        Default behavior: evenly spaced values between 0 and 1.
        """
        return torch.linspace(0, 1, self.n_perturbations, device=device).view(-1, *([1] * (len(shape) - 1)))

    @jaxtyped(typechecker=beartype)
    def perturb_tensor(self, inputs_embeds: Float[torch.Tensor, "1 l d"]) -> tuple[Float[torch.Tensor, "p l d"], None]:
        """
        Applies linear interpolation perturbation between the baseline and the original embeddings.

        Args:
            inputs_embeds (torch.Tensor):
                Embeddings of the input tokens.
                Shape: (1, l, d)
        Returns:
            perturbed_embeds (torch.Tensor):
                Perturbed embeddings.
                Shape: (p, l, d)
            mask (None):
                placeholder
        """
        # construct baselines and interpolation coefficients
        baseline: Float[torch.Tensor, "1 l d"] = self._generate_baseline(inputs_embeds)
        alphas: Float[torch.Tensor, "p 1 1"] = self._generate_alphas(inputs_embeds.shape, inputs_embeds.device)

        # interpolate between baseline and input embeddings
        perturbed_embeds: Float[torch.Tensor, "p l d"] = (1 - alphas) * inputs_embeds + alphas * baseline

        return perturbed_embeds, None


class LinearInterpolationImagePerturbator(ImageTensorPerturbator):
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
    def perturb_tensor(
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
        baseline: Float[torch.Tensor, "1 3 H W"] = self._generate_baseline(pixel_values)
        alphas: Float[torch.Tensor, "p 1 1 1"] = self._generate_alphas(pixel_values.shape, pixel_values.device)

        perturbed_embeds: Float[torch.Tensor, "p 3 H W"] = (1 - alphas) * pixel_values + alphas * baseline
        return perturbed_embeds, None
