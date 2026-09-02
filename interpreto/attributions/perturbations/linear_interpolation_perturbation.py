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

from interpreto.attributions.perturbations.base import TensorPerturbator
from interpreto.typing import TensorBaseline


class LinearInterpolationPerturbator(TensorPerturbator):
    """
    Modality-agnostic perturbator interpolating linearly between a reference point (baseline)
    and the input tensor.

    It is combined with a modality base at runtime by the explainer, which is what decides
    whether the tensor is `(1, l, d)` embeddings or `(1, 3, H, W)` pixel values.

    Serves as a base for the other interpolation-based perturbators (GradientShap), which
    override `_generate_baseline` and `_generate_alphas`.
    """

    def __init__(self, *, baseline: TensorBaseline = None, **kwargs):
        """
        Args:
            baseline (TensorBaseline, optional): The baseline value for the perturbation.
                It can be a torch.Tensor, int, float, or None. Defaults to None. The tensor
                baseline must match the shape of the input (l d for text and 3 H W for images)

        Raises:
            AssertionError: If the baseline is not a torch.Tensor, int, float, or None.
        """
        assert isinstance(baseline, (torch.Tensor, int, float, type(None)))  # noqa: UP038
        super().__init__(**kwargs)
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

    def _generate_baseline(self, processed_inputs: torch.Tensor) -> torch.Tensor:
        """
        Generates the baseline tensor for interpolation.
        Default behavior: uses a fixed baseline without noise.
        """
        baseline = self.adjust_baseline(self.baseline, processed_inputs)
        return baseline.to(processed_inputs.device).unsqueeze(0)

    def _generate_alphas(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Generates interpolation coefficients (alphas).
        Default behavior: evenly spaced values between 0 and 1.
        """
        return torch.linspace(0, 1, self.n_perturbations, device=device).view(-1, *([1] * (len(shape) - 1)))

    @jaxtyped(typechecker=beartype)
    def perturb_tensor(
        self, processed_inputs: Float[torch.Tensor, "1 l d"] | Float[torch.Tensor, "1 3 H W"]
    ) -> tuple[Float[torch.Tensor, "p l d"] | Float[torch.Tensor, "p 3 H W"], None]:
        """
        Applies linear interpolation perturbation between the baseline and the original input.

        Args:
            processed_inputs (torch.Tensor):
                tensor to interpolate: either token embeddings or pixel values.
                Shape: (1, l, d) for text, (1, 3, H, W) for images
        Returns:
            perturbed_inputs (torch.Tensor):
                Perturbed inputs, one per interpolation step.
                Shape: (p, l, d) for text, (p, 3, H, W) for images
            mask (None):
                placeholder
        """
        # construct baselines and interpolation coefficients
        baseline: Float[torch.Tensor, "1 l d"] | Float[torch.Tensor, "1 3 H W"] = self._generate_baseline(
            processed_inputs
        )
        alphas: Float[torch.Tensor, "p 1 1"] | Float[torch.Tensor, "p 1 1 1"] = self._generate_alphas(
            processed_inputs.shape, processed_inputs.device
        )

        # interpolate between the baseline and the input
        perturbed_inputs: Float[torch.Tensor, "p l d"] | Float[torch.Tensor, "p 3 H W"] = (
            1 - alphas
        ) * processed_inputs + alphas * baseline

        return perturbed_inputs, None
