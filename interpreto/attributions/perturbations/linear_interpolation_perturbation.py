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
from jaxtyping import Float, Int64, jaxtyped

from interpreto.attributions.perturbations.base import Perturbator
from interpreto.typing import TensorBaseline, TensorMapping


class LinearInterpolationPerturbator(Perturbator):
    """
    Perturbation using linear interpolation between a reference point (baseline) and the input.
    This class can serve as a base for different interpolation-based perturbators.
    """

    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        baseline: TensorBaseline = None,
        n_perturbations: int = 10,
    ):
        """
        Initializes the LinearInterpolationPerturbation instance.

        Args:
            inputs_embedder (torch.nn.Module, optional): Optional module to transform inputs into embeddings. Defaults to None.
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

        if baseline is None:
            baseline = 0

        if isinstance(baseline, (int, float)):  # noqa: UP038
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
    def perturb_embeds(self, model_inputs: TensorMapping) -> tuple[TensorMapping, None]:
        """
        Applies linear interpolation perturbation between the baseline and the original embeddings.
        """
        embeddings: Float[torch.Tensor, "1 l d"] = model_inputs["inputs_embeds"]

        baseline: Float[torch.Tensor, "1 l d"] = self._generate_baseline(embeddings)
        alphas: Float[torch.Tensor, "p 1 1"] = self._generate_alphas(embeddings.shape, embeddings.device)

        pert = (1 - alphas) * embeddings + alphas * baseline  # (p, 1, l, d)

        # Flatten (p, 1, l, d) -> (p, l, d)
        model_inputs["inputs_embeds"] = pert.view(self.n_perturbations, *embeddings.shape[1:])

        # Repeat and flatten the attention mask accordingly: (1, l) -> (p, l)
        attn: Int64[torch.Tensor, "1 l"] = model_inputs["attention_mask"]
        attn = attn.unsqueeze(0).repeat(self.n_perturbations, 1, 1).reshape(self.n_perturbations, -1)
        model_inputs["attention_mask"] = attn

        return model_inputs, None
