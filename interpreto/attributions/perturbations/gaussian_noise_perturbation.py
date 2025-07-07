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

from interpreto.attributions.perturbations.base import Perturbator
from interpreto.typing import TensorMapping


class GaussianNoisePerturbator(Perturbator):
    """
    Perturbator adding gaussian noise to the input tensor
    """

    __slots__ = ("n_perturbations", "std")

    def __init__(
        self,
        inputs_embedder: torch.nn.Module | None = None,
        n_perturbations: int = 10,
        *,
        std: float = 0.1,
    ) -> None:
        """Instantiate the perturbator.

        Args:
            inputs_embedder (torch.nn.Module | None): Optional embedder used to obtain input embeddings from input IDs.
            n_perturbations (int): Number of noisy samples to generate.
            std (float): Standard deviation of the Gaussian noise.
        """

        super().__init__(inputs_embedder)
        self.n_perturbations = n_perturbations
        self.std = std

    @jaxtyped(typechecker=beartype)
    def perturb_embeds(self, model_inputs: TensorMapping) -> tuple[TensorMapping, None]:
        """Apply Gaussian noise perturbations on ``inputs_embeds``.

        Args:
            model_inputs (TensorMapping): Mapping containing the ``inputs_embeds`` tensor and
                associated masks.

        Returns:
            tuple: The perturbed mapping and ``None`` as no mask is produced.
        """

        embeddings: Float[torch.Tensor, "b l d"] = model_inputs["inputs_embeds"]
        model_inputs["inputs_embeds"] = embeddings.repeat(self.n_perturbations, 1, 1)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(self.n_perturbations, 1)

        # add noise
        # TODO: check if we should not limit this to relevant tokens (not padding, end of sequence, etc.)
        model_inputs["inputs_embeds"] += torch.randn_like(model_inputs["inputs_embeds"]) * self.std

        return model_inputs, None
