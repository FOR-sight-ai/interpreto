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

"""
Sobol perturbations for NLP
"""

from __future__ import annotations

from enum import Enum

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from scipy.stats import qmc
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import TokenMaskBasedPerturbator
from interpreto.commons.granularity import Granularity


class SequenceSamplers(Enum):
    """
    Enumeration of available samplers for Sobol perturbations.
    """

    SOBOL = qmc.Sobol
    HALTON = qmc.Halton
    LatinHypercube = qmc.LatinHypercube


class SobolTokenPerturbator(TokenMaskBasedPerturbator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        inputs_embedder: torch.nn.Module | None = None,
        granularity: Granularity = Granularity.TOKEN,
        replace_token_id: int = 0,
        n_token_perturbations: int = 30,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
    ):
        """
        Initialize the perturbator.

        Args:
            tokenizer (PreTrainedTokenizer | None): Hugging Face tokenizer associated with the model
            inputs_embedder (torch.nn.Module | None): optional inputs embedder
            nb_token_perturbations (int): number of Monte Carlo samples perturbations for each token.
            granularity (str): granularity level of the perturbations (token, word, sentence, etc.)
            sampler (SequenceSamplers): Sobol sequence sampler, either `SOBOL`, `HALTON` or `LatinHypercube`.
        """
        super().__init__(
            tokenizer=tokenizer,
            inputs_embedder=inputs_embedder,
            granularity=granularity,
            n_perturbations=-1,  # TODO: find a better way to handle this, I guess, it should not be an attribute of the parent class
            replace_token_id=replace_token_id,
        )
        self.n_token_perturbations = n_token_perturbations
        self.sampler_class = sampler.value

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int) -> Float[torch.Tensor, "p {mask_dim}"]:
        """
        Generates a binary mask for each token in the sequence.

        Args:
            mask_dim (int): Length of the input sequence.

        Returns:
            masks (torch.Tensor): A tensor of shape ``((mask_dim + 2) * k, mask_dim)``.
        """
        # Simplify typing
        l, k = mask_dim, self.n_token_perturbations
        p = (l + 2) * k

        # Generate to random independent matrices A & B
        A: Float[torch.Tensor, k, l] = torch.Tensor(self.sampler_class(l).random(k))
        B: Float[torch.Tensor, k, l] = torch.Tensor(self.sampler_class(l).random(k))

        # Initialize C
        C: Float[torch.Tensor, l, k, l] = A.repeat(l, 1, 1)

        # C is a collection of C_i, where each C_i is a matrix of size (l, k)
        # with the i-th column being B[:, i] and the rest being A.
        indices = torch.arange(l)
        C[indices, :, indices] = B.T

        # We reshape stack all C_i, A, and B to match the expected shape from interpreto API.
        masks: Float[torch.Tensor, p, l] = torch.concat([A, B, C.view(l * k, l)], dim=0)

        return masks

        # # Generate index tensor for perturbations.
        # col_indices: Int[torch.Tensor, l * k] = torch.arange(l).repeat_interleave(k)

        # # Compute the start and end indices.
        # row_indices: Int[torch.Tensor, l * k] = torch.arange(l * k) + k

        # # Initial random mask.
        # initial_mask: Float[torch.Tensor, k, l] = torch.Tensor(self.sampler_class(l).random(k))

        # # Expand mask across all perturbation steps.
        # mask: Float[torch.Tensor, p, l] = initial_mask.repeat((l + 1, 1))

        # # Generate index tensor for perturbations.
        # col_indices: Int[torch.Tensor, l * k] = torch.arange(l).repeat_interleave(k)

        # # Compute the start and end indices.
        # row_indices: Int[torch.Tensor, l * k] = torch.arange(l * k) + k

        # # Flip the selected mask values without a loop
        # mask[row_indices, col_indices] = 1 - mask[row_indices, col_indices]

        # if self.sobol_indices_order == SobolIndicesOrders.TOTAL_ORDER.value:
        #     mask[k:] = 1 - mask[k:]

        # return mask
