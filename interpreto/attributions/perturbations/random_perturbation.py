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
Random perturbation for token-wise masking, used in LIME
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import Granularity, TokenMaskBasedPerturbator


class RandomMaskedTokenPerturbator(TokenMaskBasedPerturbator):
    """
    Perturbator adding random masking to the input tensor
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        inputs_embedder: torch.nn.Module | None = None,
        granularity: Granularity = Granularity.TOKEN,
        replace_token_id: int = 0,
        n_perturbations: int = 30,
        perturb_probability: float = 0.5,
    ):
        """
        Initialize the perturbator.

        Args:
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            inputs_embedder (torch.nn.Module | None): optional inputs embedder
            replace_token_id (int): the token id to use for replacing the masked tokens
            n_perturbations (int): the number of perturbations to generate
            perturb_probability (float): probability of perturbation
        """
        super().__init__(
            tokenizer=tokenizer,
            inputs_embedder=inputs_embedder,
            n_perturbations=n_perturbations,
            replace_token_id=replace_token_id,
            granularity=granularity,
        )
        self.perturb_probability = perturb_probability

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int) -> Float[Tensor, "{self.n_perturbations} {mask_dim}"]:
        """
        Method returning a random perturbation mask for a given input sequence.

        Args:
            mask_dim (int): The length of the sequence. Called 'l' in shapes.

        Returns:
            masks (torch.Tensor): A tensor of shape (p, l). with p the number of perturbations.
        """
        # Simplify typing
        p, l = self.n_perturbations, mask_dim

        # Generate random numbers between 0 and 1.
        rands: Float[Tensor, "{p} {l}"] = torch.rand((p, l))

        # Convert random numbers to binary masks.
        masks: Float[Tensor, "{p} {l}"] = (rands < self.perturb_probability).float()

        return masks
