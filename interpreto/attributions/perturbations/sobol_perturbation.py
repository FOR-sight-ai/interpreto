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
from transformers import PreTrainedTokenizerBase

from interpreto.attributions.perturbations.base import ImageMaskPerturbator, TextMaskPerturbator
from interpreto.commons.granularity import ImageGranularity, TextGranularity


class SequenceSamplers(Enum):
    """
    Enumeration of available samplers for Sobol perturbations.
    """

    SOBOL = qmc.Sobol
    HALTON = qmc.Halton
    LatinHypercube = qmc.LatinHypercube


class SobolTokenPerturbator(TextMaskPerturbator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        granularity: TextGranularity = TextGranularity.TOKEN,
        replace_token_id: int = 0,
        n_token_perturbations: int = 16,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
    ):
        """
        Initialize the perturbator.

        Args:
            tokenizer (PreTrainedTokenizerBase | None): Hugging Face tokenizer associated with the model
            inputs_embedder (torch.nn.Module | None): optional inputs embedder
            nb_token_perturbations (int): number of Monte Carlo samples perturbations for each token.
            granularity (str): granularity level of the perturbations (token, word, sentence, etc.)
            sampler (SequenceSamplers): Sobol sequence sampler, either `SOBOL`, `HALTON` or `LatinHypercube`.
        """
        super().__init__(
            tokenizer=tokenizer,
            granularity=granularity,
            n_perturbations=-1,  # TODO: find a better way to handle this, I guess, it should not be an attribute of the parent class
            replace_token_id=replace_token_id,
        )
        self.n_token_perturbations = n_token_perturbations
        self.sampler_class = sampler.value

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int, **kwargs) -> Float[torch.Tensor, "p {mask_dim}"]:
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
        AB: Float[torch.Tensor, k, 2 * l] = torch.Tensor(self.sampler_class(2 * l).random(k))
        A: Float[torch.Tensor, k, l] = AB[:, :l]
        B: Float[torch.Tensor, k, l] = AB[:, l:]

        # Initialize C
        C: Float[torch.Tensor, l, k, l] = A.repeat(l, 1, 1)

        # C is a collection of C_i, where each C_i is a matrix of size (l, k)
        # with the i-th column being B[:, i] and the rest being A.
        indices = torch.arange(l)
        C[indices, :, indices] = B.T

        # We reshape stack all C_i, A, and B to match the expected shape from interpreto API.
        masks: Float[torch.Tensor, p, l] = torch.concat([A, B, C.view(l * k, l)], dim=0)

        return (masks < 0.5).float()


class SobolImagePerturbator(ImageMaskPerturbator):
    """
    Perturbator producing Sobol (quasi-Monte-Carlo) masks for Sobol attribution.
    """

    def __init__(
        self,
        granularity: ImageGranularity = ImageGranularity.PATCH,
        replace_value: float = 0.0,
        n_token_perturbations: int = 16,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        patch_size: int | None = None,
    ):
        """
        Args:
            granularity (ImageGranularity): unit over which masks are defined.
            replace_value (float): baseline written into masked positions.
            n_token_perturbations (int): Monte-Carlo samples per granularity unit.
            sampler (SequenceSamplers): `SOBOL`, `HALTON`, or `LatinHypercube`.
            patch_size (int): patch side length (reconciled by the explainer).
        """
        # total p = (g + 2) * k is determined at mask time, not up front.
        super().__init__(
            granularity=granularity,
            n_perturbations=-1,
            replace_value=replace_value,
            patch_size=patch_size,
        )
        self.n_token_perturbations = n_token_perturbations
        self.sampler_class = sampler.value

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int, **kwargs) -> Float[torch.Tensor, "p {mask_dim}"]:
        """
        Generates a binary mask for each token in the sequence.

        Args:
            mask_dim (int): number of granularity units `g`.

        Returns:
            torch.Tensor: shape `((g + 2) * k, g)`.
        """
        l, k = mask_dim, self.n_token_perturbations
        p = (l + 2) * k

        # two independent random matrices A & B
        AB: Float[torch.Tensor, k, 2 * l] = torch.Tensor(self.sampler_class(2 * l).random(k))
        A: Float[torch.Tensor, k, l] = AB[:, :l]
        B: Float[torch.Tensor, k, l] = AB[:, l:]

        # C is a collection of C_i; C_i is A with its i-th column replaced by B[:, i]
        C: Float[torch.Tensor, l, k, l] = A.repeat(l, 1, 1)
        indices = torch.arange(l)
        C[indices, :, indices] = B.T

        masks: Float[torch.Tensor, p, l] = torch.concat([A, B, C.view(l * k, l)], dim=0)

        return (masks < 0.5).float()
