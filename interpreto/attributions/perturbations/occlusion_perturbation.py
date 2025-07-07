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
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import TokenMaskBasedPerturbator
from interpreto.commons.granularity import Granularity


class OcclusionPerturbator(TokenMaskBasedPerturbator):
    """
    Basic class for occlusion perturbations
    """

    __slots__ = ()

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        inputs_embedder: torch.nn.Module | None = None,
        granularity: Granularity = Granularity.TOKEN,
        replace_token_id: int = 0,
    ) -> None:
        """Instantiate the perturbator.

        Args:
            tokenizer (PreTrainedTokenizer | None): Hugging Face tokenizer associated with the model.
            inputs_embedder (torch.nn.Module | None): Optional embedder used to obtain input embeddings from input IDs.
            granularity (Granularity): Level at which occlusion should be applied.
            replace_token_id (int): Token used to replace occluded elements.
        """

        super().__init__(
            tokenizer=tokenizer,
            replace_token_id=replace_token_id,
            inputs_embedder=inputs_embedder,
            n_perturbations=-1,
            granularity=granularity,
        )

    @jaxtyped(typechecker=beartype)
    def get_mask(self, mask_dim: int) -> Float[torch.Tensor, "p l"]:
        """Return a mask performing single-token occlusions.

        Args:
            mask_dim (int): Length of the input sequence.

        Returns:
            torch.Tensor: Tensor of shape ``(mask_dim + 1, mask_dim)`` where the
                first row is all zeros (reference) and the remaining rows are the
                identity matrix.
        """

        l = mask_dim
        p = l + 1
        mask: Float[torch.Tensor, "{p} {l}"] = torch.cat([torch.zeros(1, l), torch.eye(l)], dim=0)
        assert mask.shape[0] == p
        return mask
