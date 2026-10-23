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
Inference wrapper for image retrieval (e.g., Col-family) models.
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from interpreto.attributions.inference_wrappers.inference_wrapper import ChunkMetadata, InferenceWrapper
from interpreto.typing import IncompatibilityError


class ImageRetrievalInferenceWrapper(InferenceWrapper):
    """
    Inference wrapper for image retrieval tasks (Col-family models).

    TODO
    """

    grad_key = "pixel_values"
    padding_side = "left"

    def _model_forward(self, **inputs) -> torch.Tensor:
        """
        Forward pass through the model.

        Useless by default, but opens an entry-point for easy modification via inheritance.

        Args:
            **inputs: The inputs to be passed to the model.

        Returns:
            torch.Tensor: The embeddings.
        """
        try:
            # try model forward
            return self.model(**inputs)
        except NotImplementedError as e:
            key_shapes = {k: v.shape for k, v in inputs.items()}
            raise IncompatibilityError(
                f"The model {self.model.__class__.__name__} is not compatible with the provided inputs. "
                f"Input keys and shapes: {key_shapes}. "
                f"This can happen for T5 when applying gradient-based methods, they do the forward from the `inputs_embeds`. "
            ) from e

    @jaxtyped(typechecker=beartype)
    def _extract_targets_from_logits(
        self, logits: Float[torch.Tensor, "c ..."], chunk: ChunkMetadata
    ) -> Float[torch.Tensor, "t ..."]:
        """
        The targets are the query embeddings (text).

        Args:
            logits (torch.Tensor):
                The logits from the model.
                They all correspond to the same initial input. But it is a chunk of it.
                t is the number of tokens in the query. But we use t to signify targets.
        """
        # chunks should always be a single query
        return logits[0, chunk.sequence_slice]

    def _target_logits(
        self,
        logits: Float[torch.Tensor, "c l d"],
        targets: Float[torch.Tensor, "t d"],
        chunk: ChunkMetadata,
    ) -> Float[torch.Tensor, "c t"]:
        """
        The logits correspond to the image embeddings while the targets are the query embeddings.

        We apply max similarity between the logits and the targets.
        But we do not take the sum as an output wise explanation can be interesting.
        """
        similarities = logits[:, chunk.sequence_slice].float() @ targets.float().T  # [c, l, t]
        max_sim = similarities.max(dim=1).values  # [c, t]

        return max_sim
