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
Inference wrapper for image classification (e.g., ViT) models.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.inference_wrappers.inference_wrapper import InferenceModes, InferenceWrapper


class ImageClassificationInferenceWrapper(InferenceWrapper):
    """
    Inference wrapper for image classification tasks (ViT-family models).

    Mirrors `ClassificationInferenceWrapper` with three modality-specific overrides:
    `__init__` drops the `pad_token_id` lookup (image configs have no pad token),
    `_prepare_inputs` stacks pixel tensors without padding, and `_compute_gradients`
    differentiates w.r.t. `pixel_values` and collapses the channel dimension.
    """

    padding_side = "right"  # ignored, it prevents error raising
    grad_key = "pixel_values"

    def __init__(
        self,
        model: PreTrainedModel,
        gradients: bool = False,
        input_x_gradient: bool = True,
        batch_size: int = 4,
        device: torch.device | None = None,
        mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
    ):
        self.pad_token_id = None
        super().__init__(model, gradients, input_x_gradient, batch_size, device, mode)

    @jaxtyped(typechecker=beartype)
    def _extract_targets_from_logits(self, logits: Float[torch.Tensor, "b c"], chunk) -> Int[torch.Tensor, "b 1"]:
        """
        In classification, if no targets are specified, we explain the predicted class.
        The predicted class corresponds to the highest logits.
        Therefore, the target is the argmax of the output logits for each input sample.
        """
        return logits.argmax(dim=-1, keepdim=True)

    def _target_logits(
        self, logits: Float[torch.Tensor, "b c"], targets: Int[torch.Tensor, "t"], chunk
    ) -> Float[torch.Tensor, "b t"]:
        """
        For each sample, the targets specify which logits to extract.

        The target is common between each sample.
        """
        return logits[:, targets]
