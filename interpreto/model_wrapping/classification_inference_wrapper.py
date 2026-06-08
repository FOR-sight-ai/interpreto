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
from jaxtyping import Float, Int, jaxtyped

from interpreto.model_wrapping.inference_wrapper import InferenceWrapper


class TextClassificationInferenceWrapper(InferenceWrapper):
    """
    Inference wrapper for classification tasks.
    """

    padding_side = "right"

    @jaxtyped(typechecker=beartype)
    def _extract_targets_from_logits(self, logits: Float[torch.Tensor, "b c"]) -> Int[torch.Tensor, "b 1"]:
        """
        In classification, if no targets are specified, we explain the predicted class.
        The predicted class corresponds to the highest logits.
        Therefore, the target is the argmax of the output logits for each input sample.
        """
        return logits.argmax(dim=-1, keepdim=True)

    def _target_logits(
        self, logits: Float[torch.Tensor, "b c"], targets: Int[torch.Tensor, "t"]
    ) -> Float[torch.Tensor, "b t"]:
        """
        For each sample, the targets specify which logits to extract.

        The target is common between each sample.
        """
        return logits[:, targets]
