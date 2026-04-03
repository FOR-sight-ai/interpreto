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


class GenerationInferenceWrapper(InferenceWrapper):
    """
    Inference wrapper for generation tasks.
    """

    @property
    def padding_side(self):
        return "left"

    def _prepare_inputs(self, inputs, for_gradients: bool = False):
        """
        Add position ids after padding left-padded batches.

        Decoder-only models usually infer correct positions from ``input_ids``, but
        attribution gradients are computed from ``inputs_embeds`` and therefore lose
        the tokenizer-level padding information. Recomputing positions from the
        attention mask keeps batched ``input_ids`` and ``inputs_embeds`` aligned.
        """
        padded_inputs = super()._prepare_inputs(inputs, for_gradients=for_gradients)
        if "position_ids" in padded_inputs:
            return padded_inputs
        if "attention_mask" not in padded_inputs:
            return padded_inputs

        position_ids = padded_inputs["attention_mask"].long().cumsum(dim=-1) - 1  # type: ignore[index]
        position_ids.masked_fill_(padded_inputs["attention_mask"] == 0, 0)  # type: ignore[index]
        padded_inputs["position_ids"] = position_ids
        return padded_inputs

    def _extract_targets_from_logits(self, logits):
        raise NotImplementedError(
            "GenerationInferenceWrapper does not support computing targets from logits."
            "Text generation is left to the user, the generated text should then be provided as targets."
        )

    @jaxtyped(typechecker=beartype)
    def _target_logits(
        self, logits: Float[torch.Tensor, "b l v"], targets: Int[torch.Tensor, "t"]
    ) -> Float[torch.Tensor, "b t"]:
        """
        For each output token of a chunk (as defined by `_call_batch`),
        select the logits corresponding to the initially generated text.

        The targets are shared between each element of a chunk,
        as they correspond the perturbed versions of the same input.

        Args:
            logits (torch.Tensor):
                The output logits of a generation model. (batch, seq_len, vocabulary).
                The seq_len corresponds here to the initial inputs and targets concatenated.
            targets (torch.Tensor):
                Indices of the generated tokens in the vocabulary,
                serves to extract the pertinent logits from the model's output.
                It thus corresponds to the t last tokens of the logits.

        Returns:
            targeted_logits (torch.Tensor):
                The logits corresponding to the target text given as input to `explain`.
        """
        t = targets.shape[0]

        # Select the t last logits
        # We shift the select output by one to the left as we used concatenated our targets to the inputs.
        # Therefore, the last logit vector correspond to the token generated after our target.
        # We can thus ignore it. TODO: see if we should do this modification before the forward
        last_logits: Float[torch.Tensor, f"b {t} v"] = logits[:, -(t + 1) : -1]

        # apply indexing
        return last_logits[:, torch.arange(t), targets]
