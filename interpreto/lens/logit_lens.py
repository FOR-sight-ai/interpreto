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

"""Logit Lens implementation."""

from __future__ import annotations

import torch
from torch import nn

from interpreto.concepts.splitters import AllLayersSplitter
from interpreto.typing import LensResults


class LogitLens(nn.Module):
    """Project every residual-stream state through the model prediction head.

    Logit Lens was introduced by
    [nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
    It has no learned parameters: residual states are collected in one model
    trace and projected together through the model's native prediction path.

    The prediction head was trained on final states, so early-layer scores are
    useful for rankings and within-model comparisons rather than as calibrated
    probabilities. Inference processes one text at a time; callers can iterate
    over several texts when needed. :meth:`generate` greedily completes a prompt
    and explains the generated continuation.

    Args:
        splitter (str | AllLayersSplitter): Hugging Face repository ID or configured model wrapper
            used to collect and project all layer states.
        top_k (int): Maximum number of token or class scores returned per prediction.

    Raises:
        ValueError: If ``top_k`` is not positive.

    Examples:
        >>> from interpreto import LogitLens
        >>> lens = LogitLens("hf-internal-testing/tiny-random-gpt2", top_k=3)
        >>> results = lens("Interpreto is useful.")
        >>> list(results) == lens.splitter.activation_names
        True
    """

    def __init__(self, splitter: str | AllLayersSplitter, top_k: int = 5) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("`top_k` must be positive.")

        self.splitter = splitter if isinstance(splitter, AllLayersSplitter) else AllLayersSplitter(splitter)
        self.top_k = top_k
        self.splitter._model.eval()

    def _transform(self, activations: torch.Tensor) -> torch.Tensor:
        return activations

    def _get_logits(self, inputs: str | torch.Tensor) -> torch.Tensor:
        # Stack model depths so the prediction head handles them in one call.
        activations = torch.cat(self.splitter._trace_activations(inputs), dim=0)
        return self.splitter.apply_head(self._transform(activations))

    def _format_outputs(self, logits: torch.Tensor) -> LensResults:
        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()

        top_logits, top_indices = logits.topk(min(self.top_k, logits.shape[-1]), dim=-1)
        top_scores = (top_logits - logits.logsumexp(dim=-1, keepdim=True)).exp()
        return {
            layer_name: {
                "top_indices": top_indices[index : index + 1].detach().cpu(),
                "top_scores": top_scores[index : index + 1].detach().cpu(),
            }
            for index, layer_name in enumerate(self.splitter.activation_names)
        }

    @torch.inference_mode()
    def explain(self, inputs: str, align: bool = True) -> LensResults:
        """Return top predictions at every transformer block boundary.

        Causal language-model predictions are aligned with their observed next
        tokens by default. Classification outputs are unchanged.

        Args:
            inputs (str): One text passed to the wrapped model.
            align (bool): Whether causal predictions should target the token in
                the corresponding column.

        Returns:
            LensResults: Top indices and normalized scores for each residual-stream state.
        """
        logits = self._get_logits(inputs)
        if align and logits.ndim == 3:
            if logits.shape[1] < 2:
                raise ValueError("Aligned explanations require at least two tokens.")
            logits = logits[:, :-1]
        return self._format_outputs(logits)

    @torch.inference_mode()
    def generate(
        self,
        inputs: str,
        max_new_tokens: int = 10,
        align: bool = True,
    ) -> tuple[str, LensResults]:
        """Generate a continuation and explain its tokens.

        Predictions target the generated token displayed in the same column by
        default. Set ``align=False`` to show the prediction made after each
        generated token instead.

        Args:
            inputs (str): Prompt passed to the wrapped causal language model.
            max_new_tokens (int): Maximum number of tokens to generate.
            align (bool): Whether predictions should target the generated tokens
                displayed in the same columns.

        Returns:
            tuple[str, LensResults]: Generated continuation and its layer predictions.

        Raises:
            ValueError: If ``max_new_tokens`` is not positive.
        """
        if max_new_tokens < 1:
            raise ValueError("`max_new_tokens` must be positive.")

        tokenizer = self.splitter.tokenizer
        self.splitter.dispatch()
        model = self.splitter._model
        model_inputs = tokenizer(inputs, return_tensors="pt").to(model.get_input_embeddings().weight.device)
        sequence = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)[0]
        prompt_length = model_inputs["input_ids"].shape[1]
        generated_ids = sequence[prompt_length:]
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        start = prompt_length - 1 if align else prompt_length
        stop = -1 if align else None
        logits = self._get_logits(sequence.unsqueeze(0))[:, start:stop]
        return generated_text, self._format_outputs(logits)

    def forward(self, inputs: str, align: bool = True) -> LensResults:
        """Alias for :meth:`explain`."""
        return self.explain(inputs, align)
