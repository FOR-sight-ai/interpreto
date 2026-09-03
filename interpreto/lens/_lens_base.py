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

"""Shared all-layer lens implementation."""

from __future__ import annotations

import torch
from torch import nn

from interpreto.concepts.splitters import AllLayersSplitter
from interpreto.typing import LensResults


class BaseLens(nn.Module):
    """Decode every residual-stream state exposed by an all-layer splitter."""

    def __init__(self, splitter: AllLayersSplitter, top_k: int = 5) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("`top_k` must be positive.")

        self.splitter = splitter
        self.top_k = top_k
        self.splitter._model.eval()

    def _transform(self, activations: torch.Tensor) -> torch.Tensor:
        return activations

    def _get_logits(self, inputs: str) -> torch.Tensor:
        # Stack model depths so the prediction head handles them in one call.
        activations = torch.cat(self.splitter.get_activations(inputs), dim=0)
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
    def explain(self, inputs: str) -> LensResults:
        """Return top predictions at every transformer block boundary.

        Args:
            inputs (str): Text passed to the wrapped model.

        Returns:
            LensResults: Top indices and normalized scores for each residual-stream state.
        """
        return self._format_outputs(self._get_logits(inputs))

    def forward(self, inputs: str) -> LensResults:
        """Alias for :meth:`explain`."""
        return self.explain(inputs)
