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

"""Tuned Lens implementation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from interpreto.concepts.splitters import AllLayersSplitter

from ._lens_base import BaseLens


class TunedLens(BaseLens):
    """Learn one affine residual translator for each non-final model state.

    Translators are initialized to zero, making the initial Tuned Lens identical
    to a Logit Lens. During fitting they are trained together to match the final
    model distribution.

    Args:
        splitter (AllLayersSplitter): Model wrapper used to collect and project all layer states.
        top_k (int): Maximum number of token or class scores returned per prediction.

    Examples:
        >>> from interpreto import AllLayersSplitter, TunedLens
        >>> splitter = AllLayersSplitter("hf-internal-testing/tiny-random-gpt2")
        >>> lens = TunedLens(splitter, top_k=3)
        >>> losses = lens.fit(["Interpreto is useful."], epochs=1)
        >>> results = lens("Interpreto is useful.")
    """

    def __init__(self, splitter: AllLayersSplitter, top_k: int = 5) -> None:
        super().__init__(splitter, top_k)
        hidden_size = splitter._model.config.hidden_size
        reference_parameter = next(
            parameter for parameter in splitter._model.parameters() if parameter.is_floating_point()
        )
        self.translators = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size,
                    hidden_size,
                    device=reference_parameter.device,
                    dtype=reference_parameter.dtype,
                )
                for _ in splitter.split_points
            ]
        )
        for translator in self.translators:
            nn.init.zeros_(translator.weight)
            nn.init.zeros_(translator.bias)

    def _transform(self, activations: torch.Tensor) -> torch.Tensor:
        translated = [
            activation + translator(activation)
            for translator, activation in zip(self.translators, activations[:-1], strict=True)
        ]
        return torch.stack([*translated, activations[-1]])

    @staticmethod
    def _loss(logits: torch.Tensor) -> torch.Tensor:
        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()
        target = logits[-1].softmax(dim=-1).detach().unsqueeze(0).expand_as(logits[:-1])
        return (
            F.kl_div(
                logits[:-1].log_softmax(dim=-1),
                target,
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )

    def fit(
        self,
        inputs: str | Iterable[str],
        epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """Fit every translator on a sequence of texts.

        Each text is traced independently, while all model depths are optimized
        together in one prediction-head call.

        Args:
            inputs (str | Iterable[str]): Texts used to train the translators.
            epochs (int): Number of passes over the texts.
            learning_rate (float): AdamW learning rate.
            weight_decay (float): AdamW weight decay.

        Returns:
            list[float]: Mean loss for each epoch.

        Raises:
            ValueError: If no training text is provided or `epochs` is not positive.
        """
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if not texts:
            raise ValueError("Tuned Lens fitting requires at least one text.")
        if epochs < 1:
            raise ValueError("`epochs` must be positive.")

        optimizer = torch.optim.AdamW(
            self.translators.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        model_parameters = list(self.splitter._model.parameters())
        requires_grad = [parameter.requires_grad for parameter in model_parameters]
        self.splitter._model.requires_grad_(False)
        losses = []

        try:
            self.train()
            for _ in range(epochs):
                epoch_loss = 0.0
                for text in texts:
                    activations = torch.cat(self.splitter.get_activations(text), dim=0)
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._loss(self.splitter.apply_head(self._transform(activations)))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(texts))
        finally:
            self.eval()
            for parameter, original_requires_grad in zip(model_parameters, requires_grad, strict=True):
                parameter.requires_grad_(original_requires_grad)

        return losses
