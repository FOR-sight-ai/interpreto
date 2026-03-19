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

from collections.abc import Iterable, MutableMapping
from functools import singledispatchmethod

import torch

from interpreto.model_wrapping.inference_wrapper import InferenceWrapper
from interpreto.typing import TensorMapping

# TODO: make jaxtyping in the whole file!!


class GenerationInferenceWrapper(InferenceWrapper):
    PAD_LEFT = True

    @singledispatchmethod
    def get_targeted_logits(self, model_inputs, targets, mode="logits"):
        """Return the logits associated with the target tokens.

        Args:
            model_inputs: Input mapping(s) used to compute the logits.
            targets: Token IDs of the generated part.
            mode (str): Post-processing mode applied on the logits.

        Returns:
            torch.Tensor | Iterable[torch.Tensor]: The logits selected for the target tokens.
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targeted_logits in class {self.__class__.__name__}"
        )

    @get_targeted_logits.register(MutableMapping)
    def _get_targeted_logits_from_mapping(
        self,
        model_inputs: TensorMapping,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve logits for a single batch of inputs.

        Args:
            model_inputs (TensorMapping): Full sequences including both the
                prompt and the generated continuation.
            targets (torch.Tensor): Token IDs of the continuation part with
                shape ``(batch_size, target_length)``.

        Returns:
            selected_logits (torch.Tensor): Predicted logits of shape ``(batch_size, target_length)``
            for the provided ``targets``.
        """
        # remove last target token from the model inputs
        # to avoid using the last token in the generation process
        model_inputs = {
            key: value[..., :-1, :] if key == "inputs_embeds" else value[..., :-1]
            for key, value in model_inputs.items()
        }

        # Get complete logits regardless of the input's shape.
        logits = self._get_logits_from_mapping(model_inputs)  # (l-1, v) | (n, l-1, v) | (n, p, l-1, v)

        target_length = targets.shape[-1]  # lt < l

        # assume the sequence dimension is the second-to-last.
        target_logits = logits[..., -target_length:, :]  # (n,lg,v)

        # Apply post-processing depending on selected mode
        target_logits = self.mode(target_logits)

        extended_targets = targets.expand(logits.shape[0], -1)

        if extended_targets.shape != target_logits.shape[:-1]:
            raise ValueError(
                "target logits shape without the vocabulary dimension must match the extended_targets inputs ids shape."
                f"Got {target_logits.shape[:-1]} and {extended_targets.shape}."
            )

        # For a batch case, unsqueeze the targets so that they match the logits shape.
        selected_logits = target_logits.gather(dim=-1, index=extended_targets.unsqueeze(-1)).squeeze(-1)

        return selected_logits

    @get_targeted_logits.register(Iterable)
    def _(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: Iterable[torch.Tensor],
    ):
        """Retrieve logits for each pair of inputs and targets in ``model_inputs``.

        Args:
            model_inputs (Iterable[TensorMapping]): Iterable of full input
                mappings.
            targets (Iterable[torch.Tensor]): Iterable of target token ID
                tensors.

        Returns:
            Iterable[torch.Tensor]: An iterator over the logits corresponding to
            each element of ``targets``.
        """
        # remove last target token from the model inputs
        # to avoid using the last token in the generation process
        model_inputs = [
            {key: value[..., :-1, :] if key == "inputs_embeds" else value[..., :-1] for key, value in elem.items()}
            for elem in model_inputs
        ]
        all_logits = self._get_logits_from_iterable(model_inputs)

        for logits, target in zip(all_logits, targets, strict=True):
            target_length = target.shape[-1]
            targeted_logits = logits[..., -target_length:, :]

            targeted_logits = self.mode(targeted_logits)

            extended_target = target.expand(logits.shape[0], -1).to(self.device)

            selected_logits = targeted_logits.gather(dim=-1, index=extended_target.unsqueeze(-1)).squeeze(-1)
            yield selected_logits
