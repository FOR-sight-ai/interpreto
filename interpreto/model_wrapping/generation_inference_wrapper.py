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
    def get_inputs_to_explain_and_targets(self, model_inputs, **generation_kwargs):
        """Prepare the inputs and targets for explanation in a generation setting.

        This method must be implemented for the different supported input types
        (``MutableMapping`` or ``Iterable``). It returns both the original prompt
        concatenated with the generated continuation and the token IDs of the
        generated part.

        Args:
            model_inputs (MutableMapping | Iterable[MutableMapping]): Input(s) to
                the model. Each mapping must contain ``input_ids`` or
                ``inputs_embeds`` and ``attention_mask`` as expected by Hugging
                Face models.
            **generation_kwargs: Additional arguments forwarded to
                ``model.generate()`` such as ``max_new_tokens`` or ``do_sample``.

        Returns:
            tuple[TensorMapping, torch.Tensor]: The full input mapping and the
            IDs of the generated continuation.
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_inputs_to_explain_and_targets in class {self.__class__.__name__}"
        )

    @get_inputs_to_explain_and_targets.register(MutableMapping)
    def _(self, model_inputs: TensorMapping, **generation_kwargs) -> tuple[TensorMapping, torch.Tensor]:
        """Generate continuations for a batch of sequences.

        Args:
            model_inputs (TensorMapping): Mapping with at least ``input_ids`` and
                ``attention_mask`` representing one or more sequences.
            **generation_kwargs: Keyword arguments forwarded to ``model.generate()``.

        Returns:
            full_mapping (TensorMapping): The full input mapping containing the original sequences and
                  the generated continuation.
            targets_ids (torch.Tensor): The token IDs of the generated part.
        """
        filtered_model_inputs = {key: model_inputs[key].to(self.device) for key in ("input_ids", "attention_mask")}

        full_ids = self.model.generate(**filtered_model_inputs, **generation_kwargs)
        original_length = model_inputs["attention_mask"].shape[-1]
        targets_ids = full_ids[..., original_length:]
        full_attention_mask = torch.cat(
            [model_inputs["attention_mask"].to(self.device), torch.ones_like(targets_ids)], dim=-1
        )
        full_mapping = {"input_ids": full_ids, "attention_mask": full_attention_mask}
        return full_mapping, targets_ids

    @get_inputs_to_explain_and_targets.register(Iterable)
    def _(
        self, model_inputs: Iterable[TensorMapping], **generation_kwargs
    ) -> tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]:
        """Apply :meth:`get_inputs_to_explain_and_targets` to each element.

        Args:
            model_inputs (Iterable[TensorMapping]): Iterable of input mappings.
            **generation_kwargs: Arguments forwarded to ``model.generate()``.

        Returns:
            l_full_mappings (Iterable[TensorMapping]): The full mappings for each element of ``model_inputs``.
            l_targets_ids (Iterable[torch.Tensor]): The token IDs of the generated part for each element of ``model_inputs``.
        """
        l_full_mappings, l_targets_ids = [], []
        for model_input in model_inputs:
            full_mappings, targets_ids = self.get_inputs_to_explain_and_targets(model_input, **generation_kwargs)
            l_full_mappings.append(full_mappings)
            l_targets_ids.append(targets_ids)
        return l_full_mappings, l_targets_ids

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
    def _(
        self,
        model_inputs: TensorMapping,
        targets: torch.Tensor,
    ):
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
        # TODO: remove debug lists
        logits_debug_list = list(all_logits)
        targets_debug_list = list(targets)
        for logits, target in zip(logits_debug_list, targets_debug_list, strict=True):
            # for logits, target in zip(all_logits, targets, strict=True):
            target_length = target.shape[-1]
            targeted_logits = logits[..., -target_length:, :]

            targeted_logits = self.mode(targeted_logits)

            extended_target = target.expand(logits.shape[0], -1)
            selected_logits = targeted_logits.gather(dim=-1, index=extended_target.unsqueeze(-1)).squeeze(-1)
            yield selected_logits
