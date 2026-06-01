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
Simplified model splitter for causal language models (generation).

``SplitterForGeneration`` wraps a HuggingFace generation model and splits
it at a specified layer. Activations are the per-token hidden states at the
split point, with special tokens optionally filtered out.

This class is designed for the concept pipeline on generative models.
It supports only two token-selection modes:

- **tokens** (default): returns only non-special tokens (padding, BOS, EOS, etc. removed).
- **all_tokens**: returns all token activations including special tokens but not padding.

No word/sentence aggregation is performed — that complexity lives in ``ModelWithSplitPoints``.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from math import ceil
from typing import Any

import nnsight
import torch
from jaxtyping import Bool, Float
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from interpreto.model_wrapping.base_splitter import BaseSplitter
from interpreto.typing import ConceptsActivations, LatentActivations, TensorMapping


class SplitterForGeneration(BaseSplitter):
    """A BaseSplitter specialization for causal language models (generation).

    Wraps a ``ForCausalLM`` model, splits it at a user-specified layer, and
    provides activation extraction with simple token-level granularity.

    Compared to ``ModelWithSplitPoints`` this class:
    - Only supports two activation modes: ``include_special_tokens=True/False``.
    - Does not depend on ``interpreto.commons.granularity.Granularity``.
    - Uses ``tokenizer.all_special_ids`` directly for special-token filtering.

    Arguments:
        model_or_repo_id (str | PreTrainedModel): A HuggingFace model ID or a
            pre-loaded CausalLM instance.
        split_point (str | int): The split location inside the model.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None): Tokenizer.
            Required when providing a model instance.
        config (PretrainedConfig | None): Model configuration.
        batch_size (int): Batch size for batched operations.
        device_map (torch.device | str | None): Device on which to load the model.
        output_tuple_index (int | None): Index of the hidden state in a tuple output.
        **kwargs: Additional keyword arguments forwarded to NNsight.

    Example:
        ```python
        from interpreto import SplitterForGeneration

        split_model = SplitterForGeneration(
            "gpt2",
            split_point=10,
            batch_size=8,
            device_map="auto",
        )
        activations, _ = split_model.get_activations(
            ["Hello world!", "Interpreto is magic"],
        )
        ```
    """

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_point: str | int,
        *,
        automodel: type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        output_tuple_index: int | None = None,
        **kwargs,
    ):
        """Initialize a SplitterForGeneration model wrapper.

        Raises:
            TypeError: If ``model_or_repo_id`` is a PreTrainedModel that is not a CausalLM.
        """
        if isinstance(model_or_repo_id, PreTrainedModel):
            class_name = model_or_repo_id.__class__.__name__
            if "ForCausalLM" not in class_name and "LMHeadModel" not in class_name:
                raise TypeError(
                    "The provided model is not a causal language model. "
                    "Please provide a model that inherits from `transformers.*ForCausalLM` "
                    "or `*LMHeadModel`."
                )

        super().__init__(
            model_or_repo_id,
            split_point,
            config=config,
            tokenizer=tokenizer,
            automodel=automodel if automodel is not None else AutoModelForCausalLM,  # type: ignore
            batch_size=batch_size,
            device_map=device_map,
            output_tuple_index=output_tuple_index,
            **kwargs,
        )

        # Ensure a pad token is available
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------

    def _tokenize_and_get_mask(
        self,
        inputs: list[str] | Float[torch.Tensor, "n l"],
        include_special_tokens: bool = False,
    ) -> tuple[TensorMapping, Bool[torch.Tensor, "n l"] | None]:
        """Tokenize and compute a mask of activations to keep.

        Args:
            inputs (list[str] | torch.Tensor): Inputs to the model.
                * If a list of strings, they are tokenized,
                    the mask is computed from the attention mask and optionally the special tokens mask.
                * If a tensor, it is assumed to be the input ids and the mask is a boolean ones tensor.
            include_special_tokens (bool):
                * False (default), returns a mask of non-special tokens.
                    The mask has the same shape has the token ids.
                * True, the mask only filters out padding.

        Returns:
            tokenized_inputs (dict[str, torch.Tensor]):
                The tokenized inputs.
            tokens_mask (torch.Tensor):
                The boolean mask of activations to keep.
        """
        # get input ids as a tensor
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim != 2:
                raise ValueError("Expected a 2D tensor for input_ids")
            return {"input_ids": inputs}, torch.ones_like(inputs, dtype=torch.bool)

        # embed textual inputs
        if isinstance(inputs, list):
            tokenized = self.tokenizer(
                inputs,
                return_special_tokens_mask=not include_special_tokens,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # just filters out padding
            if include_special_tokens:
                return tokenized, tokenized["attention_mask"]

            # filter out  padding and special tokens
            tokens_mask = tokenized["attention_mask"].bool() & ~tokenized.pop("special_tokens_mask").bool()
            return tokenized, tokens_mask

        raise ValueError(f"Unexpected input type: {type(inputs)}")

    def get_activations(
        self,
        inputs: list[str],
        include_special_tokens: bool = False,
        flatten_activations: bool = True,
        tqdm_bar: bool = False,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> tuple[list[LatentActivations] | LatentActivations, None]:
        """Extract per-token activations at the split point for a list of text inputs.

        Iterates over inputs in batches and delegates each batch to
        ``inputs_to_activations``.

        Args:
            inputs (list[str]): Raw text inputs.
            include_special_tokens (bool): If True, return all token activations
                (including special tokens but not padding).  If False (default),
                filter out special tokens using ``tokenizer.all_special_ids``.
            flatten_activations (bool): If True (default), flatten the activations.
                Into a single tensor (n*g, d). Where g varies if all tokens are included or not.
                If False, returns a list of sample-wise activations.
            tqdm_bar (bool): Whether to display a progress bar.
            forward_kwargs (dict[str, Any]): Additional kwargs for the model forward pass.
            **kwargs: Unused, kept for API compatibility.

        Returns:
            activations (list[LatentActivations] | LatentActivations):
                list[LatentActivations]: A list of tensors (one per sample, shape ``(l_i, d)``) and
                LatentActivations: A single tensor (n*g, d) if ``flatten_activations=True``.
            predictions (None): ``None`` (placeholder, no predicted classes for generation models).
        """
        n_batches = ceil(len(inputs) / self.batch_size)
        batch_iter = tqdm(
            range(0, len(inputs), self.batch_size),
            desc="Computing activations",
            unit="batch",
            total=n_batches,
            disable=not tqdm_bar,
        )

        sp_module = self.get(self._split_point)
        output_name = "nns_output" if hasattr(sp_module, "nns_output") else "output"

        all_activations: list[LatentActivations] = []

        with torch.no_grad():
            for start in batch_iter:
                batch_texts = inputs[start : min(start + self.batch_size, len(inputs))]

                # extract non-special tokens mask
                tokenized, tokens_mask = self._tokenize_and_get_mask(batch_texts, include_special_tokens)

                # forward till the split point
                with self.trace(tokenized, **forward_kwargs) as tracer:
                    batch_outputs = getattr(sp_module, output_name).save()
                    tracer.stop()

                batch_acts: Float[torch.Tensor, "n l d"] = self._manage_output_tuple(batch_outputs, self._split_point)

                # filter out special tokens
                for acts, mask in zip(batch_acts, tokens_mask, strict=True):
                    all_activations.append(acts.cpu()[mask])

        torch.cuda.empty_cache()
        gc.collect()

        if flatten_activations:
            return torch.cat(all_activations, dim=0), None

        return all_activations, None

    # ------------------------------------------------------------------
    # Concept-to-output gradients
    # ------------------------------------------------------------------

    def _reintegrate_activations(
        self,
        sp_module,
        module_out_name: str,
        layer_outputs: tuple[torch.Tensor] | torch.Tensor,
        raw_activations: Float[torch.Tensor, "ng d"],
        decoded_activations: Float[torch.Tensor, "ng d"],
        tokens_mask: torch.Tensor,
    ):
        """Reintegrate activations back into the full sequence.

        Args:
            sp_module: The module containing the activations.
            module_out_name (str): The name of the module output attribute to update.
            layer_outputs (tuple[torch.Tensor] | torch.Tensor): Original layer outputs, potentially tuple.
            raw_activations (Float[torch.Tensor, "ng d"]): Raw activations before decoding.
            decoded_activations (Float[torch.Tensor, "ng d"]): Decoded activations to reintegrate.
            tokens_mask (torch.Tensor | None): Mask indicating which positions to keep.
        """
        # Reintegrate decoded activations back into the full sequence and unflatten
        reconstructed = raw_activations.clone()
        index = 0
        for i, mask in enumerate(tokens_mask):
            reconstructed[i, mask] = decoded_activations[index : index + mask.sum()]
            index += mask.sum()

        # Put activations back in their tuple
        if isinstance(layer_outputs, tuple):
            layer_outputs = list(layer_outputs)  # type: ignore
            layer_outputs[self.output_tuple_index] = reconstructed  # type: ignore
        else:
            layer_outputs = reconstructed

        # Assign reconstructed activations back to the module output
        setattr(sp_module, module_out_name, layer_outputs)  # type: ignore

    def _get_concept_output_gradients(
        self,
        inputs: list[str],
        encode_activations: Callable[[LatentActivations], ConceptsActivations],
        decode_concepts: Callable[[ConceptsActivations], LatentActivations],
        targets: list[int] | None = None,
        include_special_tokens: bool = False,
        concepts_x_gradients: bool = False,
        tqdm_bar: bool = False,
        batch_size: int | None = None,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> list[Float[torch.Tensor, "t g c"]]:
        """Compute gradients of model outputs w.r.t. concept activations for generation.

        For each input, extracts full token-level activations,
        encodes them into concept space, decodes back, reintegrates, and computes the
        gradient of the logits with respect to the concept activations.

        For generation, logits have shape ``(n, l, vocab)``; we take the max over vocab
        to get ``(n, l)`` then sum over samples.

        Args:
            inputs (list[str]): Raw text inputs.
            encode_activations: Function mapping latent activations to concept space.
            decode_concepts: Function mapping concept activations back to latent space.
            targets (list[int] | None): Target token positions for which to compute gradients.
                If None, gradients are computed for all positions in the (summed) logits.
            include_special_tokens (bool): Whether to include special tokens in the activation selection.
            concepts_x_gradients (bool): If True, multiply gradients by concept activations.
            tqdm_bar (bool): Whether to display a progress bar.
            batch_size (int | None): Override the instance batch size.
            forward_kwargs (dict[str, Any]): Additional kwargs for the forward pass.
            **kwargs: Unused, kept for API compatibility.

        Returns:
            list[Float[torch.Tensor, "t g c"]]: A list of gradient tensors,
                one per sample, each of shape ``(n_targets, g_i, n_concepts)``.
        """
        grad_batch_size = batch_size or self.batch_size

        n_batches = ceil(len(inputs) / grad_batch_size)
        batch_iter = tqdm(
            range(0, len(inputs), grad_batch_size),
            desc="Computing gradients",
            unit="batch",
            total=n_batches,
            disable=not tqdm_bar,
        )
        sp_module = self.get(self._split_point)
        module_out_name = "nns_output" if hasattr(sp_module, "nns_output") else "output"

        gradients_list: list[Float[torch.Tensor, "t g c"]] = []

        for start in batch_iter:
            end = min(start + grad_batch_size, len(inputs))
            batch_texts = inputs[start:end]

            # extract non-special tokens mask
            tokens_mask: Bool[torch.Tensor, "n l"]
            tokenized, tokens_mask = self._tokenize_and_get_mask(batch_texts, include_special_tokens)

            # Forward with NNsight tracing + gradient computation
            with self.trace(tokenized, **forward_kwargs):
                # Get raw activations at split point
                layer_outputs = getattr(sp_module, module_out_name)
                raw_activations: Float[torch.Tensor, "b l d"] = self._manage_output_tuple(
                    layer_outputs, self._split_point
                )
                b, l, d = raw_activations.shape

                # Flatten and select activations of interest
                activations: Float[torch.Tensor, "bg d"] = raw_activations.flatten(0, 1)[tokens_mask.flatten()]

                # Encode activations into concepts
                concept_activations: Float[torch.Tensor, "bg c"] = encode_activations(activations)
                del activations

                # Decode concepts back into activations (n, l, d)
                decoded_activations: Float[torch.Tensor, "bg d"] = decode_concepts(concept_activations)

                # Reintegrate decoded activations back into the full sequence and into the model
                self._reintegrate_activations(
                    sp_module,
                    module_out_name,
                    layer_outputs,
                    raw_activations,
                    decoded_activations,
                    tokens_mask,
                )
                del decoded_activations, raw_activations

                # Get logits: (b, l, vocab) -> max over vocab -> (b, l) -> sum over samples -> (l,)
                logits = self.output.logits.max(dim=-1)[0].sum(dim=0)

                # Determine targets
                if targets is None:
                    current_targets = range(logits.shape[0])
                else:
                    current_targets = targets

                # Compute gradients for each target
                targets_gradients_list = []
                for t in current_targets:
                    with logits[t].backward(retain_graph=True):  # type: ignore
                        concept_grad = concept_activations.grad.clone()  # type: ignore
                        concept_activations.grad.zero_()  # type: ignore
                        if concepts_x_gradients:
                            concept_grad = concept_grad * concept_activations
                    targets_gradients_list.append(concept_grad)

                targets_gradients: Float[torch.Tensor, "bg t c"] = (
                    torch.stack(targets_gradients_list, dim=1).detach().cpu().save()  # type: ignore
                )
                del targets_gradients_list, concept_activations, logits

                # Split gradients per sample
                index = 0
                for mask in tokens_mask:
                    gradients_list.append(targets_gradients[index : index + mask.sum()].transpose(0, 1))
                    index += mask.sum()

                gc.collect()

        torch.cuda.empty_cache()  # TODO: see if it should be moved inside the loop

        return gradients_list

    # ------------------------------------------------------------------
    # Latent shape
    # ------------------------------------------------------------------

    def get_latent_shape(self) -> torch.Size:
        """Get the shape of the latent activations at the split point.

        Uses NNsight's scan to determine the shape without running a full forward pass.

        Returns:
            torch.Size: Shape of the activations at the split point (typically ``(1, l, d)``).
        """
        shape = None
        with self.scan("scan"):
            curr_module = self.get(self._split_point)
            module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"
            module = getattr(curr_module, module_out_name)
            if isinstance(module, tuple):
                for candidate in module:
                    if candidate.dim() == 3:
                        module = candidate
                        break
            shape = nnsight.save(module.shape)  # type: ignore
        return shape
