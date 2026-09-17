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
Token-level model splitter for causal and encoder text models.

``TextTokensSplitter`` wraps a Hugging Face text model and extracts hidden
states at a specified layer. Special tokens can optionally be retained, and
the resulting token representations can be pooled per input.

It supports two token-selection modes:

- **tokens** (default): returns only non-special tokens (padding, BOS, EOS, etc. removed).
- **all_tokens**: returns all token activations including special tokens but not padding.

No word/sentence grouping is performed. Retained tokens can optionally be pooled
into one representation per sample.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable
from math import ceil
from typing import Any, Literal

import torch
from jaxtyping import Bool, Float
from nnsight import save as nnsight_save
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from interpreto.concepts.splitters.base_splitter import BaseSplitter
from interpreto.typing import ConceptsActivations, LatentActivations, TensorMapping

TokenPooling = Literal[None, "mean", "max", "min", "signed_max", "first", "last"]
TextTokensTask = Literal["feature-extraction", "text-generation"]


class TextTokensSplitter(BaseSplitter):
    """A BaseSplitter specialization for token representations from text models.

    Wraps a causal or encoder text model, splits it at a user-specified layer,
    and provides token-level or pooled activation extraction.

    This class:
    - Supports two token-selection modes: ``include_special_tokens=True/False``.
    - Can pool retained token activations into one representation per sample.
    - Does not depend on ``interpreto.commons.granularity.Granularity``.

    Arguments:
        model_or_repo_id (str | PreTrainedModel): A Hugging Face model ID or a
            pre-loaded text model.
        split_point (str | int): The split location inside the model.
        task (TextTokensTask): NNsight loading task. Either ``"feature-extraction"`` or ``"text-generation"``.
            Use ``"text-generation"`` for causal language models and
            ``"feature-extraction"`` for encoder models.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None): Tokenizer.
            If None, NNsight resolves it automatically when possible.
        batch_size (int): Batch size for batched operations.
        device_map (torch.device | str | None): Device on which to load the model.
        **kwargs (dict[str, Any]): Additional keyword arguments forwarded to ``BaseSplitter.__init__`` and used for NNsight model loading.

    Example:
        ```python
        from interpreto import TextTokensSplitter

        splitter = TextTokensSplitter(
            "gpt2",
            split_point=10,          # layer index
            task="text-generation",  # task can be automatically inferred by nnsight
            batch_size=8,
            device_map="auto",       # let nnsight decide
        )
        activations, _ = splitter.get_activations(
            ["Hello world!", "Interpreto is magic"],
        )
        ```
    """

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_point: str | int,
        *,
        task: TextTokensTask | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_or_repo_id,
            split_point,
            task=task,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device_map=device_map,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _pool_activation(
        activation: LatentActivations,
        token_pooling: TokenPooling,
    ) -> LatentActivations:
        """Optionally pool one sample's activations over its sequence dimension."""
        match token_pooling:
            case None:
                pooled_activation = activation
            case "mean":
                pooled_activation = activation.mean(dim=0, keepdim=True)
            case "max":
                pooled_activation = activation.amax(dim=0, keepdim=True)
            case "min":
                pooled_activation = activation.amin(dim=0, keepdim=True)
            case "signed_max":
                indices = activation.abs().max(dim=0).indices.unsqueeze(0)
                pooled_activation = activation.gather(0, indices)
            case "first":
                pooled_activation = activation[:1]
            case "last":
                pooled_activation = activation[-1:]
            case _:
                raise ValueError(
                    f"Unknown token_pooling: {token_pooling!r}. Expected None, 'mean', 'max', 'min', "
                    "'signed_max', 'first' or 'last'."
                )
        return pooled_activation

    def _tokenize_and_get_mask(
        self,
        inputs: list[str] | Float[torch.Tensor, "n l"],
        include_special_tokens: bool = False,
    ) -> tuple[TensorMapping, Bool[torch.Tensor, "n l"]]:
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
            attention_mask: torch.Tensor = tokenized["attention_mask"]  # type: ignore

            # just filters out padding
            if include_special_tokens:
                return tokenized, attention_mask.bool()  # type: ignore

            # filter out  padding and special tokens
            tokens_mask = attention_mask.bool() & ~tokenized.pop("special_tokens_mask").bool()
            return tokenized, tokens_mask

        raise ValueError(f"Unexpected input type: {type(inputs)}")

    def inputs_to_activations(
        self,
        inputs: list[str],
        *,
        include_special_tokens: bool = False,
        flatten_activations: bool = True,
        token_pooling: TokenPooling = None,
        forward_kwargs: dict[str, Any] = {},
    ) -> list[LatentActivations] | LatentActivations:
        """Extract activations from raw inputs.

        Args:
            inputs (list[str]): Raw text inputs.
            include_special_tokens (bool): If True, return all token activations
                (including special tokens but not padding).  If False (default),
                filter out special tokens using ``tokenizer.all_special_ids``.
            flatten_activations (bool): Whether to flatten the activations into a single tensor of shape (n*g, d).
            token_pooling (TokenPooling): Optional pooling applied to the retained
                tokens of each sample.
            forward_kwargs (dict[str, Any]): Additional keyword arguments passed to the model forward pass.

        Returns:
            list[LatentActivations] | LatentActivations: Sample-wise activations
                or a single flattened tensor. Pooled samples contain one row each.
        """
        tokenized, tokens_mask = self._tokenize_and_get_mask(inputs, include_special_tokens)

        # forward till the split point
        with self.trace(tokenized, **forward_kwargs) as tracer:
            outputs = self.split_module.output.save()
            tracer.stop()

        # manage the output tuple and extract the (n, l, d) activations from it
        full_activations, _ = self._extract_hidden_state(outputs, self.split_point)

        # Filter out special tokens and move public activations to CPU.
        granular_activations = [
            self._pool_activation(acts[mask], token_pooling).detach().cpu().clone()
            for acts, mask in zip(full_activations, tokens_mask, strict=True)
        ]

        # flatten activations
        if flatten_activations:
            return torch.cat(granular_activations, dim=0)

        return granular_activations

    def get_activations(
        self,
        inputs: list[str],
        include_special_tokens: bool = False,
        flatten_activations: bool = True,
        token_pooling: TokenPooling = None,
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
            token_pooling (TokenPooling): Optional pooling applied to the retained
                tokens of each sample. Supported values are ``"mean"``, ``"max"``,
                ``"min"``, ``"signed_max"``, ``"first"``, and ``"last"``.
            tqdm_bar (bool): Whether to display a progress bar.
            forward_kwargs (dict[str, Any]): Additional kwargs for the model forward pass.
            **kwargs (dict[str, Any]): Unused, kept for API compatibility.

        Returns:
            activations (list[LatentActivations] | LatentActivations):
                list[LatentActivations]: A list of tensors (one per sample, shape ``(l_i, d)``) and
                LatentActivations: A single tensor (n*g, d) if ``flatten_activations=True``.
                With token pooling, each sample contributes one row.
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

        all_activations: list[LatentActivations] = []

        with torch.no_grad():
            for start in batch_iter:
                batch_texts = inputs[start : min(start + self.batch_size, len(inputs))]

                batch_activations = self.inputs_to_activations(
                    batch_texts,
                    include_special_tokens=include_special_tokens,
                    flatten_activations=False,
                    token_pooling=token_pooling,
                    forward_kwargs=forward_kwargs,
                )
                assert isinstance(batch_activations, list)
                all_activations.extend(batch_activations)

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
        layer_outputs: tuple[torch.Tensor] | torch.Tensor,
        raw_activations: Float[torch.Tensor, "b l d"],
        decoded_activations: Float[torch.Tensor, "ng d"],
        tokens_mask: torch.Tensor,
        tuple_index: int | None,
    ):
        """Reintegrate activations back into the full sequence.

        Args:
            sp_module: The module containing the activations.
            layer_outputs (tuple[torch.Tensor] | torch.Tensor): Original layer outputs, potentially tuple.
            raw_activations (Float[torch.Tensor, "b l d"]): Raw activations before decoding.
            decoded_activations (Float[torch.Tensor, "ng d"]): Decoded activations to reintegrate.
            tokens_mask (torch.Tensor | None): Mask indicating which positions to keep.
            tuple_index (int | None): Hidden-state index for tuple outputs.
        """
        # Reintegrate decoded activations back into the full sequence and unflatten
        reconstructed = raw_activations.clone()
        reconstructed[tokens_mask] = decoded_activations

        # Put activations back in their tuple
        if isinstance(layer_outputs, tuple):
            layer_outputs = list(layer_outputs)  # type: ignore
            layer_outputs[tuple_index] = reconstructed  # type: ignore[index]
            reconstructed = tuple(layer_outputs)  # type: ignore[assignment]

        # Assign reconstructed activations back to the module output
        sp_module.output = reconstructed  # type: ignore

    def _get_concept_output_gradients(
        self,
        inputs: list[str],
        activations_to_concepts: Callable[[LatentActivations], ConceptsActivations],
        concepts_to_activations: Callable[[ConceptsActivations], LatentActivations],
        targets: list[int] | None = None,
        include_special_tokens: bool = False,
        concepts_x_gradients: bool = True,
        tqdm_bar: bool = False,
        batch_size: int | None = None,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> list[Float[torch.Tensor, "t g c"]]:
        """Compute gradients of model outputs w.r.t. concept activations for generation.

        Only available for ``"text-generation"`` tasks.

        For each input, extracts full token-level activations,
        encodes them into concept space, decodes back, reintegrates, and computes the
        gradient of the logits with respect to the concept activations.

        For generation, logits have shape ``(1, l, vocab)``; we take the max over
        vocab to get one score per output position.

        Args:
            inputs (list[str]): Raw text inputs.
            activations_to_concepts: Function mapping latent activations to concept space.
            concepts_to_activations: Function mapping concept activations back to latent space.
            targets (list[int] | None): Target token positions for which to compute gradients.
                If None, gradients are computed for all positions in each input.
            include_special_tokens (bool): Whether to include special tokens in the activation selection.
            concepts_x_gradients (bool): If True, multiply gradients by concept activations.
            tqdm_bar (bool): Whether to display a progress bar.
            batch_size (int | None): Accepted for API compatibility; gradients
                are computed sample-wise.
            forward_kwargs (dict[str, Any]): Additional kwargs for the forward pass.
            **kwargs: Unused, kept for API compatibility.

        Returns:
            list[Float[torch.Tensor, "t g c"]]: A list of gradient tensors,
                one per sample, each of shape ``(n_targets, g_i, n_concepts)``.
        """
        if self.task != "text-generation":
            raise NotImplementedError(
                "Concept-to-output gradients are only supported for causal language models "
                f"(task='text-generation'), got task={self.task!r}."
            )

        sp_module = self.split_module
        gradients_list: list[Float[torch.Tensor, "t g c"]] = []

        for text in tqdm(inputs, desc="Computing gradients", unit="sample", disable=not tqdm_bar):
            tokenized, tokens_mask = self._tokenize_and_get_mask([text], include_special_tokens)
            current_targets = range(tokenized["input_ids"].shape[1]) if targets is None else targets

            # Forward with NNsight tracing + gradient computation
            with self.trace(tokenized, **forward_kwargs):
                # Get raw activations at split point
                layer_outputs = sp_module.output
                raw_activations, tuple_index = self._extract_hidden_state(layer_outputs, self._split_point)
                # Select activations of interest
                activations: Float[torch.Tensor, "g d"] = raw_activations[0, tokens_mask[0]]

                # Encode activations into concepts
                concept_activations: Float[torch.Tensor, "g c"] = activations_to_concepts(activations)
                concept_activations.requires_grad_(True)

                # Decode concepts back into activations
                decoded_activations: Float[torch.Tensor, "g d"] = concepts_to_activations(concept_activations).to(
                    device=raw_activations.device,
                    dtype=raw_activations.dtype,
                )

                # Reintegrate decoded activations back into the full sequence and into the model
                self._reintegrate_activations(
                    sp_module,
                    layer_outputs,
                    raw_activations,
                    decoded_activations,
                    tokens_mask,
                    tuple_index,
                )

                # Get one score per output position by taking the max over the vocabulary
                logits = self.output.logits[0].max(dim=-1)[0]

                targets_gradients_list = []
                for target_index, t in enumerate(current_targets):
                    concept_grad = torch.autograd.grad(
                        outputs=logits[t],
                        inputs=concept_activations,
                        retain_graph=target_index < len(current_targets) - 1,
                    )[0]
                    if concepts_x_gradients:
                        concept_grad = concept_grad * concept_activations
                    targets_gradients_list.append(concept_grad)

                targets_gradients: Float[torch.Tensor, "t g c"] = (
                    torch.stack(targets_gradients_list).detach().cpu().save()  # type: ignore
                )

            gradients_list.append(targets_gradients)
            gc.collect()

        torch.cuda.empty_cache()  # TODO: see if it should be moved inside the loop

        return gradients_list

    # ------------------------------------------------------------------
    # Latent shape
    # ------------------------------------------------------------------

    def get_latent_shape(self) -> torch.Size:
        """Get the shape of the latent activations at the split point.

        Uses a short real trace instead of NNsight's scan. Some causal LMs, such as
        Qwen3, run RoPE autocast checks that reject the fake/meta device used by
        scan.

        Returns:
            torch.Size: Shape of the activations at the split point (typically ``(1, l, d)``).
        """
        with self.trace("scan") as tracer:
            activations, _ = self._extract_hidden_state(self.split_module.output, self._split_point)
            shape = nnsight_save(activations.shape)  # type: ignore
            tracer.stop()
        return shape


class SplitterForGeneration(TextTokensSplitter):
    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_point: str | int,
        *,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        **kwargs,
    ):
        warnings.warn(
            "SplitterForGeneration is deprecated, use TextTokensSplitter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            task="text-generation",
            tokenizer=tokenizer,
            batch_size=batch_size,
            device_map=device_map,
            **kwargs,
        )
