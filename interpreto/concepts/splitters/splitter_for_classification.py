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
Simplified model splitter for sequence classification models.

``SplitterForClassification`` wraps a HuggingFace ``ForSequenceClassification``
model and always splits at the classification head. Activations are the
representations used by the classification head.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import cached_property
from typing import Any

import torch
from jaxtyping import Float, Int
from nnsight import save as nnsight_save
from tqdm import tqdm
from transformers import (
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from interpreto.concepts.splitters.base_splitter import BaseSplitter
from interpreto.typing import LatentActivations


class SplitterForClassification(BaseSplitter):
    """A BaseSplitter specialization for sequence classification models.

    Provides optimized implementations of activation extraction and concept gradient
    computation by exploiting the known structure of classification models:
    a backbone followed by a single classification head.

    The split point is always the classification head, and activations are
    the representations used by that head.
    """

    _HEAD_CANDIDATES = ("classifier", "classification_head", "score")

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_point: str | None = None,
        *,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        **kwargs,
    ):
        """Initialize a SplitterForClassification model wrapper.

        The wrapper loads a sequence classification model and automatically identifies
        its classification head as the split point. This simplifies the concept pipeline
        for classification models by removing the need to manually specify split points and
        exposing one representation per sample.

        Args:
            model_or_repo_id (str | PreTrainedModel): A Hugging Face model ID or a pre-loaded
                ``PreTrainedModel`` instance. Must be a sequence classification model.
            split_point (str | None): Name of the classification head module.
                If None, auto-detected by searching for common names (``"classifier"``,
                ``"classification_head"``, ``"score"``).
                For most models, one can trust the auto-detection.
                Nonetheless, there is a difference with other splitter,
                Here we use the input of the split_point not its output.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None): The tokenizer
                associated with the model. If None, it is loaded from the model repo.
            batch_size (int): Batch size for activation extraction and gradient computation.
            device_map (torch.device | str | None): Device on which to load the model
                (e.g., ``"cuda"`` or ``"cpu"``).
            **kwargs (dict[str, Any]): Additional keyword arguments forwarded to ``BaseSplitter``.

        Raises:
            ValueError: If ``model_or_repo_id`` is a PreTrainedModel that is not a
                sequence classification model.

        Example:
            ```python
            from interpreto import SplitterForClassification

            splitter = SplitterForClassification(
                "nateraw/bert-base-uncased-emotion",
                batch_size=32,
                device_map="cuda",
            )
            ```
        """
        if isinstance(model_or_repo_id, PreTrainedModel):
            if "ForSequenceClassification" not in model_or_repo_id.__class__.__name__:
                raise ValueError(
                    "The provided model is not a sequence classification model. "
                    "Please provide a model that inherits from `transformers.ForSequenceClassification`."
                )

        super().__init__(
            model_or_repo_id,
            split_point=split_point,
            task="text-classification",
            tokenizer=tokenizer,
            batch_size=batch_size,
            device_map=device_map,
            **kwargs,
        )

        # Some heads select one sequence position themselves (e.g. RoBERTa),
        # while token-wise heads leave that selection to the parent model.
        head_parameter = next(self._split_module.parameters())
        sample_input = head_parameter.new_empty(1, 2, self.config.hidden_size)
        with torch.no_grad():
            self._head_is_token_wise = self._split_module(sample_input).ndim == 3

    @BaseSplitter.split_point.setter  # type: ignore[attr-defined]
    def split_point(self, split_point: str | int | None) -> None:
        """Set the split_point corresponding to the classification head name.

        Args:
            split_point (str | None): Name of the classification head.
                If None, the first classification head is used.
        """
        if split_point is None:
            prefix = f"{self.path}."
            modules = {path.removeprefix(prefix) for path, _ in self.named_modules() if path.startswith(prefix)}
            split_point = next((name for name in self._HEAD_CANDIDATES if name in modules), None)
            if split_point is None:
                raise ValueError(
                    "No classification head found in the model. "
                    "Please specify the classification head name using the `split_point` parameter."
                )

        if not isinstance(split_point, str):
            raise ValueError(f"The provided classification head '{split_point}' is not valid.")

        BaseSplitter.split_point.fset(self, split_point)  # type: ignore[attr-defined]

    def _select_classification_representation(
        self,
        activations: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> Float[torch.Tensor, "n d"]:
        """Select the first token, or the last non-padding token for token-wise heads."""
        # The head already receives one representation per sample.
        if activations.ndim == 2:
            return activations

        # Sequence-reducing heads such as RoBERTa classify the first token.
        if not self._head_is_token_wise:
            return activations[:, 0, :]

        # Without padding information, token-wise heads classify the last token.
        pad_token_id = self.config.pad_token_id
        if input_ids is None or pad_token_id is None:
            return activations[:, -1, :]

        # With padding, select each sample's last non-padding token.
        non_padding = input_ids != pad_token_id
        token_indices = torch.arange(input_ids.shape[-1], device=input_ids.device)
        positions = (token_indices * non_padding).argmax(-1).to(activations.device)
        return activations[
            torch.arange(activations.shape[0], device=activations.device),
            positions,
        ]

    @cached_property
    def _standalone_token_template(self) -> tuple[torch.Tensor, int]:
        """Get the template of special tokens to integrate a standalone token ID."""
        input_ids = self.tokenizer("a", return_tensors="pt")["input_ids"]
        token_id = self.tokenizer("a", add_special_tokens=False)["input_ids"][0]
        position = (input_ids[0] == token_id).nonzero()[0].item()
        return input_ids, position

    def _prepare_batch(
        self,
        inputs: list[str] | Iterable[int] | torch.Tensor | BatchEncoding | dict[str, torch.Tensor] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Prepare model inputs while retaining the input IDs used for token selection."""
        if inputs is None:
            if not kwargs:
                raise ValueError("Either inputs or kwargs must be provided.")
            return dict(kwargs)

        if isinstance(inputs, torch.Tensor):
            return {"input_ids": inputs}

        if not isinstance(inputs, list):
            return dict(inputs)

        if not inputs:
            raise ValueError("List inputs cannot be empty.")

        if all(isinstance(x, str) for x in inputs):
            return dict(
                self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            )

        if all(isinstance(x, int) for x in inputs):
            # Integrate the provided token IDs into the special tokens template.
            template, position = self._standalone_token_template
            input_ids = template.expand(len(inputs), -1).clone()
            input_ids[:, position] = torch.as_tensor(inputs, dtype=input_ids.dtype)

            return {"input_ids": input_ids}

        raise TypeError("List inputs must contain only strings or only integer token IDs.")

    def inputs_to_activations(
        self,
        inputs: list[str] | Iterable[int] | torch.Tensor | BatchEncoding | dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> Float[torch.Tensor, "n d"]:
        """Compute latent classification representations from raw inputs.

        Runs the model backbone up to the classification head and extracts the
        input representation that would be fed to the classifier.

        This method does does not include batching, it is meant to be called by other methods/classes.
        In particular, it is used by the ``ModelForInputsToConcepts`` forward,
        which is batched in the ``InputsToConceptsInferenceWrapper``.

        Args:
            inputs (list[str] | Iterable[int] | torch.Tensor | BatchEncoding | dict[str, torch.Tensor] | None):
                Raw model inputs. A list of strings is tokenized normally. A list
                of integers represents standalone token IDs and receives the
                tokenizer's model-specific special tokens. A tensor is treated
                as an already-prepared batch of input IDs. BatchEncoding and
                tensor dictionaries are forwarded unchanged.
            **kwargs (dict[str, Any]): Additional keyword arguments forwarded to the trace context
                (e.g., ``truncation=True``).

        Returns:
            Float[torch.Tensor, "n d"]: The classification activations of shape
                ``(n_samples, hidden_dim)``.

        Raises:
            ValueError: If both ``inputs`` and ``kwargs`` are empty.
        """
        prepared = self._prepare_batch(inputs, kwargs)

        with self.trace(prepared, **(kwargs if inputs is not None else {})) as tracer:
            activations = self._split_module.input.save()
            tracer.stop()  # we only needed the CLS token, no need to complete the forward pass

        return self._select_classification_representation(
            activations,
            prepared.get("input_ids"),
        )

    def activations_to_outputs(
        self,
        activations: Float[torch.Tensor, "n d"],
    ) -> Float[torch.Tensor, "n cls"]:
        """Compute classification logits from latent activations.

        As activations correspond to the inputs of the classification head.
        This method just passes the activations through the classification head to obtain
        output logits.

        Args:
            activations (Float[torch.Tensor, "n d"]): Latent activations of shape
                ``(n_samples, hidden_dim)``.

        Returns:
            Float[torch.Tensor, "n cls"]: Classification logits of shape
                ``(n_samples, n_classes)``.
        """
        if not self.dispatched:
            self.dispatch()

        classification_head = self._split_module
        activations = activations.to(next(classification_head.parameters()))

        # A singleton sequence works for both composite heads, such as
        # RoBERTa's, and token-wise linear heads.
        logits = classification_head(activations.unsqueeze(1))
        return logits[:, 0] if logits.ndim == 3 else logits

    def get_activations(
        self,
        inputs: list[str] | Iterable[int] | Int[torch.Tensor, "n l"],
        tqdm_bar: bool = False,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> tuple[LatentActivations, torch.Tensor]:
        """Extract classification activations and predictions for a dataset of inputs.

        Iterates over the inputs in batches, extracting the activations at the
        classification head input and the model predictions.

        Args:
            inputs (list[str] | Iterable[int] | Int[torch.Tensor, "n l"]): Raw text
                inputs, standalone token IDs, or an already-prepared 2D tensor
                of token IDs.
            tqdm_bar (bool): Whether to display a progress bar.
            forward_kwargs (dict[str, Any]): Additional keyword arguments for
                the model forward pass (e.g., ``{"truncation": True}``).
            **kwargs (dict[str, Any]): Unused, kept for API compatibility with ``ModelWithSplitPoints``.

        Returns:
            tuple[LatentActivations, torch.Tensor]: The activations tensor of shape
                ``(n_samples, hidden_dim)`` and predicted class indices of shape ``(n_samples,)``.
        """
        activations = []
        predictions = []
        classification_head = self._split_module

        self.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), self.batch_size), disable=not tqdm_bar):
                # extract and prepare a batch of inputs
                batch = self._prepare_batch(inputs[i : i + self.batch_size], {})

                # get activations and predictions for the batch
                with self.trace(batch, **forward_kwargs):
                    batch_activations = classification_head.input.save()
                    batch_predictions = self.output.logits.argmax(dim=-1).save()  # type: ignore

                batch_activations = self._select_classification_representation(
                    batch_activations,
                    batch.get("input_ids"),
                )

                # Materialize outside the trace. This is necessary to avoid memory leaks.
                activations.append(batch_activations.detach().cpu().clone())
                predictions.append(batch_predictions.detach().cpu().clone())

                del batch, batch_activations, batch_predictions

        activations = torch.cat(activations, dim=0)
        predictions = torch.cat(predictions, dim=0)

        # free memory
        torch.cuda.empty_cache()
        gc.collect()
        return activations, predictions

    def _get_concept_output_gradients(
        self,
        inputs: list[str] | Float[torch.Tensor, "n d"],
        activations_to_concepts: Callable[[Float[torch.Tensor, "n d"]], Float[torch.Tensor, "n c"]],
        concepts_to_activations: Callable[[Float[torch.Tensor, "n c"]], Float[torch.Tensor, "n d"]],
        targets: list[int] | None = None,
        concepts_x_gradients: bool = True,
        tqdm_bar: bool = False,
        batch_size: int | None = None,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> list[Float[torch.Tensor, "t 1 c"]]:
        """Compute gradients of model outputs w.r.t. concept activations.

        For each input, encodes it into the concept space and computes the gradient
        of the specified target logits with respect to the concept activations.
        Optionally multiplies gradients by the concept activations (concepts x gradients).

        Args:
            inputs (list[str] | Float[torch.Tensor, "n d"]): Raw text inputs or
                pre-computed latent activations.
            activations_to_concepts: Function mapping latent activations to concept space.
            concepts_to_activations: Function mapping concept activations back to latent space.
            targets (list[int] | None): Target class indices for which to compute
                gradients. If None, gradients are computed for all classes.
            concepts_x_gradients (bool): If True, multiply the gradients by the concept
                activations before returning.
            tqdm_bar (bool): Whether to display a progress bar.
            batch_size (int | None): Override the instance batch size for this call.
            forward_kwargs (dict[str, Any]): Additional keyword arguments for the forward pass.
            **kwargs: Unused, kept for API compatibility.

        Returns:
            list[Float[torch.Tensor, "t 1 c"]]: A list of gradient tensors,
                one per sample, each of shape ``(n_targets, 1, n_concepts)``.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # use session to setup trace once for all batches
        gradients_list: list[Float[torch.Tensor, "t 1 c"]] = []
        for i in tqdm(range(0, len(inputs), batch_size), disable=not tqdm_bar):
            # extract and prepare a batch of inputs
            end_idx = min(i + batch_size, len(inputs))

            # get activations for the batch
            if isinstance(inputs, torch.Tensor):
                batch_activations: Float[torch.Tensor, "b d"] = inputs[i:end_idx]  # type: ignore
            else:
                with torch.no_grad():
                    batch_activations: Float[torch.Tensor, "b d"] = self.inputs_to_activations(
                        inputs[i:end_idx], **forward_kwargs
                    )

            # encode activations to concepts
            batch_concepts: Float[torch.Tensor, "b c"] = activations_to_concepts(batch_activations)
            del batch_activations
            batch_concepts.requires_grad_(True)

            # decode concepts to logits
            logits: Float[torch.Tensor, "b t_all"] = self.activations_to_outputs(
                concepts_to_activations(batch_concepts)
            )

            # specify which classes to compute gradients for
            if targets is None:
                # compute gradients for all classes
                batch_targets = range(logits.shape[1])
            else:
                batch_targets = targets

            batch_gradients_list = []
            for t, target in enumerate(batch_targets):
                # we compute gradients one target at a time to save memory and avoid jacobian computations
                target_wise_grads: Float[torch.Tensor, "b c"] = torch.autograd.grad(
                    outputs=logits[:, target].sum(),
                    inputs=batch_concepts,
                    retain_graph=t < (len(batch_targets) - 1),
                )[0].detach()

                if concepts_x_gradients:
                    # we multiply the input embeddings with their gradients before reducing them
                    target_wise_grads = target_wise_grads * batch_concepts

                batch_gradients_list.append(target_wise_grads.cpu())
            batch_gradients: Float[torch.Tensor, "b t c"] = torch.stack(batch_gradients_list, dim=1)
            del batch_gradients_list
            gradients_list.extend(list(batch_gradients.unsqueeze(2)))  # (b, t, c) -> list of (t, 1, c)

        # free memory
        torch.cuda.empty_cache()
        gc.collect()
        return gradients_list

    def get_latent_shape(self) -> torch.Size:
        """Get the shape of the exposed classification representation.

        Uses a quick trace with a dummy input to determine the classifier input
        hidden size. The splitter exposes one representation per sample even
        when the classification head receives a sequence.

        Returns:
            torch.Size: Shape ``(1, hidden_dim)``.
        """
        with self.trace("Hello world") as tracer:
            shape = nnsight_save(self._split_module.input.shape)
            tracer.stop()
        return torch.Size([1, shape[-1]])
