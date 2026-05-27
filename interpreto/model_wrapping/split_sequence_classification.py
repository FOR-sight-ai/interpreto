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

import gc
from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import LatentActivations


class SplitSequenceClassification(ModelWithSplitPoints):
    """A ModelWithSplitPoints specialization for sequence classification models.

    Provides optimized implementations of activation extraction and concept gradient
    computation by exploiting the known structure of classification models:
    a backbone followed by a single classification head.

    The split point is always the classification head, and activations are
    the CLS-token representations fed into that head.
    """

    @ModelWithSplitPoints.split_point.setter
    def split_point(self, split_point):
        """Override to store split point directly without walk_modules validation.

        The classification head is validated separately via the classification_head_name setter.
        """
        self._split_point = str(split_point)

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        classification_head_name: str | None = None,
        **kwargs,
    ):
        """Initialize a SplitSequenceClassification model wrapper.

        The wrapper loads a sequence classification model and automatically identifies
        its classification head as the split point. This simplifies the concept pipeline
        for classification models by removing the need to manually specify split points and
        forcing the granularity to be the [CLS] token.

        Args:
            model_or_repo_id (str | PreTrainedModel): A Hugging Face model ID or a pre-loaded
                ``PreTrainedModel`` instance. Must be a sequence classification model.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None): The tokenizer
                associated with the model. If None, it is loaded from the model repo.
            config (PretrainedConfig | None): Model configuration. If None, loaded automatically.
            batch_size (int): Batch size for activation extraction and gradient computation.
            device_map (torch.device | str | None): Device on which to load the model
                (e.g., ``"cuda"`` or ``"cpu"``).
            classification_head_name (str | None): Name of the classification head module.
                If None, auto-detected by searching for common names (``"classifier"``,
                ``"classification_head"``, ``"score"``).
            **kwargs: Additional keyword arguments forwarded to ``ModelWithSplitPoints``.

        Raises:
            ValueError: If ``model_or_repo_id`` is a PreTrainedModel that is not a
                sequence classification model.

        Example:
            ```python
            from interpreto import SplitSequenceClassification

            split_model = SplitSequenceClassification(
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

        # Pass a placeholder split_point; our overridden setter skips walk_modules validation.
        # The real split point is resolved after super().__init__() loads the model,
        # because the classification_head_name setter needs access to self._model.
        super().__init__(
            model_or_repo_id,
            split_point="placeholder",
            config=config,
            tokenizer=tokenizer,
            automodel=AutoModelForSequenceClassification,  # type: ignore
            batch_size=batch_size,
            device_map=device_map,
            **kwargs,
        )

        # Now self._model is available; resolve the classification head and update split_point.
        self.classification_head_name = classification_head_name  # setter auto-detects if None
        self.split_point = self.classification_head_name

    @property
    def classification_head_name(self) -> str:
        return self._classification_head_name

    @classification_head_name.setter
    def classification_head_name(self, classification_head_name: str | None) -> None:
        """Set the classification head name.

        Args:
            classification_head_name (str | None): Name of the classification head.
                If None, the first classification head is used.
        """
        sub_modules = list(self._model._modules.keys())
        if classification_head_name is None:
            resolved = None
            for candidate in ["classifier", "classification_head", "score"]:
                if candidate in sub_modules:
                    resolved = candidate
                    break
            if resolved is None:
                raise ValueError(
                    "No classification head found in the model. "
                    "Please specify the classification head name using the `classification_head_name` parameter."
                )
            self._classification_head_name = resolved
        else:
            if classification_head_name not in sub_modules:
                raise ValueError(
                    f"The provided classification head name '{classification_head_name}' is not valid. "
                    f"Existing model modules are: {', '.join(sub_modules)}."
                )
            self._classification_head_name = classification_head_name

    def __extract_cls_token(self, activations: Float[torch.Tensor, "n l d"]) -> Float[torch.Tensor, "n d"]:
        """
        Extract the CLS token from the activations.

        In some model such as Roberta the token CLS is done in the classification head,
        and is not part of the model's forward pass.
        In this case, we need to extract the CLS token from the activations.
        """
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        if padding_side == "right":
            return activations[:, 0, :]
        return activations[:, -1, :]

    def inputs_to_activations(
        self, inputs: list[str] | torch.Tensor | BatchEncoding | dict[str, torch.Tensor] | None = None, **kwargs
    ) -> Float[torch.Tensor, "n d"]:
        """Compute latent activations (CLS-token representations) from raw inputs.

        Runs the model backbone up to the classification head and extracts the
        input representation that would be fed to the classifier.

        This method does does not include batching, it is meant to be called by other methods/classes.
        In particular, it is used by the ``ModelForInputsToConcepts`` forward,
        which is batched in the ``InputsToConceptsInferenceWrapper``.

        Args:
            inputs (list[str] | torch.Tensor | BatchEncoding | dict[str, torch.Tensor] | None):
                Raw model inputs. Can be a list of strings, a tensor of input IDs,
                a BatchEncoding, or a dictionary of tensors.
            **kwargs: Additional keyword arguments forwarded to the trace context
                (e.g., ``truncation=True``).

        Returns:
            Float[torch.Tensor, "n d"]: The CLS-token activations of shape
                ``(n_samples, hidden_dim)``.

        Raises:
            ValueError: If both ``inputs`` and ``kwargs`` are empty.
        """
        if inputs is None and len(kwargs) == 0:
            raise ValueError("Either inputs or kwargs must be provided.")

        with self.trace(inputs, **kwargs) as tracer:
            activations = getattr(self, self.classification_head_name).input.save()
            tracer.stop()  # we only needed the CLS token, no need to complete the forward pass

        # force two dimensions
        if activations.ndim == 3:
            activations = self.__extract_cls_token(activations)
        return activations

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
        return getattr(self, self.classification_head_name)(activations).logits

    def get_activations(  # type: ignore
        self,
        inputs: list[str] | Int[torch.Tensor, "n l"],
        tqdm_bar: bool = False,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,  # not used, just to support the `model_with_split_points` interface
    ) -> tuple[LatentActivations, torch.Tensor]:
        """Extract CLS-token activations and predictions for a dataset of inputs.

        Iterates over the inputs in batches, extracting the activations at the
        classification head input and the model predictions.

        Args:
            inputs (list[str] | Int[torch.Tensor, "n l"]): Raw text inputs or
                tokenized input IDs.
            tqdm_bar (bool): Whether to display a progress bar.
            forward_kwargs (dict[str, Any]): Additional keyword arguments for
                the model forward pass (e.g., ``{"truncation": True}``).
            **kwargs: Unused, kept for API compatibility with ``ModelWithSplitPoints``.

        Returns:
            tuple[LatentActivations, torch.Tensor]: The activations tensor of shape
                ``(n_samples, hidden_dim)`` and predicted class indices of shape ``(n_samples,)``.
        """
        activations = []
        predictions = []
        classification_head = getattr(self, self.classification_head_name)

        self._model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), self.batch_size), disable=not tqdm_bar):
                # extract and prepare a batch of inputs
                end_idx = min(i + self.batch_size, len(inputs))
                batch = inputs[i:end_idx]
                if isinstance(batch, torch.Tensor):
                    batch = {"input_ids": batch}

                # get activations and predictions for the batch
                with self.trace(batch, **forward_kwargs):
                    batch_activations = classification_head.input.save()
                    batch_predictions = self.output.logits.argmax(dim=-1).save()  # type: ignore

                # force two dimensions
                if batch_activations.ndim == 3:
                    batch_activations = self.__extract_cls_token(batch_activations)

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

    def _get_concept_output_gradients(  # type: ignore
        self,
        inputs: list[str] | Float[torch.Tensor, "n d"],
        encode_activations: Callable[[Float[torch.Tensor, "n d"]], Float[torch.Tensor, "n c"]],
        decode_concepts: Callable[[Float[torch.Tensor, "n c"]], Float[torch.Tensor, "n d"]],
        targets: list[int] | None = None,
        concepts_x_gradients: bool = False,
        tqdm_bar: bool = False,
        batch_size: int | None = None,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,  # not used, just to support the `model_with_split_points` interface
    ) -> list[Float[torch.Tensor, "t 1 c"]]:
        """Compute gradients of model outputs w.r.t. concept activations.

        For each input, encodes it into the concept space and computes the gradient
        of the specified target logits with respect to the concept activations.
        Optionally multiplies gradients by the concept activations (concepts × gradients).

        Args:
            inputs (list[str] | Float[torch.Tensor, "n d"]): Raw text inputs or
                pre-computed latent activations.
            encode_activations: Function mapping latent activations to concept space.
            decode_concepts: Function mapping concept activations back to latent space.
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
        classification_head = getattr(self, self.classification_head_name)
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
            batch_concepts: Float[torch.Tensor, "b c"] = encode_activations(batch_activations.to(self.device))
            del batch_activations
            batch_concepts.requires_grad_(True)

            # decode concepts to logits
            try:
                logits: Float[torch.Tensor, "b t_all"] = classification_head(decode_concepts(batch_concepts))
            except IndexError:
                # we might forced two dimensions in `self.inputs_to_activations`
                logits: Float[torch.Tensor, "b t_all"] = classification_head(
                    decode_concepts(batch_concepts).unsqueeze(dim=1)
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
            gradients_list.extend(list(batch_gradients.unsqueeze(2)))  # (b, t, c) -> list of (b, 1, c)

        # free memory
        torch.cuda.empty_cache()
        gc.collect()
        return gradients_list

    def get_latent_shape(self) -> torch.Size:
        """Get the shape of the latent activations at the specified split point.

        Uses a quick trace with a dummy input to determine the classifier input shape.

        Returns:
            torch.Size: Shape of the activations at the classification head input.
        """
        with self.trace("scan") as tracer:
            shape = getattr(self, self.classification_head_name).input.shape.save()
            tracer.stop()
        return shape
