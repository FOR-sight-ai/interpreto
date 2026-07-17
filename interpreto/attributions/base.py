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
Basic standard classes for attribution methods
"""

from __future__ import annotations

import itertools
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from PIL.Image import Image as PILImage
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import (
    ImageMaskPerturbator,
    ImageTensorPerturbator,
    MaskPerturbator,
    Perturbator,
    TensorPerturbator,
    TextMaskPerturbator,
    TextTensorPerturbator,
)
from interpreto.commons import TextGranularity
from interpreto.commons.generator_tools import split_iterator
from interpreto.commons.granularity import GranularityAggregationStrategy, GranularityResizeStrategy, ImageGranularity
from interpreto.model_wrapping.classification_inference_wrapper import TextClassificationInferenceWrapper
from interpreto.model_wrapping.generation_inference_wrapper import TextGenerationInferenceWrapper
from interpreto.model_wrapping.image_classification_inference_wrapper import ImageClassificationInferenceWrapper
from interpreto.model_wrapping.inference_wrapper import InferenceModes, InferenceWrapper
from interpreto.typing import ClassificationTarget, GeneratedTarget, ModelInputs, SingleAttribution, TensorMapping


def setup_token_ids(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer | BaseImageProcessor, require_mask_token: bool = True
) -> int | None:
    """
    Setup the tokenizer and the model with the appropriate token IDs, for padding and masking.

    Returns the mask token ID.
    """

    if isinstance(tokenizer, BaseImageProcessor):
        return None

    resize_token_embeddings = False

    # ------------
    # pad_token_id
    generation_config = getattr(model, "generation_config", None)
    vocab_size = getattr(model.config, "vocab_size", None)

    resolved_pad_token_id = None
    # list possible pad_token_id candidates
    pad_token_id_candidates = [
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(generation_config, "pad_token_id", None),
        getattr(model.config, "pad_token_id", None),
        getattr(generation_config, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
    ]

    # check candidates in order
    for candidate in pad_token_id_candidates:
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None  # noqa: PLW2901 - normalized in place for brevity
        if not isinstance(candidate, int):
            continue
        if candidate < 0:
            continue
        if vocab_size is not None and candidate >= vocab_size:
            continue

        resolved_pad_token_id = candidate
        break

    if resolved_pad_token_id is None:
        # no valid pad_token_id found, create a new one
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        resize_token_embeddings = True
        resolved_pad_token_id = tokenizer.pad_token_id
    elif getattr(tokenizer, "eos_token", None) is not None and tokenizer.eos_token_id == resolved_pad_token_id:
        # it is better to set the token than the
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # set the pad_token_id
        tokenizer.pad_token_id = resolved_pad_token_id

    # propagate the pad_token_id to the model
    model.config.pad_token_id = resolved_pad_token_id
    if generation_config is not None:
        generation_config.pad_token_id = resolved_pad_token_id

    if not require_mask_token:
        return 0
    # -------------
    # mask_token_id
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None:
        # use existing mask_token_id for replacement
        replace_token_id = mask_token_id
    else:
        # create a new mask_token_id
        replace_token = "[REPLACE]"
        if replace_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([replace_token])
            resize_token_embeddings = True

        replace_token_id = tokenizer.convert_tokens_to_ids(replace_token)
        if isinstance(replace_token_id, list):
            replace_token_id = replace_token_id[0]

    if resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    return int(replace_token_id)


class ModelTask(Enum):
    """
    Enum to represent the model task type.
    """

    CLASSIFICATION = "classification"
    GENERATION = "generation"


def clone_tensor_mapping(tm: TensorMapping, detach: bool = False) -> TensorMapping:
    """
    Clone a TensorMapping, optionally detaching the tensors.

    Args:
        tm (TensorMapping): tensor mapping to clone
        detach (bool, optional): specify if new tensors must be detached. Defaults to False.

    Returns:
        TensorMapping: cloned tensor mapping
    """
    return {k: v.detach().clone() if detach else v.clone() for k, v in tm.items()}


@dataclass(slots=True)
class AttributionOutput:
    """
    Class to store the output of an attribution method.

    It contains every element needed to visualize the explanations and compute the metrics.

    Attributes:
        attributions (SingleAttribution):
            A list (n elements, with n the number of samples) of attribution score tensors:
                - `l` represents the number of elements for which attribution is computed (for NLP tasks: can be the total sequence length).
                - Shapes depend on the task:
                    - Classification (single class): `(l)`
                    - Classification (multi classes): `(c, l)`, where `c` is the number of classes.
                    - Generative models: `(l_g, l)`, where `l_g` is the length of the generated part.
                        - For non-generated elements, there are `l_g` attribution scores.
                        - For generated elements, scores are zero for previously generated tokens.
                    - Token classification: `(l_t, l)`, where `l_t` is the number of token classes. When the tokens are disturbed, l = l_t.

        elements (list[str] | torch.Tensor):
            A list or tensor representing the elements for which attributions are computed.
                - These elements can be tokens, words, sentences, or tensors of size `l`.

        model_inputs_to_explain (TensorMapping):
            The encoding post tokenization of the input text with `return_tensors="pt"`.
            In the case of generation, the target is included in it.

        targets (torch.Tensor):
            The target classes or tokens.

        model_task (ModelTask):
            An enum representing the task of the model explained: CLASSIFICATION or GENERATION.

        classes (torch.Tensor | None):
            Optional tensor of class labels.
                - For single-class classification: tensor of shape `(1)`
                - For multi-class classification: tensor of shape `(c)` where `c` is the number of classes

        granularity (TextGranularity):
            The granularity level of the explanation.

        granularity_aggregation_strategy (GranularityAggregationStrategy):
            The aggregation method used for aggregating the scores at the specified granularity.

        inference_mode (Callable[[torch.Tensor], torch.Tensor]):
            The mode used for inference.
            It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
    """

    attributions: SingleAttribution
    elements: list[str] | torch.Tensor
    model_inputs_to_explain: TensorMapping
    targets: torch.Tensor
    model_task: ModelTask
    classes: torch.Tensor | None = None
    granularity: TextGranularity = TextGranularity.DEFAULT
    granularity_aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN
    inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS

    # TODO: Harmonize even more, all attributions could be of the shape (t, l),
    # with t being either a number of class or of generated tokens.
    # It should not be a problem if some values are None or zero for generation.
    # This should be thoroughly tested.


class AttributionExplainer(ABC):
    """
    Abstract base class for attribution explainers.

    This class defines a common interface and helper methods used by various attribution explainers.
    Subclasses must implement the abstract method 'explain'.
    """

    _associated_inference_wrapper: InferenceWrapper
    base_tensor_perturbator_class: type[TensorPerturbator]
    base_mask_perturbator_class: type[MaskPerturbator]
    _model_task: ModelTask

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
        granularity: TextGranularity = TextGranularity.DEFAULT,
        granularity_aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,  # TODO: add to all classes
        use_gradient: bool = False,
        input_x_gradient: bool = True,
    ) -> None:
        """
        Initializes the AttributionExplainer.

        Args:
            model (PreTrainedModel): The model to be explained.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
            batch_size (int): The batch size used for model inference.
            perturbator (Perturbator, optional): Instance used to generate input perturbations.
                If None, the perturbator returns only the original input.
            aggregator (Aggregator, optional): Instance used to aggregate computed attribution scores.
                If None, the aggregator returns the original scores.
            device (torch.device, optional): The device on which computations are performed.
                If None, defaults to the device of the model.
            granularity (TextGranularity, optional): The level of granularity for the explanation.
                Options are: `ALL_TOKENS`, `TOKEN`, `WORD`, or `SENTENCE`.
                Defaults to TextGranularity.DEFAULT (ALL_TOKENS)
                To obtain it, `from interpreto import TextGranularity` then `TextGranularity.WORD`.
            granularity_aggregation_strategy (GranularityAggregationStrategy, optional): The method used to aggregate scores at the specified granularity,
                for gradient-based methods. Thus, it is ignored for perturbation based methods.
                Defaults to GranularityAggregationStrategy.MEAN.
                Ignored for `granularity` set to `ALL_TOKENS` or `TOKEN`.
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
            use_gradient (bool, optional): If True, computes gradients instead of inference for targeted explanations.
            input_x_gradient (bool, optional): If True and ``use_gradient`` is set, multiplies the input embeddings
                with their gradients before reducing them. Defaults to ``True``.
        """
        # set pad and mask tokens
        setup_token_ids(model, tokenizer, require_mask_token=False)
        self.tokenizer = tokenizer

        self.inference_wrapper = self._associated_inference_wrapper(
            model,
            gradients=use_gradient,
            input_x_gradient=input_x_gradient,
            batch_size=batch_size,
            device=device,
            mode=inference_mode,
        )  # type: ignore
        self.perturbator = perturbator
        self.aggregator = aggregator or Aggregator()
        self.granularity = granularity
        self.granularity_aggregation_strategy = granularity_aggregation_strategy

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model is located.
        """
        return self.inference_wrapper.device

    @device.setter
    def device(self, device: torch.device) -> None:
        """
        Sets the device on which the model is located.
        """
        self.inference_wrapper.device = device

    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): The device to which the model should be moved.
        """
        self.inference_wrapper.to(device)

    def cpu(self):
        """
        Move the model to the CPU.
        """
        self.device = torch.device("cpu")

    def cuda(self):
        """
        Move the model to the GPU.
        """
        self.device = torch.device("cuda")

    def process_model_inputs(self, model_inputs: ModelInputs) -> list[TensorMapping]:
        """
        Processes and standardizes model inputs into a list of dictionaries compatible with the model.

        This method handles various input types:
            - If a string is provided, it tokenizes the string and returns a list containing one mapping.
            - If a mapping is provided with a batch (multiple samples), it splits the batch into individual mappings.
            - If an iterable is provided, it processes each item recursively.

        Args:
            model_inputs (str, TensorMapping, or Iterable): The raw model inputs.

        Returns:
            List[TensorMapping]: A list of processed model input mappings.

        Raises:
            ValueError: If the type of model_inputs is not supported.
        """
        if isinstance(model_inputs, str):
            return [
                self.tokenizer(
                    model_inputs.rstrip(),
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True,
                )
            ]
        if isinstance(model_inputs, BatchEncoding):
            if "input_ids" not in model_inputs.keys():
                raise ValueError(
                    "The tokenized model_inputs must contain the key 'input_ids' to be processed by the attribution explainer. "
                    f"Got {model_inputs.keys()}."
                )
            if not isinstance(model_inputs["input_ids"], torch.Tensor):
                raise ValueError(
                    "The tokenized model_inputs must contain a torch.Tensor for the key 'input_ids' to be processed by the attribution explainer. "
                    "Use `model_inputs = tokenizer(..., return_tensors='pt')` to convert the model_inputs to a torch.Tensor. "
                    f"Got {type(model_inputs['input_ids'])}."
                )
            if model_inputs["input_ids"].shape[0] != 1:  # type: ignore
                raise ValueError(
                    "The tokenized model_inputs must contain a single sample to be processed by the attribution explainer. "
                    "Samples should be tokenized one by one or passed as list of texts directly. "
                    f"Got {model_inputs['input_ids'].shape[0]} samples."  # type: ignore
                )
            return [model_inputs]
        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))
        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    @abstractmethod
    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: list[TensorMapping],
        targets: torch.Tensor,
    ) -> tuple[list[TensorMapping], list[Int[torch.Tensor, "t"]]]:
        """
        Processes the inputs and targets for explanation.

        This method must be implemented by subclasses.

        Args:
            model_inputs (list[TensorMapping]): The inputs to the model.
            targets (Any): The targets to be explained.

        Returns:
            processed_inputs (list[TensorMapping]):
                The processed inputs.
            processed_targets (list[torch.Tensor]):
                The processed targets.

        Raises:
            NotImplementedError: Always raised. Subclasses must implement this method.
        """
        raise NotImplementedError(
            "Specific task subclasses must implement the 'process_inputs_to_explain_and_targets' method "
            "to correctly process inputs and targets for explanations."
        )

    def post_processing(self, contribution: Float[torch.Tensor, "t l"]):
        """
        No-op post-processing of the attribution scores, suitable for Classification.


        Args:
            contribution (Float[torch.Tensor, "t l"]): The contribution values.

        Returns:
            model_task (ModelTask): The model task.
            contribution (Float[torch.Tensor, "t l"]): The post-processed contribution values.

        """
        return contribution

    def explain(
        self,
        model_inputs: ModelInputs,
        targets: (
            torch.Tensor | Iterable[torch.Tensor] | None
        ) = None,  # TODO: create specific target type for classification and generation
    ) -> list[AttributionOutput]:
        """
        Computes attributions for NLP models.

        Process:
            1. Process and standardize the model inputs and targets.
            2. Generate perturbations for the constructed inputs, perturbations are based on the granularity.
            3. Compute scores using either gradients or targeted logits depending on the method.
            4. Aggregate the scores to obtain contribution values.

        Args:
            model_inputs (ModelInputs): Raw inputs for the model.
            targets (torch.Tensor | Iterable[torch.Tensor] | None):
                Targets for which explanations are desired.
                Further types might be supported by sub-classes.
                It depends on the task:
                    - For classification tasks, encodes the target class or classes to explain.
                    - For generation tasks, encodes the target text or tokens to explain.

        Returns:
            List[AttributionOutput]: A list of attribution outputs, one per input sample.
        """
        # Ensure the model inputs are in the correct format
        sanitized_model_inputs: list[TensorMapping] = self.process_model_inputs(model_inputs)

        # Process the inputs and targets for explanation
        # If targets are not provided, create them from model_inputs_to_explain.
        model_inputs_to_explain: list[TensorMapping]
        sanitized_targets: list[Int[torch.Tensor, "t"]]
        model_inputs_to_explain, sanitized_targets = self.process_inputs_to_explain_and_targets(
            sanitized_model_inputs,
            targets,  # type: ignore
        )

        # Create perturbation masks and perturb inputs based on the masks.
        # Inputs might be embedded during the perturbation process if the perturbator works with embeddings.
        pert_generator: Iterator[TensorMapping]
        mask_generator: Iterator[Int[torch.Tensor, "p l"] | None]
        pert_generator, mask_generator = split_iterator(self.perturbator.perturb(m) for m in model_inputs_to_explain)

        # Compute the score on perturbed inputs:
        # - If self.inference_wrapper.gradients is True, compute gradients.
        # - Otherwise, compute targeted logits.
        scores: Iterator[Float[torch.Tensor, "p t"]] = self.inference_wrapper(
            pert_generator, (a for a in sanitized_targets)
        )

        # Aggregate the scores using the aggregator function and the perturbation masks.
        # Aggregation over perturbations: (p, t), (p, l) -> (t, l)
        contributions: Iterator[Float[torch.Tensor, "t l"]] = (
            self.aggregator(score.detach(), mask) for score, mask in zip(scores, mask_generator, strict=True)
        )

        # Aggregate the score with respect to the granularity level
        # - Aggregate over the inputs for gradient-based methods: (t, l) -> (t, lg)
        # - Aggregate over the targets if the model is a generation model: (t, l) -> (tg, l)
        granular_contributions: Iterator[Float[torch.Tensor, "tg lg"]] = (
            self.granularity.granularity_score_aggregation(
                contribution=contribution.cpu(),
                granularity_aggregation_strategy=self.granularity_aggregation_strategy,
                inputs=inputs,  # type: ignore
                tokenizer=self.tokenizer,
                aggregate_inputs=self.inference_wrapper.gradients,  # Gradient-based methods
                aggregate_targets=isinstance(
                    self.inference_wrapper, TextGenerationInferenceWrapper
                ),  # Generation models
            )
            for contribution, inputs in zip(contributions, model_inputs_to_explain, strict=True)
        )

        # Decompose each input for the desired granularity level (tokens, words, sentences...)
        granular_inputs_texts: list[list[str]] = [
            self.granularity.get_decomposition(inputs, self.tokenizer, return_text=True)[0]  # type: ignore
            for inputs in model_inputs_to_explain
        ]

        # Create and return AttributionOutput objects with the contributions and decoded token sequences:
        results = []
        for contribution, model_input, elements, target in zip(
            granular_contributions, model_inputs_to_explain, granular_inputs_texts, sanitized_targets, strict=True
        ):
            # contributions post-processing
            clean_contribution = self.post_processing(contribution)

            # sanitize model_input
            model_input.pop("inputs_embeds", None)
            model_input["attention_mask"] = model_input["attention_mask"][0].unsqueeze(dim=0)

            # construct attribution output
            attribution_output = AttributionOutput(
                attributions=clean_contribution,
                elements=elements,
                model_inputs_to_explain=model_input,
                model_task=self._model_task,
                targets=target.cpu(),  # TODO: manage target device in the inference wrapper
                granularity=self.granularity,
                granularity_aggregation_strategy=self.granularity_aggregation_strategy,
                inference_mode=self.inference_wrapper.mode,
            )
            results.append(attribution_output)
        return results

    def __call__(self, model_inputs: ModelInputs, targets=None, **kwargs) -> list[AttributionOutput]:
        """
        Enables the explainer instance to be called as a function.

        Args:
            model_inputs (ModelInputs): Raw inputs for the model.
            targets: Targets for which explanations are desired. It depends on the task:
                - For classification tasks, encodes the target class or classes to explain.
                - For generation tasks, encodes the target text or tokens to explain.

        Returns:
            List[AttributionOutput]: A list of attribution outputs, one per input sample.
        """
        return self.explain(model_inputs, targets, **kwargs)


class TextClassificationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for classification models
    """

    _associated_inference_wrapper = TextClassificationInferenceWrapper
    base_tensor_perturbator_class = TextTensorPerturbator
    base_mask_perturbator_class = TextMaskPerturbator
    _model_task = ModelTask.CLASSIFICATION

    def process_targets(
        self, targets: ClassificationTarget, expected_length: int | None = None
    ) -> list[Int[torch.Tensor, "t"]]:
        """
        Normalize classification targets into a list of 1D integer tensors.

        Args:
            targets (int | torch.Tensor | Iterable[int] | Iterable[torch.Tensor]):
                The classification target(s). Supported formats include:
                - int: Interpreted as a single target.
                - torch.Tensor:
                    * 1D tensors are treated as a sequence of individual targets.
                    * 2D tensors must have shape (n, t), where `n` is the number of targets.
                - Iterable[int]: Each integer is treated as a separate target.
                - Iterable[torch.Tensor]: Each tensor must be 1D and contain integers.

            expected_length (int | None, optional):
                If specified, validates that the number of targets matches this expected length.

        Returns:
            Iterable[torch.Tensor]
                A list of 1D integer tensors, one per input instance.

        Raises:
            ValueError
                - If the number of targets does not match `expected_length`.
            TypeError
                - If the type of `targets` is unsupported.
                - If tensor targets are not 1D or 2D.
                - If tensor values are not integers.
        """
        # integer
        if isinstance(targets, int):
            if expected_length is not None and expected_length != 1:
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target is a single integer, but the length of the inputs is {expected_length}."
                )
            return [torch.tensor([targets])]

        # tensor
        if isinstance(targets, torch.Tensor):
            if targets.ndim == 1:
                # one dimensional tensors are treated as iterable of integer targets
                targets = targets.unsqueeze(-1)
            if expected_length is not None and expected_length != targets.shape[0]:
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target tensor of {targets.shape[0]} elements, but the length of the inputs is {expected_length}."
                )
            if targets.ndim != 2:
                raise TypeError(
                    "Target tensor must be one-dimensional or two-dimensional."
                    + f" Target tensor has {targets.ndim} dimensions."
                )
            if torch.is_floating_point(targets):
                raise TypeError("Target tensor must be integers.")
            return list(targets.unbind(dim=0))

        # iterable
        if isinstance(targets, Iterable):
            if expected_length is not None and len(targets) != expected_length:  # type: ignore
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target is an iterable of {len(targets)} elements, but the length of the inputs is {expected_length}."  # type: ignore
                )

            # iterable[int]
            if all(isinstance(t, int) for t in targets):
                return [torch.tensor([target]) for target in targets]

            # iterable[torch.Tensor]
            iterable_targets: list[torch.Tensor] = list(targets)  # type: ignore
            if all(isinstance(t, torch.Tensor) for t in iterable_targets):
                if any(target.ndim != 1 for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, the tensors must be one-dimensional.")
                if any(torch.is_floating_point(target) for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, they must be integers.")
                return iterable_targets

        raise TypeError(f"Target type {type(targets)} not supported.")

    @jaxtyped(typechecker=beartype)
    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: list[TensorMapping],
        targets: ClassificationTarget | None = None,
    ) -> tuple[list[TensorMapping], list[Int[torch.Tensor, "t"]]]:
        """
        Pre-processes model inputs and classification targets for explanation.

        This method ensures that:
        - If `targets` are not provided, they are computed by performing inference on `model_inputs` and selecting the predicted class using `argmax`.
        - The `targets` are then validated and converted using `self.process_targets`, ensuring the same length as `model_inputs`.

        Args:
        model_inputs : Iterable[TensorMapping]
            A batch of input mappings, typically containing tokenized inputs such as "input_ids", "attention_mask", etc.

        targets : int | torch.Tensor | Iterable[int] | Iterable[torch.Tensor] | None, optional
            Classification targets for each input. If None, targets are computed using model inference
            by selecting the index with the highest logit value for each input.

        Returns
        -------
        tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]
            - model_inputs_to_explain: List of tokenized input mappings with required explanation metadata (e.g., special tokens mask).
            - sanitized_targets: List of 1D integer tensors, each corresponding to a target label for an input.

        Raises
        ------
        ValueError
            If the provided or inferred targets do not match the number of input instances, or if their format is invalid.
        """
        sanitized_targets: list[Int[torch.Tensor, "t"]]
        if targets is None:
            # compute targets from logits if not provided
            # inputs have already been split, so we only need to select the first element
            sanitized_targets = [t[0] for t in self.inference_wrapper(model_inputs)]
        else:
            # process targets and ensure they have the same length as inputs
            sanitized_targets = self.process_targets(targets, expected_length=len(model_inputs))

        return model_inputs, sanitized_targets


class TextGenerationExplainer(AttributionExplainer):
    _associated_inference_wrapper = TextGenerationInferenceWrapper
    base_tensor_perturbator_class = TextTensorPerturbator
    base_mask_perturbator_class = TextMaskPerturbator
    _model_task = ModelTask.GENERATION

    def normalize_target_ids_with_leading_space(self, target: torch.Tensor) -> torch.Tensor:
        """Ensure target text starts with a space and return retokenized target ids."""
        target_text = self.tokenizer.decode(target, skip_special_tokens=True)
        if isinstance(target_text, str):
            if target_text.startswith(" "):
                return target
            normalized_target_text = f" {target_text}"
        elif isinstance(target_text, Iterable):
            normalized_target_text = [
                target_text_elem if target_text_elem.startswith(" ") else f" {target_text_elem}"
                for target_text_elem in target_text
            ]
        else:
            raise TypeError(
                f"Decoded target text must be a string or an iterable of strings, got {type(target_text)}."
            )
        normalized_target_ids = self.tokenizer(
            normalized_target_text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
        )["input_ids"].squeeze(dim=0)  # type: ignore
        return normalized_target_ids

    @jaxtyped(typechecker=beartype)
    def process_targets(
        self, targets: GeneratedTarget, expected_length: int | None = None
    ) -> list[Int[torch.Tensor, "t"]]:
        """
        Processes the target inputs for generative models into a standardized format.

        This function handles various input types for targets (string, TensorMapping, or Iterable)
        and converts them into a list of tensors containing token IDs.

        Args:
            targets (str, TensorMapping, torch.Tensor, or Iterable): The target texts or tokens.

        Returns:
            list[torch.Tensor]: A list of 1-D tensors representing the target token IDs.

        Raises:
            ValueError: If the target type is not supported.
        """
        if isinstance(targets, str):
            targets = self.tokenizer(
                targets if targets.startswith(" ") else " " + targets, return_tensors="pt", truncation=True
            )["input_ids"].squeeze(dim=0)  # type: ignore
            return [targets]  # type: ignore
        if isinstance(targets, MutableMapping):  # TensorMapping cannot be used in isinstance
            targets = targets["input_ids"]  # type: ignore
            if targets.dim() == 1:
                return list(self.normalize_target_ids_with_leading_space(targets))
            if targets.shape[0] > 1:
                targets = targets.split(1, dim=0)  # If the batch size > 1, we cut into a list of n mappings.
                return [self.normalize_target_ids_with_leading_space(t.squeeze(dim=0)) for t in targets]  # type: ignore
            return [self.normalize_target_ids_with_leading_space(targets.squeeze(dim=0))]
        if isinstance(targets, torch.Tensor):
            targets = targets.squeeze(dim=0)  # remove batch dimension if any
            assert targets.dim() == 1, "Target tensor must be 1-D."
            return [self.normalize_target_ids_with_leading_space(targets)]
        if isinstance(targets, Iterable):
            return list(itertools.chain(*[self.process_targets(item) for item in targets]))
        raise ValueError(
            f"type {type(targets)} not supported for method process_targets in class {self.__class__.__name__}"
        )

    @jaxtyped(typechecker=beartype)
    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: list[TensorMapping],
        targets: GeneratedTarget,
    ) -> tuple[list[TensorMapping], list[Int[torch.Tensor, "t"]]]:
        """
        Processes the inputs and targets for the generative model.
        If targets are not provided, create them with model_inputs_to_explain. Otherwise, for each input-target pair:
            a. Embed the input.
            b. Embed the target and concatenate with the input embeddings.
            c. Construct a new input mapping that includes both embeddings.
        Then, add offsets mapping and special tokens mask.

        Args:
            model_inputs (ModelInputs): The raw inputs for the generative model.
            targets (GeneratedTarget): The target texts or tokens for which explanations are desired.

        Returns:
            tuple: A tuple containing a list of processed model inputs and a list of processed targets.
        """
        # TODO: verify that inputs and targets have the same length
        sanitized_targets: list[Int[torch.Tensor, "t"]] = self.process_targets(targets)
        model_inputs_to_explain = []
        for model_input, target in zip(model_inputs, sanitized_targets, strict=True):
            target_2d = target.unsqueeze(dim=0)  # add batch dimension for concatenation with model_input
            model_inputs_to_explain.append(
                {
                    "input_ids": torch.cat(
                        [model_input["input_ids"].to(self.device), target_2d.to(self.device)], dim=1
                    ),  # type: ignore
                    "attention_mask": torch.cat(
                        [model_input["attention_mask"].to(self.device), torch.ones_like(target_2d).to(self.device)],
                        dim=1,
                    ),  # type: ignore
                }
            )

        # Convert to a `BatchEncoding` object and add offsets mapping:
        # TODO: see if it can be optimized, conversion might be necessary only for WORD and SENTENCE granularity
        model_inputs_to_explain_text = [
            self.tokenizer.decode(elem["input_ids"][0], skip_special_tokens=True) for elem in model_inputs_to_explain
        ]
        model_inputs_to_explain = [
            self.tokenizer(
                [model_inputs_to_explain_text],  # type: ignore
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
            )
            for model_inputs_to_explain_text in model_inputs_to_explain_text
        ]

        return model_inputs_to_explain, sanitized_targets  # type: ignore

    def post_processing(self, contribution: Float[torch.Tensor, "t l"]):
        """
        Generation specific post-processing of the attribution scores.

        Later generated tokens cannot be important for earlier ones, so we set them to NaN.

        Args:
            contribution (Float[torch.Tensor, "t l"]): The contribution values.

        Returns:
            model_task (ModelTask): The model task.
            contribution (Float[torch.Tensor, "t l"]): The post-processed contribution values.

        """
        t, l = contribution.shape
        mask = torch.triu(torch.ones((t, l), dtype=torch.bool), diagonal=l - t)
        contribution[mask] = float("nan")
        return contribution


@dataclass(slots=True)
class ImageAttributionOutput:
    """
    Class to store the output of an image-attribution method.

    Mirrors `AttributionOutput` with two changes for the image modality:
    `granularity` is typed as `ImageGranularity` (default `PATCH`), and instead of the
    text `elements` token labels it carries `attributions_image`, a display-ready
    pixel-resolution `(t, H, W)` map alongside the canonical `(t, g)` `attributions`.

    Attributes:
        attributions (SingleAttribution):
            Canonical per-unit attribution scores of shape `(t, g)`, where `g = H*W` (PIXEL)
            or `g = num_patches` (PATCH). This is the lossless numeric output — one scalar per
            granularity unit. Stored FLAT; patch `(row, col)` coordinates are derivable on demand
            from `model_inputs_to_explain["pixel_values"].shape[-2:]` and `patch_size`.

        attributions_image (Float[torch.Tensor, "t h w"]):
            Display-ready map of shape `(t, H, W)`, obtained by expanding `attributions` back to
            pixel resolution in `explain` (NEAREST for gradient methods, the mask resize strategy
            for masking methods). The visualization draws this directly and never interpolates
            itself.

        model_inputs_to_explain (TensorMapping):
            The output of `image_processor(image, return_tensors="pt")` (a `BatchFeature`,
            satisfies `TensorMapping`). Holds `pixel_values` of shape `(1, 3, H, W)`.

        image_mean (torch.Tensor | None):
            Per-channel mean used to normalize `pixel_values`. Together with `image_std`
            it is the affine that de-normalizes the image for visualization purposes:
            `x * image_std + image_mean`. Sourced from the explainer's `image_processor`
            when `preprocess=True`, or supplied by the user when `preprocess=False`.
            Identity `0` when no normalization was applied.

            Defaults to `None` so that an output built without it fails loudly at display
            time; an identity default would instead show a normalized image as if it were
            raw. `explain` always sets it.

        image_std (torch.Tensor | None):
            Per-channel standard deviation used to normalize `pixel_values`. See
            `image_mean`. Identity `1` when no normalization was applied.

        targets (torch.Tensor):
            The target class(es).

        classes (torch.Tensor | None):
            Optional tensor of class labels.

        granularity (ImageGranularity):
            The granularity level of the explanation. Defaults to `ImageGranularity.DEFAULT`
            (= `PATCH`).

        granularity_resize (GranularityResizeStrategy):
            The interpolation strategy used to resize per-pixel scores to the specified granularity.

        inference_mode (Callable[[torch.Tensor], torch.Tensor]):
            The mode used for inference (LOGITS, SOFTMAX, LOG_SOFTMAX). See `InferenceModes`.
    """

    attributions: SingleAttribution
    attributions_image: Float[torch.Tensor, "t h w"]
    model_inputs_to_explain: TensorMapping
    targets: torch.Tensor
    image_mean: torch.Tensor | None = None
    image_std: torch.Tensor | None = None
    classes: torch.Tensor | None = None
    granularity: ImageGranularity = ImageGranularity.DEFAULT
    granularity_resize: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR
    inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS


class ImageClassificationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for image-classification models (ViT-family).

    Mirrors `TextClassificationAttributionExplainer` with three modality-specific overrides:
    `__init__` swaps `tokenizer` for `image_processor` and drops the text-side
    `setup_token_ids` call (image configs have no pad/mask tokens),
    `process_model_inputs` accepts `BatchFeature` instead of `BatchEncoding`, and
    `explain` produces `ImageAttributionOutput` and passes `patch_size` / no tokenizer
    where the granularity API differs.
    """

    _associated_inference_wrapper = ImageClassificationInferenceWrapper
    base_tensor_perturbator_class = ImageTensorPerturbator
    base_mask_perturbator_class = ImageMaskPerturbator
    _model_task = ModelTask.CLASSIFICATION

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
        granularity: ImageGranularity = ImageGranularity.DEFAULT,
        resize_strategy: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        use_gradient: bool = False,
        input_x_gradient: bool = True,
        preprocess: bool = True,
        image_mean: Sequence[float] | float | torch.Tensor | None = None,
        image_std: Sequence[float] | float | torch.Tensor | None = None,
    ) -> None:
        """
        Args mirror `AttributionExplainer.__init__` with `tokenizer` -> `image_processor`
        and `granularity` typed as `ImageGranularity`. See parent for shared semantics.

        Additional image-only args:
            preprocess (bool): Whether interpreto runs `image_processor` on the input.
                If True, EVERY input — including the `pixel_values` of a `BatchFeature` —
                is routed through `image_processor`. If False, the input is taken as-is
                and assumed already normalized.

                NOTE: `BatchFeature` carries no provenance — it is a plain mapping, and
                `_validate_batch_feature` only checks the key, dtype and shape. So
                we cannot tell a processed `BatchFeature` from a hand-built raw one, and
                this flag is the ONLY signal. Consequence: `explain(image_processor(img))`
                under the default `preprocess=True` will double-normalize. Pass a raw
                image with `preprocess=True`, or a processed one with `preprocess=False`.

            image_mean / image_std (Sequence[float] | float | None): the affine that
                de-normalizes the image for visualization purposes, via
                `x * image_std + image_mean`. Required when `preprocess=False` (we cannot
                know what the user applied; guessing yields a plausible-looking wrong
                image). Forbidden when `preprocess=True` (we derive them from
                `image_processor`). For un-normalized input pass the identity
                `image_mean=0, image_std=1`; for a raw `[0, 255]` tensor, `image_mean=0,
                image_std=1/255`.

        Raises:
            ValueError: if `preprocess=False` and either stat is missing, or if
                `preprocess=True` and either stat is supplied.
        """
        # does not call setup_token_id because there is no need for a PAD token

        self.image_processor = image_processor
        self.preprocess = preprocess
        self.image_mean, self.image_std = self._resolve_normalization_stats(image_mean, image_std)

        self.inference_wrapper = self._associated_inference_wrapper(
            model,
            gradients=use_gradient,
            input_x_gradient=input_x_gradient,
            batch_size=batch_size,
            device=device,
            mode=inference_mode,
        )  # type: ignore
        self.perturbator = perturbator or ImageTensorPerturbator()
        self.aggregator = aggregator or Aggregator()
        self.granularity = granularity
        self.resize_strategy = resize_strategy
        # patch_size is sourced from the model config; required by ImageGranularity.PATCH
        self.patch_size = int(getattr(model.config, "patch_size", 16))
        # The explainer is the single source of truth for patch_size (it owns model.config).
        # A mask perturbator builds its (g, l) association matrix from patch_size, and the
        # explainer's (t, l) -> (t, g) aggregation interprets the result against the same g;
        # if the two disagree the mask<->score correspondence silently breaks. So we push the
        # authoritative value down, overriding the perturbator's placeholder default.
        # NOTE: this is the "version (a)" reconcile. If the isinstance wart or the
        # silently-overwritten default become a problem, switch to "version (b)" (perturbator
        # stops storing patch_size; explainer passes it into perturb() at call time).
        if isinstance(self.perturbator, ImageMaskPerturbator):
            self.perturbator.patch_size = self.patch_size
            self.perturbator.granularity_combination_strategy = self.resize_strategy

    def _resolve_normalization_stats(
        self,
        image_mean: Sequence[float] | float | torch.Tensor | None,
        image_std: Sequence[float] | float | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decide the de-normalization affine from `self.preprocess`, and validate.

        Keyed on the FLAG, not on whether processing actually ran: `preprocess=True` means
        we own the stats, so we read them off `self.image_processor` regardless of which
        input branch a given sample later takes.

        `do_normalize=False` -> identity. The processor keeps `image_mean`/`image_std` as
        class attributes even when it never applies them (`ViTImageProcessor.image_mean` is
        0.5 regardless), so reading them blindly would de-normalize an image that was never
        normalized.
        """
        if self.preprocess:
            if image_mean is not None or image_std is not None:
                raise ValueError(
                    "image_mean/image_std must not be supplied when preprocess=True: they are "
                    "derived from image_processor, so a supplied value would be silently "
                    "discarded. Either drop them, or set preprocess=False to own the "
                    "normalization yourself."
                )
            if not getattr(self.image_processor, "do_normalize", False):
                image_mean, image_std = 0.0, 1.0
            else:
                image_mean = self.image_processor.image_mean
                image_std = self.image_processor.image_std

        elif image_mean is None or image_std is None:
            raise ValueError(
                "preprocess=False requires both image_mean and image_std: the input is assumed "
                "already normalized and we cannot know which statistics were applied, so "
                "guessing would produce a plausible-looking but wrong image. Pass the affine "
                "that de-normalizes it (x * image_std + image_mean) — use image_mean=0, "
                "image_std=1 if no normalization was applied."
            )

        return (
            torch.as_tensor(image_mean, dtype=torch.float32).view(-1, 1, 1),
            torch.as_tensor(image_std, dtype=torch.float32).view(-1, 1, 1),
        )

    def process_targets(
        self, targets: ClassificationTarget, expected_length: int | None = None
    ) -> list[Int[torch.Tensor, "t"]]:
        """
        Normalize classification targets into a list of 1D integer tensors.

        Args:
            targets (int | torch.Tensor | Iterable[int] | Iterable[torch.Tensor]):
                The classification target(s). Supported formats include:
                - int: Interpreted as a single target.
                - torch.Tensor:
                    * 1D tensors are treated as a sequence of individual targets.
                    * 2D tensors must have shape (n, t), where `n` is the number of targets.
                - Iterable[int]: Each integer is treated as a separate target.
                - Iterable[torch.Tensor]: Each tensor must be 1D and contain integers.

            expected_length (int | None, optional):
                If specified, validates that the number of targets matches this expected length.

        Returns:
            Iterable[torch.Tensor]
                A list of 1D integer tensors, one per input instance.

        Raises:
            ValueError
                - If the number of targets does not match `expected_length`.
            TypeError
                - If the type of `targets` is unsupported.
                - If tensor targets are not 1D or 2D.
                - If tensor values are not integers.
        """
        # integer
        if isinstance(targets, int):
            if expected_length is not None and expected_length != 1:
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target is a single integer, but the length of the inputs is {expected_length}."
                )
            return [torch.tensor([targets])]

        # tensor
        if isinstance(targets, torch.Tensor):
            if targets.ndim == 1:
                # one dimensional tensors are treated as iterable of integer targets
                targets = targets.unsqueeze(-1)
            if expected_length is not None and expected_length != targets.shape[0]:
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target tensor of {targets.shape[0]} elements, but the length of the inputs is {expected_length}."
                )
            if targets.ndim != 2:
                raise TypeError(
                    "Target tensor must be one-dimensional or two-dimensional."
                    + f" Target tensor has {targets.ndim} dimensions."
                )
            if torch.is_floating_point(targets):
                raise TypeError("Target tensor must be integers.")
            return list(targets.unbind(dim=0))

        # iterable
        if isinstance(targets, Iterable):
            if expected_length is not None and len(targets) != expected_length:  # type: ignore
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target is an iterable of {len(targets)} elements, but the length of the inputs is {expected_length}."  # type: ignore
                )

            # iterable[int]
            if all(isinstance(t, int) for t in targets):
                return [torch.tensor([target]) for target in targets]

            # iterable[torch.Tensor]
            iterable_targets: list[torch.Tensor] = list(targets)  # type: ignore
            if all(isinstance(t, torch.Tensor) for t in iterable_targets):
                if any(target.ndim != 1 for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, the tensors must be one-dimensional.")
                if any(torch.is_floating_point(target) for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, they must be integers.")
                return iterable_targets

        raise TypeError(f"Target type {type(targets)} not supported.")

    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: list[TensorMapping],
        targets: ClassificationTarget | None = None,
    ) -> tuple[list[TensorMapping], list[Int[torch.Tensor, "t"]]]:
        """
        Pre-processes model inputs and classification targets for explanation.

        This method ensures that:
        - If `targets` are not provided, they are computed by performing inference on `model_inputs` and selecting the predicted class using `argmax`.
        - The `targets` are then validated and converted using `self.process_targets`, ensuring the same length as `model_inputs`.

        Args:
        model_inputs : Iterable[TensorMapping]
            A batch of input mappings, typically containing tokenized inputs such as "input_ids", "attention_mask", etc.

        targets : int | torch.Tensor | Iterable[int] | Iterable[torch.Tensor] | None, optional
            Classification targets for each input. If None, targets are computed using model inference
            by selecting the index with the highest logit value for each input.

        Returns
        -------
        tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]
            - model_inputs_to_explain: List of tokenized input mappings with required explanation metadata (e.g., special tokens mask).
            - sanitized_targets: List of 1D integer tensors, each corresponding to a target label for an input.

        Raises
        ------
        ValueError
            If the provided or inferred targets do not match the number of input instances, or if their format is invalid.
        """
        sanitized_targets: list[Int[torch.Tensor, "t"]]
        if targets is None:
            # compute targets from logits if not provided
            # inputs have already been split, so we only need to select the first element
            sanitized_targets = [t[0] for t in self.inference_wrapper(model_inputs)]
        else:
            # process targets and ensure they have the same length as inputs
            sanitized_targets = self.process_targets(targets, expected_length=len(model_inputs))

        return model_inputs, sanitized_targets

    def process_model_inputs(self, model_inputs: ModelInputs) -> list[TensorMapping]:
        """
        Normalize image inputs to a list of `BatchFeature` mappings, one per sample.

        `self.preprocess` decides whether the image processor runs, for EVERY input type:
            - `preprocess=True`: `PIL.Image`, `np.ndarray`, `torch.Tensor`, and the
              `pixel_values` of a `BatchFeature` are all routed through
              `self.image_processor`.
            - `preprocess=False`: `BatchFeature` is passed through after validation, and
              `torch.Tensor` of shape `(1, 3, H, W)` or `(3, H, W)` (auto-unsqueezed) is
              wrapped as `BatchFeature` directly. Both are assumed already normalized.
            - Iterables of any of the above are flattened recursively.

        A `BatchFeature` carries no provenance — `_validate_batch_feature` checks the key,
        dtype and shape, none of which say whether the processor ever ran. So a
        hand-built raw `BatchFeature` is indistinguishable from a processed one, and the
        flag is the only signal we have. Hence `explain(image_processor(img))` under the
        default `preprocess=True` double-normalizes; that is the documented cost of the
        flag meaning what it says.

        Mirrors text `process_model_inputs` with `BatchFeature` for `BatchEncoding`
        and `pixel_values` for `input_ids`. The `str` branch is replaced by raw-image
        branches; text has no `preprocess` flag because tokenization cannot be skipped.
        """
        # check BatchFeature BEFORE Iterable — BatchFeature is a UserDict and would
        # otherwise match the Iterable branch and recurse into its keys.
        if isinstance(model_inputs, BatchFeature):
            validated = self._validate_batch_feature(model_inputs)
            if not self.preprocess:
                return [validated]
            # MVP is ViT-classification-only, where pixel_values is the only key. Re-processing
            # rebuilds from pixel_values alone, so any other key a BatchFeature legally carries
            # (bool_masked_pos, interpolate_pos_encoding) would be dropped here. Nothing in scope
            # produces them; revisit if a head that needs them comes into scope.
            processed = self.image_processor(validated["pixel_values"], return_tensors="pt")
            return [self._validate_batch_feature(processed)]

        # Raw-image types (PIL.Image is NOT Iterable on the class level; np.ndarray and
        # torch.Tensor ARE — so we must catch them before the Iterable branch).
        if isinstance(model_inputs, (PILImage, np.ndarray, torch.Tensor)):
            if self.preprocess:
                processed = self.image_processor(model_inputs, return_tensors="pt")
                return [self._validate_batch_feature(processed)]

            # preprocess=False: only Tensor is allowed; assume already normalized.
            if not isinstance(model_inputs, torch.Tensor):
                raise ValueError(
                    f"When preprocess=False, raw inputs must be torch.Tensor, got {type(model_inputs)}. "
                    "Either set preprocess=True, or pre-process via image_processor and pass BatchFeature."
                )
            tensor = model_inputs
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[1] != 3 or tensor.shape[0] != 1:
                raise ValueError(
                    f"When preprocess=False, expected pixel_values of shape (1, 3, H, W), got {tuple(tensor.shape)}."
                )
            return [BatchFeature(data={"pixel_values": tensor})]

        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))

        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    def _validate_batch_feature(self, bf: BatchFeature) -> BatchFeature:
        """Sanity-check that a BatchFeature has a single-sample (1, 3, H, W) pixel_values tensor."""
        if "pixel_values" not in bf.keys():
            raise ValueError(
                "The BatchFeature must contain the key 'pixel_values' to be processed by the "
                f"image attribution explainer. Got {list(bf.keys())}."
            )
        if not isinstance(bf["pixel_values"], torch.Tensor):
            raise ValueError(
                "The BatchFeature must hold a torch.Tensor for 'pixel_values'. "
                "Use `image_processor(image, return_tensors='pt')`. "
                f"Got {type(bf['pixel_values'])}."
            )
        if bf["pixel_values"].ndim != 4 or bf["pixel_values"].shape[0] != 1:
            raise ValueError(
                "The BatchFeature must hold a single-sample (1, 3, H, W) pixel_values tensor. "
                "Pre-process images one by one, or pass an iterable. "
                f"Got shape {tuple(bf['pixel_values'].shape)}."
            )
        if bf["pixel_values"].shape[1] not in {1, 3}:
            raise ValueError("The BatchFeature number of channels (C) should be either 1 or 3")
        return bf

    def explain(
        self,
        model_inputs: ModelInputs,
        targets: ClassificationTarget | None = None,
    ) -> list[ImageAttributionOutput]:
        """
        Compute attributions for image-classification models.

        Mirrors `AttributionExplainer.explain` with image-side substitutions:
        - `granularity_resize` (the image counterpart of text's `granularity_score_aggregation`)
          is called with `patch_size` instead of `tokenizer`, and spatially interpolates the
          per-pixel field rather than reducing each unit to a scalar.
        - `get_decomposition` uses `return_coordinates=True` (image equivalent of `return_text`).
        - The `attention_mask` post-processing block is dropped (no attention_mask for ViT).
        - The output is `ImageAttributionOutput` instead of `AttributionOutput`.
        - The generation `aggregate_targets` branch is dropped (image MVP is classification-only).
        """
        sanitized_model_inputs: list[TensorMapping] = self.process_model_inputs(model_inputs)

        model_inputs_to_explain: list[TensorMapping]
        sanitized_targets: list[Int[torch.Tensor, "t"]]
        model_inputs_to_explain, sanitized_targets = self.process_inputs_to_explain_and_targets(
            sanitized_model_inputs,
            targets,  # type: ignore
        )

        # Perturbations + scores: same flow as text. The default Perturbator() yields
        # the input unchanged with a None mask, which is what gradient methods need.
        pert_generator: Iterator[TensorMapping]
        mask_generator: Iterator[Int[torch.Tensor, "p g"] | None]
        pert_generator, mask_generator = split_iterator(self.perturbator.perturb(m) for m in model_inputs_to_explain)

        # Gradient methods yield signed, per-channel scores (p, t, d, l); mask methods yield (p, t).
        scores: Iterator[Float[torch.Tensor, "p t d l"] | Float[torch.Tensor, "p t"]] = self.inference_wrapper(
            pert_generator, sanitized_targets
        )

        # Aggregate over perturbations: (p, t, d, l) -> (t, d, l) for gradient methods (the cross-
        #   perturbation statistic — mean/var/squared-mean — lands per channel, so VarGrad/SquareGrad
        #   get the faithful formula);
        #   (p, t), (p, g) -> (t, g) for masking methods where g = gh * gw.
        aggregated: Iterator[Float[torch.Tensor, "t d l"] | Float[torch.Tensor, "t g"]] = (
            self.aggregator(score.detach(), mask) for score, mask in zip(scores, mask_generator, strict=True)
        )

        # Collapse the 3 channels of gradient scores to a per-pixel value: (t, d, l) -> (t, l).
        # Signed mean (no abs): for mean/trapezoid methods this keeps direction (positive = pushes
        # the target logit up), matching the signed attributions the mask methods produce; for
        # VarGrad/SquareGrad the per-channel statistic is already non-negative so the sign is moot.
        # Done here, after the per-perturbation statistic, never inside the aggregator (mask methods
        # share the aggregator and have no channel axis).
        contributions: Iterator[Float[torch.Tensor, "t l"] | Float[torch.Tensor, "t g"]] = (
            contribution.mean(dim=1) if self.inference_wrapper.gradients else contribution
            for contribution in aggregated
        )

        # Aggregate over inputs for gradient-based methods: (t, l) -> (t, g).
        # Image granularity signature: patch_size instead of tokenizer; no aggregate_targets.
        granular_contributions: Iterator[Float[torch.Tensor, "t g"]] = (
            self.granularity.granularity_resize(
                contribution=contribution,
                granularity_resize_strategy=self.resize_strategy,
                inputs=inputs,
                patch_size=self.patch_size,
                aggregate_inputs=self.inference_wrapper.gradients,
            )
            for contribution, inputs in zip(contributions, model_inputs_to_explain, strict=True)
        )

        # Display map: expand the (t, g) granularity scores back to a pixel-resolution (t, H, W)
        # field so the visualization receives a ready-to-draw map and never interpolates itself.
        # Gradient methods discarded sub-patch detail in the l->g pool, so we re-expand with NEAREST
        # (honest flat blocks, no invented detail); mask methods reuse the strategy that expanded
        # their perturbation masks, so the heatmap matches what actually perturbed the model.
        display_strategy: GranularityResizeStrategy = (
            GranularityResizeStrategy.NEAREST if self.inference_wrapper.gradients else self.resize_strategy
        )

        results: list[ImageAttributionOutput] = []
        for contribution, model_input, target in zip(
            granular_contributions,
            model_inputs_to_explain,
            sanitized_targets,
            strict=True,
        ):
            attributions_image: Float[torch.Tensor, "t h w"] = self.granularity.resize_to_image(
                contribution=contribution,
                resize_strategy=display_strategy,
                inputs=model_input,
                patch_size=self.patch_size,
            )

            attribution_output = ImageAttributionOutput(
                attributions=contribution.cpu(),
                attributions_image=attributions_image.cpu(),
                model_inputs_to_explain=model_input,
                targets=target.cpu(),
                image_mean=self.image_mean,
                image_std=self.image_std,
                granularity=self.granularity,
                granularity_resize=self.resize_strategy,
                inference_mode=self.inference_wrapper.mode,
            )
            results.append(attribution_output)
        return results


class FactoryGeneratedMeta(ABCMeta):
    """
    Metaclass to distinguish classes generated by the MultitaskExplainerMixin.
    """


class MultitaskExplainerMixin(AttributionExplainer):
    """
    Mixin class to generate the appropriate Explainer based on the model type.
    """

    def __new__(cls, model: PreTrainedModel, *args: Any, **kwargs: Any) -> AttributionExplainer:
        if isinstance(cls, FactoryGeneratedMeta):
            return super().__new__(cls)  # type: ignore
        if model.__class__.__name__.endswith("ForSequenceClassification"):
            t = FactoryGeneratedMeta(
                "Classification" + cls.__name__, (cls, TextClassificationAttributionExplainer), {}
            )
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
            t = FactoryGeneratedMeta("Generation" + cls.__name__, (cls, TextGenerationExplainer), {})
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        if model.__class__.__name__.endswith("ForImageClassification"):
            t = FactoryGeneratedMeta(
                "Classification" + cls.__name__, (cls, ImageClassificationAttributionExplainer), {}
            )
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        raise NotImplementedError(
            "Model type not supported for Explainer. Use a ModelForSequenceClassification, a ModelForCausalLM model or a LMHeadModel model."
        )
