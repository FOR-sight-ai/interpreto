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
from abc import abstractmethod
from collections.abc import Callable, Iterable, MutableMapping
from enum import Enum
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import Perturbator
from interpreto.commons import Granularity
from interpreto.commons.generator_tools import split_iterator
from interpreto.commons.granularity import GranularityAggregationStrategy
from interpreto.model_wrapping.classification_inference_wrapper import ClassificationInferenceWrapper
from interpreto.model_wrapping.generation_inference_wrapper import GenerationInferenceWrapper
from interpreto.model_wrapping.inference_wrapper import InferenceModes, InferenceWrapper
from interpreto.typing import ClassificationTarget, GeneratedTarget, ModelInputs, TensorMapping

SingleAttribution = (
    Float[torch.Tensor, "l"] | Float[torch.Tensor, "l c"] | Float[torch.Tensor, "l l_g"] | Float[torch.Tensor, "l l_t"]
)


class ModelTask(Enum):
    """
    Enum to represent the model task type.
    """

    SINGLE_CLASS_CLASSIFICATION = "single-class classification"
    MULTI_CLASS_CLASSIFICATION = "multi-class classification"
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


class AttributionOutput:
    """
    Class to store the output of an attribution method.
    """

    __slots__ = ("attributions", "elements", "model_task", "classes")

    def __init__(
        self,
        attributions: SingleAttribution,
        elements: list[str] | torch.Tensor,
        model_task: ModelTask,
        classes: torch.Tensor | None = None,
    ):
        """
        Initializes an AttributionOutput instance.

        # TODO: Harmonize even more, all attributions could be of the shape (t, l),
        # with t being either a number of class or of generated tokens.
        # It should not be a problem if some values are None or zero for generation.
        # This should be thoroughly tested.

        Args:
            attributions (Iterable[SingleAttribution]): A list (n elements, with n the number of samples) of attribution score tensors:
                - `l` represents the number of elements for which attribution is computed (for NLP tasks: can be the total sequence length).
                - Shapes depend on the task:
                    - Classification (single class): `(l)`
                    - Classification (multi classes): `(c, l)`, where `c` is the number of classes.
                    - Generative models: `(l_g, l)`, where `l_g` is the length of the generated part.
                        - For non-generated elements, there are `l_g` attribution scores.
                        - For generated elements, scores are zero for previously generated tokens.
                    - Token classification: `(l_t, l)`, where `l_t` is the number of token classes. When the tokens are disturbed, l = l_t.
            elements (Iterable[list[str]] | Iterable[torch.Tensor]): A list or tensor representing the elements for which attributions are computed.
                - These elements can be tokens, words, sentences, or tensors of size `l`.
            model_task (ModelTask): An enum representing the task of the model explained, such as SINGLE_CLASS_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, or GENERATION.
            classes (torch.Tensor | None): Optional tensor of class labels.
                - For single-class classification: tensor of shape `(1)`
                - For multi-class classification: tensor of shape `(c)` where `c` is the number of classes
        """
        self.attributions = attributions
        self.elements = elements
        self.model_task = model_task
        self.classes = classes

    def __repr__(self):
        return (
            f"AttributionOutput("
            f"attributions={repr(self.attributions)}, "
            f"elements={repr(self.elements)}, "
            f"model_task='{self.model_task}', "
            f"classes={repr(self.classes)})"
        )

    def __str__(self):
        return (
            f"AttributionOutput("
            f"attributions={self.attributions}, "
            f"elements={self.elements}, "
            f"model_task='{self.model_task}', "
            f"classes={self.classes})"
        )


class AttributionExplainer:
    """
    Abstract base class for attribution explainers.

    This class defines a common interface and helper methods used by various attribution explainers.
    Subclasses must implement the abstract method 'explain'.
    """

    _associated_inference_wrapper = InferenceWrapper

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
        granularity: Granularity = Granularity.DEFAULT,
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
            granularity (Granularity, optional): The level of granularity for the explanation.
                Options are: `ALL_TOKENS`, `TOKEN`, `WORD`, or `SENTENCE`.
                Defaults to Granularity.DEFAULT (ALL_TOKENS)
                To obtain it, `from interpreto import Granularity` then `Granularity.WORD`.
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
        self.use_gradient = use_gradient
        self.input_x_gradient = input_x_gradient
        if not hasattr(self, "tokenizer"):
            model, _ = self._set_tokenizer(model, tokenizer)
        self.inference_wrapper = self._associated_inference_wrapper(
            model, batch_size=batch_size, device=device, mode=inference_mode
        )  # type: ignore
        self.perturbator = perturbator or Perturbator()
        self.perturbator.to(self.device)
        self.aggregator = aggregator or Aggregator()
        self.granularity = granularity
        self.granularity_aggregation_strategy = granularity_aggregation_strategy
        self.inference_wrapper.pad_token_id = self.tokenizer.pad_token_id

    def _set_tokenizer(self, model, tokenizer) -> tuple[PreTrainedModel, int]:
        self.tokenizer = tokenizer
        # replace token for perturbations
        replace_token = "[REPLACE]"
        if replace_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([replace_token])

        # add a pad token if it does not exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # resize model with new tokens
        model.resize_token_embeddings(len(self.tokenizer))

        replace_token_id = self.tokenizer.convert_tokens_to_ids(replace_token)
        if isinstance(replace_token_id, list):
            replace_token_id = replace_token_id[0]

        return model, replace_token_id

    def get_scores(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: Iterable[torch.Tensor],
    ) -> Iterable[torch.Tensor]:
        """
        Computes scores for the given perturbations and targets.

        Args:
            pert_generator (Iterable[TensorMapping]): An iterable of perturbed model inputs.
            targets (torch.Tensor): The target classes or tokens.

        Returns:
            Iterable[torch.Tensor]: The computed scores.
        """
        if self.use_gradient:
            return self.inference_wrapper.get_gradients(model_inputs, targets, input_x_gradient=self.input_x_gradient)
        return self.inference_wrapper.get_targeted_logits(model_inputs, targets)

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
            return [self.tokenizer(model_inputs, return_tensors="pt", return_offsets_mapping=True)]
        if isinstance(
            model_inputs, BatchEncoding
        ):  # we cant use TensorMapping in the isinstance so we use MutableMapping.
            splitted_encodings = []
            for i, enc in enumerate(model_inputs.encodings):  # type: ignore  # one Encoding per row
                data_i = {
                    k: (v[i].unsqueeze(0) if isinstance(v, torch.Tensor) else [v[i]]) for k, v in model_inputs.items()
                }
                splitted_encodings.append(
                    BatchEncoding(
                        data=data_i,  # tensors/arrays for that row
                        encoding=enc,  # its Encoding (keeps word_ids, offsets…) necessary for granularity
                        tensor_type="pt",  # keep tensors if you had them
                    )
                )
            return splitted_encodings
        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))
        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    @abstractmethod
    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: torch.Tensor | Iterable[torch.Tensor] | None = None,
        **model_kwargs: Any,
    ) -> tuple[Iterable[TensorMapping], Iterable[Float[torch.Tensor, "n t"]]]:
        """
        Processes the inputs and targets for explanation.

        This method must be implemented by subclasses.

        Args:
            model_inputs (Iterable[TensorMapping]): The inputs to the model.
            targets (Any): The targets to be explained.
            model_kwargs (Any): Additional model-specific arguments.

        Returns:
            tuple: A tuple of (processed_inputs, processed_targets).

        Raises:
            NotImplementedError: Always raised. Subclasses must implement this method.
        """
        raise NotImplementedError(
            "Specific task subclasses must implement the 'process_inputs_to_explain_and_targets' method "
            "to correctly process inputs and targets for explanations."
        )

    def explain(
        self,
        model_inputs: ModelInputs,
        targets: torch.Tensor
        | Iterable[torch.Tensor]
        | None = None,  # TODO: create specific target type for classification and generation
        **model_kwargs: Any,
    ) -> Iterable[AttributionOutput]:
        """
        Computes attributions for NLP models.

        Process:
            1. Process and standardize the model inputs.
            2. Create the tokenizer's pad token if not already set and add it to the inference wrapper.
            3. If targets are not provided, create them. Otherwise, for each input-target pair, process them.
            4. Decompose the inputs based on the desired granularity and decode tokens.
            5. Generate perturbations for the constructed inputs.
            6. Compute scores using either gradients (if use_gradient is True) or targeted logits.
            7. Aggregate the scores to obtain contribution values.

        Args:
            model_inputs (ModelInputs): Raw inputs for the model.
            targets (torch.Tensor | Iterable[torch.Tensor] | None): Targets for which explanations are desired.
            Further types might be supported by sub-classes.
            It depends on the task:
                - For classification tasks, encodes the target class or classes to explain.
                - For generation tasks, encodes the target text or tokens to explain.

        Returns:
            List[AttributionOutput]: A list of attribution outputs, one per input sample.
        """
        # Ensure the model inputs are in the correct format
        sanitized_model_inputs: Iterable[TensorMapping] = self.process_model_inputs(model_inputs)

        # Process the inputs and targets for explanation
        # If targets are not provided, create them from model_inputs_to_explain.
        model_inputs_to_explain: Iterable[TensorMapping]
        sanitized_targets: Iterable[Float[torch.Tensor, "n t"]]
        model_inputs_to_explain, sanitized_targets_gen = self.process_inputs_to_explain_and_targets(
            sanitized_model_inputs, targets, **model_kwargs
        )
        sanitized_targets = list(sanitized_targets_gen)

        # Create perturbation masks and perturb inputs based on the masks.
        # Inputs might be embedded during the perturbation process if the perturbator works with embeddings.
        pert_generator: Iterable[TensorMapping]
        mask_generator: Iterable[torch.Tensor | None]
        pert_generator, mask_generator = split_iterator(self.perturbator.perturb(m) for m in model_inputs_to_explain)

        # Compute the score on perturbed inputs:
        # - If use_gradient is True, compute gradients.
        # - Otherwise, compute targeted logits.
        scores: Iterable[torch.Tensor] = self.get_scores(
            pert_generator, (a.to(self.device) for a in sanitized_targets)
        )

        # Aggregate the scores using the aggregator function and the perturbation masks.
        contributions = (
            self.aggregator(score.detach(), mask.to(self.device) if mask is not None else None)
            for score, mask in zip(scores, mask_generator, strict=True)
        )

        if self.use_gradient:
            granular_contributions = [
                Granularity.aggregate_score_for_gradient_method(
                    contribution.cpu(),
                    self.granularity,
                    self.granularity_aggregation_strategy,  # type: ignore
                    inputs,  # type: ignore
                    self.tokenizer,
                )
                for contribution, inputs in zip(contributions, model_inputs_to_explain, strict=True)
            ]
        else:
            granular_contributions = contributions

        # Decompose each input for the desired granularity level (tokens, words, sentences...)
        granular_inputs_texts: list[list[str]] = [
            Granularity.get_decomposition(t, self.granularity, self.tokenizer, return_text=True)[0]  # type: ignore
            for t in model_inputs_to_explain
        ]  # type: ignore

        # Create and return AttributionOutput objects with the contributions and decoded token sequences:
        results = []
        for contribution, elements, target in zip(
            granular_contributions, granular_inputs_texts, sanitized_targets, strict=True
        ):
            if self.inference_wrapper.__class__.__name__ == "GenerationInferenceWrapper":
                model_task = ModelTask.GENERATION
                classes = None
            elif self.inference_wrapper.__class__.__name__ == "ClassificationInferenceWrapper":
                classes = target
                if contribution.shape[0] == 1:
                    model_task = ModelTask.SINGLE_CLASS_CLASSIFICATION
                else:
                    model_task = ModelTask.MULTI_CLASS_CLASSIFICATION
            else:
                raise NotImplementedError(
                    f"Model type {self.inference_wrapper.model.__class__.__name__} not supported for AttributionExplainer."
                )
            attribution_output = AttributionOutput(
                attributions=contribution, elements=elements, model_task=model_task, classes=classes
            )
            results.append(attribution_output)
        return results

    def __call__(self, model_inputs: ModelInputs, targets=None, **kwargs) -> Iterable[AttributionOutput]:
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


class ClassificationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for classification models
    """

    _associated_inference_wrapper = ClassificationInferenceWrapper
    inference_wrapper: ClassificationInferenceWrapper

    def process_targets(
        self, targets: ClassificationTarget, expected_length: int | None = None
    ) -> Iterable[Int[torch.Tensor, "t"]]:
        """
        Normalize classification targets into a list of 1D integer tensors.

        Parameters
        ----------
        targets : int | Int[torch.Tensor, "n"] | Int[torch.Tensor, "n t"] | Iterable[int] | Iterable[Int[torch.Tensor, "t"]]
            The classification target(s). Supported formats include:
            - A single integer: Interpreted as a single target.
            - A 1D or 2D integer torch.Tensor:
                * 1D tensors are treated as a sequence of individual targets.
                * 2D tensors must have shape (n, t), where `n` is the number of targets.
            - An iterable of integers: Each integer is treated as a separate target.
            - An iterable of 1D integer torch.Tensors: Each tensor must be 1D and contain integers.

        expected_length : int | None, optional
            If specified, validates that the number of targets matches this expected length.

        Returns
        -------
        Iterable[Int[torch.Tensor, "t"]]
            A list of 1D integer tensors, one per input instance.

        Raises
        ------
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
            if targets.ndim != 2:  # actually verified by jaxtyping
                raise TypeError(
                    "Target tensor must be one-dimensional or two-dimensional."
                    + f" Target tensor has {targets.ndim} dimensions."
                )
            if torch.is_floating_point(targets):  # actually verified by jaxtyping
                raise TypeError("Target tensor must be integers.")
            return targets.unbind(dim=0)

        # iterable
        if isinstance(targets, Iterable):
            if expected_length is not None and len(targets) != expected_length:  # type: ignore
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target is an iterable of {len(targets)} elements, but the length of the inputs is {expected_length}."  # type: ignore
                )

            # iterable[int]
            if all(isinstance(t, int) for t in targets):  # actually verified by jaxtyping
                return [torch.tensor(target) for target in targets]

            # iterable[torch.Tensor]
            iterable_targets: Iterable[torch.Tensor] = targets  # type: ignore
            if all(isinstance(t, torch.Tensor) for t in iterable_targets):  # actually verified by jaxtyping
                if any(target.ndim != 1 for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, the tensors must be one-dimensional.")
                if any(torch.is_floating_point(target) for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, they must be integers.")
                return iterable_targets

        raise TypeError(f"Target type {type(targets)} not supported.")

    @jaxtyped(typechecker=beartype)
    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: ClassificationTarget | None = None,
        **model_kwargs: Any,
    ) -> tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]:
        """
        Pre-processes model inputs and classification targets for explanation.

        This method ensures that:
        - If `targets` are not provided, they are computed by performing inference on `model_inputs` and selecting the predicted class using `argmax`.
        - The `targets` are then validated and converted using `self.process_targets`, ensuring the same length as `model_inputs`.

        Parameters
        ----------
        model_inputs : Iterable[TensorMapping]
            A batch of input mappings, typically containing tokenized inputs such as "input_ids", "attention_mask", etc.

        targets : int | torch.Tensor | Iterable[int] | Iterable[torch.Tensor] | None, optional
            Classification targets for each input. If None, targets are computed using model inference
            by selecting the index with the highest logit value for each input.

        **model_kwargs : Any
            Additional keyword arguments passed to the model during inference, if targets are inferred.

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
        if targets is None:
            # compute targets from logits if not provided
            sanitized_targets: Iterable[torch.Tensor] = self.inference_wrapper.get_targets(model_inputs)  # type: ignore
        else:
            # process targets and ensure they have the same length as inputs
            expected_targets_length = len(model_inputs)  # type: ignore
            sanitized_targets: Iterable[torch.Tensor] = self.process_targets(targets, expected_targets_length)  # type: ignore

        return model_inputs, sanitized_targets


class GenerationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for generation models
    """

    _associated_inference_wrapper = GenerationInferenceWrapper
    inference_wrapper: GenerationInferenceWrapper

    @jaxtyped(typechecker=beartype)
    def process_targets(self, targets: GeneratedTarget, expected_length: int | None = None) -> list[torch.Tensor]:
        """
        Processes the target inputs for generative models into a standardized format.

        This function handles various input types for targets (string, TensorMapping, or Iterable)
        and converts them into a list of tensors containing token IDs.

        Args:
            targets (str, TensorMapping, torch.Tensor, or Iterable): The target texts or tokens.

        Returns:
            List[torch.Tensor]: A list of tensors representing the target token IDs.

        Raises:
            ValueError: If the target type is not supported.
        """
        if isinstance(targets, str):
            return [self.tokenizer(targets, return_tensors="pt")["input_ids"]]  # type: ignore
        if isinstance(targets, MutableMapping):  # TensorMapping cannot be used in isinstance
            targets = targets["input_ids"]
            if targets.dim() == 1:
                return list(
                    targets.unsqueeze(0)
                )  # If there's only one sentence in the mapping, we standardize by giving it a first batch dimension of 1
            if targets.shape[0] > 1:
                return list(
                    targets.split(1, dim=0)
                )  # If the batch dimension n is greater than 1, we cut the batch into a list of size n with mappings having a batch of 1.
            return [targets]
        if isinstance(targets, torch.Tensor):
            return [targets]
        if isinstance(targets, Iterable):
            return list(itertools.chain(*[self.process_targets(item) for item in targets]))
        raise ValueError(
            f"type {type(targets)} not supported for method process_targets in class {self.__class__.__name__}"
        )

    @jaxtyped(typechecker=beartype)
    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: GeneratedTarget | None = None,
        **model_kwargs,
    ) -> tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]:
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
            model_kwargs (dict): Additional arguments for the generation process.

        Returns:
            tuple: A tuple containing a list of processed model inputs and a list of processed targets.
        """
        # TODO: verify that inputs and targets have the same length
        sanitized_targets: list[torch.Tensor]
        if targets is None:
            model_inputs_to_explain, sanitized_targets = self.inference_wrapper.get_inputs_to_explain_and_targets(
                model_inputs, **model_kwargs
            )
        else:
            sanitized_targets = self.process_targets(targets)
            model_inputs_to_explain = []
            for model_input, target in zip(model_inputs, sanitized_targets, strict=True):
                model_inputs_to_explain.append(
                    {
                        "input_ids": torch.cat([model_input["input_ids"], target], dim=1),  # type: ignore
                        "attention_mask": torch.cat([model_input["attention_mask"], torch.ones_like(target)], dim=1),  # type: ignore
                    }
                )

        # Convert to a `BatchEncoding` object and add offsets mapping:
        # TODO: see if it can be optimized, conversion might be necessary only for WORD and SENTENCE granularity
        model_inputs_to_explain_text = [
            self.tokenizer.decode(elem["input_ids"][0], skip_special_tokens=True) for elem in model_inputs_to_explain
        ]
        model_inputs_to_explain = [
            self.tokenizer(
                [model_inputs_to_explain_text],
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            for model_inputs_to_explain_text in model_inputs_to_explain_text
        ]

        return model_inputs_to_explain, sanitized_targets


class FactoryGeneratedMeta(type):
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
            t = FactoryGeneratedMeta("Classification" + cls.__name__, (cls, ClassificationAttributionExplainer), {})
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
            t = FactoryGeneratedMeta("Generation" + cls.__name__, (cls, GenerationAttributionExplainer), {})
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        raise NotImplementedError(
            "Model type not supported for Explainer. Use a ModelForSequenceClassification, a ModelForCausalLM model or a LMHeadModel model."
        )
