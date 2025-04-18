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
from collections.abc import Iterable, MutableMapping
from copy import deepcopy
from typing import Any

import torch
from jaxtyping import Float
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.perturbations.base import Perturbator
from interpreto.commons.generator_tools import split_iterator
from interpreto.commons.granularity import GranularityLevel
from interpreto.commons.model_wrapping.classification_inference_wrapper import ClassificationInferenceWrapper
from interpreto.commons.model_wrapping.generation_inference_wrapper import GenerationInferenceWrapper
from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper
from interpreto.typing import Generated_Target, ModelInputs, TensorMapping

SingleAttribution = (
    Float[torch.Tensor, "l"] | Float[torch.Tensor, "l c"] | Float[torch.Tensor, "l l_g"] | Float[torch.Tensor, "l l_t"]
)


class AttributionOutput:
    """
    Class to store the output of an attribution method.
    """

    __slots__ = ("attributions", "elements")

    def __init__(
        self,
        attributions: SingleAttribution,
        elements: list[str] | torch.Tensor | None = None,
    ):
        """
        Initializes an AttributionOutput instance.

        Args:
            attributions (Iterable[SingleAttribution]): A list (n elements, with n the number of samples) of attribution score tensors:
                - `l` represents the number of elements for which attribution is computed (for NLP tasks: can be the total sequence length).
                - Shapes depend on the task:
                    - Classification (single class): `(l,)`
                    - Classification (all classes): `(l, c)`, where `c` is the number of classes.
                    - Generative models: `(l_g, l)`, where `l_g` is the length of the generated part.
                        - For non-generated elements, there are `l_g` attribution scores.
                        - For generated elements, scores are zero for previously generated tokens.
                    - Token classification: `(l, l_t)`, where `l_t` is the number of token classes. When the tokens are disturbed, l = l_t.
            elements (Iterable[list[str]] | Iterable[torch.Tensor] | None, optional): A list or tensor representing the elements for which attributions are computed.
                - These elements can be tokens, words, sentences, or tensors of size `l`.
        """
        self.attributions = attributions
        self.elements = elements

    def __repr__(self):
        return f"AttributionOutput(attributions={repr(self.attributions)}, elements={repr(self.elements)})"

    def __str__(self):
        return f"AttributionOutput(attributions={self.attributions}, elements={self.elements})"


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
        batch_size: int,
        # inference_wrapper: InferenceWrapper,
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        use_gradient: bool = False,
        device: torch.device | None = None,
        granularity_level: GranularityLevel = GranularityLevel.DEFAULT,
    ) -> None:
        """
        Initializes the AttributionExplainer.

        Args:
            TODO : update docstring
            perturbator (Perturbator, optional): An instance for generating input perturbations.
                Defaults to a Perturbator if not provided.
            aggregator (Aggregator, optional): An instance used to aggregate computed attribution scores.
            use_gradient (bool): If True, use gradient-based methods for computing attributions. Defaults to False (using inference-based methods).
            device (torch.device, optional): The device on which computations will be performed.
            granularity_level (GranularityLevel): The level of granularity for the explanation (e.g., token, word, sentence).
        """
        self.tokenizer = tokenizer
        self.inference_wrapper = self._associated_inference_wrapper(model, batch_size=batch_size, device=device)
        self.perturbator = perturbator or Perturbator()
        self.aggregator = aggregator or Aggregator()
        self.use_gradient = use_gradient
        self.device = device
        self.granularity_level = granularity_level

        if self.use_gradient:
            self.get_scores = self.inference_wrapper.get_gradients
        else:
            self.get_scores = self.inference_wrapper.get_targeted_logits

        # TODO : check this line, eventually move it
        self.inference_wrapper.pad_token_id = self.tokenizer.pad_token_id

    def process_model_inputs(self, model_inputs: ModelInputs) -> list[TensorMapping]:
        """
        Processes and standardizes model inputs into a list of dictionaries compatible with the model.

        This method handles various input types:
            - If a string is provided, it tokenizes the string and returns a list containing one mapping.
            - If a mapping is provided with a batch (multiple samples), it splits the batch into individual mappings.
            - If an iterable is provided, it processes each item recursively.

        Args:
            model_inputs (str, MutableMapping, or Iterable): The raw model inputs.

        Returns:
            List[MutableMapping]: A list of processed model input mappings.

        Raises:
            ValueError: If the type of model_inputs is not supported.
        """
        if isinstance(model_inputs, str):
            return [
                self.tokenizer(
                    model_inputs, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True
                )
            ]
        if isinstance(model_inputs, MutableMapping):
            n = model_inputs["attention_mask"].shape[0]
            if n > 1:
                return [{key: value[i].unsqueeze(0) for key, value in model_inputs.items()} for i in range(n)]
            return [model_inputs]
        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))
        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    def explain(
        self, model_inputs: ModelInputs, targets: Generated_Target, **generation_kwargs
    ) -> Iterable[AttributionOutput]:
        """
        Computes attributions for generative models.

        Process:
            1. Move the model to the designated device.
            2. Process and standardize the model inputs.
            3. Create the tokenizer's pad token if not already set and add it to the inference wrapper.
            4. If targets are not provided, create them. Otherwise, for each input-target pair, process them.
            5. Generate perturbations for the constructed inputs.
            6. Compute scores using either gradients (if use_gradient is True) or targeted logits.
            7. Aggregate the scores to obtain contribution values.
            8. Decompose the inputs based on the desired granularity and decode tokens.

        Args:
            model_inputs (ModelInputs): Raw inputs for the generative model.
            targets (ModelInputs, optional): Target texts or tokens for which explanations are desired.

        Returns:
            List[AttributionOutput]: A list of attribution outputs, one per input sample.
        """
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_wrapper.to(self.device)

        model_inputs = self.process_model_inputs(model_inputs)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inference_wrapper.pad_token_id = self.tokenizer.pad_token_id

        model_inputs_to_explain, targets = self.process_inputs_to_explain_and_targets(
            model_inputs, targets, **generation_kwargs
        )

        pert_generator, mask_generator = split_iterator(self.perturbator.perturb(m) for m in model_inputs_to_explain)

        # scores = list(self.get_scores(pert_generator, targets))
        if self.use_gradient:
            # Compute gradients for each perturbed input.
            scores = list(self.inference_wrapper.get_gradients(pert_generator, targets))
        else:
            # Compute targeted logits for each perturbed input.
            scores = list(self.inference_wrapper.get_targeted_logits(pert_generator, targets))

        # Retrieve the perturbation masks.
        masks = list(mask_generator)

        # Aggregate the scores using the aggregator to obtain contribution values.

        # TODO : check if we need to add a squeeze(0) here (in generation version we have but not in classification)
        contributions = [
            self.aggregator(score.unsqueeze(0), mask).squeeze(0)
            for score, mask in zip(scores, masks, strict=True)  # generation version
        ]
        # contributions = [self.aggregator(score, mask).squeeze(0) for score, mask in zip(scores, masks, strict=True)] #classification version

        # Decompose each input for the desired granularity level.
        decompositions = [
            GranularityLevel.get_decomposition(t, self.granularity_level) for t in model_inputs_to_explain
        ]

        # Create and return AttributionOutput objects with the contributions and decoded token sequences:
        return [
            AttributionOutput(
                c,
                [
                    self.tokenizer.decode(
                        token_ids, skip_special_tokens=self.granularity_level is not GranularityLevel.ALL_TOKENS
                    )
                    for token_ids in d[0]
                ],
            )
            for c, d in zip(contributions, decompositions, strict=True)
        ]

    def __call__(
        self, model_inputs: ModelInputs, targets: ModelInputs | torch.Tensor | None = None
    ) -> Iterable[AttributionOutput]:
        """
        Enables the explainer instance to be called as a function.

        Args:
            model_inputs (ModelInputs): The inputs to the model.
            targets (torch.Tensor, optional): The target classes or tokens.

        Returns:
            Any: The computed attributions.
        """
        return self.explain(model_inputs, targets)


class ClassificationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for classification models
    """

    _associated_inference_wrapper = ClassificationInferenceWrapper

    def process_inputs_to_explain_and_targets(
        self, model_inputs: ModelInputs, targets: torch.tensor | None
    ) -> tuple[list[TensorMapping], list[torch.Tensor]]:
        logits = torch.stack(list(self.inference_wrapper.get_logits(deepcopy(model_inputs))))
        if targets is None:
            targets = logits.argmax(dim=-1)

        # TODO : change call to process_target
        targets = ClassificationInferenceWrapper.process_target(targets, logits.shape[:-1])
        return model_inputs, targets


class GenerationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for generation models
    """

    _associated_inference_wrapper = GenerationInferenceWrapper

    def process_targets(self, targets: Generated_Target) -> list[torch.Tensor]:
        """
        Processes the target inputs for generative models into a standardized format.

        This function handles various input types for targets (string, MutableMapping, or Iterable)
        and converts them into a list of tensors containing token IDs.

        Args:
            targets (str, MutableMapping, torch.Tensor, or Iterable): The target texts or tokens.

        Returns:
            List[torch.Tensor]: A list of tensors representing the target token IDs.

        Raises:
            ValueError: If the target type is not supported.
        """
        if isinstance(targets, str):
            return [self.tokenizer(targets, return_tensors="pt")["input_ids"]]
        if isinstance(targets, MutableMapping):
            targets = targets["input_ids"]
            if targets.shape[0] > 1:
                return list(targets.split(1, dim=0))
            return [targets]
        if isinstance(targets, torch.Tensor):
            return [targets]
        if isinstance(targets, Iterable):
            return list(itertools.chain(*[self.process_targets(item) for item in targets]))
        raise ValueError(
            f"type {type(targets)} not supported for method process_targets in class {self.__class__.__name__}"
        )

    def process_inputs_to_explain_and_targets(
        self, model_inputs: ModelInputs, targets: Generated_Target, generation_kwargs: dict[str, Any]
    ) -> tuple[list[TensorMapping], list[torch.Tensor]]:
        """
        Processes the inputs and targets for the generative model.
        If targets are not provided, create them with model_inputs_to_explain. Otherwise, for each input-target pair:
            a. Embed the input.
            b. Embed the target and concatenate with the input embeddings.
            c. Construct a new input mapping that includes both embeddings.

        Args:
            model_inputs (ModelInputs): The raw inputs for the generative model.
            targets (Generated_Target): The target texts or tokens for which explanations are desired.
            generation_kwargs (dict): Additional arguments for the generation process.

        Returns:
            tuple: A tuple containing a list of processed model inputs and a list of processed targets.
        """
        if targets is None:
            model_inputs_to_explain, targets = self.inference_wrapper.get_inputs_to_explain_and_targets(
                model_inputs, **generation_kwargs
            )
        else:
            targets = self.process_targets(targets)
            model_inputs_to_explain = []
            for model_input, target in zip(model_inputs, targets, strict=True):
                embed_model_input = self.inference_wrapper.embed(model_input)
                with torch.no_grad():
                    target_embed = self.inference_wrapper.model.get_input_embeddings()(target)
                model_inputs_to_explain.append(
                    {
                        "inputs_embeds": torch.cat([embed_model_input["inputs_embeds"], target_embed], dim=1),
                        "attention_mask": torch.cat(
                            [embed_model_input["attention_mask"], torch.ones_like(target)], dim=1
                        ),
                    }
                )
        # Add offsets mapping:
        model_inputs_to_explain_text = [
            self.tokenizer.decode(model_input_to_explain["input_ids"][0])
            for model_input_to_explain in model_inputs_to_explain
        ]
        model_inputs_to_explain = [
            self.tokenizer([model_inputs_to_explain_text], return_tensors="pt", return_offsets_mapping=True)
            for model_inputs_to_explain_text in model_inputs_to_explain_text
        ]

        # Decompose each input for the desired granularity level.
        # TODO: move this in a better place
        if self.granularity_level == GranularityLevel.TOKEN:
            self.granularity_level = GranularityLevel.ALL_TOKENS  # equal for generative models

        return model_inputs_to_explain, targets


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
