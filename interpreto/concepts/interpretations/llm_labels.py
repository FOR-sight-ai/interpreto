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

import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Literal, NamedTuple

import torch
from jaxtyping import Float

from interpreto.commons.granularity import GranularityAggregationStrategy
from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.concepts.interpretations.base import (
    BaseConceptInterpretationMethod,
)
from interpreto.model_wrapping.llm_interface import LLMInterface, Role
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity
from interpreto.typing import ConceptsActivations, LatentActivations


class SamplingMethod(Enum):
    TOP = "top"
    QUANTILE = "quantile"
    RANDOM = "random"

    def sample_examples(
        self,
        concept_activations: Float[torch.Tensor, "ng"],
        k_examples: int,
        k_quantile: int = 5,
    ) -> list[int]:
        """Select examples to provide to the LLM for labeling a concept, based on the
        concept activations and the sampling method.

        Args:
            concept_activations (Float[torch.Tensor, "ng"]): concepts activations for each granular
                text, first dimension is number of texts, second is number of concepts.
            k_examples (int): number of examples to select, when possible.
            k_quantile (int, optional): number of quantiles to use for sampling the inputs, if `sampling_method` is `QUANTILE`. Defaults to 5.

        Raises:
            NotImplementedError: when `self.sampling_method` is not one of the supported methods.

        Returns:
            list[int]: the indexes of the examples to provide to the LLM for labeling the concept.
        """
        if self == SamplingMethod.TOP:
            inputs_idx = _sample_top(
                concept_activations=concept_activations,
                k_examples=k_examples,
            )
        elif self == SamplingMethod.QUANTILE:
            inputs_idx = _sample_quantile(
                concept_activations=concept_activations,
                k_examples=k_examples,
                k_quantile=k_quantile,
            )
        elif self == SamplingMethod.RANDOM:
            inputs_idx = _sample_random(
                concept_activations=concept_activations,
                k_examples=k_examples,
            )
        else:
            raise NotImplementedError(f"Sampling method {self} is not implemented.")
        return inputs_idx


class Example(NamedTuple):
    texts: list[str] | str
    activations: list[int] | int


class LLMLabels(BaseConceptInterpretationMethod):
    """Code [:octicons-mark-github-24: `concepts/interpretations/llm_labels.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/llm_labels.py)

    Implement the automatic labeling method using a language model (LLM) to provide a short textual description given some examples of what activate the concept.
    This method was first introduced in [^1], we implement here the step 1 of the method.

    [^1]:
        Steven Bills*, Nick Cammarata*, Dan Mossing*, Henk Tillman*, Leo Gao*, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu*, William Saunders*
        [Language models can explain neurons in language models](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
        2023.

    Arguments:
        concept_explainer (ConceptEncoderExplainer):
            The fitted concept explainer used for encoding activations.

        activation_granularity (ActivationGranularity):
            The granularity of the activations to use for the interpretation.
            See :method:`interpreto.model_wrapping.model_with_split_points.ModelWithSplitPoints.get_activations` for more details.

        aggregation_strategy (GranularityAggregationStrategy):
            The aggregation strategy to use for the activations.
            See :method:`interpreto.model_wrapping.model_with_split_points.ModelWithSplitPoints.get_activations` for more details.

        llm_interface (LLMInterface):
            The LLM interface to use for the interpretation.

        concept_encoding_batch_size (int):
            The batch size to use for the concept encoding.

        sampling_method (SAMPLING_METHOD):
            The method to use for sampling the inputs provided to the LLM.

        k_examples (int):
            The number of inputs to use for the interpretation.

        k_context (int):
            The number of context tokens to use around the concept tokens.
            In the prompt, in the examples, the k context tokens before and after the concept token are selected.
            It is recommended to set it to between 5 and 10 for TOKEN and WORD granularities.
            However, if the granularity is CLS_TOKEN or SAMPLE,
            or `use_unique_words=True` or `use_vocab=True`,
            it will be forced to 0.
            Indeed, in these cases the context do not make sense.

        use_vocab (bool):
            If True, the interpretation will be computed from the vocabulary of the model.

        use_unique_words (bool):
            If True, the interpretation will be computed from the unique words of the inputs.
            Incompatible with `use_vocab=True`.
            Default unique words selects all different word from the input.
            It can be tuned through the `unique_words_kwargs` argument.

        unique_words_kwargs (dict):
            The kwargs to pass to the `extract_unique_words` function.
            See [`extract_unique_words`][interpreto.concepts.interpretations.extract_unique_words] for more details.
            Possible arguments are `count_min_threshold`, `lemmatize`, `words_to_ignore`.

        k_quantile (int):
            The number of quantiles to use for sampling the inputs, if `sampling_method` is `QUANTILE`.

        system_prompt (str | None):
            The system prompt to use for the LLM. If None, a default prompt is used.

        concept_model_device (torch.device | str | None):
            The device to use for the concept model forward pass.
            If None, does not change the device.
    """

    def __init__(
        self,
        *,
        concept_explainer: ConceptEncoderExplainer,
        activation_granularity: ActivationGranularity = ActivationGranularity.TOKEN,
        aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        llm_interface: LLMInterface,
        concept_encoding_batch_size: int = 1024,
        sampling_method: SamplingMethod = SamplingMethod.TOP,
        k_examples: int = 30,
        k_context: int = 0,
        use_vocab: bool = False,
        use_unique_words: bool = False,
        unique_words_kwargs: dict = {},
        k_quantile: int = 5,
        system_prompt: str | None = None,
        concept_model_device: torch.device | str | None = None,
    ):
        super().__init__(
            concept_explainer=concept_explainer,
            activation_granularity=activation_granularity,
            aggregation_strategy=aggregation_strategy,
            concept_encoding_batch_size=concept_encoding_batch_size,
            use_vocab=use_vocab,
            use_unique_words=use_unique_words,
            unique_words_kwargs=unique_words_kwargs,
            concept_model_device=concept_model_device,
        )

        if k_context > 0 and (
            use_vocab
            or use_unique_words
            or self.activation_granularity
            in [
                ActivationGranularity.SAMPLE,
                ActivationGranularity.CLS_TOKEN,
            ]
        ):
            k_context = 0
            warnings.warn(
                "k_context is set to 0 because use_vocab or use_unique_words or activation_granularity is SAMPLE or CLS_TOKEN."
                "With these granularities, it is not possible to provide context around the granular inputs.",
                stacklevel=2,
            )

        self.llm_interface = llm_interface
        self.sampling_method = sampling_method
        self.k_examples = k_examples
        self.k_context = k_context
        self.k_quantile = k_quantile

        if system_prompt is None:
            if self.k_context > 0:
                self.system_prompt = SYSTEM_PROMPT_WITH_CONTEXT
            else:
                self.system_prompt = SYSTEM_PROMPT_WITHOUT_CONTEXT
        else:
            self.system_prompt = system_prompt

    def interpret(
        self,
        concepts_indices: int | list[int] | Literal["all"],
        inputs: list[str] | None = None,
        latent_activations: dict[str, torch.Tensor] | LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, str | None]:
        """
        Give the interpretation of the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and a short textual description.
        The granularity of input examples is determined by the `activation_granularity` class attribute.


        Args:
            concepts_indices (int | list[int] | Literal["all"]):
                The indices of the concepts to interpret. If "all", all concepts are interpreted.

            inputs (list[str] | None):
                The inputs to use for the interpretation.
                Necessary if not `use_vocab`,as examples are extracted from the inputs.

            latent_activations (dict[str, torch.Tensor] | Float[torch.Tensor, "nl d"] | None):
                The latent activations matching the inputs. If not provided,
                it is computed from the inputs.

            concepts_activations (Float[torch.Tensor, "nl cpt"] | None):
                The concepts activations matching the inputs. If not provided,
                it is computed from the inputs or latent activations.

        Returns:
            Mapping[int, str | None]: The textual labels of the concepts indices.
        """
        sure_concepts_indices: list[int]
        granular_inputs: list[str]
        sure_concepts_activations: Float[torch.Tensor, "nl cpt"]
        granular_sample_ids: list[int]
        (
            sure_concepts_indices,
            granular_inputs,
            sure_concepts_activations,
            granular_sample_ids,
        ) = self.get_granular_inputs_and_concept_activations(
            concepts_indices=concepts_indices,
            inputs=inputs,
            latent_activations=latent_activations,
            concepts_activations=concepts_activations,
        )

        labels: Mapping[int, str | None] = {}
        for concept_idx in sure_concepts_indices:
            example_idx = self.sampling_method.sample_examples(
                concept_activations=sure_concepts_activations[:, concept_idx],
                k_examples=self.k_examples,
                k_quantile=self.k_quantile,
            )
            examples = _format_examples(
                example_ids=example_idx,
                inputs=granular_inputs,
                concept_activations=sure_concepts_activations[:, concept_idx],
                sample_ids=granular_sample_ids,
                k_context=self.k_context,
            )
            example_prompt = _build_example_prompt(examples)
            prompt: list[tuple[Role, str]] = [
                (Role.SYSTEM, self.system_prompt),
                (Role.USER, example_prompt),
                (Role.ASSISTANT, ""),
            ]
            label = self.llm_interface.generate(prompt)
            labels[concept_idx] = label
        return labels


def _sample_top(
    concept_activations: Float[torch.Tensor, "ng"],
    k_examples: int,
) -> list[int]:
    """Select the k_examples that most activate the concept. The number of selected
    sample might be lower than k_examples if there are not enough non-zero activations.

    Args:
        concept_activations (Float[torch.Tensor, &quot;ng&quot;]): concept activation values for each granular text
        k_examples (int): number of examples to select, when possible

    Raises:
        ValueError: if concept_activations is not a 1D tensor.

    Returns:
        list[int]: the indexes of the examples to provide to the LLM for labeling the concept.
    """
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )
    non_zero_samples = torch.argwhere(concept_activations != 0).squeeze(-1)
    k_examples = min(k_examples, non_zero_samples.size(0))
    inputs_indices = non_zero_samples[torch.topk(concept_activations[non_zero_samples], k=k_examples).indices]

    return inputs_indices.tolist()  # type: ignore


def _sample_random(
    concept_activations: Float[torch.Tensor, "ng"],
    k_examples: int,
) -> list[int]:
    """Select k_examples randomly from the non-zero activations of the concept.

    Args:
        concept_activations (Float[torch.Tensor, &quot;ng&quot;]): concept activation values for each granular text
        k_examples (int): number of examples to select, when possible

    Raises:
        ValueError: if concept_activations is not a 1D tensor.

    Returns:
        list[int]: the indexes of the examples to provide to the LLM for labeling the concept.
    """
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )
    non_zero_samples = torch.argwhere(concept_activations != 0).squeeze(-1)
    inputs_indices = non_zero_samples[torch.randperm(len(non_zero_samples))][:k_examples]
    return inputs_indices.tolist()  # type: ignore


def _sample_quantile(
    concept_activations: Float[torch.Tensor, "ng"], k_examples: int, k_quantile: int = 5
) -> list[int]:
    """Select k_examples/k_quantile examples from each quantile of the concept activations.

    Args:
        concept_activations (Float[torch.Tensor, &quot;ng&quot;]): concept activation values for each granular text
        k_examples (int): number of examples to select, when possible. Should be higher than k_quantile.
        k_quantile (int, optional): number of quantiles. Defaults to 5.

    Raises:
        ValueError: if k_examples is lower than k_quantile.
        ValueError: if concept_activations is not a 1D tensor.

    Returns:
        list[int]: the indexes of the examples to provide to the LLM for labeling the concept.
    """
    if k_examples < k_quantile:
        raise ValueError(f"k_examples ({k_examples}) should be greater than k_quantile ({k_quantile}).")
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )

    non_zero_samples = torch.argwhere(concept_activations != 0).squeeze(-1)
    if non_zero_samples.size(0) < k_quantile:
        warnings.warn(
            "Not enough non-zero samples to compute quantiles. Using all non-zero samples.",
            stacklevel=2,
        )
        return non_zero_samples.tolist()  # type: ignore

    quantile_size = non_zero_samples.size(0) // k_quantile
    samples_per_quantile = k_examples // k_quantile

    sorted_indexes = torch.argsort(concept_activations, descending=True)[: non_zero_samples.size(0)]
    sample_indices: list[int] = []
    for i in range(k_quantile):
        if i == k_quantile - 1:
            # Last quantile (minimally activating samples) may have more samples
            quantile_samples = sorted_indexes[i * quantile_size :]
        else:
            quantile_samples = sorted_indexes[i * quantile_size : (i + 1) * quantile_size]
        selected_samples = quantile_samples[torch.randperm(len(quantile_samples))[:samples_per_quantile]]
        sample_indices.extend(selected_samples.tolist())  # type: ignore
    return sample_indices


def _format_examples(
    example_ids: list[int],
    inputs: list[str],
    concept_activations: Float[torch.Tensor, "ng"],
    sample_ids: list[int],  # (ng)
    k_context: int,
) -> list[Example]:
    """Format examples for the LLM input. If k_context > 0, it will add context around
    the selected text (for instance tokens before and after the selected token). Concept
    activations are normalized to a scale of 0 to 10, 10 being the maximum activation
    for the concept in the set of inputs.

    Args:
        example_ids (list[int]): selected example ids to provide to the LLM.
        inputs (list[str]): the list of all granular texts from the inputs, flattened.
        concept_activations (Float[torch.Tensor, &quot;ng&quot;]): the concept activations for each granular text.
        sample_ids (list[int]): the id of which sample each granular text belongs to.

    Raises:
        ValueError: if concept_activations is not a 1D tensor
        ValueError: if the length of inputs, sample_ids, and concept_activations do not match.

    Returns:
        list[Example]: list of Example objects, each containing the texts and normalized concept
        activations for the LLM input.
    """
    if len(concept_activations.size()) > 1:
        raise ValueError(
            f"concept_activations should be a 1D tensor, got tensor of shape {concept_activations.size()}"
        )
    if len(inputs) != len(sample_ids) or len(inputs) != concept_activations.size(0):
        raise ValueError(
            f"The number of inputs ({len(inputs)}), sample_ids ({len(sample_ids)}), and concept_activations ({concept_activations.size(0)}) should be the same."
        )

    max_act = concept_activations.max().item()
    examples: list[Example] = []
    for example_id in example_ids:
        if k_context > 0:
            left_idx = max(0, example_id - k_context)
            right_idx = example_id + k_context + 1
            # Select context from the same sample, it won't select tokens/sentences/texts from other samples
            sample_idx = sample_ids[example_id]
            example = Example(
                texts=[
                    text
                    for text, id in zip(
                        inputs[left_idx:right_idx],
                        sample_ids[left_idx:right_idx],
                        strict=False,
                    )
                    if id == sample_idx
                ],
                activations=[
                    round(act.item() / max_act * 10)
                    for act, id in zip(
                        concept_activations[left_idx:right_idx],
                        sample_ids[left_idx:right_idx],
                        strict=False,
                    )
                    if id == sample_idx
                ],
            )
        else:
            example = Example(
                texts=inputs[example_id],
                activations=round(concept_activations[example_id].item() / max_act * 10),
            )
        examples.append(example)
    return examples


def _build_example_prompt(examples: list[Example]) -> str:
    """The prompt containing the examples of text activating a concept, formatted for the LLM.

    For examples provided with a context, the format is:

    Example 1:  the dog <<eats>> the cat
    Activations: ("the", 0), (" dog", 2), (" eats", 10), (" the", 2), (" cat", 0)
    Example 2:  it was a <<delicious>> meal, but
    Activations: ("it", 0), (" was", 0), (" a", 1), (" delicious", 9), (" meal", 8), (",", 0), (" but", 0)

    For examples without context, the format is:

    Example 1:  The dog eats the cat (activation : 6)
    Example 2:  it was a delicious meal (activation : 4)

    Args:
        examples (list[Example]): List of Example objects containing the texts and activations.

    Raises:
        ValueError: if the types of texts and activations in the Example objects are not as expected.

    Returns:
        str: prompt containing the formatted examples for the LLM.
    """
    example_prompts: list[str] = []

    # Text without context and only unique words or tokens
    if all(isinstance(e.texts, str) for e in examples):
        return ", ".join([f'("{e.texts}", {e.activations})' for e in examples])

    for i, example in enumerate(examples):
        if isinstance(example.texts, str) and isinstance(example.activations, int):
            # Text without context
            example_prompts.append(f"Example {i + 1}: {example.texts} (activation: {example.activations})")
        elif isinstance(example.texts, list) and isinstance(example.activations, list):
            # Text with context
            max_text_pos = example.activations.index(max(example.activations))
            example_prompts.append(
                f"Example {i + 1}: "
                + "".join(example.texts[:max_text_pos])
                + f" <<{example.texts[max_text_pos]}>> "
                + "".join(example.texts[max_text_pos + 1 :])
            )
            example_prompts.append(
                "Activations: "
                + ", ".join(
                    [
                        f'("{text}", {activation})'
                        for text, activation in zip(example.texts, example.activations, strict=False)
                    ]
                )
            )
        else:
            raise ValueError(
                f"example.text is {type(example)} and example.activations is {example.texts}, expected str with int or list[str] with list[int]."
            )
    return "\n".join(example_prompts)


# From https://github.com/EleutherAI/delphi/blob/article_version/sae_auto_interp/explainers/default/prompts.py
SYSTEM_PROMPT_WITH_CONTEXT = """Your role is to label the concepts/patterns present in the different examples.

You will be given a list of text examples on which special tokens are selected and between delimiters like <<this>>.
How important each token is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

Hard rules:
- The label should summarize the concept linking the examples together. Give a single label describing all examples highlighted tokens.
- The label should be between 1 and 5 words long.
- The shorter the label the better. The best is a word.
- Do not make a sentence.
- The label should be the most precise possible. The goal is to be able to differentiate between concepts.
- The label should encompass most examples. But you can ignore the non-informative ones.
- Do not mention the marker tokens (<< >>) in your explanation. Nor refer to the importance.
- Only focus on the content and the label.
- Never ever give labels with more than 5 words, they would be cut out.

Some examples: 'blue', 'positive sentiment and enthusiasm', 'legal entities', 'medical places', 'hate', 'noun phrase', 'ion or iou sounds', 'questions' final words'...
"""

SYSTEM_PROMPT_WITHOUT_CONTEXT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 5 words. 5 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
"""
