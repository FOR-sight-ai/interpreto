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
Tests for `interpreto.concepts.interpretation.llm_labels` methods
for `ConceptEncoderExplainer` and `ConceptAutoEncoderExplainer`
using the `NeuronsAsConcepts` concept explainer
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForMaskedLM

from interpreto import ModelWithSplitPoints
from interpreto.concepts import NeuronsAsConcepts
from interpreto.concepts.interpretations import LLMLabels
from interpreto.concepts.interpretations.llm_labels import (
    Example,
    SamplingMethod,
    _build_example_prompt,
    _format_examples,
    _sample_quantile,
    _sample_random,
    _sample_top,
)
from interpreto.model_wrapping.llm_interface import LLMInterface, Role
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def concept_activations() -> torch.Tensor:
    return torch.tensor([0.1, 0.5, 0.0, 8.5, 7.2, 0.0, 0.0, 1.4, 0.1, 3.8])


def test_sample_top(concept_activations: torch.Tensor):
    # Test output type and length
    selected_idx = _sample_top(
        concept_activations=concept_activations,
        k_examples=2,
    )
    assert isinstance(selected_idx, list)
    assert all(isinstance(id, int) for id in selected_idx)
    assert len(selected_idx) == 2
    assert 3 in selected_idx
    assert 4 in selected_idx

    # Test that 0 values are not kept
    selected_idx = _sample_top(
        concept_activations=concept_activations,
        k_examples=9,
    )
    assert isinstance(selected_idx, list)
    assert 2 not in selected_idx
    assert 5 not in selected_idx
    assert 6 not in selected_idx

    # Test that there is no repetition
    selected_idx = _sample_top(
        concept_activations=concept_activations,
        k_examples=15,
    )
    assert isinstance(selected_idx, list)
    assert len(set(selected_idx)) == len(selected_idx)

    # Test raising error
    with pytest.raises(ValueError):
        _sample_top(
            concept_activations=torch.rand(10, 5),
            k_examples=4,
        )


def test_sample_quantile(concept_activations: torch.Tensor):
    # Test output type and length
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=2,
        k_quantile=2,
    )
    assert isinstance(selected_idx, list)
    assert all(isinstance(id, int) for id in selected_idx)
    assert len(selected_idx) == 2

    # Test that 0 values are not kept
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=9,
        k_quantile=2,
    )
    assert isinstance(selected_idx, list)
    assert len(selected_idx) == 7  # min(k_examples//k_quantile * k_examples, num non zero samples)
    assert 2 not in selected_idx
    assert 5 not in selected_idx
    assert 6 not in selected_idx

    # Test that there is no repetition
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=15,
        k_quantile=2,
    )
    assert isinstance(selected_idx, list)
    assert len(set(selected_idx)) == len(selected_idx)

    # Test raising error
    with pytest.raises(ValueError):
        _sample_quantile(
            concept_activations=torch.rand(10, 5),
            k_examples=4,
            k_quantile=2,
        )
    with pytest.raises(ValueError):
        _sample_quantile(
            concept_activations=torch.rand(10, 5),
            k_examples=2,
            k_quantile=4,
        )

    # Test quantile
    selected_idx = _sample_quantile(
        concept_activations=concept_activations,
        k_examples=3,
        k_quantile=3,
    )
    assert len(selected_idx) == 3
    assert 3 in selected_idx or 4 in selected_idx  # 1st quantile
    assert 9 in selected_idx or 7 in selected_idx  # 2nd quantile
    assert 1 in selected_idx or 0 in selected_idx or 8 in selected_idx  # 3rd quantile


def test_sample_random(concept_activations: torch.Tensor):
    # Test output type and length
    selected_idx = _sample_random(
        concept_activations=concept_activations,
        k_examples=2,
    )
    assert isinstance(selected_idx, list)
    assert all(isinstance(id, int) for id in selected_idx)
    assert len(selected_idx) == 2

    # Test that 0 values are not kept
    selected_idx = _sample_random(
        concept_activations=concept_activations,
        k_examples=9,
    )
    assert isinstance(selected_idx, list)
    assert 2 not in selected_idx
    assert 5 not in selected_idx
    assert 6 not in selected_idx

    # Test that there is no repetition
    selected_idx = _sample_random(
        concept_activations=concept_activations,
        k_examples=15,
    )
    assert isinstance(selected_idx, list)
    assert len(set(selected_idx)) == len(selected_idx)

    # Test raising error
    with pytest.raises(ValueError):
        _sample_random(
            concept_activations=torch.rand(10, 5),
            k_examples=4,
        )


def test_format_examples():
    examples = _format_examples(
        example_ids=[3, 2, 6],
        inputs=["This", " is", " a", " test", "Another", " sentence", " with", " more", " words"],
        concept_activations=torch.tensor([0.1, 0.0, 8.5, 5.3, 7.2, 0.0, 4.2, 0.0, 0.1]),
        sample_ids=[0, 0, 0, 0, 1, 1, 1, 1, 1],
        k_context=1,  # 1 on the right and 1 on the left
    )
    assert isinstance(examples, list)
    assert len(examples) == 3
    assert examples[0].texts == [" a", " test"]
    assert examples[0].activations[0] == 10
    assert examples[0].activations[1] == 6
    assert examples[1].texts == [" is", " a", " test"]
    assert examples[2].texts == [" sentence", " with", " more"]
    # Test sentences without context
    examples = _format_examples(
        example_ids=[0, 2],
        inputs=[
            "This is a test with sentences",
            "Another sentence with more words",
            "And another one",
        ],
        concept_activations=torch.tensor([8.5, 0.0, 2.1]),
        sample_ids=[0, 1, 2],
        k_context=0,  # no context
    )
    assert isinstance(examples, list)
    assert len(examples) == 2
    assert examples[0].texts == "This is a test with sentences"
    assert isinstance(examples[0].activations, int)
    assert examples[0].activations == 10
    assert examples[1].texts == "And another one"
    assert isinstance(examples[1].activations, int)


def test_build_example_prompt():
    # Test with examples without context
    examples = [
        Example(
            texts="This is a test with sentences",
            activations=10,
        ),
        Example(
            texts="This is another sentence",
            activations=4,
        ),
    ]
    prompt = _build_example_prompt(examples)
    assert isinstance(prompt, str)
    assert (
        prompt
        == "Example 1: This is a test with sentences (activation: 10)\nExample 2: This is another sentence (activation: 4)"
    )

    # Test with examples with context
    examples = [
        Example(
            texts=["This", " is", " a", " test"],
            activations=[2, 6, 10, 0],
        ),
        Example(
            texts=["Another", " sentence", " with", " more", " words"],
            activations=[0, 2, 4, 0, 0],
        ),
    ]
    prompt = _build_example_prompt(examples)
    assert isinstance(prompt, str)
    assert (
        prompt
        == 'Example 1: This is << a>>  test\nActivations: ("This", 2), (" is", 6), (" a", 10), (" test", 0)\nExample 2: Another sentence << with>>  more words\nActivations: ("Another", 0), (" sentence", 2), (" with", 4), (" more", 0), (" words", 0)'
    )

    # Test raising error
    with pytest.raises(ValueError):
        _build_example_prompt(
            [
                Example(
                    texts=["This", " is", " a", " test"],
                    activations=2,
                ),
            ]
        )
    with pytest.raises(ValueError):
        _build_example_prompt(
            [
                Example(
                    texts="This is a test",
                    activations=[2, 3],
                ),
            ]
        )


@pytest.fixture
def splitted_encoder() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_points=["bert.encoder.layer.1.output"],
        automodel=AutoModelForMaskedLM,  # type: ignore
    )


class LLMInterfaceMock(LLMInterface):
    def generate(self, prompt: list[tuple[Role, str]]) -> str | None:
        return "mock answer"


def test_llm_labels_concept_selection(splitted_encoder: ModelWithSplitPoints):
    """
    Test that the `interpret` method works as expected
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    hidden_size = 32
    split = "bert.encoder.layer.1.output"
    concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder, split_point=split).concept_model
    interpretation_method = LLMLabels(
        model_with_split_points=splitted_encoder,
        split_point=split,
        concept_model=concept_model,
        activation_granularity=ActivationGranularity.TOKEN,
        llm_interface=LLMInterfaceMock(),
        sampling_method=SamplingMethod.TOP,
        k_examples=2,
    )

    labels = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=["a", "b", "c"],
        concepts_activations=torch.rand(3, hidden_size).to(DEVICE),  # Fake activations
    )
    assert isinstance(labels, dict)
    assert len(labels) == 3
    assert 5 in labels
    assert 0 in labels
    assert 13 in labels
    assert all(isinstance(label, str) for label in labels.values())

    labels = interpretation_method.interpret(
        concepts_indices=0,
        inputs=["a", "b", "c"],  # Fake tokens
        concepts_activations=torch.rand(3, hidden_size).to(DEVICE),  # Fake activations
    )
    assert isinstance(labels, dict)
    assert len(labels) == 1
    assert 0 in labels
    assert all(isinstance(label, str) for label in labels.values())


@pytest.mark.parametrize(
    "activation_granularity",
    [
        ModelWithSplitPoints.activation_granularities.TOKEN,
        ModelWithSplitPoints.activation_granularities.WORD,
        ModelWithSplitPoints.activation_granularities.SENTENCE,
        ModelWithSplitPoints.activation_granularities.SAMPLE,
    ],
)
def test_llm_labels_granularity(splitted_encoder: ModelWithSplitPoints, activation_granularity: ActivationGranularity):
    split = "bert.encoder.layer.1.output"
    concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder, split_point=split).concept_model
    interpretation_method = LLMLabels(
        model_with_split_points=splitted_encoder,
        split_point=split,
        concept_model=concept_model,
        activation_granularity=activation_granularity,
        llm_interface=LLMInterfaceMock(),
        sampling_method=SamplingMethod.TOP,
        k_examples=2,
    )

    texts = [
        "This is a test. This is another sentence. And yet another one.",
        "This is a second test. This is another sentence. And yet another one.",
    ]
    activations = splitted_encoder.get_activations(texts, activation_granularity=activation_granularity)[split]

    # just verify that everything works, not the content of the labels
    labels = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=texts,
        latent_activations=activations,
    )
    assert isinstance(labels, dict)
    assert len(labels) == 3
    assert 5 in labels
    assert 0 in labels
    assert 13 in labels
    assert all(isinstance(label, str) for label in labels.values())


def test_llm_labels_sources(splitted_encoder: ModelWithSplitPoints):
    """
    Test the different sources
    """
    # generating data with duplicates and different lengths
    n_tokens = 6
    k = 3
    n_samples = k * 5
    assert k <= n_tokens  # otherwise the test will break
    joined_tokens_list = [" ".join([f"{i + j}" for j in range(n_tokens)]) for i in range(n_samples)]
    larger_input = " ".join(["test" for _ in range(2 * n_tokens)])
    joined_tokens_list.append(larger_input)

    split = "bert.encoder.layer.1.output"
    concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder, split_point=split)

    interpretation_method = LLMLabels(
        model_with_split_points=splitted_encoder,
        split_point=split,
        concept_model=concept_explainer.concept_model,
        activation_granularity=ActivationGranularity.TOKEN,
        llm_interface=LLMInterfaceMock(),
        sampling_method=SamplingMethod.TOP,
        k_examples=2,
    )

    # getting the activations
    activations = splitted_encoder.get_activations(
        inputs=joined_tokens_list, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
    )[split]

    # From input
    labels = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=joined_tokens_list,
    )
    assert isinstance(labels, dict)
    assert len(labels) == 3
    assert 5 in labels
    assert 0 in labels
    assert 13 in labels
    assert all(isinstance(label, str) for label in labels.values())

    labels = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=joined_tokens_list,
        latent_activations=activations,
    )
    assert isinstance(labels, dict)
    assert len(labels) == 3
    assert 5 in labels
    assert 0 in labels
    assert 13 in labels
    assert all(isinstance(label, str) for label in labels.values())

    labels = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=joined_tokens_list,
        concepts_activations=activations,
    )
    assert isinstance(labels, dict)
    assert len(labels) == 3
    assert 5 in labels
    assert 0 in labels
    assert 13 in labels
    assert all(isinstance(label, str) for label in labels.values())


def test_llm_labels_from_vocabulary(splitted_encoder: ModelWithSplitPoints):
    """
    Test that interpretations can be obtained from the vocabulary
    """
    hidden_size = 32
    nb_concepts = 3

    split = "bert.encoder.layer.1.output"
    concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder, split_point=split)

    interpretation_method = LLMLabels(
        model_with_split_points=splitted_encoder,
        split_point=split,
        concept_model=concept_explainer.concept_model,
        activation_granularity=ActivationGranularity.TOKEN,
        llm_interface=LLMInterfaceMock(),
        sampling_method=SamplingMethod.TOP,
        k_examples=2,
        use_vocab=True,
    )
    label = interpretation_method.interpret(
        concepts_indices=torch.randperm(hidden_size)[:nb_concepts].tolist(),
    )
    assert len(label) == nb_concepts


def test_llm_labels_call_from_concept_module(splitted_encoder: ModelWithSplitPoints):
    """
    Test that LLMLabels can be called from the concept module
    """
    hidden_size = 32
    nb_concepts = 3

    split = "bert.encoder.layer.1.output"
    concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder, split_point=split)

    label = concept_explainer.interpret(
        interpretation_method=LLMLabels,
        activation_granularity=ActivationGranularity.TOKEN,
        concepts_indices=torch.randperm(hidden_size)[:nb_concepts].tolist(),
        use_vocab=False,
        inputs=["This is a sentence", "This is another sentence"],
        sampling_method=SamplingMethod.TOP,
        k_context=0,
        llm_interface=LLMInterfaceMock(),
    )

    assert len(label) == nb_concepts
    # TODO : verify that some methods are called


def test_llm_labels_error_raising(splitted_encoder: ModelWithSplitPoints):
    """
    Test that the `TopKInputs` class raises an error when needed
    """

    concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder).concept_model

    method = LLMLabels(
        model_with_split_points=splitted_encoder,
        split_point="bert.encoder.layer.1.output",
        concept_model=concept_model,
        activation_granularity=ActivationGranularity.TOKEN,
        use_vocab=False,
        llm_interface=LLMInterfaceMock(),
        sampling_method=SamplingMethod.TOP,
    )

    # When use_vocab=False and inputs is not provided
    with pytest.raises(ValueError):
        method.interpret(concepts_indices=0)

    # wrong indices
    for wrong_indices in [-1, [-1, 0, 1], ["?"], (0, 1, 2), "str"]:
        with pytest.raises(ValueError):
            method.interpret(
                concepts_indices=wrong_indices, inputs=["a sentence", "another sentence", "yet another sentence"]
            )
