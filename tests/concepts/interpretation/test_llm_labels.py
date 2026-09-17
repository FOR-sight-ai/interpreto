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

from interpreto import SplitterForClassification, TextTokensSplitter
from interpreto.commons.llm_interface import LLMInterface, Role
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def classification_splitter():
    return SplitterForClassification("hf-internal-testing/tiny-random-bert", device_map=DEVICE)


@pytest.fixture(scope="module")
def token_splitter():
    return TextTokensSplitter(
        "hf-internal-testing/tiny-random-gpt2",
        split_point=1,
        task="text-generation",
        batch_size=8,
        device_map=DEVICE,
    )


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
    assert prompt == '("This is a test with sentences", 10), ("This is another sentence", 4)'

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


class LLMInterfaceMock(LLMInterface):
    def generate(self, prompt: list[tuple[Role, str]]) -> str | None:
        return "mock answer"


def test_llm_labels_concept_selection(classification_splitter: SplitterForClassification):
    """
    Test that the `interpret` method works as expected
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    hidden_size = 32
    concept_explainer = NeuronsAsConcepts(splitter=classification_splitter)
    interpretation_method = LLMLabels(
        concept_explainer=concept_explainer,
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


def test_llm_labels_token_and_pooled_modes(
    classification_splitter: SplitterForClassification,
    token_splitter: TextTokensSplitter,
    sentences: list[str],
):
    """Token and pooled representations both produce labels."""
    concept_explainer = NeuronsAsConcepts(token_splitter)
    token_method = LLMLabels(
        concept_explainer=concept_explainer,
        llm_interface=LLMInterfaceMock(),
        k_examples=2,
        k_context=1,
    )
    assert token_method.k_context == 1
    assert len(token_method.interpret(concepts_indices=[0, 5, 13], inputs=sentences)) == 3

    with pytest.warns(UserWarning, match="k_context is set to 0"):
        pooled_method = LLMLabels(
            concept_explainer=concept_explainer,
            llm_interface=LLMInterfaceMock(),
            k_examples=2,
            k_context=1,
            token_pooling="mean",
        )
    assert pooled_method.k_context == 0
    assert len(pooled_method.interpret(concepts_indices=[0, 5, 13], inputs=sentences)) == 3

    with pytest.warns(UserWarning, match="k_context is set to 0"):
        classification_method = LLMLabels(
            concept_explainer=NeuronsAsConcepts(classification_splitter),
            llm_interface=LLMInterfaceMock(),
            k_context=1,
        )
    assert classification_method.k_context == 0


def test_llm_labels_sources(token_splitter: TextTokensSplitter, sentences: list[str]):
    """
    Test the different sources
    """
    concept_explainer = NeuronsAsConcepts(splitter=token_splitter)

    interpretation_method = LLMLabels(
        concept_explainer=concept_explainer,
        llm_interface=LLMInterfaceMock(),
        sampling_method=SamplingMethod.TOP,
        k_examples=2,
    )

    # getting the activations
    activations, _ = token_splitter.get_activations(sentences)

    # From input
    labels = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=sentences,
    )
    assert isinstance(labels, dict)
    assert len(labels) == 3
    assert 5 in labels
    assert 0 in labels
    assert 13 in labels
    assert all(isinstance(label, str) for label in labels.values())

    labels = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=sentences,
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
        inputs=sentences,
        concepts_activations=activations,
    )
    assert isinstance(labels, dict)
    assert len(labels) == 3
    assert 5 in labels
    assert 0 in labels
    assert 13 in labels
    assert all(isinstance(label, str) for label in labels.values())


def test_llm_labels_from_vocabulary(classification_splitter: SplitterForClassification):
    """
    Test that interpretations can be obtained from the vocabulary
    """
    hidden_size = 32
    nb_concepts = 3

    concept_explainer = NeuronsAsConcepts(splitter=classification_splitter)

    interpretation_method = LLMLabels(
        concept_explainer=concept_explainer,
        llm_interface=LLMInterfaceMock(),
        sampling_method=SamplingMethod.TOP,
        k_examples=2,
        use_vocab=True,
    )
    label = interpretation_method.interpret(
        concepts_indices=torch.randperm(hidden_size)[:nb_concepts].tolist(),
    )
    assert len(label) == nb_concepts


def test_llm_labels_call_from_concept_module(token_splitter: TextTokensSplitter, sentences: list[str]):
    """
    Test that LLMLabels can be called from the concept module
    """
    hidden_size = 32
    nb_concepts = 3

    concept_explainer = NeuronsAsConcepts(splitter=token_splitter)

    label = LLMLabels(
        concept_explainer=concept_explainer,
        use_vocab=False,
        sampling_method=SamplingMethod.TOP,
        k_context=0,
        llm_interface=LLMInterfaceMock(),
    ).interpret(
        concepts_indices=torch.randperm(hidden_size)[:nb_concepts].tolist(),
        inputs=sentences[:2],
    )

    assert len(label) == nb_concepts
    # TODO : verify that some methods are called


def test_llm_labels_error_raising(classification_splitter: SplitterForClassification):
    """
    Test that the `TopKInputs` class raises an error when needed
    """

    concept_explainer = NeuronsAsConcepts(
        splitter=classification_splitter,
    )
    method = LLMLabels(
        concept_explainer=concept_explainer,
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
