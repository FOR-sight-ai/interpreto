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

"""Tests for splitter-derived concept interpretation examples."""

import pytest
import torch

from interpreto import SplitterForClassification
from interpreto.concepts import NeuronsAsConcepts, TopKInputs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def classification_splitter():
    return SplitterForClassification("hf-internal-testing/tiny-random-bert", device_map=DEVICE)


def test_examples_follow_classification_splitter(classification_splitter, sentences: list[str]):
    """Classification representations map to one example per input."""
    method = TopKInputs(concept_explainer=NeuronsAsConcepts(classification_splitter))

    granular_inputs, sample_ids = method.get_granular_inputs(sentences)

    assert granular_inputs == sentences
    assert sample_ids == list(range(len(sentences)))


def test_examples_follow_token_splitter(text_tokens_splitter, sentences: list[str]):
    """Token examples use the same mask and row order as activation extraction."""
    method = TopKInputs(concept_explainer=NeuronsAsConcepts(text_tokens_splitter))

    granular_inputs, sample_ids = method.get_granular_inputs(sentences)
    per_sample, _ = text_tokens_splitter.get_activations(sentences, flatten_activations=False)

    assert len(granular_inputs) == sum(activations.shape[0] for activations in per_sample)
    assert sample_ids == [
        sample_id for sample_id, activations in enumerate(per_sample) for _ in range(activations.shape[0])
    ]


def test_pooling_maps_generation_to_inputs(text_tokens_splitter, sentences: list[str]):
    """Pooled token representations map back to whole input examples."""
    method = TopKInputs(
        concept_explainer=NeuronsAsConcepts(text_tokens_splitter),
        token_pooling="mean",
    )

    granular_inputs, sample_ids = method.get_granular_inputs(sentences)
    concept_activations = method.concepts_activations_from_source(inputs=sentences)

    assert granular_inputs == sentences
    assert sample_ids == list(range(len(sentences)))
    assert concept_activations.shape[0] == len(sentences)


def test_precomputed_activations_must_match_interpretation_mode(classification_splitter, sentences: list[str]):
    """Precomputed rows cannot silently use a different representation mode."""
    method = TopKInputs(concept_explainer=NeuronsAsConcepts(classification_splitter))
    mismatched = torch.rand(len(sentences) + 1, classification_splitter.config.hidden_size)

    with pytest.raises(ValueError, match="does not match"):
        method.get_granular_inputs_and_concept_activations(
            concepts_indices=[0],
            inputs=sentences,
            concepts_activations=mismatched,
        )
