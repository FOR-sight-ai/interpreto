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
Tests for `interpreto.concepts.interpretation.topk_inputs` methods
for `ConceptEncoderExplainer` and `ConceptAutoEncoderExplainer`
using the `NeuronsAsConcepts` concept explainer
"""

from __future__ import annotations

from itertools import product

import pytest
import torch

from interpreto import ModelWithSplitPoints
from interpreto.concepts import NeuronsAsConcepts
from interpreto.concepts.interpretations import TopKInputs
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_topk_inputs_from_activations(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test that the `_topk_inputs_from_concepts_activations` method works as expected
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    nb_concepts = 10
    n_tokens = 6
    k = 3
    n_samples = k * nb_concepts
    assert k <= n_tokens  # otherwise the test will break

    # generating data and fake activations
    # ['word_s0_t0', 'word_s0_t1',... 'word_s0_tn_tokens', 'word_s1_t0', 'word_s1_t1', ... 'word_s1_tn_tokens', ...]
    # sentences_list = [" ".join([f"word_s{i}_t{j}" for j in range(n_tokens)]) for i in range(n_samples)]
    words_list_list = [f"word_s{i}_t{j}" for i, j in product(range(n_samples), range(n_tokens))]
    fake_activations = torch.zeros(n_samples, n_tokens, nb_concepts)
    for i in range(nb_concepts):
        for j in range(k):
            fake_activations[i, (i + j) % n_tokens, i] = k - j
    fake_activations = fake_activations.view(-1, nb_concepts)

    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    splitted_encoder_ml.split_points = split
    concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split).concept_model

    # initializing the interpreter
    interpretation_method = TopKInputs(
        model_with_split_points=splitted_encoder_ml,
        split_point=split,
        concept_model=concept_model,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
    )

    # extracting concept interpretations
    all_top_k_tokens = interpretation_method._topk_inputs_from_concepts_activations(
        inputs=words_list_list,
        concepts_activations=fake_activations,
        concepts_indices=list(range(nb_concepts)),
    )

    assert isinstance(all_top_k_tokens, dict) and len(all_top_k_tokens) == nb_concepts

    for c in range(nb_concepts):
        # correct format dict[int, list[tuple[str, float]]]
        assert isinstance(all_top_k_tokens[c], dict) and len(all_top_k_tokens[c]) == k

        # check the exact words returned
        for i, (w, a) in enumerate(all_top_k_tokens[c].items()):
            assert w == f"word_s{c}_t{(c + i) % n_tokens}"
            assert a == k - i

    indices = [0, 2, 4]
    subset_top_k_tokens = interpretation_method._topk_inputs_from_concepts_activations(
        inputs=words_list_list,
        concepts_activations=fake_activations,
        concepts_indices=indices,
    )

    assert isinstance(subset_top_k_tokens, dict) and len(subset_top_k_tokens) == 3
    for c in indices:
        assert subset_top_k_tokens[c] == all_top_k_tokens[c]

    index = 0
    single_top_k_tokens = interpretation_method._topk_inputs_from_concepts_activations(
        inputs=words_list_list,
        concepts_activations=fake_activations,
        concepts_indices=[index],
    )
    assert isinstance(single_top_k_tokens, dict) and len(single_top_k_tokens) == 1
    assert single_top_k_tokens[index] == all_top_k_tokens[index]


@pytest.mark.parametrize(
    "activation_granularity",
    [
        ModelWithSplitPoints.activation_granularities.TOKEN,
        ModelWithSplitPoints.activation_granularities.WORD,
        ModelWithSplitPoints.activation_granularities.SENTENCE,
        ModelWithSplitPoints.activation_granularities.SAMPLE,
    ],
)
def test_topk_inputs_granularity(
    splitted_encoder_ml: ModelWithSplitPoints, huge_text: list[str], activation_granularity: ActivationGranularity
):
    """
    Test that the `interpret` method works as expected for different activation granularities
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    splitted_encoder_ml.split_points = split
    concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split).concept_model

    # getting the activations
    activations = splitted_encoder_ml.get_activations(huge_text, activation_granularity=activation_granularity)[split]

    # initializing the interpreter
    interpretation_method = TopKInputs(
        model_with_split_points=splitted_encoder_ml,
        split_point=split,
        concept_model=concept_model,
        activation_granularity=activation_granularity,
        k=2,
    )

    topk_inputs = interpretation_method.interpret(
        concepts_indices=[0, 5, 13],
        inputs=huge_text,
        latent_activations=activations,
    )

    flattened_huge_text = ". ".join(huge_text).replace("\n", " ")

    assert isinstance(topk_inputs, dict) and len(topk_inputs) == 3
    for c in [0, 5, 13]:
        assert c in topk_inputs
        assert len(topk_inputs[c]) == 2
        for key in topk_inputs[c].keys():
            new_key = key[2:] if key.startswith("##") else key
            assert new_key.lower().replace("\n", " ") in flattened_huge_text.lower()


def test_topk_inputs_concepts_selection(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test that the concept selection works as expected
    Fake activations are given to the `NeuronsAsConcepts` explainer
    """
    hidden_size = 32
    n_tokens = 6
    k = 3
    n_samples = k * 5
    assert k <= n_tokens  # otherwise the test will break

    # generating data (these have no importance)
    joined_tokens_list = [" ".join([f"{i}{j}" for j in range(n_tokens)]) for i in range(n_samples)]
    larger_input = " ".join(["test" for _ in range(2 * n_tokens)])
    joined_tokens_list.append(larger_input)

    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    splitted_encoder_ml.split_points = split
    concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split)

    # getting the activations
    activations = splitted_encoder_ml.get_activations(
        joined_tokens_list, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
    )

    # extracting concept interpretations
    all_top_k_tokens = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
        latent_activations=activations,
    )

    assert isinstance(all_top_k_tokens, dict) and len(all_top_k_tokens) == hidden_size

    for c in range(hidden_size):
        # correct format dict[int, list[tuple[str, float]]]
        assert isinstance(all_top_k_tokens[c], dict) and len(all_top_k_tokens[c]) == k

    indices = [0, 2, 4]
    subset_top_k_tokens = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
        concepts_indices=indices,
        inputs=joined_tokens_list,
        latent_activations=activations,
    )

    assert isinstance(subset_top_k_tokens, dict) and len(subset_top_k_tokens) == 3
    for c in indices:
        assert subset_top_k_tokens[c] == all_top_k_tokens[c]

    index = 0
    single_top_k_tokens = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
        concepts_indices=index,
        inputs=joined_tokens_list,
        latent_activations=activations,
    )
    assert isinstance(single_top_k_tokens, dict) and len(single_top_k_tokens) == 1
    assert single_top_k_tokens[index] == all_top_k_tokens[index]


def test_topk_inputs_sources(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test that different sources give the same results
    """
    hidden_size = 32
    n_tokens = 6
    k = 3
    n_samples = k * 5
    assert k <= n_tokens  # otherwise the test will break

    # generating data with duplicates and different lengths
    joined_tokens_list = [" ".join([f"{i + j}" for j in range(n_tokens)]) for i in range(n_samples)]
    larger_input = " ".join(["test" for _ in range(2 * n_tokens)])
    joined_tokens_list.append(larger_input)

    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    splitted_encoder_ml.split_points = split
    concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split)

    # getting the activations
    activations = splitted_encoder_ml.get_activations(
        joined_tokens_list, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
    )

    # getting the top k tokens
    top_k_inputs = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
    )
    top_k_latent = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
        latent_activations=activations,
    )
    top_k_concept = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
        concepts_indices="all",
        inputs=joined_tokens_list,
        concepts_activations=activations[split],
    )

    assert list(range(hidden_size)) == list(top_k_latent.keys())
    assert list(range(hidden_size)) == list(top_k_concept.keys())
    assert list(range(hidden_size)) == list(top_k_inputs.keys())

    for top_latent, top_concept, top_input in zip(
        top_k_latent.values(), top_k_concept.values(), top_k_inputs.values(), strict=True
    ):
        assert len(top_latent) == k
        assert top_latent == top_concept == top_input


def test_topk_inputs_from_vocabulary(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test that interpretations can be obtained from the vocabulary
    """
    k = 2
    hidden_size = 32
    nb_concepts = 3

    # initializing the explainer
    split = "bert.encoder.layer.1.output"
    splitted_encoder_ml.split_points = split
    concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml, split_point=split)

    top_k_vocabulary = concept_explainer.interpret(
        interpretation_method=TopKInputs,
        activation_granularity=TopKInputs.activation_granularities.TOKEN,
        k=k,
        concepts_indices=torch.randperm(hidden_size)[:nb_concepts].tolist(),
        use_vocab=True,
    )

    assert len(top_k_vocabulary) == nb_concepts

    vocabulary = list(splitted_encoder_ml.tokenizer.get_vocab().keys())
    for topk in top_k_vocabulary.values():
        assert len(topk) == k
        for token in topk:
            assert token in vocabulary


def test_topk_inputs_error_raising(
    splitted_encoder_ml: ModelWithSplitPoints, activations_dict: dict[str, torch.Tensor]
):
    """
    Test that the `TopKInputs` class raises an error when needed
    """
    # getting the activations
    activations = next(iter(activations_dict.values()))  # dictionary with only one element

    concept_model = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml).concept_model

    some_texts_for_interpretation = ["a sentence", "another sentence", "yet another sentence"]

    # When use_vocab=False and inputs is not provided
    with pytest.raises(ValueError):
        method = TopKInputs(
            model_with_split_points=splitted_encoder_ml,
            concept_model=concept_model,
            activation_granularity=TopKInputs.activation_granularities.TOKEN,
            use_vocab=False,
        )
        method.interpret(
            concepts_indices=0,
        )

    # wrong indices
    for wrong_indices in [-1, activations.shape[1], [-1, 0, 1], ["?"], (0, 1, 2), "str other than 'all'"]:
        with pytest.raises(ValueError):
            method = TopKInputs(
                model_with_split_points=splitted_encoder_ml,
                concept_model=concept_model,
                activation_granularity=TopKInputs.activation_granularities.TOKEN,
            )
            method.interpret(
                concepts_indices=wrong_indices,
                inputs=some_texts_for_interpretation,
                concepts_activations=activations,
            )


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM

    splitted_encoder_ml = ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_points=["bert.encoder.layer.1.output"],
        automodel=AutoModelForMaskedLM,  # type: ignore
    )
    sentences = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. sed do eiusmod tempor incididunt\n\nut labore et dolore magna aliqua.",
        "Interpreto is magical",
        "Testing interpreto",
    ]
    activation_dict = splitted_encoder_ml.get_activations(
        sentences, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
    )

    test_topk_inputs_from_activations(splitted_encoder_ml)
    test_topk_inputs_from_vocabulary(splitted_encoder_ml)
    test_topk_inputs_concepts_selection(splitted_encoder_ml)
    test_topk_inputs_sources(splitted_encoder_ml)
    test_topk_inputs_error_raising(splitted_encoder_ml, activation_dict)  # type: ignore
    test_topk_inputs_granularity(
        splitted_encoder_ml, sentences * 10, ModelWithSplitPoints.activation_granularities.SAMPLE
    )
