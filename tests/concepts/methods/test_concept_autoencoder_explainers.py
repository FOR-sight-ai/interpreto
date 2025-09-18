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
Tests for interpreto.concepts.methods.concept_bottleneck methods
"""

from __future__ import annotations

import pytest
import torch

from interpreto import Granularity
from interpreto.concepts import (
    BatchTopKSAEConcepts,
    Cockatiel,
    ConceptAutoEncoderExplainer,
    # ConvexNMFConcepts,
    DictionaryLearningConcepts,
    ICAConcepts,
    JumpReLUSAEConcepts,
    KMeansConcepts,
    NeuronsAsConcepts,
    NMFConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SparsePCAConcepts,
    SVDConcepts,
    TopKSAEConcepts,
    VanillaSAEConcepts,
)
from interpreto.concepts.methods.overcomplete import DictionaryLearningExplainer, SAEExplainer
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity, ModelWithSplitPoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_CONCEPT_METHODS = [
    BatchTopKSAEConcepts,
    Cockatiel,
    # ConvexNMFConcepts,
    DictionaryLearningConcepts,
    ICAConcepts,
    JumpReLUSAEConcepts,
    KMeansConcepts,
    NeuronsAsConcepts,
    NMFConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SparsePCAConcepts,
    SVDConcepts,
    TopKSAEConcepts,
    VanillaSAEConcepts,
]


def test_cbe_fit_failure_cases(multi_split_model: ModelWithSplitPoints):
    """Test failure cases in matching the CBE with a ModelWithSplitPoints"""

    # Raise when no split is provided and the model has more than one split
    with pytest.raises(ValueError, match="If the model has more than one split point"):
        _ = Cockatiel(multi_split_model, nb_concepts=3)


@pytest.mark.parametrize("method_class", ALL_CONCEPT_METHODS)
def test_overcomplete_cbe(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_dict: dict[str, torch.Tensor],
    method_class: type[ConceptAutoEncoderExplainer],
):
    """Test SAEExplainer and DictionaryLearningExplainer"""
    split, activations = next(iter(activations_dict.items()))  # dictionary with only one element
    n = activations.shape[0]
    d = activations.shape[1]
    nb_concepts = 3

    # iterate over all methods from the namedtuple listing them
    if method_class == NeuronsAsConcepts:
        cbe = method_class(splitted_encoder_ml)  # type: ignore
    elif method_class in [Cockatiel, NMFConcepts]:
        cbe = method_class(
            splitted_encoder_ml,
            nb_concepts=nb_concepts,  # type: ignore
            device=DEVICE,  # type: ignore
            force_relu=True,  # type: ignore
        )  # type: ignore
        cbe.fit(activations)
    elif issubclass(method_class, SAEExplainer):
        cbe = method_class(splitted_encoder_ml, nb_concepts=nb_concepts, device=DEVICE)
        cbe.fit(activations, nb_epochs=1, batch_size=1, device=DEVICE)
    elif issubclass(method_class, DictionaryLearningExplainer):
        cbe = method_class(
            splitted_encoder_ml,
            nb_concepts=nb_concepts,
            device=DEVICE,
        )
        cbe.fit(activations)
    else:
        raise ValueError(f"Unknown method_class {method_class}")

    assert hasattr(cbe, "concept_model"), f"Explainer {method_class.__name__} missing attribute 'concept_model'"
    assert hasattr(cbe.concept_model, "nb_concepts"), f"Concept model in {method_class.__name__} missing 'nb_concepts'"
    assert hasattr(cbe, "model_with_split_points"), (
        f"Explainer {method_class.__name__} missing 'model_with_split_points'"
    )
    assert cbe.concept_model.fitted, f"Concept model in {method_class.__name__} not fitted"
    assert cbe.is_fitted, f"Explainer {method_class.__name__} reports not fitted"
    assert cbe.split_point == split, f"Split point mismatch: expected {split}, got {cbe.split_point}"
    assert hasattr(cbe, "has_differentiable_concept_encoder"), (
        f"Explainer {method_class.__name__} missing 'has_differentiable_concept_encoder'"
    )
    assert hasattr(cbe, "has_differentiable_concept_decoder"), (
        f"Explainer {method_class.__name__} missing 'has_differentiable_concept_decoder'"
    )

    concepts = cbe.encode_activations(activations)
    assert concepts is not None, f"{method_class.__name__}.encode_activations returned None"
    reconstructed_activations = cbe.decode_concepts(concepts)
    assert reconstructed_activations is not None, f"{method_class.__name__}.decode_concepts returned None"
    assert reconstructed_activations.shape == (n, d), (
        f"Explainer {method_class.__name__} encode-decode reconstructed activations shape mismatch: ",
        f"got {tuple(reconstructed_activations.shape)}, expected {(n, d)}",
    )

    dictionary = cbe.get_dictionary()
    assert dictionary is not None, f"{method_class.__name__}.get_dictionary returned None"
    if method_class == NeuronsAsConcepts:
        assert cbe.concept_model.nb_concepts == d, (
            f"nb_concepts mismatch for NeuronsAsConcepts: got {cbe.concept_model.nb_concepts}, expected {d}"
        )
        assert concepts.shape == (n, d), (
            f"Concepts shape mismatch for NeuronsAsConcepts: got {tuple(concepts.shape)}, expected {(n, d)}"
        )
        assert torch.allclose(dictionary, torch.eye(d)), "Dictionary not identity for NeuronsAsConcepts"
    else:
        assert cbe.concept_model.nb_concepts == nb_concepts, (
            f"{method_class.__name__}.nb_concepts mismatch: got {cbe.concept_model.nb_concepts}, expected {nb_concepts}"
        )
        assert concepts.shape == (n, nb_concepts), (
            f"{method_class.__name__}: Concepts shape mismatch: got {tuple(concepts.shape)}, expected {(n, nb_concepts)}"
        )
        assert dictionary.shape == (nb_concepts, d), (
            f"{method_class.__name__}: Dictionary shape mismatch: got {tuple(dictionary.shape)}, expected {(nb_concepts, d)}"
        )


@pytest.mark.parametrize("method_class", ALL_CONCEPT_METHODS)
@pytest.mark.parametrize(
    "granularity",
    [
        ModelWithSplitPoints.activation_granularities.TOKEN,
        ModelWithSplitPoints.activation_granularities.WORD,
        ModelWithSplitPoints.activation_granularities.SENTENCE,
    ],
)
def test_concept_output_gradient(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_dict: dict[str, torch.Tensor],
    sentences: list[str],
    method_class: type[ConceptAutoEncoderExplainer],
    granularity: ActivationGranularity,
):
    split, activations = next(iter(activations_dict.items()))
    nb_concepts = 3

    if method_class == NeuronsAsConcepts:
        cbe = method_class(splitted_encoder_ml)  # type: ignore
        concepts_dim = activations.shape[1]
    elif method_class in [Cockatiel, NMFConcepts]:
        cbe = method_class(
            splitted_encoder_ml,
            nb_concepts=nb_concepts,  # type: ignore
            device=DEVICE,  # type: ignore
            force_relu=True,  # type: ignore
        )  # type: ignore
        cbe.fit(activations)
        concepts_dim = nb_concepts
    elif issubclass(method_class, SAEExplainer):
        cbe = method_class(splitted_encoder_ml, nb_concepts=nb_concepts, device=DEVICE)
        cbe.fit(activations, nb_epochs=1, batch_size=1, device=DEVICE)
        concepts_dim = nb_concepts
    elif issubclass(method_class, DictionaryLearningExplainer):
        cbe = method_class(splitted_encoder_ml, nb_concepts=nb_concepts, device=DEVICE)
        cbe.fit(activations)
        concepts_dim = nb_concepts
    else:
        raise ValueError(f"Unknown method_class {method_class}")

    if not cbe.has_differentiable_concept_decoder:
        pytest.skip("Skipping test for method_class that does not have a differentiable concept decoder")

    gradients = cbe.concept_output_gradient(
        sentences,
        targets=[0],
        activation_granularity=granularity,
    )
    assert gradients is not None, f"{method_class.__name__}.concept_output_gradient returned None"
    assert isinstance(gradients, list), (
        f"{method_class.__name__}.concept_output_gradient returned type {type(gradients)} instead of list"
    )
    assert len(gradients) == len(sentences), (
        f"Gradients list length mismatch: got {len(gradients)}, expected {len(sentences)}"
    )
    for grad, sentence in zip(gradients, sentences, strict=True):
        assert grad is not None, "A gradient entry is None"
        assert isinstance(grad, torch.Tensor), f"Gradient entry has type {type(grad)} instead of torch.Tensor"

        tokenizer = splitted_encoder_ml.tokenizer
        tokens = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        indices_list = Granularity.get_indices(tokens, granularity.value, tokenizer)  # type: ignore
        nb_granularity_elements = len(indices_list[0])
        assert grad.shape == (1, nb_granularity_elements, concepts_dim), (
            "Gradient shape mismatch: got "
            f"{tuple(grad.shape)}, expected {(1, nb_granularity_elements, concepts_dim)} for sentence '{sentence}'"
        )


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM

    from interpreto import ModelWithSplitPoints

    sentences: list[str] = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Interpreto is magical",
        "Testing interpreto",
    ]
    splitted_encoder_ml: ModelWithSplitPoints = ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_points=["bert.encoder.layer.1.output"],
        automodel=AutoModelForMaskedLM,  # type: ignore
        device_map=DEVICE,
    )
    activations_dict: dict[str, torch.Tensor] = splitted_encoder_ml.get_activations(
        sentences, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS
    )  # type: ignore
    test_overcomplete_cbe(
        splitted_encoder_ml=splitted_encoder_ml,
        activations_dict=activations_dict,
        method_class=KMeansConcepts,
    )
    test_concept_output_gradient(
        splitted_encoder_ml=splitted_encoder_ml,
        activations_dict=activations_dict,
        sentences=sentences,
        method_class=NeuronsAsConcepts,
        granularity=ModelWithSplitPoints.activation_granularities.WORD,
    )
