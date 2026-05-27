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
from interpreto.concepts.methods.sklearn_wrappers import SkLearnWrapperExplainer
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


@pytest.mark.parametrize("method_class", ALL_CONCEPT_METHODS)
def test_overcomplete_cbe(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations: torch.Tensor,
    method_class: type[ConceptAutoEncoderExplainer],
):
    """Test SAEExplainer and DictionaryLearningExplainer"""
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
    elif issubclass(method_class, (DictionaryLearningExplainer, SkLearnWrapperExplainer)):
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
        ModelWithSplitPoints.activation_granularities.CLS_TOKEN,
        ModelWithSplitPoints.activation_granularities.TOKEN,
        ModelWithSplitPoints.activation_granularities.WORD,
        ModelWithSplitPoints.activation_granularities.SENTENCE,
    ],
)
def test_concept_output_gradient(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations: torch.Tensor,
    sentences: list[str],
    method_class: type[ConceptAutoEncoderExplainer],
    granularity: ActivationGranularity,
):
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
    elif issubclass(method_class, (DictionaryLearningExplainer, SkLearnWrapperExplainer)):
        cbe = method_class(splitted_encoder_ml, nb_concepts=nb_concepts, device=DEVICE)
        cbe.fit(activations)
        concepts_dim = nb_concepts
    else:
        raise ValueError(f"Unknown method_class {method_class}")

    if not cbe.has_differentiable_concept_decoder:
        pytest.skip(f"Skipping test for {method_class.__name__} that does not have a differentiable concept decoder")

    gradients = cbe.concept_output_gradient(
        sentences,
        targets=None,
        activation_granularity=granularity,
        concepts_x_gradients=True,
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
        if granularity == ModelWithSplitPoints.activation_granularities.CLS_TOKEN:
            nb_granularity_elements = 1
        else:
            indices_list = granularity.value.get_indices(tokens, tokenizer)  # type: ignore
            nb_granularity_elements = len(indices_list[0])
        assert grad.shape[1:] == (nb_granularity_elements, concepts_dim), (
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
        split_point="bert.encoder.layer.1.output",
        automodel=AutoModelForMaskedLM,  # type: ignore
        device_map=DEVICE,
    )
    activations, _ = splitted_encoder_ml.get_activations(
        sentences, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS
    )
    test_overcomplete_cbe(
        splitted_encoder_ml=splitted_encoder_ml,
        activations=activations,  # type: ignore
        method_class=KMeansConcepts,
    )
    test_concept_output_gradient(
        splitted_encoder_ml=splitted_encoder_ml,
        activations=activations,  # type: ignore
        sentences=sentences,
        method_class=SemiNMFConcepts,
        granularity=ModelWithSplitPoints.activation_granularities.CLS_TOKEN,
    )
