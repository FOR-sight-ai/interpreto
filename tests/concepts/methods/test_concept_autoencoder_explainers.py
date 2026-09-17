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

from interpreto import SplitterForClassification, TextTokensSplitter
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
    splitted_encoder_ml: SplitterForClassification,
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
    assert hasattr(cbe, "splitter"), f"Explainer {method_class.__name__} missing 'splitter'"
    assert cbe.concept_model.fitted, f"Concept model in {method_class.__name__} not fitted"
    assert cbe.is_fitted, f"Explainer {method_class.__name__} reports not fitted"
    assert hasattr(cbe, "has_differentiable_concept_encoder"), (
        f"Explainer {method_class.__name__} missing 'has_differentiable_concept_encoder'"
    )
    assert hasattr(cbe, "has_differentiable_concept_decoder"), (
        f"Explainer {method_class.__name__} missing 'has_differentiable_concept_decoder'"
    )

    concepts = cbe.activations_to_concepts(activations)
    assert concepts is not None, f"{method_class.__name__}.activations_to_concepts returned None"
    reconstructed_activations = cbe.concepts_to_activations(concepts)
    assert reconstructed_activations is not None, f"{method_class.__name__}.concepts_to_activations returned None"
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
def test_concept_output_gradient(
    splitted_encoder_ml: SplitterForClassification,
    activations: torch.Tensor,
    sentences: list[str],
    method_class: type[ConceptAutoEncoderExplainer],
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
        concepts_x_gradients=True,
    )
    assert gradients is not None, f"{method_class.__name__}.concept_output_gradient returned None"
    assert isinstance(gradients, list), (
        f"{method_class.__name__}.concept_output_gradient returned type {type(gradients)} instead of list"
    )
    assert len(gradients) == len(sentences), (
        f"Gradients list length mismatch: got {len(gradients)}, expected {len(sentences)}"
    )
    for grad in gradients:
        assert grad is not None, "A gradient entry is None"
        assert isinstance(grad, torch.Tensor), f"Gradient entry has type {type(grad)} instead of torch.Tensor"
        assert grad.shape[1:] == (1, concepts_dim), (
            f"Gradient shape mismatch: got {tuple(grad.shape)}, expected {(1, 1, concepts_dim)}"
        )


def test_mixed_precision_encode_decode_preserves_gradients(
    sentences: list[str],
):
    """Concept inputs are cast to model dtype without breaking gradients."""
    splitter = SplitterForClassification("hf-internal-testing/tiny-random-bert", device_map=DEVICE)
    activations, _ = splitter.get_activations(sentences)
    explainer = PCAConcepts(splitter, nb_concepts=3, device=DEVICE)
    explainer.fit(activations)

    high_precision_activations = activations.to(torch.float64).requires_grad_(True)
    concepts = explainer.activations_to_concepts(high_precision_activations)
    decoded = explainer.concepts_to_activations(concepts.to(torch.float64))

    assert concepts.dtype == torch.float32
    assert decoded.dtype == torch.float32
    gradients = torch.autograd.grad(decoded.sum(), high_precision_activations)[0]
    assert torch.isfinite(gradients).all()


def test_sae_fit_normalizes_dtype_without_moving_dataset(monkeypatch):
    """SAE fitting keeps the dataset on its input device for batched transfer."""
    splitter = SplitterForClassification("hf-internal-testing/tiny-random-bert", device_map=DEVICE)
    explainer = VanillaSAEConcepts(splitter, nb_concepts=3, device=DEVICE)
    activations = torch.randn(4, splitter.config.hidden_size, dtype=torch.float64)
    observed = {}

    def inspect_dataloader(**kwargs):
        dataset_activations = kwargs["dataloader"].dataset.tensors[0]
        observed["device"] = dataset_activations.device
        observed["dtype"] = dataset_activations.dtype
        return {}

    monkeypatch.setattr("interpreto.concepts.methods.overcomplete.oc_sae.train_sae", inspect_dataloader)

    explainer.fit(activations, nb_epochs=1, batch_size=2)

    assert observed == {"device": activations.device, "dtype": torch.float32}


def test_concept_output_gradient_uses_splitter_contract(
    splitted_encoder_ml: SplitterForClassification, sentences: list[str]
):
    """Concept gradients delegate shape semantics to the task-specific splitter."""
    explainer = NeuronsAsConcepts(splitted_encoder_ml)

    gradients = explainer.concept_output_gradient(sentences[:2], targets=[0], normalization=False)

    assert len(gradients) == 2
    assert all(gradient.shape == (1, 1, splitted_encoder_ml.config.hidden_size) for gradient in gradients)


def test_generation_concept_output_gradient_uses_splitter_contract(
    text_tokens_splitter: TextTokensSplitter, sentences: list[str]
):
    """Generation splitters retain their token-level gradient dimension."""
    explainer = NeuronsAsConcepts(text_tokens_splitter)

    gradients = explainer.concept_output_gradient(sentences[:1], targets=[0], normalization=False)

    assert len(gradients) == 1
    assert gradients[0].shape[0] == 1
    assert gradients[0].shape[1] > 1
    assert gradients[0].shape[2] == text_tokens_splitter.config.hidden_size


if __name__ == "__main__":
    sentences: list[str] = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Interpreto is magical",
        "Testing interpreto",
    ]
    splitted_encoder_ml = SplitterForClassification(
        "hf-internal-testing/tiny-random-bert",
        device_map=DEVICE,
    )
    activations, _ = splitted_encoder_ml.get_activations(sentences)
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
    )
