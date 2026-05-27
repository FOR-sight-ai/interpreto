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

import pytest
import torch

from interpreto import SplitSequenceClassification as SSC
from interpreto.attributions import (
    KernelShap,
    Lime,
    Occlusion,
    Sobol,
)
from interpreto.attributions.base import AttributionExplainer
from interpreto.concepts import (
    BatchTopKSAEConcepts,
    ConceptAutoEncoderExplainer,
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_IDS = [
    "hf-internal-testing/tiny-random-albert",
    "hf-internal-testing/tiny-random-bart",
    "hf-internal-testing/tiny-random-bert",
    "hf-internal-testing/tiny-random-distilbert",
    "hf-internal-testing/tiny-random-ElectraModel",
    "hf-internal-testing/tiny-random-roberta",
    "hf-internal-testing/tiny-random-t5",
    "hf-internal-testing/tiny-random-gpt2",
]

# Perturbation based methods:
attribution_method_kwargs = {
    Occlusion: {},
    KernelShap: {"n_perturbations": 3},
    Lime: {"n_perturbations": 3},
    Sobol: {"n_token_perturbations": 3},
}

# Cycle over repo_id, concept_explainer_class, attribution_explainer_class at least once
REPRESENTATIVE_CASES = [
    # ()"hf-internal-testing/tiny-random-distilbert", PCAConcepts, KernelShap),  # used in fast test
    ("hf-internal-testing/tiny-random-albert", BatchTopKSAEConcepts, Occlusion),
    ("hf-internal-testing/tiny-random-bart", DictionaryLearningConcepts, KernelShap),
    ("hf-internal-testing/tiny-random-bert", ICAConcepts, Lime),
    ("hf-internal-testing/tiny-random-distilbert", JumpReLUSAEConcepts, Sobol),
    ("hf-internal-testing/tiny-random-ElectraModel", KMeansConcepts, Occlusion),
    ("hf-internal-testing/tiny-random-roberta", NeuronsAsConcepts, KernelShap),
    ("hf-internal-testing/tiny-random-t5", PCAConcepts, Lime),
    ("hf-internal-testing/tiny-random-gpt2", SemiNMFConcepts, Sobol),
    ("hf-internal-testing/tiny-random-albert", SparsePCAConcepts, Occlusion),
    ("hf-internal-testing/tiny-random-bart", SVDConcepts, KernelShap),
    ("hf-internal-testing/tiny-random-bert", TopKSAEConcepts, Lime),
    ("hf-internal-testing/tiny-random-distilbert", VanillaSAEConcepts, Sobol),
]


def test_inputs_to_concepts_attributions_fast(sentences):
    """
    Test using the input-to-concepts model in an attribution explainer.

    This test is fast because it only test with one model and explainer.
    While the other tests are slow because they test with multiple models and explainers.
    """
    inputs_to_concepts_attributions("hf-internal-testing/tiny-random-distilbert", PCAConcepts, KernelShap, sentences)


@pytest.mark.slow
@pytest.mark.parametrize("repo_id, concepts_explainer_class, attribution_explainer_class", REPRESENTATIVE_CASES)
def test_inputs_to_concepts_attributions_slow(
    repo_id, concepts_explainer_class, attribution_explainer_class, sentences
):
    """
    Test using the input-to-concepts model in an attribution explainer.

    This test is slow because it tests with multiple models and explainers.

    Not all combinations are tested, there are far too many, but we cycle over the REPRESENTATIVE_CASES.
    """
    inputs_to_concepts_attributions(repo_id, concepts_explainer_class, attribution_explainer_class, sentences)


def inputs_to_concepts_attributions(
    repo_id: str,
    concepts_explainer_class: type[ConceptAutoEncoderExplainer],
    attribution_explainer_class: type[AttributionExplainer],
    sentences: list[str],
):
    """
    Test that the `get_activations` and `_get_concept_output_gradients` methods return the expected shapes.
    """
    # extract kwargs
    attribution_kwargs = attribution_method_kwargs[attribution_explainer_class]

    # Split the model
    split_model = SSC(repo_id, device_map=DEVICE)

    # Fit the explainer
    if concepts_explainer_class != NeuronsAsConcepts:
        kwargs = {} if concepts_explainer_class != NMFConcepts else {"force_relu": True}
        concepts_explainer = concepts_explainer_class(split_model, nb_concepts=3, **kwargs)  # type: ignore
        activations, _ = split_model.get_activations(sentences)
        concepts_explainer.fit(activations)
    else:
        concepts_explainer = NeuronsAsConcepts(split_model)  # type: ignore

    # Instantiate the attribution explainer
    attribution_explainer = attribution_explainer_class(
        concepts_explainer.inputs_to_concepts, split_model.tokenizer, **attribution_kwargs
    )

    # Compute the attributions
    attribution_outputs = attribution_explainer.explain(sentences)

    assert isinstance(attribution_outputs, list)
    assert len(attribution_outputs) == len(sentences)
    for output in attribution_outputs:
        assert isinstance(output.attributions, torch.Tensor)
        assert isinstance(output.elements, list)
        assert output.attributions.shape == (concepts_explainer.concept_model.nb_concepts, len(output.elements))


if __name__ == "__main__":
    test_inputs_to_concepts_attributions_slow(
        "hf-internal-testing/tiny-random-gpt2",
        SemiNMFConcepts,
        Sobol,
        sentences=["Hello world!", "Can you hear me?", "Yes, who is it?", "It's me!"],
    )
