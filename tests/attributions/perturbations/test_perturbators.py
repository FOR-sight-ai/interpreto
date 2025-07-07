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

from collections.abc import MutableMapping

import pytest
import torch

from interpreto.attributions.perturbations import (
    GaussianNoisePerturbator,
    LinearInterpolationPerturbator,
    OcclusionPerturbator,
    RandomMaskedTokenPerturbator,
    ShapTokenPerturbator,
    SobolTokenPerturbator,
)
from interpreto.attributions.perturbations.base import TokenMaskBasedPerturbator
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers
from interpreto.commons.granularity import Granularity

embeddings_perturbators = [
    GaussianNoisePerturbator,
    LinearInterpolationPerturbator,
]

tokens_perturbators = [
    OcclusionPerturbator,
    RandomMaskedTokenPerturbator,
    ShapTokenPerturbator,
    SobolTokenPerturbator,
]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.parametrize("perturbator_class", embeddings_perturbators)
def test_embeddings_perturbators(perturbator_class, sentences, bert_model, bert_tokenizer):
    """test all perturbators respect the API"""
    assert not issubclass(perturbator_class, TokenMaskBasedPerturbator)
    p = 10
    d = 32
    inputs_embedder = bert_model.get_input_embeddings()

    perturbator = perturbator_class(inputs_embedder=inputs_embedder, n_perturbations=p)
    perturbator.to(DEVICE)

    for sent in sentences:
        elem = bert_tokenizer(sent, return_tensors="pt", return_offsets_mapping=True)
        elem.to(DEVICE)
        assert isinstance(elem, MutableMapping)
        l = elem["input_ids"].shape[1]

        perturbed_inputs, masks = perturbator.perturb(elem)

        assert isinstance(perturbed_inputs, MutableMapping)

        # inputs are embedded before perturbation
        assert "inputs_embeds" in perturbed_inputs.keys()
        assert isinstance(perturbed_inputs["inputs_embeds"], torch.Tensor)
        assert perturbed_inputs["inputs_embeds"].shape == (p, l, d)

        assert "attention_mask" in perturbed_inputs.keys()
        assert isinstance(perturbed_inputs["attention_mask"], torch.Tensor)
        assert perturbed_inputs["attention_mask"].shape == (p, l)

        assert torch.all(torch.isclose(perturbed_inputs["attention_mask"], elem["attention_mask"], atol=1e-5))


@pytest.mark.parametrize("perturbator_class", tokens_perturbators)
def test_token_perturbators(perturbator_class, sentences, bert_model, bert_tokenizer):
    """test all perturbators respect the API"""
    assert issubclass(perturbator_class, TokenMaskBasedPerturbator)
    p = 10
    inputs_embedder = bert_model.get_input_embeddings()

    replace_token = "[REPLACE]"
    if replace_token not in bert_tokenizer.get_vocab():
        bert_tokenizer.add_tokens([replace_token])
        bert_model.resize_token_embeddings(len(bert_tokenizer))
    replace_token_id = bert_tokenizer.convert_tokens_to_ids(replace_token)  # type: ignore

    if perturbator_class in [OcclusionPerturbator, SobolTokenPerturbator]:
        # the number of perturbations depends on the sequence length
        perturbator = perturbator_class(
            tokenizer=bert_tokenizer,
            inputs_embedder=inputs_embedder,
            granularity=Granularity.ALL_TOKENS,
            replace_token_id=replace_token_id,
        )
    else:
        perturbator = perturbator_class(
            tokenizer=bert_tokenizer,
            inputs_embedder=inputs_embedder,
            granularity=Granularity.ALL_TOKENS,
            replace_token_id=replace_token_id,
            n_perturbations=p,
        )
    perturbator.to(DEVICE)

    for sent in sentences:
        elem = bert_tokenizer(sent, return_tensors="pt", return_offsets_mapping=True)
        elem.to(DEVICE)
        assert isinstance(elem, MutableMapping)
        l = elem["input_ids"].shape[1]

        perturbed_inputs, masks = perturbator.perturb(elem)

        assert isinstance(perturbed_inputs, MutableMapping)

        # inputs should not have been embedded
        assert "input_ids" in perturbed_inputs.keys()
        assert isinstance(perturbed_inputs["input_ids"], torch.Tensor)
        if isinstance(perturbator, OcclusionPerturbator):
            real_p = l + 1
        elif isinstance(perturbator, SobolTokenPerturbator):
            k = 30  # default value
            real_p = (l + 2) * k
        else:
            real_p = p

        assert perturbed_inputs["input_ids"].shape == (real_p, l)

        assert isinstance(masks, torch.Tensor)
        assert masks.shape[0] == perturbed_inputs["input_ids"].shape[0]

        assert "attention_mask" in perturbed_inputs.keys()
        assert isinstance(perturbed_inputs["attention_mask"], torch.Tensor)
        assert torch.all(torch.isclose(perturbed_inputs["attention_mask"], elem["attention_mask"], atol=1e-5))
        assert perturbed_inputs["attention_mask"].shape == (real_p, l)


# TODO: test apply mask

# TODO: test granularity functions and resulting masks

# TODO: test granularity perturbation application on inputs


def test_linear_interpolation_perturbation_adjust_baseline():
    inputs = torch.randn(4, 3, 10)

    # Test with None baseline
    baseline = LinearInterpolationPerturbator.adjust_baseline(None, inputs)
    assert torch.all(baseline == 0)
    assert baseline.shape == inputs.shape[1:]

    # Test with float baseline
    baseline = LinearInterpolationPerturbator.adjust_baseline(0.5, inputs)
    assert torch.all(baseline == 0.5)
    assert baseline.shape == inputs.shape[1:]

    # Test with tensor baseline
    baseline_tensor = torch.randn(3, 10)
    baseline = LinearInterpolationPerturbator.adjust_baseline(baseline_tensor, inputs)
    assert torch.all(baseline == baseline_tensor)
    assert baseline.shape == inputs.shape[1:]


def test_linear_interpolation_perturbation_adjust_baseline_invalid():
    inputs = torch.randn(4, 3, 10)

    # Test with invalid baseline type
    with pytest.raises(TypeError):
        LinearInterpolationPerturbator.adjust_baseline("invalid", inputs)  # type: ignore

    # Test with mismatched tensor shape
    baseline_tensor = torch.randn(2, 10)
    with pytest.raises(ValueError):
        LinearInterpolationPerturbator.adjust_baseline(baseline_tensor, inputs)

    # Test with mismatched tensor dtype
    baseline_tensor = torch.randn(3, 10, dtype=torch.float64)
    with pytest.raises(ValueError):
        LinearInterpolationPerturbator.adjust_baseline(baseline_tensor, inputs)


# TODO: adapt following tests to new perturbation API
# def test_tokens_perturbators(tokenizer, inputs_embedding, sentences):
#     token_perturbators = [
#         # TODO : fill
#     ]

#     def assert_mask_pertinence(reference, perturbations, mask):
#         # Check that a masks corresponds to the perturbation
#         # TODO : generalize this function to various shapes and use cases and put it in test utils for other tests
#         emb_size = perturbations.shape[-1]
#         diff = torch.ne(reference, perturbations).sum(axis=-1) / emb_size
#         binary_mask = mask.ne(0)
#         assert torch.equal(binary_mask, diff), "Mask is incoherent with applied perturbation"

#     sentences_embeddings = [
#         inputs_embedding(torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))) for sent in sentences
#     ]

#     for perturbator in token_perturbators:
#         # Test single sentence
#         clean_sentence_embeddings = sentences_embeddings[0]

#         # perturb the sentence
#         perturbed_sentences_embeddings, mask = perturbator.perturb(sentences[0])
#         assert perturbed_sentences_embeddings.shape[0] == 1, "Perturbation should have a batch dimension"

#         # Check that the mask fits with the perturbation
#         assert_mask_pertinence(clean_sentence_embeddings, perturbed_sentences_embeddings.squeeze(0), mask)

#         # Test batch of sentences
#         for index, (emb, mask) in enumerate(perturbator.perturb(sentences)):
#             assert_mask_pertinence(sentences_embeddings[index], emb.squeeze(0), mask)


@pytest.mark.parametrize(
    "sampler",
    [SequenceSamplers.SOBOL, SequenceSamplers.HALTON, SequenceSamplers.LatinHypercube],
)
def test_sobol_masks(sampler):
    k = 10
    perturbator = SobolTokenPerturbator(
        inputs_embedder=None,
        n_token_perturbations=k,
        sampler=sampler,
    )

    for l in range(2, 20, 3):
        mask = perturbator.get_mask(l)
        assert mask.shape == ((l + 2) * k, l)
        A = mask[:k]
        B = mask[k : 2 * k]
        C = mask[2 * k :].view(l, k, l)

        # verify token-wise mask compared to the initial mask
        for i in range(l):
            assert torch.all(torch.isclose(C[i, :, i], B[:, i], atol=1e-5))
            if i != 0:
                assert torch.all(torch.isclose(C[i, :, :i], A[:, :i], atol=1e-5))
            if i != l - 1:
                assert torch.all(torch.isclose(C[i, :, i + 1 :], A[:, i + 1 :], atol=1e-5))


def test_occlusion_masks():
    perturbator = OcclusionPerturbator(inputs_embedder=None)
    for l in range(2, 20, 3):
        mask = perturbator.get_mask(l)
        assert torch.equal(mask, torch.cat([torch.zeros(1, l), torch.eye(l)], dim=0))


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sentences = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Interpreto is magical",
        "Testing interpreto",
    ]
    bert_model = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert")
    bert_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")

    test_token_perturbators(
        SobolTokenPerturbator,
        sentences=sentences,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
    )
