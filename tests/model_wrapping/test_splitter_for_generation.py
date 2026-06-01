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

"""Tests for ``SplitterForGeneration``."""

import pytest
import torch

from interpreto import SplitterForGeneration as SFG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = "hf-internal-testing/tiny-random-gpt2"
SPLIT_POINT = 1


@pytest.fixture(scope="module")
def split_gen():
    return SFG(
        REPO_ID,
        split_point=SPLIT_POINT,
        batch_size=2,
        device_map=DEVICE,
    )


def test_loading_possibilities(gpt2_model, gpt2_tokenizer, bert_model, bert_tokenizer):
    """Generation splitters can be loaded from repos or causal LM instances only."""
    with pytest.raises(ValueError):
        SFG(gpt2_model, split_point=SPLIT_POINT)

    with pytest.raises(TypeError, match="not a causal language model"):
        SFG(bert_model, split_point=SPLIT_POINT, tokenizer=bert_tokenizer)

    split_model = SFG(gpt2_model, split_point=SPLIT_POINT, tokenizer=gpt2_tokenizer, device_map=DEVICE)
    assert split_model.split_point is not None, "split_point should be resolved for a pre-loaded causal LM"
    assert split_model.tokenizer.pad_token is not None, "generation splitters should ensure a pad token exists"

    split_model = SFG(REPO_ID, split_point=SPLIT_POINT, device_map=DEVICE)
    assert split_model.split_point is not None, "split_point should be resolved when loading from a repo id"
    assert split_model.tokenizer.pad_token is not None, "generation splitters should ensure a pad token exists"


def test_get_latent_shape(split_gen: SFG):
    """``get_latent_shape`` returns the scanned split-point hidden-state shape."""
    shape = split_gen.get_latent_shape()
    expected_hidden = split_gen._model.config.hidden_size

    assert len(shape) == 3, f"Latent shape should be 3D, got {shape}"
    assert shape[0] == 1, f"Latent shape should include the scan batch dimension, got {shape}"
    assert shape[-1] == expected_hidden, f"Hidden size mismatch: got {shape[-1]}, expected {expected_hidden}"


def test_get_activations_returns_flattened_tokens_by_default(split_gen: SFG, sentences: list[str]):
    """Default activation extraction returns flattened non-special token activations."""
    activations, predictions = split_gen.get_activations(sentences)

    assert predictions is None, "Generation splitters should not return predicted classes"
    assert isinstance(activations, torch.Tensor), "Flattened activations should be returned as a tensor"
    assert activations.ndim == 2, f"Expected flattened token activations with shape (ng, d), got {activations.shape}"
    assert activations.shape[-1] == split_gen._model.config.hidden_size


@pytest.mark.parametrize("include_all_tokens", [False, True])
def test_flatten_activations_matches_sample_wise_activations(
    split_gen: SFG,
    sentences: list[str],
    include_all_tokens: bool,
):
    """Flattened activations should be the concatenation of the sample-wise activations."""
    flattened_acts, flattened_predictions = split_gen.get_activations(
        sentences,
        include_all_tokens=include_all_tokens,
    )
    sample_wise_acts, sample_wise_predictions = split_gen.get_activations(
        sentences,
        include_all_tokens=include_all_tokens,
        flatten_activations=False,
    )

    assert flattened_predictions is None
    assert sample_wise_predictions is None
    assert isinstance(flattened_acts, torch.Tensor), "Flattened activations should be a tensor"
    assert isinstance(sample_wise_acts, list), "Sample-wise activations should be returned as a list"
    assert len(sample_wise_acts) == len(sentences), (
        f"Expected one activation tensor per input, got {len(sample_wise_acts)} for {len(sentences)} inputs"
    )

    expected_flattened_acts = torch.cat(sample_wise_acts, dim=0)
    assert flattened_acts.shape == expected_flattened_acts.shape, (
        f"Flattened activation shape mismatch: got {flattened_acts.shape}, expected {expected_flattened_acts.shape}"
    )
    assert torch.allclose(flattened_acts, expected_flattened_acts, atol=1e-5), (
        "Flattened activations do not match concatenated sample-wise activations"
    )


def test_get_activation_and_gradient(split_gen: SFG, sentences: list[str]):
    """Activation and concept-output gradient shapes follow the generation splitter contract."""
    hidden = split_gen._model.config.hidden_size
    nb_concepts = 2 * hidden
    initial = torch.randn(nb_concepts, hidden)
    decoder_weights = torch.linalg.qr(initial)[0].to(DEVICE)
    encoder_weights = decoder_weights.T

    sample_wise_acts, predictions = split_gen.get_activations(sentences, flatten_activations=False)
    assert predictions is None
    assert isinstance(sample_wise_acts, list)

    grads_list = split_gen._get_concept_output_gradients(
        sentences,
        encode_activations=lambda x: x @ encoder_weights,
        decode_concepts=lambda x: x @ decoder_weights,
        targets=None,
    )

    assert len(grads_list) == len(sentences), (
        f"Gradients list length mismatch: got {len(grads_list)}, expected {len(sentences)}"
    )
    for grads, activations in zip(grads_list, sample_wise_acts, strict=True):
        assert grads.ndim == 3, f"Expected gradients with shape (t, g, c), got {grads.shape}"
        assert grads.shape[1] == activations.shape[0], (
            f"Granularity dimension mismatch: got {grads.shape[1]}, expected {activations.shape[0]}"
        )
        assert grads.shape[-1] == nb_concepts, (
            f"Concept dimension mismatch: got {grads.shape[-1]}, expected {nb_concepts}"
        )


def test_get_concept_output_gradients_with_explicit_targets(split_gen: SFG, sentences: list[str]):
    """Explicit generation targets control the first gradient dimension."""
    hidden = split_gen._model.config.hidden_size
    identity = torch.eye(hidden).to(DEVICE)
    targets = [0, 1]

    grads_list = split_gen._get_concept_output_gradients(
        sentences[:1],
        encode_activations=lambda x: x @ identity,
        decode_concepts=lambda x: x @ identity,
        targets=targets,
    )

    assert len(grads_list) == 1
    assert grads_list[0].ndim == 3, f"Expected gradients with shape (t, g, c), got {grads_list[0].shape}"
    assert grads_list[0].shape[0] == len(targets)
    assert grads_list[0].shape[-1] == hidden


def test_batching(split_gen: SFG, huge_text: list[str]):
    """Activation extraction works with more inputs than the configured batch size."""
    inputs = huge_text[:11]
    activations, predictions = split_gen.get_activations(inputs, flatten_activations=False)

    assert predictions is None
    assert isinstance(activations, list)
    assert len(activations) == len(inputs)
