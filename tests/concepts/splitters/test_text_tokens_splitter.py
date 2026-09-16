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

"""Tests for ``TextTokensSplitter``."""

import pytest
import torch

from interpreto import TextTokensSplitter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATION_REPO_ID = "hf-internal-testing/tiny-random-gpt2"
ENCODER_REPO_ID = "hf-internal-testing/tiny-random-bert"


@pytest.fixture(scope="module")
def generation_splitter():
    return TextTokensSplitter(
        GENERATION_REPO_ID,
        split_point=1,
        task="text-generation",
        batch_size=2,
        device_map=DEVICE,
    )


@pytest.fixture(scope="module")
def encoder_splitter():
    return TextTokensSplitter(
        ENCODER_REPO_ID,
        split_point="encoder.layer.1",
        task="feature-extraction",
        batch_size=2,
        device_map=DEVICE,
    )


def test_loading_preloaded_models(
    gpt2_model,
    gpt2_tokenizer,
    bert_model,
    bert_tokenizer,
):
    """The explicit task supports pre-loaded text models and task inference."""
    generation = TextTokensSplitter(
        gpt2_model,
        split_point=1,
        # task="text-generation",
        tokenizer=gpt2_tokenizer,
        device_map=DEVICE,
    )
    encoder = TextTokensSplitter(
        bert_model,
        split_point="bert.encoder.layer.1",
        # task="feature-extraction",
        tokenizer=bert_tokenizer,
        device_map=DEVICE,
    )

    assert generation.split_point == "transformer.h.1", "Wrong split point mapping"
    assert generation.task == "text-generation", "Wrong task inference"
    assert encoder.split_point == "bert.encoder.layer.1", "Wrong split point mapping"
    assert encoder.task != "text-generation", "Wrong task inference"


def test_generation_task_is_inferred(gpt2_model, gpt2_tokenizer):
    """NNsight infers text generation for a pre-loaded causal model."""
    splitter = TextTokensSplitter(
        gpt2_model,
        split_point=1,
        tokenizer=gpt2_tokenizer,
        device_map=DEVICE,
    )

    assert splitter.task == "text-generation"


@pytest.mark.parametrize("splitter", ["generation_splitter", "encoder_splitter"])
def test_token_extraction_for_text_model_tasks(request, splitter, sentences):
    """Both supported tasks expose flattened or sample-wise token activations."""
    splitter = request.getfixturevalue(splitter)
    flattened, predictions = splitter.get_activations(sentences)
    per_sample, _ = splitter.get_activations(sentences, flatten_activations=False)

    assert predictions is None
    assert flattened.shape == (sum(len(activations) for activations in per_sample), splitter.config.hidden_size)
    assert torch.allclose(flattened, torch.cat(per_sample), atol=1e-5)


def test_encoder_values_match_retained_tokens(encoder_splitter: TextTokensSplitter, sentences):
    """Encoder activations preserve the values and order of retained tokens."""
    tokenized, tokens_mask = encoder_splitter._tokenize_and_get_mask(sentences)
    with encoder_splitter.trace(tokenized):
        layer_outputs = encoder_splitter.split_module.output.save()
    full_activations, _ = encoder_splitter._extract_hidden_state(
        layer_outputs,
        encoder_splitter.split_point,
    )

    per_sample, _ = encoder_splitter.get_activations(sentences, flatten_activations=False)

    for activations, full_sample, mask in zip(per_sample, full_activations, tokens_mask, strict=True):
        assert torch.allclose(activations, full_sample[mask].cpu(), atol=1e-5)


def test_encoder_pooling_ignores_padding(encoder_splitter: TextTokensSplitter):
    """Pooling is applied after padding and special-token removal."""
    texts = ["Hi", "Interpreto is the latin for 'to interpret' and much longer"]
    pooled, _ = encoder_splitter.get_activations(texts, token_pooling="mean")
    per_sample, _ = encoder_splitter.get_activations(texts, flatten_activations=False)

    assert pooled.shape == (len(texts), encoder_splitter.config.hidden_size)
    for pooled_sample, activations in zip(pooled, per_sample, strict=True):
        assert torch.allclose(pooled_sample, activations.mean(dim=0), atol=1e-5)


def test_encoder_concept_output_gradients_are_unsupported(
    encoder_splitter: TextTokensSplitter,
    sentences,
):
    """Feature-extraction models do not expose causal-LM gradient targets."""
    identity = torch.eye(encoder_splitter.config.hidden_size)

    with pytest.raises(NotImplementedError, match="only supported for causal language models"):
        encoder_splitter._get_concept_output_gradients(
            sentences[:1],
            activations_to_concepts=lambda x: x @ identity,
            concepts_to_activations=lambda x: x @ identity,
            targets=[0],
        )
