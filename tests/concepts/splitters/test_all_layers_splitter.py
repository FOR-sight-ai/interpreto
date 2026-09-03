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
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from interpreto.concepts.splitters import AllLayersSplitter

SLOW_MODEL_CASES = [
    pytest.param(
        model_name,
        AutoModelForSequenceClassification,
        id=model_name.rsplit("/", 1)[-1],
    )
    for model_name in [
        "hf-internal-testing/tiny-random-distilbert",
        "hf-internal-testing/tiny-random-ElectraModel",
        "hf-internal-testing/tiny-random-roberta",
        "hf-internal-testing/tiny-xlm-roberta",
    ]
] + [
    pytest.param(
        model_name,
        AutoModelForCausalLM,
        id=model_name.rsplit("/", 1)[-1],
    )
    for model_name in [
        "hf-internal-testing/tiny-random-gpt_neo",
        "hf-internal-testing/tiny-random-gptj",
        "hf-internal-testing/tiny-random-CodeGenForCausalLM",
        "hf-internal-testing/tiny-random-FalconModel",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "hf-internal-testing/tiny-random-MistralForCausalLM",
        "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    ]
]


def _assert_activations_and_head(model, tokenizer, layer_path=None):
    """Check all-layer extraction and faithful execution of the model tail."""
    text = "Interpreto is useful."
    model.eval()
    splitter = AllLayersSplitter(model, tokenizer=tokenizer)
    activations = splitter.get_activations(text)

    if layer_path is not None:
        assert splitter.split_points == [
            f"model.{layer_path}.{index}" for index in range(len(model.get_submodule(layer_path)))
        ]
    assert len(activations) == len(splitter.split_points) + 1
    assert all(activation.ndim == 3 and activation.shape == activations[0].shape for activation in activations)

    with torch.no_grad():
        expected_logits = model(**tokenizer(text, return_tensors="pt")).logits
    layer_logits = splitter.apply_head(torch.cat(activations))
    assert layer_logits.shape[0] == len(activations)
    torch.testing.assert_close(layer_logits[-1:], expected_logits)


def test_all_layers_splitter_bert_fast(bert_model, bert_tokenizer):
    """BERT exposes every block and projects the final output faithfully."""
    _assert_activations_and_head(bert_model, bert_tokenizer, "bert.encoder.layer")


def test_all_layers_splitter_gpt2_fast(gpt2_model, gpt2_tokenizer):
    """GPT-2 exposes every block and projects the final output faithfully."""
    _assert_activations_and_head(gpt2_model, gpt2_tokenizer, "transformer.h")


def test_all_layers_splitter_head_preserves_activation_gradients(gpt2_model, gpt2_tokenizer):
    """The native model tail remains differentiable for Tuned Lens fitting."""
    splitter = AllLayersSplitter(gpt2_model, tokenizer=gpt2_tokenizer)
    activation = splitter.get_activations("Interpreto is useful.")[-1].detach().requires_grad_()

    splitter.apply_head(activation).square().mean().backward()

    assert activation.grad is not None
    assert torch.isfinite(activation.grad).all()
    assert torch.any(activation.grad != 0)


def test_all_layers_splitter_head_skips_transformer_blocks(gpt2_model, gpt2_tokenizer):
    """Applying the head does not recompute transformer blocks."""
    splitter = AllLayersSplitter(gpt2_model, tokenizer=gpt2_tokenizer)
    activations = torch.cat(splitter.get_activations("Interpreto is useful."))
    block_calls = []

    with gpt2_model.transformer.h[0].attn.register_forward_hook(lambda *_: block_calls.append(None)):
        splitter.apply_head(activations)

    assert not block_calls


@pytest.mark.slow
@pytest.mark.parametrize(("model_name", "automodel"), SLOW_MODEL_CASES)
def test_all_layers_splitter_slow(model_name, automodel):
    """All-layer extraction supports common classification and generation models."""
    model = automodel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _assert_activations_and_head(model, tokenizer)
