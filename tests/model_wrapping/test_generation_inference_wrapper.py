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
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpreto.model_wrapping.generation_inference_wrapper import GenerationInferenceWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTENCES = ["Hello, my dog is cute", "Hello, my cat is cute"]
GENERATION_MODELS = ["hf-internal-testing/tiny-random-LlamaForCausalLM", "hf-internal-testing/tiny-random-gpt2"]
TARGET_LENGTH = 2


def prepare_generation_wrapper(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    inference_wrapper = GenerationInferenceWrapper(model, batch_size=5, device=DEVICE)
    inference_wrapper.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, inference_wrapper


def compute_reference_targeted_logits(
    model: AutoModelForCausalLM,
    model_inputs,
    targets: torch.Tensor,
    mode,
) -> torch.Tensor:
    trimmed_inputs = {
        key: value[..., :-1, :] if key == "inputs_embeds" else value[..., :-1] for key, value in model_inputs.items()
    }

    with torch.no_grad():
        logits = model(**trimmed_inputs).logits

    target_logits = logits[..., -targets.shape[-1] :, :]
    target_logits = mode(target_logits)
    expanded_targets = targets.expand(logits.shape[0], -1)
    return target_logits.gather(dim=-1, index=expanded_targets.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("model_name", GENERATION_MODELS)
def test_generation_inference_wrapper_single_sentence(model_name):
    model, tokenizer, inference_wrapper = prepare_generation_wrapper(model_name)

    tokens = tokenizer(SENTENCES[0], return_tensors="pt")
    tokens.to(DEVICE)
    targets = tokens["input_ids"][..., -TARGET_LENGTH:]

    reference_scores = compute_reference_targeted_logits(
        model,
        tokens,
        targets,
        inference_wrapper.mode,
    )

    test_scores_mapping = inference_wrapper.get_targeted_logits(tokens.copy(), targets)
    test_scores_iterable = next(inference_wrapper.get_targeted_logits([tokens.copy()], [targets]))

    assert torch.allclose(reference_scores, test_scores_mapping, atol=1e-5)
    assert torch.allclose(reference_scores, test_scores_iterable, atol=1e-5)


@pytest.mark.parametrize("model_name", GENERATION_MODELS)
def test_generation_inference_wrapper_multiple_sentences(model_name):
    model, tokenizer, inference_wrapper = prepare_generation_wrapper(model_name)

    batch_tokens = tokenizer(SENTENCES, return_tensors="pt", padding=True, truncation=True)
    batch_tokens.to(DEVICE)
    batch_targets = batch_tokens["input_ids"][..., -TARGET_LENGTH:]

    reference_batch_scores = compute_reference_targeted_logits(
        model,
        batch_tokens,
        batch_targets,
        inference_wrapper.mode,
    )
    test_batch_scores = inference_wrapper.get_targeted_logits(batch_tokens.copy(), batch_targets)

    assert torch.allclose(reference_batch_scores, test_batch_scores, atol=1e-5)

    tokenized_sentences = [tokenizer(sentence, return_tensors="pt") for sentence in SENTENCES]
    for tokens in tokenized_sentences:
        tokens.to(DEVICE)

    targets_list = [tokens["input_ids"][..., -TARGET_LENGTH:] for tokens in tokenized_sentences]
    reference_iterable_scores = [
        compute_reference_targeted_logits(model, tokens, targets, inference_wrapper.mode)
        for tokens, targets in zip(tokenized_sentences, targets_list, strict=True)
    ]
    test_iterable_scores = list(
        inference_wrapper.get_targeted_logits([tokens.copy() for tokens in tokenized_sentences], targets_list)
    )

    for reference_scores, test_scores in zip(reference_iterable_scores, test_iterable_scores, strict=True):
        assert torch.allclose(reference_scores, test_scores, atol=1e-5)


@pytest.mark.parametrize("model_name", GENERATION_MODELS)
def test_generation_inference_wrapper_with_inputs_embeds(model_name):
    model, tokenizer, inference_wrapper = prepare_generation_wrapper(model_name)

    tokens = tokenizer(SENTENCES[0], return_tensors="pt")
    tokens.to(DEVICE)

    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(tokens["input_ids"])

    model_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": tokens["attention_mask"],
    }
    targets = tokens["input_ids"][..., -TARGET_LENGTH:]

    reference_scores = compute_reference_targeted_logits(
        model,
        model_inputs,
        targets,
        inference_wrapper.mode,
    )
    test_scores = inference_wrapper.get_targeted_logits(model_inputs.copy(), targets)

    assert torch.allclose(reference_scores, test_scores, atol=1e-5)


def test_generation_inference_wrapper_unsupported_input_type():
    inference_wrapper = object.__new__(GenerationInferenceWrapper)

    with pytest.raises(NotImplementedError, match="not supported"):
        inference_wrapper.get_targeted_logits(1, torch.tensor([[0]]))
