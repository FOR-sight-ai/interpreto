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

generation_models = ["hf-internal-testing/tiny-random-LlamaForCausalLM", "hf-internal-testing/tiny-random-gpt2"]


def prepare_model_and_tokenizer(model_name: str):
    """
    Helper function to prepare the tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inference_wrapper = GenerationInferenceWrapper(model, batch_size=5, device=DEVICE)
    return tokenizer, model, inference_wrapper


@pytest.mark.parametrize("model_name", generation_models)
def test_generation_inference_wrapper_single_sentence(model_name, sentences):
    """
    Tests the all function of the generation inference wrapper with a single sentence input.

    The test ensures:
      - The full model input is constructed by appending the target to the original input.
      - The first part of the full input exactly matches the original input.
      - The second part of the full input exactly matches the generated target.
      - The logits, targeted logits, and gradient matrix have the expected shapes.
    """

    # Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(
        sentences[0], return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )
    model_inputs.to(DEVICE)
    model_inputs_length = model_inputs["input_ids"].shape[1]
    max_length = model_inputs_length + 12

    full_model_inputs, target = inference_wrapper.get_inputs_to_explain_and_targets(
        model_inputs, max_length=max_length, do_sample=False
    )
    target_length = target.shape[1]
    full_shape = full_model_inputs["input_ids"].shape[1]
    # Check that the first part of the full input is equal to the original input.
    assert full_model_inputs["input_ids"][:, :-target_length].equal(model_inputs["input_ids"]), (
        "For a single sentence, the first part of the full input does not match the original input."
    )

    # Check that the second part of the full input is equal to the target.
    assert full_model_inputs["input_ids"][:, -target_length:].equal(target), (
        "For a single sentence, the target part of the full input does not match the provided target."
    )

    # Check that the full input has the expected shape given by the max_length.
    assert full_model_inputs["input_ids"].shape[1] == max_length, (
        "The full input shape doesn't match the expected shape given by max_length."
    )

    logits = inference_wrapper.get_logits(full_model_inputs)
    # Check that the logits shape is correct.
    assert logits.shape == (1, full_shape, model.config.vocab_size), (
        "For a single sentence, the logits shape is incorrect."
    )

    targeted_logits = inference_wrapper.get_targeted_logits(full_model_inputs, target)
    # Check that the targeted logits shape is correct.
    assert targeted_logits.shape == (1, target_length), (
        "For a single sentence, the targeted logits shape is incorrect."
    )

    grad_matrix = inference_wrapper.get_gradients(full_model_inputs, target)
    # Check that the gradient matrix shape is correct.
    assert isinstance(grad_matrix, torch.Tensor)
    assert grad_matrix.shape == (1, target_length, full_shape), (
        "For a single sentence, the gradient matrix shape is incorrect."
    )


@pytest.mark.parametrize("model_name", generation_models)
def test_generation_inference_wrapper_multiple_sentences(model_name, sentences):
    """
    Tests all function of the generation inference wrapper with multiple sentences input.

    The test ensures:
      - The full model input is constructed by appending the generated targets to the original inputs.
      - The first part of the full input exactly matches the original input.
      - The second part of the full input exactly matches the provided target.
      - The logits, targeted logits, and gradient matrix have the expected shapes for all sentences.
    """
    # Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_sentences = len(sentences)
    model_inputs = tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )
    model_inputs.to(DEVICE)
    model_inputs_length = model_inputs["input_ids"].shape[1]
    max_length = model_inputs_length + 12

    full_model_inputs, target = inference_wrapper.get_inputs_to_explain_and_targets(
        model_inputs, max_length=max_length, do_sample=False
    )
    target_length = target.shape[1]
    full_shape = full_model_inputs["input_ids"].shape[1]
    # Check that the first part of the full input is equal to the original input.
    assert full_model_inputs["input_ids"][:, :-target_length].equal(model_inputs["input_ids"]), (
        "For multiple sentences, the first part of the full input does not match the original input."
    )

    # Check that the second part of the full input is equal to the target.
    assert full_model_inputs["input_ids"][:, -target_length:].equal(target), (
        "For multiple sentences, the target part of the full input does not match the provided target."
    )

    # Check that the full input has the expected shape given by the max_length.
    assert full_model_inputs["input_ids"].shape[1] == max_length, (
        "The full input shape doesn't match the expected shape given by max_length."
    )

    logits = inference_wrapper.get_logits(full_model_inputs)
    # Check that the logits shape is correct.
    assert logits.shape == (n_sentences, full_shape, model.config.vocab_size), (
        "For multiple sentences, the logits shape is incorrect."
    )

    targeted_logits = inference_wrapper.get_targeted_logits(full_model_inputs, target)
    # Check that the targeted logits shape is correct.
    assert targeted_logits.shape == (n_sentences, target_length), (
        "For multiple sentences, the targeted logits shape is incorrect."
    )

    grad_matrix = inference_wrapper.get_gradients(full_model_inputs, target)
    # Check that the gradient matrix shape is correct.
    assert isinstance(grad_matrix, torch.Tensor)
    assert grad_matrix.shape == (n_sentences, target_length, full_shape), (
        "For multiple sentences, the gradient matrix shape is incorrect."
    )


@pytest.mark.parametrize("model_name", generation_models)
def test_generation_inference_wrapper_multiple_mappings(model_name, sentences):
    """
    Tests all function of the generation inference wrapper with multiple mappings.

    This test verifies that:
      - The full model inputs are correctly constructed by appending the target to the original inputs.
      - The first part of each full input matches the corresponding original input.
      - The second part of each full input exactly matches the generated target.
      - The shapes of the logits, targeted logits, and gradient matrix are as expected for each mapping.
    """
    # Model preparation
    tokenizer, model, inference_wrapper = prepare_model_and_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inference_wrapper.pad_token_id = tokenizer.pad_token_id
    n_sentences = len(sentences)
    nb_split = 2
    if nb_split >= n_sentences:
        raise ValueError("nb_split must be less than the number of sentences (n_sentences).")

    model_inputs1 = tokenizer(
        sentences[:nb_split], return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )
    model_inputs2 = tokenizer(
        sentences[nb_split:], return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )
    model_inputs1.to(DEVICE)
    model_inputs2.to(DEVICE)
    model_inputs = [model_inputs1, model_inputs2]
    model_inputs_length1 = model_inputs1["input_ids"].shape[1]
    model_inputs_length2 = model_inputs2["input_ids"].shape[1]
    max_length = max(model_inputs_length1, model_inputs_length2) + 6

    full_model_inputs, target = inference_wrapper.get_inputs_to_explain_and_targets(
        model_inputs, max_length=max_length, do_sample=False
    )

    target_length1 = target[0].shape[1]
    target_length2 = target[1].shape[1]
    full_shape1 = full_model_inputs[0]["input_ids"].shape[1]
    full_shape2 = full_model_inputs[1]["input_ids"].shape[1]

    # check that the first part of the full input is equal to the original input:
    assert full_model_inputs[0]["input_ids"][:, :-target_length1].equal(model_inputs[0]["input_ids"]), (
        "The first part of the full input for mapping 0 does not match the original input."
    )
    assert full_model_inputs[1]["input_ids"][:, :-target_length2].equal(model_inputs[1]["input_ids"]), (
        "The first part of the full input for mapping 1 does not match the original input."
    )

    # check that the second part of the full input is equal to the target:
    assert full_model_inputs[0]["input_ids"][:, -target_length1:].equal(target[0]), (
        "The target part of the full input for mapping 0 does not match the target."
    )
    assert full_model_inputs[1]["input_ids"][:, -target_length2:].equal(target[1]), (
        "The target part of the full input for mapping 1 does not match the target."
    )

    # Check that the full input has the expected shape given by the max_length.
    assert full_model_inputs[0]["input_ids"].shape[1] == max_length, (
        "The full input shape doesn't match the expected shape given by max_length."
    )
    assert full_model_inputs[1]["input_ids"].shape[1] == max_length, (
        "The full input shape doesn't match the expected shape given by max_length."
    )

    logits = inference_wrapper.get_logits(full_model_inputs)
    logits2 = list(logits)
    # check that the logits shape is correct:
    assert logits2[0].shape == (nb_split, full_shape1, model.config.vocab_size), (
        "Logits shape for mapping 0 is incorrect."
    )
    assert logits2[1].shape == (n_sentences - nb_split, full_shape2, model.config.vocab_size), (
        "Logits shape for mapping 1 is incorrect."
    )

    targeted_logits = inference_wrapper.get_targeted_logits(full_model_inputs, target)
    targeted_logits2 = list(targeted_logits)
    # check that the targeted logits shape is correct:
    assert targeted_logits2[0].shape == (nb_split, target_length1), "Targeted logits shape for mapping 0 is incorrect."
    assert targeted_logits2[1].shape == (n_sentences - nb_split, target_length2), (
        "Targeted logits shape for mapping 1 is incorrect."
    )

    grad_matrix = iter(inference_wrapper.get_gradients(full_model_inputs, target))
    # check that the grad matrix shape is correct:
    assert next(grad_matrix).shape == (nb_split, target_length1, full_shape1), (
        "Gradient matrix shape for mapping 0 is incorrect."
    )
    assert next(grad_matrix).shape == (n_sentences - nb_split, target_length2, full_shape2), (
        "Gradient matrix shape for mapping 1 is incorrect."
    )
