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
GENERATION_MODELS = [
    "hf-internal-testing/tiny-random-gpt_neo",
    "hf-internal-testing/tiny-random-gptj",
    "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "hf-internal-testing/tiny-random-FalconModel",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
]


def test_generation_wrapper_fast():
    """Test generation wrapper with a single model for fast tests."""
    test_generation_wrapper("hf-internal-testing/tiny-random-gpt2")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", GENERATION_MODELS)
def test_generation_wrapper(model_name):
    # sentences divided in two batches which we could see as different samples in interpreto
    # for each sample there are several perturbed sentences
    sentences = [
        ["first sample with target"] * 2,
        ["second sample, longer, with longer target"] * 4,
    ]

    # Model preparation
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    embedder = model.get_input_embeddings()
    inference_wrapper = GenerationInferenceWrapper(model, batch_size=3, device=DEVICE)

    # Construct inputs
    with torch.no_grad():
        tokens = [tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(DEVICE) for s in sentences]
        logits = [model(**t).logits for t in tokens]
        embeddings = []
        targets = []
        targeted_logits = []
        for token, l in zip(tokens, logits, strict=True):
            # embeddings
            e = token.copy()
            input_ids = e.pop("input_ids")
            e["inputs_embeds"] = embedder(input_ids)
            embeddings.append(e)

            # targets (only one target per sample, second half of the sequence)
            t = input_ids[0, input_ids.shape[1] // 2 :]
            targets.append(t)

            # Reference values
            start = l.shape[1] - t.shape[0] - 1
            end = l.shape[1] - 1
            targeted_logits.append(l[:, torch.arange(start, end), t])

    # Compute elements with the wrapper
    test_targeted_logits = list(inference_wrapper(tokens, targets))
    inference_wrapper.gradients = True
    test_gradients = list(inference_wrapper(embeddings, targets))

    for i in range(len(sentences)):
        assert torch.allclose(targeted_logits[i], test_targeted_logits[i], atol=1e-5), (
            "Generation targeted logits are not correct"
        )
        grads_shape = (
            len(sentences[i]),
            targets[i].shape[0],
            embeddings[i]["inputs_embeds"].shape[1],
        )  # (b, n_targets, seq_len) or (b, t, l)
        assert grads_shape == test_gradients[i].shape, "Classification gradients have wrong shape."

    with pytest.raises(ValueError):
        # "inputs_embeds" are required for gradients
        inference_wrapper.gradients = True
        next(inference_wrapper(tokens, targets))

    with pytest.raises(NotImplementedError):
        # targets are required for generation
        inference_wrapper.gradients = False
        next(inference_wrapper(tokens))


if __name__ == "__main__":
    test_generation_wrapper("hf-internal-testing/tiny-random-gpt2")
    test_generation_wrapper("hf-internal-testing/tiny-random-LlamaForCausalLM")
