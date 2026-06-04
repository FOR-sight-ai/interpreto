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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interpreto.attributions.base import setup_token_ids
from interpreto.attributions.inference_wrappers.classification_inference_wrapper import ClassificationInferenceWrapper
from interpreto.typing import IncompatibilityError

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFICATION_MODELS = [
    "hf-internal-testing/tiny-random-albert",
    "hf-internal-testing/tiny-random-bart",
    "hf-internal-testing/tiny-random-distilbert",
    "hf-internal-testing/tiny-random-ElectraModel",
    "hf-internal-testing/tiny-random-roberta",
    "hf-internal-testing/tiny-random-t5",
    "hf-internal-testing/tiny-xlm-roberta",
]


def test_classification_wrapper_fast():
    """Test classification wrapper with a single model for fast tests."""
    test_classification_wrapper("hf-internal-testing/tiny-random-bert")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_wrapper(model_name):
    # sentences divided in two batches which we could see as different samples in interpreto
    # for each sample there are several perturbed sentences
    sentences = [
        ["first sample"] * 2,
        ["second sample, longer"] * 4,
    ]

    # Model preparation
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    setup_token_ids(model, tokenizer)
    model.eval()
    embedder = model.get_input_embeddings()
    inference_wrapper = ClassificationInferenceWrapper(model, batch_size=3, device=DEVICE)

    # Construct inputs
    with torch.no_grad():
        tokens = [tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(DEVICE) for s in sentences]
        logits = [model(**t).logits for t in tokens]
        targets = [l[0].argmax(dim=-1, keepdim=True) for l in logits]
        embeddings = []
        for t in tokens:
            e = t.copy()
            e["inputs_embeds"] = embedder(e.pop("input_ids"))
            embeddings.append(e)

    # Reference values
    expected_targets = [l.argmax(dim=-1, keepdim=True) for l in logits]
    targeted_logits = [l.gather(dim=-1, index=t) for l, t in zip(logits, expected_targets, strict=True)]

    # Compute elements with the wrapper
    test_targets = list(inference_wrapper(tokens))
    test_targeted_logits = list(inference_wrapper(tokens, targets))
    inference_wrapper.gradients = True
    try:
        test_gradients = list(inference_wrapper(embeddings, targets))
    except IncompatibilityError:
        test_gradients = "ignore"

    for i in range(len(sentences)):
        assert torch.allclose(expected_targets[i].cpu(), test_targets[i], atol=1e-5), (
            "Classification targets are not argmax"
        )
        assert torch.allclose(targeted_logits[i], test_targeted_logits[i], atol=1e-5), (
            "Classification targeted logits are not correct"
        )
        grads_shape = (
            len(sentences[i]),
            targets[i].shape[0],
            embeddings[i]["inputs_embeds"].shape[1],
        )  # (b, n_targets, seq_len) or (b, t, l)
        if test_gradients != "ignore":
            assert grads_shape == test_gradients[i].shape, "Classification gradients have wrong shape."


if __name__ == "__main__":
    test_classification_wrapper("hf-internal-testing/tiny-random-roberta")
    test_classification_wrapper("hf-internal-testing/tiny-random-bart")
