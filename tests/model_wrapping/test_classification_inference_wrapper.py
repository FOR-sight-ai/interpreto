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

import torch

from interpreto.model_wrapping.classification_inference_wrapper import ClassificationInferenceWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_classification_inference_wrapper_single_sentence(bert_model, bert_tokenizer, sentences):
    # Model preparation
    inference_wrapper = ClassificationInferenceWrapper(bert_model, batch_size=5, device=DEVICE)
    inference_wrapper.pad_token_id = bert_tokenizer.pad_token_id

    # Reference values
    tokens = bert_tokenizer(sentences[0], return_tensors="pt", padding=True, truncation=True)
    tokens.to(DEVICE)
    logits = bert_model(**tokens).logits
    target = logits.argmax(dim=-1)
    predefined_targets = torch.randperm(logits.shape[-1]).to(DEVICE)
    scores = logits.index_select(dim=-1, index=predefined_targets)

    # Tests
    assert torch.equal(logits, inference_wrapper.get_logits(tokens.copy()))
    assert torch.equal(logits, next(inference_wrapper.get_logits([tokens.copy()])))
    assert torch.equal(target, inference_wrapper.get_targets(tokens.copy()))  # type: ignore
    assert torch.equal(target, next(inference_wrapper.get_targets([tokens.copy()])))  # type: ignore
    assert torch.equal(scores, inference_wrapper.get_targeted_logits(tokens.copy(), predefined_targets))  # type: ignore
    assert torch.equal(scores, next(inference_wrapper.get_targeted_logits([tokens.copy()], [predefined_targets])))  # type: ignore


def test_classification_inference_wrapper_multiple_sentences(bert_model, bert_tokenizer, sentences):
    # Model preparation
    inference_wrapper = ClassificationInferenceWrapper(bert_model, batch_size=5, device=DEVICE)

    ### Reference values
    tokens = bert_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    tokens.to(DEVICE)
    logits = bert_model(**tokens).logits
    targets = logits.argmax(dim=-1)
    predefined_targets = torch.randperm(logits.shape[-1]).to(DEVICE)
    target_logits = torch.gather(logits, dim=-1, index=predefined_targets.unsqueeze(0).expand(logits.shape[0], -1))

    ### Tests
    test_logits = torch.stack(list(inference_wrapper.get_logits(tokens.copy())))
    test_targets = torch.stack(list(inference_wrapper.get_targets(tokens.copy())))
    test_target_logits = torch.stack(list(inference_wrapper.get_targeted_logits(tokens.copy(), predefined_targets)))

    assert torch.all(torch.isclose(logits, test_logits, atol=1e-5))
    assert torch.all(torch.isclose(targets, test_targets, atol=1e-5))
    assert torch.all(torch.isclose(target_logits, test_target_logits, atol=1e-5))


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    bert_model = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert")
    bert_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    sentences = ["Hello, my dog is cute", "Hello, my cat is cute"]
    test_classification_inference_wrapper_single_sentence(bert_model, bert_tokenizer, sentences)
    test_classification_inference_wrapper_multiple_sentences(bert_model, bert_tokenizer, sentences)
