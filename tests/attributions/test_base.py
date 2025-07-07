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

from interpreto.attributions.base import ClassificationAttributionExplainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyInferenceWrapper:
    def get_targets(self, _):
        # For each input, return a dummy tensor with logits
        return [torch.tensor([1]), torch.tensor([0])]


def test_process_targets(bert_model, bert_tokenizer):
    """
    Test the process_targets method for different input types.
    """
    explainer = ClassificationAttributionExplainer(bert_model, bert_tokenizer, batch_size=2, device=DEVICE)

    # Single integer
    result = explainer.process_targets(3, expected_length=1)
    assert len(result) == 1  # type: ignore
    assert torch.equal(result[0], torch.tensor([3]))  # type: ignore

    # Single integer with mismatch
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs is 2"):
        explainer.process_targets(3, expected_length=2)

    # 1D tensor
    result = explainer.process_targets(torch.tensor([1, 2, 3]), expected_length=3)
    assert len(result) == 3  # type: ignore
    assert all(torch.equal(r.squeeze(), torch.tensor(v)) for r, v in zip(result, [1, 2, 3], strict=True))  # type: ignore

    # 2D tensor
    tensor = torch.tensor([[1], [2], [3]])
    result = explainer.process_targets(tensor, expected_length=3)
    assert len(result) == 3  # type: ignore
    assert all(r.shape == (1,) for r in result)

    # Tensor with floats
    tensor = torch.tensor([[1.0], [2.0]])
    with pytest.raises(TypeError, match="Target tensor must be integers."):
        explainer.process_targets(tensor)

    # Tensor with invalid ndim
    tensor = torch.tensor([[[1]]])
    with pytest.raises(TypeError, match="Target tensor must be one-dimensional or two-dimensional."):
        explainer.process_targets(tensor)

    # Iterable of ints
    result = explainer.process_targets([1, 2, 3], expected_length=3)
    assert len(result) == 3  # type: ignore
    assert all(torch.equal(r, torch.tensor(v)) for r, v in zip(result, [1, 2, 3], strict=True))

    # Iterable of ints with mismatch
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs is 2"):
        explainer.process_targets([1, 2, 3], expected_length=2)

    # Iterable of tensors
    tensors = [torch.tensor([1]), torch.tensor([2])]
    result = explainer.process_targets(tensors, expected_length=2)
    assert result == tensors

    # Iterable of float tensors
    tensors = [torch.tensor([1.0]), torch.tensor([2.0])]
    with pytest.raises(TypeError, match="must be integers"):
        explainer.process_targets(tensors)

    # Iterable of mixed-dim tensors
    tensors = [torch.tensor([[1]]), torch.tensor([2])]
    with pytest.raises(TypeError, match="must be one-dimensional"):
        explainer.process_targets(tensors)

    # Unsupported type
    with pytest.raises(TypeError, match="Target type .* not supported"):
        explainer.process_targets("invalid_input")  # type: ignore


def test_process_inputs_to_explain_and_targets(bert_model, bert_tokenizer):
    explainer = ClassificationAttributionExplainer(bert_model, tokenizer=bert_tokenizer, batch_size=2, device=DEVICE)
    explainer.inference_wrapper = DummyInferenceWrapper()  # type: ignore

    # Model input example
    model_inputs = [
        {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 2742, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        },
        {
            "input_ids": torch.tensor([[101, 1045, 2293, 2070, 3185, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        },
    ]

    # 1. Case with explicit targets (ints)
    targets = [0, 1]
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(
        model_inputs, targets=targets
    )
    assert len(processed_inputs) == 2  # type: ignore
    assert len(processed_targets) == 2  # type: ignore
    assert all(isinstance(t, torch.Tensor) for t in processed_targets)

    # 2. Case with explicit targets (tensor)
    targets_tensor = torch.tensor([1, 0])
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(
        model_inputs, targets=targets_tensor
    )
    assert len(processed_targets) == 2  # type: ignore
    assert all(isinstance(t, torch.Tensor) for t in processed_targets)

    # 3. Case with no targets (should use logits + argmax)
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(model_inputs)
    processed_targets = list(processed_targets).copy()
    assert len(processed_targets) == 2
    assert [t.item() for t in processed_targets] == [1, 0]

    # 4. Mismatched targets
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs"):
        explainer.process_inputs_to_explain_and_targets(model_inputs, targets=[1])  # Only one target for two inputs

    # 5. Already tokenized input
    model_inputs_masked = [
        {
            "input_ids": torch.tensor([[101, 2023, 1037, 3185, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        },
        {
            "input_ids": torch.tensor([[101, 2064, 2017, 2066, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        },
    ]
    targets = [0, 1]
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(model_inputs_masked, targets)
