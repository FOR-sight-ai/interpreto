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
from tests.conftest import multi_split_model
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification

from interpreto.model_wrapping.model_with_split_points import InitializationError, ModelWithSplitPoints

BERT_SPLIT_POINTS = [
    "cls.predictions.transform.LayerNorm",
    "bert.encoder.layer.1.output",
    "bert.encoder.layer.3.attention.self.query",
]

BERT_SPLIT_POINTS_SORTED = [
    "bert.encoder.layer.1.output",
    "bert.encoder.layer.3.attention.self.query",
    "cls.predictions.transform.LayerNorm",
]


def test_order_split_points(multi_split_model: ModelWithSplitPoints):
    """
    Test the sort_paths method upon split assignment
    """
    multi_split_model.split_points = BERT_SPLIT_POINTS  # type: ignore
    # Assert the ordered split points match expected order
    assert multi_split_model.split_points == BERT_SPLIT_POINTS_SORTED, (
        f"Failed for split points: {BERT_SPLIT_POINTS}\n"
        f"Expected: {BERT_SPLIT_POINTS_SORTED}\n"
        f"Got:      {multi_split_model.split_points}"
    )


def test_loading_possibilities(bert_model, gpt2_model):
    """
    Test loading model with and without split points
    """
    # Load model with split points
    model_with_split_points = ModelWithSplitPoints(bert_model, "bert.encoder.layer.1")
    assert model_with_split_points.split_points == ["bert.encoder.layer.1"]
    # Load model without split points
    model_without_split_points = ModelWithSplitPoints(
        "bert-base-cased",
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
        split_points="bert.encoder.layer.1",
    )
    assert model_without_split_points.split_points == ["bert.encoder.layer.1"]
    # Load model with split points
    model_with_split_points = ModelWithSplitPoints(gpt2_model, "transformer.h.1")
    assert model_with_split_points.split_points == ["transformer.h.1"]
    # Load model without split points
    model_without_split_points = ModelWithSplitPoints(
        "gpt2",
        model_autoclass=AutoModelForCausalLM,  # type: ignore
        split_points="transformer.h.1",
    )
    assert model_without_split_points.split_points == ["transformer.h.1"]

    with pytest.raises(InitializationError):
        # Model id with no auto class
        ModelWithSplitPoints("gpt2", "transformer.h.1")


def test_activation_equivalence_batched_text_token_inputs(multi_split_model: ModelWithSplitPoints):
    """
    Test the equivalence of activations for text and token inputs
    """
    multi_split_model.split_points = BERT_SPLIT_POINTS  # type: ignore
    inputs_str = ["Hello, my dog is cute", "The cat is on the [MASK]"]
    inputs_tensor = multi_split_model.tokenizer(inputs_str, return_tensors="pt", padding=True, truncation=True)

    activations_str = multi_split_model.get_activations(inputs_str)
    activations_tensor = multi_split_model.get_activations(inputs_tensor)

    for k in activations_str.keys():
        assert torch.allclose(activations_str[k], activations_tensor[k])  # type: ignore


# TODO: This test was removed because we do not currently handle splitting over layers that return
# outputs that are not tensors.
# def test_index_by_layer_idx(multi_split_model: ModelWithSplitPoints):
#    """Test indexing by layer idx"""
#    split_points_with_layer_idx: list[str | int] = list(BERT_SPLIT_POINTS)
#    split_points_with_layer_idx[1] = 1  # instead of bert.encoder.layer.1
#    multi_split_model.split_points = split_points_with_layer_idx
#    assert multi_split_model.split_points == BERT_SPLIT_POINTS_SORTED, (
#        f"Failed for split_points: {BERT_SPLIT_POINTS}\n"
#        f"Expected: {BERT_SPLIT_POINTS_SORTED}\n"
#        f"Got:      {multi_split_model.split_points}"
#    )

if __name__ == "__main__":
    multi_split_model = ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_points=[
            "cls.predictions.transform.LayerNorm",
            "bert.encoder.layer.1.output",
            "bert.encoder.layer.3.attention.self.query",
        ],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
    )
    bert_model = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert")
    gpt2_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")

    test_order_split_points(multi_split_model)
    test_loading_possibilities(bert_model, gpt2_model)
    test_activation_equivalence_batched_text_token_inputs(multi_split_model)
