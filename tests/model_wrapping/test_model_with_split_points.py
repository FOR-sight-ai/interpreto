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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from interpreto import Granularity, ModelWithSplitPoints
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity, InitializationError

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


def test_loading_possibilities(bert_model, bert_tokenizer, gpt2_model, gpt2_tokenizer):
    """
    Test loading model with and without split points
    """
    # ----
    # BERT
    # Load model with split points
    with pytest.raises(ValueError):  # tokenizer is not set
        ModelWithSplitPoints(bert_model, "bert.encoder.layer.1")
    model_with_split_points = ModelWithSplitPoints(
        bert_model, split_points="bert.encoder.layer.1", tokenizer=bert_tokenizer
    )
    assert model_with_split_points.split_points == ["bert.encoder.layer.1"]
    # Load model without split points
    model_without_split_points = ModelWithSplitPoints(
        "bert-base-cased",
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
        split_points="bert.encoder.layer.1",
    )
    assert model_without_split_points.split_points == ["bert.encoder.layer.1"]

    # ----
    # GPT2
    # Load model with split points
    with pytest.raises(ValueError):  # tokenizer is not set
        ModelWithSplitPoints(gpt2_model, "transformer.h.1")
    model_with_split_points = ModelWithSplitPoints(
        gpt2_model, split_points="transformer.h.1", tokenizer=gpt2_tokenizer
    )
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
    inputs_tensor = multi_split_model.tokenizer(
        inputs_str, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )

    activations_str = multi_split_model.get_activations(inputs_str)
    activations_tensor = multi_split_model.get_activations(inputs_tensor)

    for k in activations_str.keys():
        assert torch.allclose(activations_str[k], activations_tensor[k])  # type: ignore


def test_get_activations_selection_strategies(
    splitted_encoder_ml: ModelWithSplitPoints,
    sentences: list[str],
):
    """Validate output shapes and values for all activation selection strategies."""

    tokenizer = splitted_encoder_ml.tokenizer
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )

    batch, seq_len = tokens["input_ids"].shape  # type: ignore
    hidden = splitted_encoder_ml._model.config.hidden_size
    total_token_len = sum([len(indices) for indices in Granularity.get_indices(tokens, Granularity.TOKEN, tokenizer)])
    total_word_len = sum([len(indices) for indices in Granularity.get_indices(tokens, Granularity.WORD, tokenizer)])

    expected_shapes = {
        # splitted_encoder_ml.activation_granularities.ALL: (batch, seq_len, hidden),
        splitted_encoder_ml.activation_granularities.CLS_TOKEN: (batch, hidden),
        splitted_encoder_ml.activation_granularities.ALL_TOKENS: (batch * seq_len, hidden),
        splitted_encoder_ml.activation_granularities.TOKEN: (total_token_len, hidden),
        splitted_encoder_ml.activation_granularities.WORD: (total_word_len, hidden),
        splitted_encoder_ml.activation_granularities.SENTENCE: (len(sentences) + 1, hidden),
        splitted_encoder_ml.activation_granularities.SAMPLE: (batch, hidden),
    }

    split = splitted_encoder_ml.split_points[0]

    for strategy, shape in expected_shapes.items():
        activations = splitted_encoder_ml.get_activations(sentences, activation_granularity=strategy)
        assert activations[split].shape == shape


@pytest.mark.parametrize(
    "strategy",
    [
        # ModelWithSplitPoints.activation_granularities.ALL,
        ModelWithSplitPoints.activation_granularities.CLS_TOKEN,
        ModelWithSplitPoints.activation_granularities.ALL_TOKENS,
        ModelWithSplitPoints.activation_granularities.TOKEN,
        ModelWithSplitPoints.activation_granularities.WORD,
        ModelWithSplitPoints.activation_granularities.SENTENCE,
        ModelWithSplitPoints.activation_granularities.SAMPLE,
    ],
)
def test_batching(splitted_encoder_ml: ModelWithSplitPoints, huge_text: list[str], strategy: ActivationGranularity):
    splitted_encoder_ml.get_activations(huge_text, activation_granularity=strategy)


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

ALL_MODEL_LOADERS = {
    "hf-internal-testing/tiny-random-albert": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-bart": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-bert": AutoModelForSequenceClassification,
    # "hf-internal-testing/tiny-random-DebertaV2Model": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-distilbert": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-ElectraModel": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-roberta": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-t5": AutoModelForSeq2SeqLM,
    # "hf-internal-testing/tiny-xlm-roberta": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-gpt2": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-gpt_neo": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-gptj": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-CodeGenForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-FalconModel": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-LlamaForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-MistralForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-Starcoder2ForCausalLM": AutoModelForCausalLM,
}

ALL_MODEL_SPLIT_POINTS = {
    "hf-internal-testing/tiny-random-albert": ["albert.encoder.albert_layer_groups.1.albert_layers.0.ffn_output"],
    "hf-internal-testing/tiny-random-bart": ["model.decoder.layers.1.fc2"],
    "hf-internal-testing/tiny-random-bert": ["bert.encoder.layer.1.output"],
    # "hf-internal-testing/tiny-random-DebertaV2Model": ["todo"],
    "hf-internal-testing/tiny-random-distilbert": ["distilbert.transformer.layer.1.ffn"],
    "hf-internal-testing/tiny-random-ElectraModel": ["electra.encoder.layer.1.output"],
    "hf-internal-testing/tiny-random-roberta": ["roberta.encoder.layer.1.output"],
    "hf-internal-testing/tiny-random-t5": ["decoder.block.1.layer.2"],
    "hf-internal-testing/tiny-xlm-roberta": ["roberta.encoder.layer.1.output"],
    "hf-internal-testing/tiny-random-gpt2": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-gpt_neo": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-gptj": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-CodeGenForCausalLM": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-FalconModel": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM": ["model.layers.1.mlp"],
    "hf-internal-testing/tiny-random-LlamaForCausalLM": ["model.layers.1.mlp"],
    "hf-internal-testing/tiny-random-MistralForCausalLM": ["model.layers.1.mlp"],
    "hf-internal-testing/tiny-random-Starcoder2ForCausalLM": ["model.layers.1.mlp"],
}

# A small subset to run on CI:
CI_MODEL_LOADERS = [
    "hf-internal-testing/tiny-random-bert",
    "hf-internal-testing/tiny-random-gpt2",
    "hf-internal-testing/tiny-random-roberta",
    "hf-internal-testing/tiny-random-t5",
]

STRATEGIES = [
    ModelWithSplitPoints.activation_granularities.ALL,
    ModelWithSplitPoints.activation_granularities.CLS_TOKEN,
    ModelWithSplitPoints.activation_granularities.ALL_TOKENS,
    ModelWithSplitPoints.activation_granularities.TOKEN,
    ModelWithSplitPoints.activation_granularities.WORD,
    ModelWithSplitPoints.activation_granularities.SENTENCE,
    ModelWithSplitPoints.activation_granularities.SAMPLE,
]


@pytest.mark.parametrize("model_name", CI_MODEL_LOADERS)
def test_activations_short(model_name, huge_text: list[str]):
    evaluate_activations(model_name, huge_text)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [k for k in ALL_MODEL_LOADERS.keys() if k not in CI_MODEL_LOADERS])
def test_activations_long(model_name, huge_text: list[str]):
    evaluate_activations(model_name, huge_text)


def evaluate_activations(model_name, huge_text: list[str]):
    """Tests all combinations of models and loaders with an attribution method"""

    model = ALL_MODEL_LOADERS[model_name].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=ALL_MODEL_SPLIT_POINTS[model_name],
        model_autoclass=ALL_MODEL_LOADERS[model_name],
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
    )

    for strategy in STRATEGIES:
        if (
            ALL_MODEL_LOADERS[model_name] != AutoModelForSequenceClassification
            and strategy == ModelWithSplitPoints.activation_granularities.CLS_TOKEN
        ):
            # CLS_TOKEN is only supported for sequence classification models
            continue
        splitted_model.get_activations(huge_text, activation_granularity=strategy)


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM

    sentences = [
        "Interpreto is the latin for 'to interpret'. But it also sounds like a spell from the Harry Potter books.",
        "Interpreto is magical",
        "Testing interpreto",
    ]

    splitted_encoder_ml = ModelWithSplitPoints(
        "bert-base-uncased",
        split_points=["bert.encoder.layer.2.output"],
        model_autoclass=AutoModelForSequenceClassification,  # type: ignore
        device_map="auto",
        batch_size=4,
    )
    multi_split_model = ModelWithSplitPoints(
        "bert-base-uncased",
        split_points=[
            "cls.predictions.transform.LayerNorm",
            "bert.encoder.layer.1.output",
            "bert.encoder.layer.3.attention.self.query",
        ],
        model_autoclass=AutoModelForMaskedLM,  # type: ignore
        device_map="cuda",
        batch_size=4,
    )

    bert_model = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert")
    bert_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    gpt2_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

    test_order_split_points(multi_split_model)
    test_loading_possibilities(bert_model, bert_tokenizer, gpt2_model, gpt2_tokenizer)
    test_activation_equivalence_batched_text_token_inputs(multi_split_model)
    test_get_activations_selection_strategies(splitted_encoder_ml, sentences)
    test_batching(splitted_encoder_ml, sentences * 10, ModelWithSplitPoints.activation_granularities.CLS_TOKEN)
    evaluate_activations("hf-internal-testing/tiny-random-t5", sentences * 100)
