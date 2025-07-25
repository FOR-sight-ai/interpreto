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

import itertools

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
    "bert.encoder.layer.1",
    "bert.encoder.layer.3.attention.self.query",
]

BERT_SPLIT_POINTS_SORTED = [
    "bert.encoder.layer.1",
    "bert.encoder.layer.3.attention.self.query",
    "cls.predictions.transform.LayerNorm",
]

GRANULARITIES = [
    ActivationGranularity.ALL,
    ActivationGranularity.ALL_TOKENS,
    ActivationGranularity.CLS_TOKEN,
    ActivationGranularity.TOKEN,
    ActivationGranularity.WORD,
    ActivationGranularity.SENTENCE,
    ActivationGranularity.SAMPLE,
]

AGGREGATIONS = [
    ModelWithSplitPoints.aggregation_strategies.MEAN,
    ModelWithSplitPoints.aggregation_strategies.SUM,
    ModelWithSplitPoints.aggregation_strategies.MAX,
    ModelWithSplitPoints.aggregation_strategies.SIGNED_MAX,
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


def test_pad_and_concat():
    """Validate ``pad_and_concat`` left and right padding."""
    tensors = [
        torch.zeros(1, 2, 3),
        torch.ones(1, 3, 3),
    ]
    out_right = ModelWithSplitPoints.pad_and_concat(tensors, "right", 0.5)
    out_left = ModelWithSplitPoints.pad_and_concat(tensors, "left", -1.0)

    assert out_right.shape == (2, 3, 3)
    assert out_right[0, -1].tolist() == [0.5, 0.5, 0.5]
    assert out_left.shape == (2, 3, 3)
    assert out_left[0, 0].tolist() == [-1.0, -1.0, -1.0]


def test_manage_output_tuple():
    """Ensure ``_manage_output_tuple`` extracts the 3-D tensor from a tuple."""
    model = ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_points=["bert.encoder.layer.1.output"],
        model_autoclass=AutoModelForSequenceClassification,  # type: ignore
    )
    tensor = torch.zeros(1, 2, 3)
    other = torch.zeros(1, 2)
    out = model._manage_output_tuple((other, tensor), "dummy")  # type: ignore
    assert out.shape == tensor.shape
    assert model.output_tuple_index == 1

    with pytest.raises(TypeError):
        model._manage_output_tuple(42, "dummy")  # type: ignore


def test_get_split_activations(splitted_encoder_ml: ModelWithSplitPoints, sentences: list[str]):
    """Test activation extraction for a specific split."""
    acts = splitted_encoder_ml.get_activations(sentences)
    split = splitted_encoder_ml.split_points[0]
    extracted = splitted_encoder_ml.get_split_activations(acts, split)
    assert torch.equal(extracted, acts[split])

    with pytest.raises(ValueError):
        splitted_encoder_ml.get_split_activations({}, "unknown")  # type: ignore

    with pytest.raises(TypeError):
        splitted_encoder_ml.get_split_activations(42)  # type: ignore


def test_get_latent_shape(splitted_encoder_ml: ModelWithSplitPoints, sentences: list[str]):
    """Shapes returned by ``get_latent_shape`` match activation shapes."""
    shapes = splitted_encoder_ml.get_latent_shape(sentences)
    acts = splitted_encoder_ml.get_activations(sentences, activation_granularity=ActivationGranularity.ALL)
    for sp in splitted_encoder_ml.split_points:
        assert shapes[sp] == acts[sp].shape


def test_activation_selection_and_reintegration_with_bert(bert_model, bert_tokenizer, sentences):
    activation_selection_and_reintegration(bert_model, bert_tokenizer, "bert.encoder.layer.1.output", sentences)


def test_activation_selection_and_reintegration_with_gpt2(gpt2_model, gpt2_tokenizer, sentences):
    activation_selection_and_reintegration(gpt2_model, gpt2_tokenizer, "transformer.h.1.mlp", sentences)


def activation_selection_and_reintegration(model, tokenizer, split_point, sentences):
    """
    Test that the selection then reintegration of raw activations are coherent.

    Raw activations have a shape (n, l, d) where n is the batch size, l is the sequence length, and d is the hidden dimension.

    Selected activations have a shape (ng, d) where ng is the number of granularity levels.
    Except for the `ALL` activation granularity which dos not impact the shape.

    Reintegrated activations should match back the raw activations.

    We also reselect after reintegration to ensure that the selection is idempotent.
    """
    # ---------------------------
    # Compute initial activations
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    mwsp = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=[split_point],
        model_autoclass=type(model),
        batch_size=2,
    )
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    activations = mwsp.get_activations(tokens, activation_granularity=ActivationGranularity.ALL)[split_point]

    # -----------------------------------------------------------
    # Define expected shapes for the different granularity levels
    batch, seq_len = tokens["input_ids"].shape
    hidden = mwsp._model.config.hidden_size
    total_token_len = sum(len(idx) for idx in Granularity.get_indices(tokens, Granularity.TOKEN, tokenizer))
    total_word_len = sum(len(idx) for idx in Granularity.get_indices(tokens, Granularity.WORD, tokenizer))

    expected = {
        ActivationGranularity.ALL: (batch, seq_len, hidden),
        ActivationGranularity.CLS_TOKEN: (batch, hidden),
        # ActivationGranularity.ALL_TOKENS: cannot be tested as the output shape depends on the batch size
        ActivationGranularity.TOKEN: (total_token_len, hidden),
        ActivationGranularity.WORD: (total_word_len, hidden),
        ActivationGranularity.SENTENCE: (len(sentences) + 1, hidden),
        ActivationGranularity.SAMPLE: (batch, hidden),
    }

    # -------------------------------------------------------
    # List all plausible granularity/aggregation combinations
    granularities_with_aggregations = list(
        itertools.product(
            (ActivationGranularity.WORD, ActivationGranularity.SENTENCE),
            AGGREGATIONS,
        )
    )
    granularities_without_aggregations = [
        (ActivationGranularity.ALL, None),
        (ActivationGranularity.ALL_TOKENS, None),
        (ActivationGranularity.TOKEN, None),
    ]

    if model.__class__.__name__.endswith("ForSequenceClassification"):
        granularities_without_aggregations.append((ActivationGranularity.CLS_TOKEN, None))

    # --------------------------------------------------------------------------------
    # Test selection and reintegration for all combinations of granularity/aggregation
    for granularity, aggregation in granularities_without_aggregations + granularities_with_aggregations:
        # ------------------
        # Select activations
        selected_activations, indices = mwsp._apply_selection_strategy(
            inputs=tokens,
            activations=activations.clone(),
            activation_granularity=granularity,
            aggregation_strategy=aggregation,
        )
        # ensure that the shape of the selected activations matches the expected shape
        if granularity != ActivationGranularity.ALL_TOKENS:
            # the ALL_TOKENS granularity shape depends on the batch size and cannot be tested
            assert selected_activations.shape == expected[granularity]

        # -----------------------
        # Reintegrate activations
        reconstructed_activations = mwsp._reintegrate_selected_activations(
            activations.clone(),
            selected_activations,
            activation_granularity=granularity,
            aggregation_strategy=aggregation,
            granularity_indices=indices,
        )
        # ensure that the shape of the reintegrated activations matches the initial shape
        assert reconstructed_activations.shape == activations.shape
        # ensure that the reintegrated activations match the initial activations
        if aggregation is None:
            # if activations are aggregated, we cannot get back to the exact original activations
            assert torch.allclose(reconstructed_activations, activations, atol=1e-5)

        # -----------------------
        # Reselect activations to ensure verify that the aggregation is idempotent
        reselected_activations, indices = mwsp._apply_selection_strategy(
            inputs=tokens,
            activations=reconstructed_activations.clone(),
            activation_granularity=granularity,
            aggregation_strategy=aggregation,
        )
        # ensure that the shape of the reselected activations matches the selected activations
        assert reselected_activations.shape == selected_activations.shape
        # ensure that the reselected activations match the selected activations
        assert torch.allclose(reselected_activations, selected_activations, atol=1e-5)


def test_get_activation_and_gradient_with_bert(bert_model, bert_tokenizer, sentences):
    activation_selection_and_reintegration(bert_model, bert_tokenizer, "bert.encoder.layer.1.output", sentences)


def test_get_activation_and_gradient_with_gpt2(gpt2_model, gpt2_tokenizer, sentences):
    activation_selection_and_reintegration(gpt2_model, gpt2_tokenizer, "transformer.h.1.mlp", sentences)


def get_activation_and_gradient(model, tokenizer, split_point, sentences):
    """
    Test that the `get_activations` and `get_concepts_output_gradients` methods return the expected shapes.
    """
    # ----------------------------
    # Add a padding token for gpt2
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # --------------------------------------------------------
    # Setup the model with split points, tokenizer, and tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mwsp = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=[split_point],
        model_autoclass=type(model),
        batch_size=2,
        device_map=device,
    )
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )

    # -----------------------------------------------------------
    # Define expected shapes for the different granularity levels
    batch, seq_len = tokens["input_ids"].shape
    hidden = mwsp._model.config.hidden_size
    total_token_len = sum(len(idx) for idx in Granularity.get_indices(tokens, Granularity.TOKEN, tokenizer))
    total_word_len = sum(len(idx) for idx in Granularity.get_indices(tokens, Granularity.WORD, tokenizer))

    granularities_expected_shapes = {
        # ActivationGranularity.ALL_TOKENS: cannot be tested as the output shape depends on the batch size
        ActivationGranularity.TOKEN: (total_token_len, hidden),
        ActivationGranularity.WORD: (total_word_len, hidden),
        ActivationGranularity.SENTENCE: (len(sentences) + 1, hidden),
        ActivationGranularity.SAMPLE: (batch, hidden),
    }

    if model.__class__.__name__.endswith("ForSequenceClassification"):
        granularities_expected_shapes[ActivationGranularity.CLS_TOKEN] = (batch, hidden)

    # ----------------------------------------------
    # Define a concept encoder/decoder weight matrix
    # We want W@W.T to be approximately the identity matrix
    nb_concepts = 2 * hidden
    initial = torch.randn(nb_concepts, hidden)
    decoder_weights = torch.linalg.qr(initial)[0].to(device)
    encoder_weights = decoder_weights.T

    # ---------------------------------------------------------------------------------
    # Test get_activations and get_concepts_output_gradients for all granularity levels
    for granularity, expected_shape in granularities_expected_shapes.items():
        # ---------------
        # Get activations
        activations = mwsp.get_activations(sentences, activation_granularity=granularity)[split_point]
        assert activations.shape == expected_shape

        if granularity in [ActivationGranularity.ALL, ActivationGranularity.SAMPLE]:
            # ALL and SAMPLE granularities are not compatible with gradients
            continue

        if granularity == ActivationGranularity.CLS_TOKEN:
            indices_list = [[[0]]] * len(sentences)  # type: ignore
        else:
            indices_list = Granularity.get_indices(tokens, granularity.value, tokenizer)  # type: ignore

        # -------------
        # Get gradients
        if granularity in [ActivationGranularity.WORD, ActivationGranularity.SENTENCE]:
            aggregations = AGGREGATIONS
        else:
            aggregations = [None]
        for aggregation in aggregations:
            grads_list = mwsp.get_concepts_output_gradients(
                sentences,
                encode_activations=lambda x: x @ encoder_weights,
                decode_activations=lambda x: x @ decoder_weights,
                activation_granularity=granularity,
                aggregation_strategy=aggregation,
                targets=None,
            )
            assert len(grads_list) == len(sentences)  # there should be as many gradients as inputs
            for grads, indices in zip(grads_list, indices_list, strict=True):
                # we expect the shape of the gradients to be (t, g, c)
                # with t the number of targets, ng the number of granularity elements concatenated, and c the number of concepts
                assert grads.shape[1] == len(indices)  # number of granularity elements
                assert grads.shape[2] == nb_concepts  # number of concepts


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
def test_index_by_layer_idx(multi_split_model: ModelWithSplitPoints):
    """Test indexing by layer idx"""
    split_points_with_layer_idx: list = list(BERT_SPLIT_POINTS)
    split_points_with_layer_idx[1] = 1  # instead of bert.encoder.layer.1
    multi_split_model.split_points = split_points_with_layer_idx
    assert multi_split_model.split_points == BERT_SPLIT_POINTS_SORTED, (
        f"Failed for split_points: {BERT_SPLIT_POINTS}\n"
        f"Expected: {BERT_SPLIT_POINTS_SORTED}\n"
        f"Got:      {multi_split_model.split_points}"
    )


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
def test_activations_and_gradients_on_models_short(model_name, sentences: list[str]):
    evaluate_activations_and_gradients(model_name, sentences)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [k for k in ALL_MODEL_LOADERS.keys() if k not in CI_MODEL_LOADERS])
def test_activations_and_gradients_on_models_long(model_name, sentences: list[str]):
    evaluate_activations_and_gradients(model_name, sentences)


def evaluate_activations_and_gradients(model_name, sentences: list[str]):
    """Tests model with split points get activations and gradients with a large variety of models"""

    model = ALL_MODEL_LOADERS[model_name].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=ALL_MODEL_SPLIT_POINTS[model_name],
        model_autoclass=ALL_MODEL_LOADERS[model_name],
        device_map=device,
        batch_size=8,
    )

    # ----------------------------------------------
    # Define a concept encoder/decoder weight matrix
    # We want W@W.T to be approximately the identity matrix
    hidden = splitted_model._model.config.hidden_size
    nb_concepts = 2 * hidden
    initial = torch.randn(nb_concepts, hidden)
    decoder_weights = torch.linalg.qr(initial)[0].to(device)
    encoder_weights = decoder_weights.T

    for strategy in STRATEGIES:
        if (
            ALL_MODEL_LOADERS[model_name] != AutoModelForSequenceClassification
            and strategy == ModelWithSplitPoints.activation_granularities.CLS_TOKEN
        ):
            # CLS_TOKEN is only supported for sequence classification models
            continue
        splitted_model.get_activations(sentences, activation_granularity=strategy)

        if strategy in [
            ModelWithSplitPoints.activation_granularities.ALL,
            ModelWithSplitPoints.activation_granularities.SAMPLE,
        ]:
            # ALL and SAMPLE granularities are not compatible with gradients
            continue

        splitted_model.get_concepts_output_gradients(
            sentences,
            encode_activations=lambda x: x @ encoder_weights,
            decode_activations=lambda x: x @ decoder_weights,
            activation_granularity=strategy,
            targets=[0],
        )


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
            "bert.encoder.layer.1",
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
    test_batching(splitted_encoder_ml, sentences * 10, ModelWithSplitPoints.activation_granularities.CLS_TOKEN)
    evaluate_activations_and_gradients("hf-internal-testing/tiny-random-t5", sentences * 100)
    get_activation_and_gradient(bert_model, bert_tokenizer, "bert.encoder.layer.1.output", sentences)
    get_activation_and_gradient(gpt2_model, gpt2_tokenizer, "transformer.h.1.mlp", sentences)
    activation_selection_and_reintegration(bert_model, bert_tokenizer, "bert.encoder.layer.1.output", sentences)
    activation_selection_and_reintegration(gpt2_model, gpt2_tokenizer, "transformer.h.1.mlp", sentences)
    test_index_by_layer_idx(multi_split_model)
