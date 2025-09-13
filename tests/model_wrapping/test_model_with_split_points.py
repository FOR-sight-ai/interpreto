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

from interpreto import Granularity
from interpreto import ModelWithSplitPoints as MWSP
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity, InitializationError

AG = MWSP.activation_granularities

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
    MWSP.aggregation_strategies.MEAN,
    MWSP.aggregation_strategies.SUM,
    MWSP.aggregation_strategies.MAX,
    MWSP.aggregation_strategies.SIGNED_MAX,
]


def test_order_split_points(multi_split_model: MWSP):
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
        MWSP(bert_model, "bert.encoder.layer.1")
    model_with_split_points = MWSP(bert_model, split_points="bert.encoder.layer.1", tokenizer=bert_tokenizer)
    assert model_with_split_points.split_points == ["bert.encoder.layer.1"], (
        f"split_points mismatch: got {model_with_split_points.split_points}, expected ['bert.encoder.layer.1']"
    )
    # Load model without split points
    model_without_split_points = MWSP(
        "bert-base-cased",
        automodel=AutoModelForMaskedLM,  # type: ignore
        split_points="bert.encoder.layer.1",
    )
    assert model_without_split_points.split_points == ["bert.encoder.layer.1"], (
        f"split_points mismatch: got {model_without_split_points.split_points}, expected ['bert.encoder.layer.1']"
    )

    # ----
    # GPT2
    # Load model with split points
    with pytest.raises(ValueError):  # tokenizer is not set
        MWSP(gpt2_model, "transformer.h.1")
    model_with_split_points = MWSP(gpt2_model, split_points="transformer.h.1", tokenizer=gpt2_tokenizer)
    assert model_with_split_points.split_points == ["transformer.h.1"], (
        f"split_points mismatch: got {model_with_split_points.split_points}, expected ['transformer.h.1']"
    )
    # Load model without split points
    model_without_split_points = MWSP(
        "gpt2",
        automodel=AutoModelForCausalLM,  # type: ignore
        split_points="transformer.h.1",
    )
    assert model_without_split_points.split_points == ["transformer.h.1"], (
        f"split_points mismatch: got {model_without_split_points.split_points}, expected ['transformer.h.1']"
    )

    with pytest.raises(InitializationError):
        # Model id with no auto class
        MWSP("gpt2", "transformer.h.1")


def test_pad_and_concat():
    """Validate ``pad_and_concat`` left and right padding."""
    tensors = [
        torch.zeros(1, 2, 3),
        torch.ones(1, 3, 3),
    ]
    out_right = MWSP._pad_and_concat(tensors, "right", 0.5)
    out_left = MWSP._pad_and_concat(tensors, "left", -1.0)

    assert out_right.shape == (2, 3, 3), "right padding shape mismatch"
    assert out_right[0, -1].tolist() == [0.5, 0.5, 0.5], "right padding values mismatch"
    assert out_left.shape == (2, 3, 3), "left padding shape mismatch"
    assert out_left[0, 0].tolist() == [-1.0, -1.0, -1.0], "left padding values mismatch"


def test_manage_output_tuple():
    """Ensure ``_manage_output_tuple`` extracts the 3-D tensor from a tuple."""
    model = MWSP(
        "hf-internal-testing/tiny-random-bert",
        split_points=["bert.encoder.layer.1.output"],
        automodel=AutoModelForSequenceClassification,  # type: ignore
    )
    tensor = torch.zeros(1, 2, 3)
    other = torch.zeros(1, 2)
    out = model._manage_output_tuple((other, tensor), "dummy")  # type: ignore
    assert out.shape == tensor.shape, (
        f"output tuple shape mismatch, got {out.shape}, expected {tensor.shape}. The 3D tensor should be extracted."
    )
    assert model.output_tuple_index == 1, "index should correspond to the 3D tensor"

    with pytest.raises(TypeError):
        model._manage_output_tuple(42, "dummy")  # type: ignore


def test_get_split_activations(splitted_encoder_ml: MWSP, sentences: list[str]):
    """Test activation extraction for a specific split."""
    acts = splitted_encoder_ml.get_activations(sentences, activation_granularity=AG.ALL)
    assert acts is not None, "get_activations returned None"
    split = splitted_encoder_ml.split_points[0]
    assert split in acts, f"Split '{split}' not found in activations dict"
    extracted = splitted_encoder_ml.get_split_activations(acts, split)
    assert extracted is not None, "get_split_activations returned None"
    assert torch.equal(extracted, acts[split]), "Extracted activations do not match activations dict at split"

    with pytest.raises(ValueError):
        splitted_encoder_ml.get_split_activations({}, "unknown")  # type: ignore

    with pytest.raises(TypeError):
        splitted_encoder_ml.get_split_activations(42)  # type: ignore


def test_get_latent_shape(splitted_encoder_ml: MWSP, sentences: list[str]):
    """Shapes returned by ``get_latent_shape`` match activation shapes."""
    shapes = splitted_encoder_ml.get_latent_shape(sentences)
    acts = splitted_encoder_ml.get_activations(sentences, activation_granularity=ActivationGranularity.ALL)
    assert shapes is not None, "get_latent_shape returned None"
    assert acts is not None, "get_activations returned None"
    for sp in splitted_encoder_ml.split_points:
        assert sp in shapes, f"Split '{sp}' missing in shapes"
        assert sp in acts, f"Split '{sp}' missing in activations"
        assert shapes[sp] == acts[sp].shape, (
            f"Latent shape mismatch for split '{sp}': shapes={shapes[sp]}, activations={acts[sp].shape}"
        )


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
    mwsp = MWSP(
        model,
        tokenizer=tokenizer,
        split_points=[split_point],
        automodel=type(model),
        batch_size=2,
    )
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    acts_dict = mwsp.get_activations(tokens, activation_granularity=ActivationGranularity.ALL)
    assert acts_dict is not None, "get_activations returned None"
    assert split_point in acts_dict, f"Split '{split_point}' not found in activations dict"
    activations = acts_dict[split_point]
    assert activations is not None, "Activations at split are None"

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
        # ---------------------------
        # Extract granularity indices
        granularity_indices = mwsp._get_granularity_indices(tokens, granularity)

        # ------------------
        # Select activations
        selected_activations = mwsp._apply_selection_strategy(
            activations=activations.clone(),
            granularity_indices=granularity_indices,
            activation_granularity=granularity,
            aggregation_strategy=aggregation,
        )
        assert selected_activations is not None, f"_apply_selection_strategy returned None for {granularity}"
        # ensure that the shape of the selected activations matches the expected shape
        if granularity != ActivationGranularity.ALL_TOKENS:
            # the ALL_TOKENS granularity shape depends on the batch size and cannot be tested
            assert selected_activations.shape == expected[granularity], (
                f"Selected shape mismatch for {granularity}: got {tuple(selected_activations.shape)}, "
                f"expected {expected[granularity]}"
            )

        # -----------------------
        # Reintegrate activations
        reconstructed_activations = mwsp._reintegrate_selected_activations(
            activations.clone(),
            selected_activations,
            activation_granularity=granularity,
            aggregation_strategy=aggregation,
            granularity_indices=granularity_indices,
        )
        assert reconstructed_activations is not None, (
            f"_reintegrate_selected_activations returned None, for {granularity}"
        )
        # ensure that the shape of the reintegrated activations matches the initial shape
        assert reconstructed_activations.shape == activations.shape, (
            f"Reintegrated shape mismatch: got {tuple(reconstructed_activations.shape)}, "
            f"expected {tuple(activations.shape)} for {granularity}"
        )
        # ensure that the reintegrated activations match the initial activations
        if aggregation is None:
            # if activations are aggregated, we cannot get back to the exact original activations
            assert torch.allclose(reconstructed_activations, activations, atol=1e-5), (
                "Reintegrated activations differ from original beyond tolerance for {granularity}"
            )

        # -----------------------
        # Reselect activations to ensure verify that the aggregation is idempotent
        reselected_activations = mwsp._apply_selection_strategy(
            activations=reconstructed_activations.clone(),
            granularity_indices=granularity_indices,
            activation_granularity=granularity,
            aggregation_strategy=aggregation,
        )
        assert reselected_activations is not None, (
            "_apply_selection_strategy (reselect) returned None for {granularity}"
        )
        # ensure that the shape of the reselected activations matches the selected activations
        assert reselected_activations.shape == selected_activations.shape, (
            f"Reselected shape mismatch: got {tuple(reselected_activations.shape)}, "
            f"expected {tuple(selected_activations.shape)}, for {granularity}"
        )
        # ensure that the reselected activations match the selected activations
        assert torch.allclose(reselected_activations, selected_activations, atol=1e-5), (
            "Reselected activations differ from initially selected beyond tolerance, for {granularity}"
        )


def test_get_activation_and_gradient_with_bert(bert_model, bert_tokenizer, sentences):
    activation_selection_and_reintegration(bert_model, bert_tokenizer, "bert.encoder.layer.1.output", sentences)


def test_get_activation_and_gradient_with_gpt2(gpt2_model, gpt2_tokenizer, sentences):
    activation_selection_and_reintegration(gpt2_model, gpt2_tokenizer, "transformer.h.1.mlp", sentences)


def get_activation_and_gradient(model, tokenizer, split_point, sentences):
    """
    Test that the `get_activations` and `_get_concept_output_gradients` methods return the expected shapes.
    """
    # ----------------------------
    # Add a padding token for gpt2
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # --------------------------------------------------------
    # Setup the model with split points, tokenizer, and tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mwsp = MWSP(
        model,
        tokenizer=tokenizer,
        split_points=[split_point],
        automodel=type(model),
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
    # Test get_activations and _get_concept_output_gradients for all granularity levels
    for granularity, expected_shape in granularities_expected_shapes.items():
        # ---------------
        # Get activations
        include_predicted_classes = isinstance(mwsp._model, AutoModelForSequenceClassification)
        activations_dict = mwsp.get_activations(
            sentences, activation_granularity=granularity, include_predicted_classes=include_predicted_classes
        )
        assert activations_dict is not None, f"get_activations returned None for {granularity}"
        assert split_point in activations_dict, f"Split '{split_point}' missing in activations dict for {granularity}"
        activations = activations_dict[split_point]
        assert activations is not None, f"Activations at split are None for {granularity}"
        assert activations.shape == expected_shape, (
            f"Activations shape mismatch for {granularity}: got {tuple(activations.shape)}, expected {expected_shape} for {granularity}"
        )
        if include_predicted_classes:
            predictions = activations_dict["predictions"]
            assert predictions is not None, f"Predictions entry is None for {granularity}"
            assert predictions.shape[0] == expected_shape[0], (
                f"Predictions batch mismatch: got {predictions.shape[0]}, expected {expected_shape[0]} for {granularity}"
            )

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
            grads_list = mwsp._get_concept_output_gradients(
                sentences,
                encode_activations=lambda x: x @ encoder_weights,
                decode_concepts=lambda x: x @ decoder_weights,
                activation_granularity=granularity,
                aggregation_strategy=aggregation,
                targets=None,
            )
            assert grads_list is not None, (
                f"_get_concept_output_gradients returned None for {granularity} and {aggregation}"
            )
            assert len(grads_list) == len(sentences), (
                f"Gradients list length mismatch: got {len(grads_list)}, expected {len(sentences)} for {granularity} and {aggregation}"
            )  # there should be as many gradients as inputs
            for grads, indices in zip(grads_list, indices_list, strict=True):
                assert grads is not None, f"A gradients tensor is None for {granularity} and {aggregation}"
                # we expect the shape of the gradients to be (t, g, c)
                # with t the number of targets, ng the number of granularity elements concatenated, and c the number of concepts
                assert grads.shape[1] == len(indices), (
                    f"Gradient granularity dimension mismatch: got {grads.shape[1]}, expected {len(indices)} for {granularity} and {aggregation}"
                )  # number of granularity elements
                assert grads.shape[2] == nb_concepts, (
                    f"Gradient concepts dimension mismatch: got {grads.shape[2]}, expected {nb_concepts} for {granularity} and {aggregation}"
                )  # number of concepts


def test_activation_equivalence_batched_text_token_inputs(multi_split_model: MWSP):
    """
    Test the equivalence of activations for text and token inputs
    """
    multi_split_model.split_points = BERT_SPLIT_POINTS  # type: ignore
    inputs_str = ["Hello, my dog is cute", "The cat is on the [MASK]"]
    inputs_tensor = multi_split_model.tokenizer(
        inputs_str, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )

    activations_str = multi_split_model.get_activations(inputs_str, activation_granularity=AG.ALL)
    activations_tensor = multi_split_model.get_activations(inputs_tensor, activation_granularity=AG.ALL)
    assert activations_str is not None and activations_tensor is not None, "get_activations returned None"

    for k in activations_str.keys():
        assert k in activations_tensor, f"Key '{k}' missing in tensor activations"
        assert torch.allclose(activations_str[k], activations_tensor[k]), (
            f"Mismatch between text and token activations for key '{k}'"
        )  # type: ignore


@pytest.mark.parametrize(
    "strategy",
    [
        # AG.ALL,
        AG.CLS_TOKEN,
        AG.ALL_TOKENS,
        AG.TOKEN,
        AG.WORD,
        AG.SENTENCE,
        AG.SAMPLE,
    ],
)
def test_batching(splitted_encoder_ml: MWSP, huge_text: list[str], strategy: ActivationGranularity):
    splitted_encoder_ml.get_activations(huge_text, activation_granularity=strategy)


def test_index_by_layer_idx(multi_split_model: MWSP):
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
    "hf-internal-testing/tiny-random-bert": AutoModelForMaskedLM,
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
    "hf-internal-testing/tiny-random-albert": [1],
    "hf-internal-testing/tiny-random-bart": ["model.decoder.layers.1.fc2"],
    "hf-internal-testing/tiny-random-bert": ["bert.encoder.layer.1.output"],
    # "hf-internal-testing/tiny-random-DebertaV2Model": ["todo"],
    "hf-internal-testing/tiny-random-distilbert": ["distilbert.transformer.layer.1.ffn"],
    "hf-internal-testing/tiny-random-ElectraModel": ["electra.encoder.layer.1.output"],
    "hf-internal-testing/tiny-random-roberta": [1],
    "hf-internal-testing/tiny-random-t5": ["decoder.block.1.layer.2"],
    "hf-internal-testing/tiny-xlm-roberta": ["roberta.encoder.layer.1.output"],
    "hf-internal-testing/tiny-random-gpt2": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-gpt_neo": [1],
    "hf-internal-testing/tiny-random-gptj": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-CodeGenForCausalLM": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-FalconModel": ["transformer.h.1.mlp"],
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM": ["model.layers.1.mlp"],
    "hf-internal-testing/tiny-random-LlamaForCausalLM": ["model.layers.1.mlp"],
    "hf-internal-testing/tiny-random-MistralForCausalLM": [1],
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
    AG.ALL,
    AG.CLS_TOKEN,
    AG.ALL_TOKENS,
    AG.TOKEN,
    AG.WORD,
    AG.SENTENCE,
    AG.SAMPLE,
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
    splitted_model = MWSP(
        model,
        tokenizer=tokenizer,
        split_points=ALL_MODEL_SPLIT_POINTS[model_name],
        automodel=ALL_MODEL_LOADERS[model_name],
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
        if ALL_MODEL_LOADERS[model_name] != AutoModelForSequenceClassification and strategy == AG.CLS_TOKEN:
            # CLS_TOKEN is only supported for sequence classification models
            continue
        acts = splitted_model.get_activations(sentences, activation_granularity=strategy)
        assert acts is not None, f"get_activations returned None for granularity {strategy} on {model_name}"

        if strategy in [
            AG.ALL,
            AG.SAMPLE,
        ]:
            # ALL and SAMPLE granularities are not compatible with gradients
            continue

        grads = splitted_model._get_concept_output_gradients(
            sentences,
            encode_activations=lambda x: x @ encoder_weights,
            decode_concepts=lambda x: x @ decoder_weights,
            activation_granularity=strategy,
            targets=[0],
        )
        assert grads is not None, (
            f"_get_concept_output_gradients returned None for granularity {strategy} on {model_name}"
        )


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM

    sentences = [
        "Interpreto is the latin for 'to interpret'. But it also sounds like a spell from the Harry Potter books.",
        "Interpreto is magical",
        "Testing interpreto",
    ]

    splitted_encoder_ml = MWSP(
        "gpt2",
        split_points=2,
        automodel=AutoModelForCausalLM,  # type: ignore
        device_map="auto",
        batch_size=4,
    )

    splitted_encoder_ml = MWSP(
        "bert-base-uncased",
        split_points=["bert.encoder.layer.2.output"],
        automodel=AutoModelForSequenceClassification,  # type: ignore
        device_map="cuda",
        batch_size=4,
    )
    multi_split_model = MWSP(
        "bert-base-uncased",
        split_points=[
            "cls.predictions.transform.LayerNorm",
            "bert.encoder.layer.1",
            "bert.encoder.layer.3.attention.self.query",
        ],
        automodel=AutoModelForMaskedLM,  # type: ignore
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
    test_batching(splitted_encoder_ml, sentences * 10, AG.CLS_TOKEN)
    evaluate_activations_and_gradients("hf-internal-testing/tiny-random-t5", sentences * 100)
    get_activation_and_gradient(bert_model, bert_tokenizer, "bert.encoder.layer.1.output", sentences)
    get_activation_and_gradient(gpt2_model, gpt2_tokenizer, "transformer.h.1.mlp", sentences)
    activation_selection_and_reintegration(bert_model, bert_tokenizer, "bert.encoder.layer.1.output", sentences)
    activation_selection_and_reintegration(gpt2_model, gpt2_tokenizer, "transformer.h.1.mlp", sentences)
    test_index_by_layer_idx(multi_split_model)
