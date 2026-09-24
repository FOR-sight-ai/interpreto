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

"""Tests for the lens methods."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
)

import interpreto.visualizations.lens as lens_visualizations
from interpreto import LogitLens, ModelWithSplitPoints, TunedLens, plot_lens
from interpreto.lens import LogitLens as LensLogitLens
from interpreto.lens import TunedLens as LensTunedLens

DEVICE = "cpu"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CAUSAL_LANGUAGE_MODEL_CASES = [
    pytest.param(
        "hf-internal-testing/tiny-random-gpt2",
        "transformer.h.1",
        id="tiny-random-gpt2",
    ),
    pytest.param(
        "hf-internal-testing/tiny-random-gpt_neo",
        "transformer.h.1",
        marks=pytest.mark.slow,
        id="tiny-random-gpt_neo",
    ),
]

MASKED_LANGUAGE_MODEL_CASES = [
    pytest.param(
        "hf-internal-testing/tiny-random-bert",
        "bert.encoder.layer.1.output",
        id="tiny-random-bert",
    ),
]

POOLER_CLASSIFICATION_CASES = [
    pytest.param(
        "hf-internal-testing/tiny-random-bert",
        "bert.encoder.layer.1.output",
        id="tiny-random-bert",
    ),
]

SEQUENCE_AWARE_CLASSIFICATION_CASES = [
    pytest.param(
        "hf-internal-testing/tiny-random-roberta",
        "roberta.encoder.layer.1.output",
        id="tiny-random-roberta",
    ),
    pytest.param(
        "hf-internal-testing/tiny-xlm-roberta",
        "roberta.encoder.layer.1.output",
        marks=pytest.mark.slow,
        id="tiny-xlm-roberta",
    ),
]

TUNED_LANGUAGE_MODEL_CASES = [
    pytest.param(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
        3,
        id="causal-language-model",
    ),
    pytest.param(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForMaskedLM,
        "bert.encoder.layer.1.output",
        4,
        id="masked-language-model",
    ),
]

TUNED_SEQUENCE_CLASSIFICATION_CASES = [
    pytest.param(
        "hf-internal-testing/tiny-random-bert",
        "bert.encoder.layer.1.output",
        id="pooler-based-classification",
    ),
    pytest.param(
        "hf-internal-testing/tiny-random-roberta",
        "roberta.encoder.layer.1.output",
        marks=pytest.mark.slow,
        id="sequence-aware-classification",
    ),
]


def _build_model_with_split_points(model_name, automodel, split_point):
    model = automodel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_point=split_point,
        batch_size=2,
        device_map=DEVICE,
    )


def _tokenize_texts(tokenizer, texts):
    text_batch = [texts] if isinstance(texts, str) else texts
    return tokenizer(
        text_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


def _get_model_module(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    current_module = model
    for path_element in module_name.split("."):
        if path_element.isdigit():
            current_module = current_module[int(path_element)]
        else:
            current_module = getattr(current_module, path_element)
    return current_module


def _capture_displayed_html(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    displayed_html = []

    monkeypatch.setattr(lens_visualizations, "HTML", lambda html: html)
    monkeypatch.setattr(lens_visualizations, "display", displayed_html.append)
    return displayed_html


def _assert_metrics_close(metrics, expected_metrics):
    assert metrics.keys() == expected_metrics.keys()
    for metric_name, expected_value in expected_metrics.items():
        if isinstance(expected_value, float):
            assert metrics[metric_name] == pytest.approx(expected_value, rel=1e-6, abs=1e-7)
        else:
            assert metrics[metric_name] == expected_value


def test_lens_exports_are_available():
    assert LogitLens is LensLogitLens
    assert TunedLens is LensTunedLens
    assert plot_lens is lens_visualizations.plot_lens


def test_lens_api_uses_the_wrapped_split_point():
    for method in [LogitLens.explain, LogitLens.metrics, LogitLens.__call__]:
        assert "split_point" not in inspect.signature(method).parameters

    assert "split_point" not in inspect.signature(TunedLens.__init__).parameters
    assert "split_point" not in inspect.signature(TunedLens.fit).parameters


@pytest.mark.parametrize(("model_name", "split_point"), CAUSAL_LANGUAGE_MODEL_CASES)
def test_logit_lens_supports_causal_language_models(model_name, split_point, sentences):
    model_with_split_points = _build_model_with_split_points(
        model_name,
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    explanations = lens.explain(sentences[1])
    metrics = lens.metrics(sentences[1])[split_point]
    layer_output = explanations[split_point]

    assert set(layer_output) == {"top_indices", "top_scores"}
    assert layer_output["top_indices"].shape == layer_output["top_scores"].shape
    assert layer_output["top_indices"].shape[0] == 1
    assert layer_output["top_indices"].shape[-1] == 3
    assert lens.model_head is not None
    assert lens.model_pre_head is not None
    assert lens.split_point == split_point
    assert metrics["target_source"] == "next_token"
    assert metrics["nb_evaluated_elements"] > 0
    assert 0.0 <= metrics["mean_target_score"] <= 1.0
    assert 0.0 <= metrics["mean_max_score"] <= 1.0
    assert metrics["target_cross_entropy"] >= 0.0
    assert metrics["perplexity"] >= 1.0
    assert metrics["kl_divergence_to_model"] > -1e-6


def test_logit_lens_resolves_the_standard_gpt2_projection(sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    explanations = lens.explain(sentences[1])

    assert lens.head_name == "lm_head"
    assert lens.pre_head_name == "transformer.ln_f"
    assert explanations["transformer.h.1"]["top_indices"].shape[-1] == 3


def test_logit_lens_synchronizes_existing_padding_ids():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    tokenizer_pad_token_id = model_with_split_points.tokenizer.pad_token_id

    LogitLens(model_with_split_points, top_k=3)

    assert model_with_split_points._model.config.pad_token_id == tokenizer_pad_token_id
    assert model_with_split_points._model.generation_config.pad_token_id == tokenizer_pad_token_id


def test_logit_lens_causal_outputs_are_batch_companion_invariant():
    split_point = "transformer.h.1"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    short_text = "Short input."
    long_text = "This is a substantially longer companion input for the same decoder batch."
    single_output = lens.explain(short_text)[split_point]
    batched_output = lens.explain([long_text, short_text])[split_point]
    short_length = single_output["top_indices"].shape[1]

    assert torch.equal(batched_output["top_indices"][1, :short_length], single_output["top_indices"][0])
    assert torch.allclose(
        batched_output["top_scores"][1, :short_length],
        single_output["top_scores"][0],
        atol=1e-6,
    )


def test_logit_lens_rejects_left_padded_causal_batch_encodings():
    split_point = "transformer.h.1"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    short_text = "Short input."
    long_text = "This is a substantially longer companion input for the same decoder batch."
    original_padding_side = lens.tokenizer.padding_side
    try:
        lens.tokenizer.padding_side = "left"
        batched_inputs = _tokenize_texts(lens.tokenizer, [long_text, short_text])
    finally:
        lens.tokenizer.padding_side = original_padding_side

    with pytest.raises(ValueError, match="right padding"):
        lens.explain(batched_inputs)


@pytest.mark.parametrize(("model_name", "split_point"), MASKED_LANGUAGE_MODEL_CASES)
def test_logit_lens_supports_masked_language_models(model_name, split_point, sentences):
    model_with_split_points = _build_model_with_split_points(
        model_name,
        AutoModelForMaskedLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=4)

    explanations = lens.explain(sentences[1])
    metrics = lens.metrics(sentences[1])[split_point]
    layer_output = explanations[split_point]

    assert set(layer_output) == {"top_indices", "top_scores"}
    assert layer_output["top_indices"].shape == layer_output["top_scores"].shape
    assert layer_output["top_indices"].shape[0] == 1
    assert layer_output["top_indices"].shape[-1] == 4
    assert "perplexity" not in metrics


def test_logit_lens_evaluates_provided_targets_at_masked_positions():
    split_point = "bert.encoder.layer.1.output"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForMaskedLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=4)
    model_inputs = _tokenize_texts(lens.tokenizer, "Interpreto is useful.")
    target_position = 1
    targets = torch.full_like(model_inputs["input_ids"], -100)
    targets[0, target_position] = model_inputs["input_ids"][0, target_position]
    model_inputs["input_ids"][0, target_position] = lens.tokenizer.mask_token_id

    metrics = lens.metrics(model_inputs, targets=targets)[split_point]

    assert metrics["target_source"] == "provided_targets"
    assert metrics["nb_evaluated_elements"] == 1


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(torch.tensor([[1.5]]), id="floating-token-id"),
        pytest.param(torch.tensor([[True]]), id="boolean-token-id"),
    ],
)
def test_logit_lens_rejects_non_integer_language_model_targets(targets):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForMaskedLM,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=4)
    model_inputs = BatchEncoding(
        {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
    )

    with pytest.raises(TypeError, match="integer token ids"):
        lens.metrics(model_inputs, targets=targets)


@pytest.mark.parametrize(("model_name", "split_point"), POOLER_CLASSIFICATION_CASES)
def test_logit_lens_supports_sequence_classification_with_internal_pooler(
    model_name,
    split_point,
    sentences,
):
    model_with_split_points = _build_model_with_split_points(
        model_name,
        AutoModelForSequenceClassification,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=2)

    explanations = lens.explain(sentences[:2])
    metrics = lens.metrics(sentences[:2], targets=[1, 0])[split_point]
    layer_output = explanations[split_point]

    assert set(layer_output) == {"top_indices", "top_scores"}
    assert layer_output["top_indices"].shape == (2, 2)
    assert layer_output["top_scores"].shape == (2, 2)
    assert lens.head_name == "classifier"
    assert lens.pre_head_name == "bert.pooler"
    assert lens.model_head is not None
    assert lens.model_pre_head is not None
    assert metrics["target_source"] == "provided_targets"
    assert metrics["nb_evaluated_elements"] == 2
    assert 0.0 <= metrics["mean_target_score"] <= 1.0
    assert metrics["target_cross_entropy"] >= 0.0
    assert 0.0 <= metrics["target_accuracy"] <= 1.0
    assert metrics["kl_divergence_to_model"] > -1e-6


@pytest.mark.parametrize(("model_name", "split_point"), SEQUENCE_AWARE_CLASSIFICATION_CASES)
def test_logit_lens_supports_sequence_aware_classification_heads(model_name, split_point, sentences):
    model_with_split_points = _build_model_with_split_points(
        model_name,
        AutoModelForSequenceClassification,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=2)

    explanations = lens.explain(sentences[1])
    layer_output = explanations[split_point]

    assert set(layer_output) == {"top_indices", "top_scores"}
    assert layer_output["top_indices"].shape == (1, 2)
    assert layer_output["top_scores"].shape == (1, 2)
    assert lens.pooling_strategy is None
    assert lens.model_pre_head is None


def test_logit_lens_preserves_native_roberta_position_ids(sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-roberta",
        AutoModelForSequenceClassification,
        "roberta.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)
    model_inputs = _tokenize_texts(lens.tokenizer, sentences[:2])

    forward_inputs = lens._prepare_model_forward_inputs(model_inputs)

    assert "position_ids" not in forward_inputs


def test_logit_lens_rejects_left_padded_classification_inputs(sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)
    original_padding_side = lens.tokenizer.padding_side
    try:
        lens.tokenizer.padding_side = "left"
        model_inputs = _tokenize_texts(lens.tokenizer, sentences[:2])
    finally:
        lens.tokenizer.padding_side = original_padding_side

    with pytest.raises(ValueError, match="right padding"):
        lens.explain(model_inputs)


@pytest.mark.parametrize(
    ("num_labels", "problem_type"),
    [
        pytest.param(1, "regression", id="regression"),
        pytest.param(2, "multi_label_classification", id="multi-label"),
    ],
)
def test_logit_lens_rejects_non_single_label_sequence_tasks(num_labels, problem_type):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    model_with_split_points._model.config.num_labels = num_labels
    model_with_split_points._model.config.problem_type = problem_type

    with pytest.raises(NotImplementedError, match="single-label"):
        LogitLens(model_with_split_points, top_k=2)


def test_lens_does_not_forward_supervision_tensors():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)
    model_inputs = _tokenize_texts(lens.tokenizer, "Interpreto is useful.")
    model_inputs["labels"] = torch.tensor([[0.0, 1.0]])

    lens.metrics(model_inputs)

    assert lens.model.config.problem_type is None


def test_logit_lens_metrics_can_preserve_gradients(sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    for parameter in lens.model.parameters():
        parameter.grad = None

    differentiable_metrics = lens.metrics(sentences[1], differentiable=True)["transformer.h.1"]
    loss = differentiable_metrics["target_cross_entropy"]

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    loss.backward()

    assert any(parameter.grad is not None for parameter in lens.model.parameters())


def test_logit_lens_causal_metrics_exclude_right_padding_transitions():
    split_point = "transformer.h.1"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    token_ids = [
        token_id for token_id in range(lens.model.config.vocab_size) if token_id not in lens.tokenizer.all_special_ids
    ][:8]
    model_inputs = BatchEncoding(
        {
            "input_ids": torch.tensor([token_ids[:4], token_ids[4:]]),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 0, 0],
                    [1, 1, 1, 1],
                ]
            ),
        }
    )

    metrics = lens.metrics(model_inputs)[split_point]

    assert metrics["nb_evaluated_elements"] == 4


def test_logit_lens_batches_lists_without_changing_results(sentences):
    split_point = "transformer.h.1"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    model_with_split_points.batch_size = len(sentences)
    expected_explanations = lens.explain(sentences)[split_point]
    expected_metrics = lens.metrics(sentences)[split_point]

    model_with_split_points.batch_size = 2
    forward_calls = []
    handle = lens.model.register_forward_pre_hook(lambda _module, _args: forward_calls.append(None))
    try:
        explanations = lens.explain(sentences)[split_point]
        metrics = lens.metrics(sentences)[split_point]
    finally:
        handle.remove()

    assert len(forward_calls) == 4
    assert torch.equal(explanations["top_indices"], expected_explanations["top_indices"])
    assert torch.allclose(explanations["top_scores"], expected_explanations["top_scores"])
    _assert_metrics_close(metrics, expected_metrics)


def test_logit_lens_batches_sequence_classification_targets(sentences):
    split_point = "bert.encoder.layer.1.output"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=2)
    targets = [0, 1, 0]
    model_with_split_points.batch_size = len(sentences)
    expected_metrics = lens.metrics(sentences, targets=targets)[split_point]

    model_with_split_points.batch_size = 1
    metrics = lens.metrics(sentences, targets=targets)[split_point]

    _assert_metrics_close(metrics, expected_metrics)


def test_logit_lens_temporarily_uses_evaluation_mode(sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    model_with_split_points._model.train()
    forward_training_modes = []
    lens = LogitLens(model_with_split_points, top_k=3)
    handle = lens.model.register_forward_pre_hook(lambda module, _args: forward_training_modes.append(module.training))
    try:
        lens.explain(sentences[1])
        lens.metrics(sentences[1])
    finally:
        handle.remove()

    assert forward_training_modes == [False, False]
    assert lens.model.training


def test_logit_lens_explain_stops_at_the_split_and_disables_cache(sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    forward_kwargs = []
    later_block_calls = []

    def _capture_forward_kwargs(_module, _args, kwargs):
        forward_kwargs.append(kwargs)

    model_handle = lens.model.register_forward_pre_hook(_capture_forward_kwargs, with_kwargs=True)
    later_block_handle = lens.model.transformer.h[2].register_forward_pre_hook(
        lambda _module, _args: later_block_calls.append(None)
    )
    try:
        lens.explain(sentences[1])
    finally:
        model_handle.remove()
        later_block_handle.remove()

    assert len(forward_kwargs) == 1
    assert forward_kwargs[0]["use_cache"] is False
    assert not later_block_calls


def test_logit_lens_rejects_invalid_top_k_values():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )

    with pytest.raises(TypeError):
        LogitLens(model_with_split_points, top_k=True)

    with pytest.raises(TypeError):
        LogitLens(model_with_split_points, top_k=1.5)

    with pytest.raises(ValueError):
        LogitLens(model_with_split_points, top_k=0)

    with pytest.raises(ValueError):
        LogitLens(model_with_split_points, top_k=-1)


@pytest.mark.parametrize(
    ("config_attribute", "value"),
    [
        ("final_logit_softcapping", 30.0),
        ("logits_soft_cap", 30.0),
        ("logit_scale", 0.0625),
        ("logits_scaling", 16.0),
    ],
)
def test_logit_lens_rejects_functional_language_model_logit_transforms(config_attribute, value):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    setattr(model_with_split_points._model.config, config_attribute, value)

    with pytest.raises(NotImplementedError, match=config_attribute):
        LogitLens(model_with_split_points, top_k=3)


@pytest.mark.parametrize(
    ("model_name", "automodel", "split_point", "top_k"),
    [
        pytest.param(
            "hf-internal-testing/tiny-random-gpt2",
            AutoModelForCausalLM,
            "transformer.h.1",
            1001,
            id="causal-language-model",
        ),
        pytest.param(
            "hf-internal-testing/tiny-random-bert",
            AutoModelForMaskedLM,
            "bert.encoder.layer.1.output",
            2000,
            id="masked-language-model",
        ),
        pytest.param(
            "hf-internal-testing/tiny-random-bert",
            AutoModelForSequenceClassification,
            "bert.encoder.layer.1.output",
            3,
            id="sequence-classification",
        ),
    ],
)
def test_logit_lens_rejects_top_k_above_output_size(model_name, automodel, split_point, top_k, sentences):
    model_with_split_points = _build_model_with_split_points(model_name, automodel, split_point)
    lens = LogitLens(model_with_split_points, top_k=top_k)

    with pytest.raises(ValueError):
        lens.explain(sentences[1])


def test_logit_lens_requires_explicit_pooling_for_linear_classification_heads():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )

    with pytest.raises(ValueError, match="pooling_strategy"):
        LogitLens(
            model_with_split_points,
            head_name="classifier",
            pre_head_name=None,
            pooling_strategy=None,
            top_k=2,
        )


def test_logit_lens_does_not_infer_an_incomplete_decoder_classifier_suffix():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForSequenceClassification,
        "transformer.h.1",
    )

    with pytest.raises(ValueError, match="faithful sequence-classification projection"):
        LogitLens(model_with_split_points, pooling_strategy="last", top_k=2)


@pytest.mark.parametrize("pooling_strategy", ["cls", "mean", "last"])
def test_logit_lens_supports_explicit_sequence_classification_pooling(pooling_strategy, sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(
        model_with_split_points,
        head_name="classifier",
        pre_head_name=None,
        pooling_strategy=pooling_strategy,
        top_k=2,
    )

    layer_output = lens.explain(sentences[1])["bert.encoder.layer.1.output"]

    assert layer_output["top_indices"].shape == (1, 2)
    assert layer_output["top_scores"].shape == (1, 2)


@pytest.mark.parametrize(
    ("pooling_strategy", "expected"),
    [
        pytest.param("cls", [[2.0], [5.0]], id="first-valid-token"),
        pytest.param("last", [[3.0], [6.0]], id="last-valid-token"),
    ],
)
def test_logit_lens_honors_explicit_head_pooling_with_left_padding(pooling_strategy, expected):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(
        model_with_split_points,
        head_name="classifier",
        pre_head_name=None,
        pooling_strategy=pooling_strategy,
        top_k=2,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    model_inputs = BatchEncoding(
        {
            "attention_mask": torch.tensor(
                [
                    [0, 0, 1, 1],
                    [0, 1, 1, 0],
                ]
            )
        }
    )

    pooled_hidden_states = lens._pool_hidden_states(hidden_states, model_inputs)

    assert lens.head_name == "classifier"
    assert lens.pre_head_name is None
    assert lens.pooling_strategy == pooling_strategy
    assert torch.equal(pooled_hidden_states, torch.tensor(expected))


def test_logit_lens_rejects_token_classification_models():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForTokenClassification,
        "bert.encoder.layer.1.output",
    )

    with pytest.raises(NotImplementedError):
        LogitLens(model_with_split_points)


def test_logit_lens_rejects_unsupported_encoder_decoder_models():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-t5",
        AutoModelForSeq2SeqLM,
        "decoder.block.1.layer.2",
    )

    with pytest.raises(ValueError, match="Unsupported model type"):
        LogitLens(model_with_split_points)


def test_logit_lens_uses_existing_eos_token_for_padding_without_resizing(
    monkeypatch: pytest.MonkeyPatch,
    sentences,
):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    resize_calls = []

    def _forbid_resize(*args, **kwargs):
        resize_calls.append((args, kwargs))
        raise AssertionError("The lens methods should not resize token embeddings.")

    monkeypatch.setattr(lens.model, "resize_token_embeddings", _forbid_resize)

    lens.explain(sentences[1])

    assert lens.tokenizer.pad_token is not None
    assert lens.tokenizer.pad_token == lens.tokenizer.eos_token
    assert not resize_calls


def test_logit_lens_requires_tensor_backed_batch_encodings():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    model_inputs = BatchEncoding(
        {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
        }
    )

    with pytest.raises(TypeError, match="return_tensors"):
        lens.explain(model_inputs)


def test_logit_lens_top_scores_match_softmax_top_k(sentences):
    split_point = "transformer.h.1"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    captured_hidden_states = {}

    def _capture_hidden_states(_module, _args, output):
        captured_hidden_states["value"] = output[0] if isinstance(output, tuple) else output

    split_module = _get_model_module(lens.model, split_point)
    handle = split_module.register_forward_hook(_capture_hidden_states)
    try:
        with torch.no_grad():
            lens.model(
                input_ids=model_inputs["input_ids"].to(lens.model_device),
                attention_mask=model_inputs["attention_mask"].to(lens.model_device),
            )
    finally:
        handle.remove()

    hidden_states = captured_hidden_states["value"].to(lens.model_device)
    if lens.model_pre_head is not None:
        hidden_states = lens.model_pre_head(hidden_states)
    projected_logits = lens.model_head(hidden_states)
    if isinstance(projected_logits, tuple):
        projected_logits = projected_logits[0]

    expected_top_outputs = torch.topk(torch.softmax(projected_logits, dim=-1), k=3, dim=-1)
    explanations = lens.explain(model_inputs)[split_point]

    assert torch.allclose(explanations["top_scores"], expected_top_outputs.values.detach().cpu())
    assert torch.equal(explanations["top_indices"], expected_top_outputs.indices.detach().cpu())


def test_logit_lens_upcasts_low_precision_scores():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    logits = torch.randn(1, 2, lens.model.config.vocab_size, dtype=torch.float16)

    top_outputs = lens._compute_top_outputs(logits, expected_ndim=3, output_name="vocabulary")

    assert top_outputs["top_scores"].dtype == torch.float32


def test_plot_lens_keeps_hover_friendly_language_model_tokens(
    monkeypatch: pytest.MonkeyPatch,
    sentences,
):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    results = lens.explain(model_inputs)
    displayed_html = _capture_displayed_html(monkeypatch)
    plot_lens(
        results,
        model_inputs,
        tokenizer=lens.tokenizer,
        task=lens.task,
    )
    html = displayed_html[0]

    assert len(displayed_html) == 1
    assert "vocabulary scores decoded from that position" in html
    assert "Top vocabulary scores" in html
    assert "lens-tooltip-title" in html
    assert "lens-token-stream" in html
    assert "Sample 1" in html


def test_plot_lens_keeps_sequence_classification_card_view(
    monkeypatch: pytest.MonkeyPatch,
    sentences,
):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    results = lens.explain(model_inputs)
    displayed_html = _capture_displayed_html(monkeypatch)
    plot_lens(
        results,
        model_inputs,
        tokenizer=lens.tokenizer,
        task=lens.task,
    )
    html = displayed_html[0]
    top_label = str(int(results["bert.encoder.layer.1.output"]["top_indices"][0, 0]))

    assert len(displayed_html) == 1
    assert "Current top class:" in html
    assert "lens-prediction-bar" in html
    assert "Sample 1" in html
    assert top_label in html
    assert "LABEL_" not in html


def test_plot_lens_accepts_label_names_css_and_save_path(
    monkeypatch: pytest.MonkeyPatch,
    sentences,
    tmp_path,
):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    results = lens.explain(model_inputs)
    displayed_html = _capture_displayed_html(monkeypatch)
    output_path = tmp_path / "lens.html"
    plot_lens(
        results,
        model_inputs,
        tokenizer=lens.tokenizer,
        task=lens.task,
        label_names={0: "negative", 1: "positive"},
        custom_css=".lens-shell { color: black; }",
        save_path=output_path,
    )

    assert len(displayed_html) == 1
    assert "negative" in displayed_html[0]
    assert "positive" in displayed_html[0]
    assert ".lens-shell { color: black; }" in displayed_html[0]
    assert output_path.read_text(encoding="utf-8") == displayed_html[0]


def test_plot_lens_rejects_misaligned_results(sentences):
    split_point = "transformer.h.1"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    model_inputs = _tokenize_texts(lens.tokenizer, sentences)
    results = lens.explain(model_inputs)
    malformed_results = {
        split_point: {
            "top_indices": results[split_point]["top_indices"],
            "top_scores": results[split_point]["top_scores"][:, :-1],
        }
    }

    with pytest.raises(ValueError, match="matching shapes"):
        plot_lens(
            malformed_results,
            model_inputs,
            tokenizer=lens.tokenizer,
            task=lens.task,
        )


@pytest.mark.parametrize(
    ("model_name", "automodel", "split_point", "top_k"),
    TUNED_LANGUAGE_MODEL_CASES,
)
def test_tuned_lens_fits_and_restores_for_language_models(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    model_name,
    automodel,
    split_point,
    top_k,
    sentences,
):
    model_with_split_points = _build_model_with_split_points(
        model_name,
        automodel,
        split_point,
    )
    lens = TunedLens(model_with_split_points, top_k=top_k)

    initial_weight = lens.translator.weight.detach().clone()
    lens.model.train()
    forward_training_modes = []
    handle = lens.model.register_forward_pre_hook(lambda module, _args: forward_training_modes.append(module.training))
    try:
        history = lens.fit(
            sentences[:2],
            epochs=1,
            batch_size=2,
        )
    finally:
        handle.remove()
    explanations = lens.explain(sentences[1])
    layer_output = explanations[split_point]

    assert forward_training_modes == [False]
    assert lens.model.training
    assert history["epochs"] == 1
    assert history["split_point"] == split_point
    assert len(history["loss"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"])).all()
    assert not torch.allclose(initial_weight, lens.translator.weight)
    assert layer_output["top_indices"].shape == layer_output["top_scores"].shape
    assert layer_output["top_indices"].shape[0] == 1
    assert layer_output["top_indices"].shape[-1] == top_k

    checkpoint_path = tmp_path / f"{split_point.replace('.', '_')}.pt"
    lens.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_arguments = {}

    def _record_safe_load(*_args, **kwargs):
        load_arguments.update(kwargs)
        return checkpoint

    monkeypatch.setattr(torch, "load", _record_safe_load)
    restored_lens = TunedLens.from_checkpoint(model_with_split_points, checkpoint_path)
    restored_output = restored_lens.explain(sentences[1])[split_point]

    assert checkpoint["format_version"] == 1
    assert checkpoint["metadata"]["split_point"] == split_point
    assert checkpoint["metadata"]["model_name_or_path"] == model_name
    assert "translator" in checkpoint
    assert "translators" not in checkpoint
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["translator"].values())
    assert load_arguments["weights_only"] is True
    assert restored_lens.initialization_mode == lens.initialization_mode
    assert torch.allclose(restored_output["top_scores"], layer_output["top_scores"])


def test_tuned_lens_supports_multiple_initialization_modes():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    default_lens = TunedLens(model_with_split_points, initialization_mode="default")
    xavier_lens = TunedLens(model_with_split_points, initialization_mode="xavier")
    logit_lens_init = TunedLens(model_with_split_points, initialization_mode="logit_lens")

    hidden_size = logit_lens_init.hidden_size
    expected_identity = torch.eye(hidden_size)

    assert default_lens.initialization_mode == "default"
    assert xavier_lens.initialization_mode == "xavier"
    assert logit_lens_init.initialization_mode == "logit_lens"
    assert default_lens.model_head is not None
    assert logit_lens_init.model_pre_head is not None
    assert torch.allclose(logit_lens_init.translator.weight.detach().cpu(), expected_identity)
    assert torch.allclose(
        logit_lens_init.translator.bias.detach().cpu(),
        torch.zeros(hidden_size),
    )
    assert torch.allclose(
        xavier_lens.translator.bias.detach().cpu(),
        torch.zeros(hidden_size),
    )
    assert not torch.allclose(
        xavier_lens.translator.weight.detach().cpu(),
        expected_identity,
    )


def test_tuned_lens_rejects_invalid_fit_configuration(sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = TunedLens(model_with_split_points)
    invalid_configurations = [
        ({"epochs": True}, TypeError),
        ({"epochs": 0}, ValueError),
        ({"learning_rate": True}, TypeError),
        ({"learning_rate": 0.0}, ValueError),
        ({"weight_decay": True}, TypeError),
        ({"weight_decay": -1.0}, ValueError),
        ({"batch_size": True}, TypeError),
        ({"batch_size": 0}, ValueError),
    ]

    for fit_kwargs, expected_exception in invalid_configurations:
        with pytest.raises(expected_exception):
            lens.fit(sentences[1], **fit_kwargs)

    with pytest.raises(ValueError, match="Empty"):
        lens.fit([])


def test_tuned_lens_tokenizes_raw_training_inputs_by_batch(monkeypatch: pytest.MonkeyPatch, sentences):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = TunedLens(model_with_split_points)
    batch_lengths = []
    tokenize_texts = lens._tokenize_texts

    def _record_batch(texts):
        batch_lengths.append(len(texts))
        return tokenize_texts(texts)

    monkeypatch.setattr(lens, "_tokenize_texts", _record_batch)

    lens.fit(sentences, epochs=1, batch_size=2)

    assert batch_lengths == [2, 1]


def test_tuned_lens_rejects_mismatched_fit_logits():
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = TunedLens(model_with_split_points)
    projected_logits = torch.randn(1, 3, lens.model.config.vocab_size)
    target_logits = torch.randn(1, 2, lens.model.config.vocab_size)
    model_inputs = BatchEncoding(
        {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
    )

    with pytest.raises(ValueError, match="same shape"):
        lens._language_model_loss(projected_logits, target_logits, model_inputs)


def test_tuned_lens_rejects_malformed_checkpoints(tmp_path):
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1",
    )
    lens = TunedLens(model_with_split_points)
    checkpoint_path = tmp_path / "valid.pt"
    lens.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mismatched_metadata = dict(checkpoint["metadata"])
    mismatched_metadata["split_point"] = "transformer.h.0"
    mismatched_model_metadata = dict(checkpoint["metadata"])
    mismatched_model_metadata["model_name_or_path"] = "different/model"
    malformed_checkpoints = [
        [],
        {**checkpoint, "format_version": 2},
        {"format_version": 1, "metadata": checkpoint["metadata"]},
        {**checkpoint, "metadata": mismatched_metadata},
        {**checkpoint, "metadata": mismatched_model_metadata},
    ]

    for checkpoint_index, malformed_checkpoint in enumerate(malformed_checkpoints):
        malformed_path = tmp_path / f"malformed_{checkpoint_index}.pt"
        torch.save(malformed_checkpoint, malformed_path)
        with pytest.raises(ValueError):
            TunedLens.from_checkpoint(model_with_split_points, malformed_path)


@pytest.mark.parametrize(("model_name", "split_point"), TUNED_SEQUENCE_CLASSIFICATION_CASES)
def test_tuned_lens_fits_for_sequence_classification_models(model_name, split_point, sentences):
    model_with_split_points = _build_model_with_split_points(
        model_name,
        AutoModelForSequenceClassification,
        split_point,
    )
    lens = TunedLens(model_with_split_points, top_k=2)

    history = lens.fit(
        sentences[:2],
        epochs=1,
        batch_size=2,
    )
    explanations = lens.explain(sentences[1])
    layer_output = explanations[split_point]

    assert history["split_point"] == split_point
    assert len(history["loss"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"])).all()
    assert layer_output["top_indices"].shape == (1, 2)
    assert layer_output["top_scores"].shape == (1, 2)


def test_lens_notebook_uses_a_generic_kernelspec():
    notebook_path = REPOSITORY_ROOT / "docs/notebooks/lens_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["metadata"]["kernelspec"]["name"] == "python3"


def test_lens_notebook_does_not_contain_error_outputs():
    notebook_path = REPOSITORY_ROOT / "docs/notebooks/lens_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error"
