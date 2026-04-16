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

import json
from pathlib import Path

import pytest
import torch
import interpreto.visualizations.lens as lens_visualizations
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from interpreto import LogitLens, ModelWithSplitPoints, TunedLens
from interpreto.lens import LogitLens as LensLogitLens
from interpreto.lens import TunedLens as LensTunedLens
from interpreto.visualizations import display_lens_results

DEVICE = "cpu"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CAUSAL_LANGUAGE_MODEL_CASES = [
    pytest.param(
        "hf-internal-testing/tiny-random-gpt2",
        "transformer.h.1.mlp",
        id="tiny-random-gpt2",
    ),
    pytest.param(
        "hf-internal-testing/tiny-random-gpt_neo",
        "transformer.h.1.mlp",
        marks=pytest.mark.slow,
        id="tiny-random-gpt_neo",
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


def _build_model_with_split_points(
    model_name: str,
    automodel,
    split_point: str,
) -> ModelWithSplitPoints:
    model = automodel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=split_point,
        batch_size=2,
        device_map=DEVICE,
    )


def _tokenize_texts(tokenizer, texts: str | list[str]):
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
    displayed_html: list[str] = []

    monkeypatch.setattr(lens_visualizations, "HTML", lambda html: html)
    monkeypatch.setattr(lens_visualizations, "display", displayed_html.append)
    return displayed_html


def test_lens_exports_are_available() -> None:
    assert LogitLens is LensLogitLens
    assert TunedLens is LensTunedLens


@pytest.mark.parametrize(("model_name", "split_point"), CAUSAL_LANGUAGE_MODEL_CASES)
def test_logit_lens_supports_causal_language_models(
    model_name: str,
    split_point: str,
    sentences: list[str],
) -> None:
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
    assert metrics["target_source"] == "next_token"
    assert metrics["nb_evaluated_elements"] > 0
    assert 0.0 <= metrics["mean_target_probability"] <= 1.0
    assert 0.0 <= metrics["mean_max_probability"] <= 1.0
    assert metrics["target_cross_entropy"] >= 0.0
    assert metrics["perplexity"] >= 1.0
    assert metrics["kl_divergence_to_model"] > -1e-6


def test_logit_lens_resolves_the_standard_gpt2_projection(sentences: list[str]) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1.mlp",
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    explanations = lens.explain(sentences[1])

    assert lens.head_name == "lm_head"
    assert lens.pre_head_name == "transformer.ln_f"
    assert explanations["transformer.h.1.mlp"]["top_indices"].shape[-1] == 3


def test_logit_lens_supports_masked_language_models(sentences: list[str]) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForMaskedLM,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=4)

    explanations = lens.explain(sentences[1])
    layer_output = explanations["bert.encoder.layer.1.output"]

    assert set(layer_output) == {"top_indices", "top_scores"}
    assert layer_output["top_indices"].shape == layer_output["top_scores"].shape
    assert layer_output["top_indices"].shape[0] == 1
    assert layer_output["top_indices"].shape[-1] == 4


def test_logit_lens_supports_sequence_classification_with_internal_pooler(
    sentences: list[str],
) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)

    explanations = lens.explain(sentences[:2])
    metrics = lens.metrics(sentences[:2], targets=[1, 0])["bert.encoder.layer.1.output"]
    layer_output = explanations["bert.encoder.layer.1.output"]

    assert set(layer_output) == {"top_indices", "top_scores"}
    assert layer_output["top_indices"].shape == (2, 2)
    assert layer_output["top_scores"].shape == (2, 2)
    assert lens.head_name == "classifier"
    assert lens.pre_head_name == "bert.pooler"
    assert lens.model_head is not None
    assert lens.model_pre_head is not None
    assert metrics["target_source"] == "provided_targets"
    assert metrics["nb_evaluated_elements"] == 2
    assert 0.0 <= metrics["mean_target_probability"] <= 1.0
    assert metrics["target_cross_entropy"] >= 0.0
    assert 0.0 <= metrics["target_accuracy"] <= 1.0
    assert metrics["kl_divergence_to_model"] > -1e-6


@pytest.mark.parametrize(("model_name", "split_point"), SEQUENCE_AWARE_CLASSIFICATION_CASES)
def test_logit_lens_supports_sequence_aware_classification_heads(
    model_name: str,
    split_point: str,
    sentences: list[str],
) -> None:
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


def test_logit_lens_metrics_can_preserve_gradients(sentences: list[str]) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1.mlp",
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    for parameter in lens.model.parameters():
        parameter.grad = None

    differentiable_metrics = lens.metrics(
        sentences[1],
        differentiable=True,
    )["transformer.h.1.mlp"]
    loss = differentiable_metrics["target_cross_entropy"]
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    loss.backward()

    assert any(parameter.grad is not None for parameter in lens.model.parameters())


def test_logit_lens_rejects_invalid_top_k_values() -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1.mlp",
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
    ("model_name", "automodel", "split_point", "top_k"),
    [
        pytest.param(
            "hf-internal-testing/tiny-random-gpt2",
            AutoModelForCausalLM,
            "transformer.h.1.mlp",
            1001,
            id="causal-language-model",
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
def test_logit_lens_rejects_top_k_above_output_size(
    model_name: str,
    automodel,
    split_point: str,
    top_k: int,
    sentences: list[str],
) -> None:
    model_with_split_points = _build_model_with_split_points(model_name, automodel, split_point)
    lens = LogitLens(model_with_split_points, top_k=top_k)

    with pytest.raises(ValueError):
        lens.explain(sentences[1])


def test_logit_lens_requires_explicit_pooling_for_linear_classification_heads() -> None:
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


@pytest.mark.parametrize("pooling_strategy", ["cls", "mean", "last"])
def test_logit_lens_supports_explicit_sequence_classification_pooling(
    pooling_strategy: str,
    sentences: list[str],
) -> None:
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


def test_logit_lens_rejects_token_classification_models() -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForTokenClassification,
        "bert.encoder.layer.1.output",
    )

    try:
        LogitLens(model_with_split_points)
    except NotImplementedError:
        return

    raise AssertionError("Token classification should not be accepted by LogitLens.")


def test_logit_lens_rejects_unsupported_encoder_decoder_models() -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-t5",
        AutoModelForSeq2SeqLM,
        "decoder.block.1.layer.2",
    )

    with pytest.raises(ValueError, match="Unsupported model type"):
        LogitLens(model_with_split_points)


def test_logit_lens_uses_existing_eos_token_for_padding_without_resizing(
    monkeypatch: pytest.MonkeyPatch,
    sentences: list[str],
) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1.mlp",
    )
    lens = LogitLens(model_with_split_points, top_k=3)
    resize_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _forbid_resize(*args: object, **kwargs: object) -> None:
        resize_calls.append((args, kwargs))
        raise AssertionError("The lens methods should not resize token embeddings.")

    monkeypatch.setattr(lens.model, "resize_token_embeddings", _forbid_resize)

    lens.explain(sentences[1])

    assert lens.tokenizer.pad_token is not None
    assert lens.tokenizer.pad_token == lens.tokenizer.eos_token
    assert not resize_calls


def test_logit_lens_top_scores_match_softmax_top_k(sentences: list[str]) -> None:
    split_point = "transformer.h.1.mlp"
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        split_point,
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    captured_hidden_states: dict[str, torch.Tensor] = {}

    def _capture_hidden_states(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
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
    expected_scores = expected_top_outputs.values.detach().cpu()
    expected_indices = expected_top_outputs.indices.detach().cpu()
    explanations = lens.explain(model_inputs)[split_point]

    assert torch.allclose(explanations["top_scores"], expected_scores)
    assert torch.equal(explanations["top_indices"], expected_indices)


def test_logit_lens_language_model_renderer_keeps_hover_friendly_tokens(
    monkeypatch: pytest.MonkeyPatch,
    sentences: list[str],
) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1.mlp",
    )
    lens = LogitLens(model_with_split_points, top_k=3)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    results = lens.explain(model_inputs)
    displayed_html = _capture_displayed_html(monkeypatch)
    display_lens_results(
        results,
        model_inputs,
        tokenizer=lens.tokenizer,
        task=lens.task,
    )
    assert len(displayed_html) == 1
    html = displayed_html[0]
    top_index = int(results["transformer.h.1.mlp"]["top_indices"][0, 0, 0])
    decoded_token = lens.tokenizer.convert_ids_to_tokens([top_index])[0]
    decoded_token = decoded_token.replace("Ġ", " ").replace("▁", " ").replace("</w>", "")

    assert "Hover a token to see what this layer currently predicts." in html
    assert "lens-tooltip-title" in html
    assert "lens-token-stream" in html
    assert "Sample 1" in html
    if decoded_token:
        assert decoded_token in html


def test_logit_lens_sequence_classification_renderer_keeps_card_view(
    monkeypatch: pytest.MonkeyPatch,
    sentences: list[str],
) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    results = lens.explain(model_inputs)
    displayed_html = _capture_displayed_html(monkeypatch)
    display_lens_results(
        results,
        model_inputs,
        tokenizer=lens.tokenizer,
        task=lens.task,
    )
    assert len(displayed_html) == 1
    html = displayed_html[0]
    top_label = str(int(results["bert.encoder.layer.1.output"]["top_indices"][0, 0]))

    assert "Current top class:" in html
    assert "lens-prediction-bar" in html
    assert "Sample 1" in html
    assert top_label in html
    assert "LABEL_" not in html


def test_logit_lens_sequence_classification_renderer_accepts_explicit_label_names(
    monkeypatch: pytest.MonkeyPatch,
    sentences: list[str],
) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = LogitLens(model_with_split_points, top_k=2)

    model_inputs = _tokenize_texts(lens.tokenizer, sentences[1])
    results = lens.explain(model_inputs)
    displayed_html = _capture_displayed_html(monkeypatch)
    display_lens_results(
        results,
        model_inputs,
        tokenizer=lens.tokenizer,
        task=lens.task,
        label_names={0: "negative", 1: "positive"},
    )
    assert len(displayed_html) == 1
    html = displayed_html[0]

    assert "negative" in html
    assert "positive" in html


def test_tuned_lens_fits_and_restores_for_causal_language_models(
    tmp_path,
    sentences: list[str],
) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1.mlp",
    )
    lens = TunedLens(model_with_split_points, top_k=3)

    initial_weight = lens.translators["transformer.h.1.mlp"].weight.detach().clone()
    history = lens.fit(
        sentences[:2],
        epochs=1,
        batch_size=2,
    )

    assert history["epochs"] == 1
    assert history["split_points"] == ["transformer.h.1.mlp"]
    assert len(history["loss"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"])).all()
    assert not torch.allclose(initial_weight, lens.translators["transformer.h.1.mlp"].weight)

    checkpoint_path = tmp_path / "tuned_lens.pt"
    lens.save(checkpoint_path)
    restored_lens = TunedLens.from_checkpoint(model_with_split_points, checkpoint_path)

    assert restored_lens.initialization_mode == lens.initialization_mode

    restored_output = restored_lens.explain(sentences[1])["transformer.h.1.mlp"]
    current_output = lens.explain(sentences[1])["transformer.h.1.mlp"]

    assert torch.allclose(restored_output["top_scores"], current_output["top_scores"])


def test_tuned_lens_supports_multiple_initialization_modes() -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-gpt2",
        AutoModelForCausalLM,
        "transformer.h.1.mlp",
    )
    default_lens = TunedLens(model_with_split_points, initialization_mode="default")
    xavier_lens = TunedLens(model_with_split_points, initialization_mode="xavier")
    logit_lens_init = TunedLens(model_with_split_points, initialization_mode="logit_lens")

    split_point = "transformer.h.1.mlp"
    hidden_size = logit_lens_init.hidden_size
    expected_identity = torch.eye(hidden_size)

    assert default_lens.initialization_mode == "default"
    assert xavier_lens.initialization_mode == "xavier"
    assert logit_lens_init.initialization_mode == "logit_lens"
    assert default_lens.model_head is not None
    assert logit_lens_init.model_pre_head is not None
    assert torch.allclose(logit_lens_init.translators[split_point].weight.detach().cpu(), expected_identity)
    assert torch.allclose(
        logit_lens_init.translators[split_point].bias.detach().cpu(),
        torch.zeros(hidden_size),
    )
    assert torch.allclose(
        xavier_lens.translators[split_point].bias.detach().cpu(),
        torch.zeros(hidden_size),
    )
    assert not torch.allclose(
        xavier_lens.translators[split_point].weight.detach().cpu(),
        expected_identity,
    )


def test_tuned_lens_fits_for_sequence_classification(sentences: list[str]) -> None:
    model_with_split_points = _build_model_with_split_points(
        "hf-internal-testing/tiny-random-bert",
        AutoModelForSequenceClassification,
        "bert.encoder.layer.1.output",
    )
    lens = TunedLens(model_with_split_points, top_k=2)

    history = lens.fit(
        sentences[:2],
        epochs=1,
        batch_size=2,
    )
    explanations = lens.explain(sentences[1])
    layer_output = explanations["bert.encoder.layer.1.output"]

    assert len(history["loss"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"])).all()
    assert layer_output["top_indices"].shape == (1, 2)
    assert layer_output["top_scores"].shape == (1, 2)


def test_lens_notebook_uses_a_generic_kernelspec() -> None:
    notebook_path = REPOSITORY_ROOT / "docs/notebooks/lens_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["metadata"]["kernelspec"]["name"] == "python3"


def test_lens_notebook_does_not_contain_error_outputs() -> None:
    notebook_path = REPOSITORY_ROOT / "docs/notebooks/lens_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error"


def test_fetch_head_is_not_tracked_with_the_lens_changes() -> None:
    assert not (REPOSITORY_ROOT / "FETCH_HEAD").exists()
