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

"""Tests for Logit Lens and Tuned Lens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import interpreto.visualizations.lens as lens_visualizations
from interpreto import AllLayersSplitter, LogitLens, TunedLens, plot_lens

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gpt2_splitter(gpt2_model, gpt2_tokenizer):
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.eval()
    return AllLayersSplitter(gpt2_model, tokenizer=gpt2_tokenizer)


@pytest.fixture(scope="module")
def bert_splitter(bert_model, bert_tokenizer):
    bert_model.eval()
    return AllLayersSplitter(bert_model, tokenizer=bert_tokenizer)


def _expected_top_k(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    top_logits, top_indices = logits.topk(top_k, dim=-1)
    top_scores = (top_logits - logits.logsumexp(dim=-1, keepdim=True)).exp()
    return top_indices, top_scores


def test_logit_lens_processes_all_layers_in_one_head_call(gpt2_splitter, monkeypatch):
    apply_head = gpt2_splitter.apply_head
    head_inputs = []

    def record_head_input(activations):
        head_inputs.append(activations)
        return apply_head(activations)

    monkeypatch.setattr(gpt2_splitter, "apply_head", record_head_input)
    results = LogitLens(gpt2_splitter, top_k=3)("Interpreto is useful.")

    assert list(results) == gpt2_splitter.activation_names
    assert len(head_inputs) == 1
    assert head_inputs[0].shape[0] == len(gpt2_splitter.activation_names)
    assert all(output["top_indices"].shape == output["top_scores"].shape for output in results.values())
    assert all(output["top_indices"].shape[:1] == (1,) for output in results.values())
    assert all(output["top_indices"].shape[-1] == 3 for output in results.values())


def test_logit_lens_final_output_matches_the_model(gpt2_splitter):
    text = "Interpreto is useful."
    results = LogitLens(gpt2_splitter, top_k=3)(text)
    model_inputs = gpt2_splitter.tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = gpt2_splitter._model(**model_inputs).logits
    expected_indices, expected_scores = _expected_top_k(logits, top_k=3)
    final_output = results[gpt2_splitter.activation_names[-1]]

    torch.testing.assert_close(final_output["top_indices"], expected_indices)
    torch.testing.assert_close(final_output["top_scores"], expected_scores)


def test_logit_lens_uses_the_same_path_for_classification(bert_splitter):
    results = LogitLens(bert_splitter)("Interpreto is useful.")

    assert list(results) == bert_splitter.activation_names
    assert all(output["top_indices"].shape == (1, 2) for output in results.values())


def test_logit_lens_requires_a_positive_top_k(gpt2_splitter):
    with pytest.raises(ValueError, match="positive"):
        LogitLens(gpt2_splitter, top_k=0)


def test_tuned_lens_starts_as_a_logit_lens(gpt2_splitter):
    text = "Interpreto is useful."
    logit_results = LogitLens(gpt2_splitter, top_k=3)(text)
    tuned_lens = TunedLens(gpt2_splitter, top_k=3)
    tuned_results = tuned_lens(text)

    assert len(tuned_lens.translators) == len(gpt2_splitter.split_points)
    for layer_name in gpt2_splitter.activation_names:
        torch.testing.assert_close(
            tuned_results[layer_name]["top_indices"],
            logit_results[layer_name]["top_indices"],
        )
        torch.testing.assert_close(
            tuned_results[layer_name]["top_scores"],
            logit_results[layer_name]["top_scores"],
        )


def test_tuned_lens_fits_every_layer_together(gpt2_splitter, monkeypatch):
    texts = ["Interpreto is useful.", "Lens methods expose intermediate predictions."]
    lens = TunedLens(gpt2_splitter, top_k=3)
    parameters_before_fit = [parameter.detach().clone() for parameter in lens.parameters()]
    apply_head = gpt2_splitter.apply_head
    head_depths = []

    def record_head_input(activations):
        head_depths.append(activations.shape[0])
        return apply_head(activations)

    monkeypatch.setattr(gpt2_splitter, "apply_head", record_head_input)
    losses = lens.fit(texts, epochs=1, learning_rate=1e-2)

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses)).all()
    assert head_depths == [len(gpt2_splitter.activation_names)] * len(texts)
    assert any(
        not torch.equal(before, after) for before, after in zip(parameters_before_fit, lens.parameters(), strict=True)
    )

    tuned_final = lens(texts[0])[gpt2_splitter.activation_names[-1]]
    logit_final = LogitLens(gpt2_splitter, top_k=3)(texts[0])[gpt2_splitter.activation_names[-1]]
    torch.testing.assert_close(tuned_final["top_indices"], logit_final["top_indices"])
    torch.testing.assert_close(tuned_final["top_scores"], logit_final["top_scores"])


def test_plot_lens_renders_every_layer(gpt2_splitter, monkeypatch):
    text = "Interpreto is useful."
    results = LogitLens(gpt2_splitter, top_k=3)(text)
    displayed_html = []
    monkeypatch.setattr(lens_visualizations, "HTML", lambda html: html)
    monkeypatch.setattr(lens_visualizations, "display", displayed_html.append)

    plot_lens(results, text, tokenizer=gpt2_splitter.tokenizer)

    assert len(displayed_html) == 1
    assert all(layer_name in displayed_html[0] for layer_name in gpt2_splitter.activation_names)


def test_plot_lens_renders_class_names(bert_splitter, monkeypatch):
    results = LogitLens(bert_splitter)("Interpreto is useful.")
    displayed_html = []
    monkeypatch.setattr(lens_visualizations, "HTML", lambda html: html)
    monkeypatch.setattr(lens_visualizations, "display", displayed_html.append)

    plot_lens(results, "Interpreto is useful.", tokenizer=bert_splitter.tokenizer, label_names={0: "no", 1: "yes"})

    assert "no" in displayed_html[0]
    assert "yes" in displayed_html[0]


def test_lens_notebook_is_portable_and_has_no_error_outputs():
    notebook = json.loads((REPOSITORY_ROOT / "docs" / "notebooks" / "lens_notebook.ipynb").read_text())

    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert all(
        output.get("output_type") != "error" for cell in notebook["cells"] for output in cell.get("outputs", [])
    )
