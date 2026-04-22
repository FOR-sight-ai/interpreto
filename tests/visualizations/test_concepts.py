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

from __future__ import annotations

import json
import re

import torch

from interpreto.visualizations import plot_concepts


def _extract_payload(html: str, viz_name: str) -> dict:
    match = re.search(rf"new {viz_name}\((.*?)\);", html, re.DOTALL)
    assert match, f"Expected {viz_name} call in HTML."
    call_args = match.group(1)
    json_literal_match = re.search(r'"(?:\\.|[^"\\])*"', call_args)
    assert json_literal_match, "Expected JSON payload in visualization call."
    json_string = json.loads(json_literal_match.group(0))
    return json.loads(json_string)


def _render_concepts_html(tmp_path, **kwargs: object) -> str:
    save_path = tmp_path / "concepts.html"
    plot_concepts(save_path=save_path, **kwargs)
    return save_path.read_text(encoding="utf-8")


def test_plot_concepts_classification_global_payload_keys(tmp_path):
    classes_names = ["A", "B"]
    concepts_labels = {0: "alpha", 1: "beta"}
    concepts_importances = torch.tensor([[0.5, 0.0], [0.1, 0.3]])

    html = _render_concepts_html(
        tmp_path,
        concepts_importances=concepts_importances,
        concepts_labels=concepts_labels,
        classes_names=classes_names,
    )
    payload = _extract_payload(html, "ClassificationConceptsBarPlotVisualization")

    assert {"classes", "concepts", "concept_color", "onclick_colormap"} <= set(payload.keys())
    assert [entry["name"] for entry in payload["classes"]] == ["A", "B"]
    assert len(payload["concepts"]) == 2
    assert payload["concepts"][0][0]["label"] == "alpha"
    assert payload["concepts"][1][0]["id"] == 1
    assert len(payload["onclick_colormap"]) == 2


def test_plot_concepts_classification_local_classwise_labels(tmp_path):
    sample = ["Great", "food", "but", "slow"]
    classes_names = ["Neg", "Pos"]
    concepts_labels = {
        0: {0: "Neg sentiment", 1: "Service issues"},
        1: {0: "Pos sentiment", 1: "Food quality"},
    }
    concepts_activations = {
        0: [[0.8, 0.2]],
        1: [[0.1, 0.9]],
    }
    concepts_importances = {
        0: [0.2, 0.8],
        1: [0.7, 0.3],
    }

    html = _render_concepts_html(
        tmp_path,
        sample=sample,
        classes_names=classes_names,
        concepts_activations=concepts_activations,
        concepts_importances=concepts_importances,
        concepts_labels=concepts_labels,
    )
    payload = _extract_payload(html, "ClassificationLocalConceptsVisualization")

    assert {
        "classes",
        "sample",
        "labels",
        "labels_by_class",
        "importances",
    } <= set(payload.keys())
    assert payload["sample"] == sample
    assert [entry["name"] for entry in payload["classes"]] == ["Neg", "Pos"]
    assert payload["labels_by_class"]["0"][0] == "Neg sentiment"
    assert payload["labels_by_class"]["1"][1] == "Food quality"
    assert payload["labels"] == ["Neg sentiment", "Service issues"]
    if "activations_by_class" in payload:
        assert payload["activations_by_class"]["1"] == [[0.1, 0.9]]


def test_plot_concepts_classification_local_uses_static_root(tmp_path):
    html = _render_concepts_html(
        tmp_path,
        sample=["A", "sample"],
        classes_names=["Neg", "Pos"],
        concepts_activations=[[0.4, 0.2], [0.1, 0.3]],
        concepts_importances=[[0.6, -0.2], [-0.4, 0.5]],
        concepts_labels={0: "Sentiment", 1: "Topic"},
        top_k=1,
    )

    assert re.search(r"<div id='classification-local-[^']+'></div>", html)
    assert "<h3>Classes</h3>" not in html
    assert "<h3>Concepts</h3>" not in html
    assert "<h3>Sample</h3>" not in html


def test_plot_concepts_generation_local_payload_keys(tmp_path):
    sample = ["Test", "generation", "concepts", "visualization"]
    concepts_labels = {0: "A", 1: "E"}
    concepts_activations = torch.tensor(
        [
            [0.1, 0.0],
            [0.0, 0.2],
            [0.3, 0.1],
            [0.0, 0.4],
        ]
    )
    concepts_importances = torch.tensor(
        [
            [0.2, 0.1],
            [0.0, 0.3],
        ]
    )

    html = _render_concepts_html(
        tmp_path,
        sample=sample,
        concepts_activations=concepts_activations,
        concepts_importances=concepts_importances,
        concepts_labels=concepts_labels,
        top_k=2,
    )
    payload = _extract_payload(html, "GenerationLocalConceptsVisualization")

    assert {"sample", "labels", "activations", "importances", "top_k"} <= set(payload.keys())
    assert payload["sample"] == sample
    assert payload["labels"][0] == "A"
    assert payload["top_k"] == 2
    assert len(payload["importances"]) == 2
