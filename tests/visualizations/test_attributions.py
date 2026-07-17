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

# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupery et Universite Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupery,
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

from interpreto.attributions.base import AttributionOutput, ModelTask
from interpreto.visualizations import plot_attributions


def _extract_payload(html: str, viz_name: str) -> dict:
    match = re.search(rf"new {viz_name}\((.*?)\);", html, re.DOTALL)
    assert match, f"Expected {viz_name} call in HTML."
    call_args = match.group(1)
    json_literal_match = re.search(r'"(?:\\.|[^"\\])*"', call_args)
    assert json_literal_match, "Expected JSON payload in visualization call."
    json_string = json.loads(json_literal_match.group(0))
    return json.loads(json_string)


def _render_attributions_html(tmp_path, attribution_output: AttributionOutput, **kwargs: object) -> str:
    save_path = tmp_path / "attributions.html"
    plot_attributions(attribution_output, save_path=save_path, **kwargs)
    return save_path.read_text(encoding="utf-8")


def test_plot_attributions_classification_single_class_payload_keys(tmp_path):
    attribution_output = AttributionOutput(
        model_inputs_to_explain=None,
        attributions=torch.tensor([[0.1, -0.2, 0.3]]),
        elements=["1", "2", "3"],
        targets=torch.tensor([[1]]),
        model_task=ModelTask.CLASSIFICATION,
    )

    html = _render_attributions_html(
        tmp_path,
        attribution_output,
        classes_names={1: "positive"},
    )
    payload = _extract_payload(html, "ClassificationVisualization")

    assert {"classes", "inputs", "outputs", "custom_style", "onclick_colormap"} <= set(payload.keys())
    assert payload["inputs"]["words"] == ["1", "2", "3"]
    assert payload["outputs"]["words"] is None
    assert payload["classes"][0]["id"] == 1
    assert payload["classes"][0]["name"] == "positive"
    assert payload["onclick_colormap"] == ["#ff0000", "#0000ff"]


def test_plot_attributions_classification_multiclass_uses_targets(tmp_path):
    attribution_output = AttributionOutput(
        model_inputs_to_explain=None,
        attributions=torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]]),
        elements=["a", "b", "c"],
        targets=torch.tensor([3, 5]),
        model_task=ModelTask.CLASSIFICATION,
    )

    html = _render_attributions_html(
        tmp_path,
        attribution_output,
        classes_names={3: "Class A", 5: "Class B"},
    )
    payload = _extract_payload(html, "ClassificationVisualization")

    class_ids = [entry["id"] for entry in payload["classes"]]
    assert class_ids == [3, 5]
    assert [entry["name"] for entry in payload["classes"]] == ["Class A", "Class B"]
    assert all("color" in entry for entry in payload["classes"])
    assert len(payload["inputs"]["attributions"][0]) == len(attribution_output.elements)
    assert len(payload["inputs"]["attributions"][0][0]) == 2


def test_plot_attributions_classification_accepts_list_names(tmp_path):
    attribution_output = AttributionOutput(
        model_inputs_to_explain=None,
        attributions=torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]]),
        elements=["a", "b", "c"],
        targets=torch.tensor([0, 1]),
        model_task=ModelTask.CLASSIFICATION,
    )

    html = _render_attributions_html(
        tmp_path,
        attribution_output,
        classes_names=["negative", "positive"],
    )
    payload = _extract_payload(html, "ClassificationVisualization")

    assert [entry["name"] for entry in payload["classes"]] == ["negative", "positive"]


def test_plot_attributions_generation_includes_nulls(tmp_path):
    nan = float("nan")
    attribution_output = AttributionOutput(
        model_inputs_to_explain=None,
        attributions=torch.tensor(
            [
                [0.1, nan, 0.2, nan],
                [0.0, 0.3, nan, 0.4],
            ]
        ),
        elements=["in1", "in2", "out1", "out2"],
        targets=torch.tensor([0, 1]),
        model_task=ModelTask.GENERATION,
    )

    html = _render_attributions_html(tmp_path, attribution_output)
    payload = _extract_payload(html, "GenerationVisualization")

    assert {"classes", "inputs", "outputs", "custom_style"} <= set(payload.keys())
    assert payload["inputs"]["words"] == ["in1", "in2"]
    assert payload["outputs"]["words"] == ["out1", "out2"]
    assert payload["classes"][0]["name"] == "None"
    assert payload["inputs"]["attributions"][0][1][0] is None
    assert payload["outputs"]["attributions"][0][1][0] is None
