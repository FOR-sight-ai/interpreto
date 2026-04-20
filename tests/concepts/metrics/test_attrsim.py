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

import pytest
import torch

from interpreto.attributions.base import AttributionOutput, ModelTask
from interpreto.concepts.metrics.simulatability.attrsim import AttrSim, PromptSetting


def _build_attribution_output(
    attributions: torch.Tensor,
    *,
    elements: list[str] | None = None,
) -> AttributionOutput:
    if elements is None:
        elements = [f"t{i}" for i in range(attributions.shape[-1])]

    return AttributionOutput(
        attributions=attributions,
        elements=elements,
        model_inputs_to_explain={"input_ids": torch.tensor([[1, 2, 3]])},
        targets=torch.tensor([0]),
        model_task=ModelTask.CLASSIFICATION,
        classes=torch.tensor([0, 1]),
    )  # type: ignore


def test_prompt_settings_defaults_and_presets():
    default_setting = PromptSetting()
    assert default_setting.lp_samples is False
    assert default_setting.lp_attributions is False
    assert default_setting.lp_contrastive_attributions is False

    assert AttrSim.prompt_types.L1_baseline_without_lp.value == PromptSetting()
    assert AttrSim.prompt_types.L2_baseline_with_lp.value.lp_samples is True
    assert AttrSim.prompt_types.L2_baseline_with_lp.value.lp_attributions is False
    assert AttrSim.prompt_types.E1_attribution_with_lp.value.lp_samples is True
    assert AttrSim.prompt_types.E1_attribution_with_lp.value.lp_attributions is True


def test_format_attribution_for_pred_handles_singleton_class_axis():
    attribution = _build_attribution_output(torch.tensor([[0.2, -0.5, 0.1]]))

    rendered = AttrSim._format_attribution_for_pred(attribution_output=attribution, pred_index=2, top_k=2)

    assert "t1: -0.500" in rendered
    assert "t0: +0.200" in rendered


def test_construct_prompt_with_enum_setting():
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2", "s3"]
    predictions = torch.tensor([0, 1, 0, 1])
    labels = torch.tensor([0, 0, 1, 1])
    attributions = [_build_attribution_output(torch.tensor([[0.1, -0.2, 0.3]])) for _ in samples]

    system_prompt, user_prompts, model_predictions = metric.construct_prompt(
        setting=AttrSim.prompt_types.E1_attribution_with_lp,
        interesting_samples=samples,
        corresponding_predictions=predictions,
        corresponding_labels=labels,
        nb_learning_samples=2,
        corresponding_attribution=attributions,
    )

    assert "Attributions for A:" in system_prompt
    assert len(user_prompts) == 2
    assert model_predictions == ["A", "B"]


def test_construct_prompt_validates_lengths():
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2"]
    predictions = torch.tensor([0, 1, 0])
    labels = torch.tensor([0, 1, 0])
    attributions = [_build_attribution_output(torch.tensor([[0.1, -0.2, 0.3]])) for _ in range(2)]

    with pytest.raises(ValueError, match="same length"):
        metric.construct_prompt(
            setting=AttrSim.prompt_types.E1_attribution_with_lp,
            interesting_samples=samples,
            corresponding_predictions=predictions,
            corresponding_labels=labels,
            nb_learning_samples=1,
            corresponding_attribution=attributions,
        )


def test_construct_prompt_rejects_too_many_learning_samples():
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1"]
    predictions = torch.tensor([0, 1])
    labels = torch.tensor([0, 1])
    attributions = [_build_attribution_output(torch.tensor([[0.1, -0.2, 0.3]])) for _ in samples]

    with pytest.raises(ValueError, match="nb_learning_samples"):
        metric.construct_prompt(
            setting=AttrSim.prompt_types.E1_attribution_with_lp,
            interesting_samples=samples,
            corresponding_predictions=predictions,
            corresponding_labels=labels,
            nb_learning_samples=2,
            corresponding_attribution=attributions,
        )


def test_construct_prompt_with_contrastive_attributions():
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2", "s3"]
    predictions = torch.tensor([0, 1, 0, 1])
    labels = torch.tensor([0, 0, 1, 1])  # sample 1 and 2 are misclassified
    attributions = [
        _build_attribution_output(torch.tensor([[0.4, -0.1, 0.2], [0.1, 0.2, -0.3]])),
        _build_attribution_output(torch.tensor([[0.2, -0.5, 0.1], [0.6, -0.1, -0.2]])),
        _build_attribution_output(torch.tensor([[0.1, -0.3, 0.5], [-0.2, 0.7, 0.1]])),
        _build_attribution_output(torch.tensor([[0.2, 0.2, -0.2], [0.3, -0.4, 0.1]])),
    ]

    system_prompt, user_prompts, model_predictions = metric.construct_prompt(
        setting=AttrSim.prompt_types.C1_contrastive_attribution_with_lp,
        interesting_samples=samples,
        corresponding_predictions=predictions,
        corresponding_labels=labels,
        nb_learning_samples=2,
        corresponding_attribution=attributions,
    )

    assert "Contrastive Attributions supporting B rather than A" in system_prompt
    assert "Attributions for A" in system_prompt
    assert len(user_prompts) == 2
    assert model_predictions == ["A", "B"]


def test_construct_prompt_contrastive_requires_classwise_attribution_for_miss():
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2"]
    predictions = torch.tensor([0, 1, 0])
    labels = torch.tensor([0, 0, 1])  # one miss in LP when nb_learning_samples=2
    attributions = [
        _build_attribution_output(torch.tensor([[0.1, -0.2, 0.3]])),
        _build_attribution_output(torch.tensor([[0.2, -0.5, 0.1]])),  # singleton axis, not classwise
        _build_attribution_output(torch.tensor([[0.4, -0.1, 0.2]])),
    ]

    with pytest.raises(ValueError, match="require class-wise attributions"):
        metric.construct_prompt(
            setting=AttrSim.prompt_types.C1_contrastive_attribution_with_lp,
            interesting_samples=samples,
            corresponding_predictions=predictions,
            corresponding_labels=labels,
            nb_learning_samples=2,
            corresponding_attribution=attributions,
        )
