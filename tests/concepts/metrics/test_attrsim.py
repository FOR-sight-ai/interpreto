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


def _build_attr_output(
    attributions: torch.Tensor,
    *,
    elements: list[str] | None = None,
    classes: torch.Tensor | None = None,
) -> AttributionOutput:
    if elements is None:
        elements = [f"t{i}" for i in range(attributions.shape[-1])]

    if classes is None:
        if attributions.ndim == 2:
            classes = torch.arange(attributions.shape[0])
        else:
            classes = torch.tensor([0])

    return AttributionOutput(
        attributions=attributions,
        elements=elements,
        model_inputs_to_explain={"input_ids": torch.tensor([[1, 2, 3]])},
        targets=torch.tensor([0]),
        model_task=ModelTask.CLASSIFICATION,
        classes=classes,
    )


def test_prompt_settings_defaults_and_presets():
    """Check AttrSim defaults and shipped prompt presets."""
    default_setting = PromptSetting()

    assert default_setting.lp_samples is False
    assert default_setting.lp_attributions is False
    assert default_setting.lp_contrastive_attributions is False
    assert default_setting.attribution_top_k == 6
    assert default_setting.attribution_onlypositivevalues is True

    assert AttrSim.prompt_types.L1_baseline_without_lp.value == PromptSetting()
    assert AttrSim.prompt_types.L2_baseline_with_lp.value.lp_samples is True
    assert AttrSim.prompt_types.E1_attribution_with_lp.value.lp_attributions is True
    assert AttrSim.prompt_types.C1_contrastive_attribution_with_lp.value.lp_contrastive_attributions is True


def test_format_attribution_singleton_axis():
    """Singleton class axis should be handled and normalized correctly."""
    output = _build_attr_output(torch.tensor([[0.2, -0.5, 0.1]]))

    rendered = AttrSim._format_attribution_for_pred(
        attribution_output=output,
        pred_index=2,
        top_k=2,
        only_positive_values=False,
    )

    assert "t1: -0.625" in rendered
    assert "t0: +0.250" in rendered


def test_format_attr_vector_positive_only():
    """Positive-only mode keeps top positive normalized scores."""
    rendered = AttrSim._format_attr_vector(
        elements=["a", "b", "c"],
        attr_vector=torch.tensor([0.1, -0.9, 0.2]),
        top_k=2,
        only_positive_values=True,
    )

    assert "c: +0.167" in rendered
    assert "a: +0.083" in rendered
    assert "b: -0.750" not in rendered


def test_construct_prompt_with_attribution_lp():
    """Prompt construction should include LP attributions and evaluation targets."""
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2", "s3"]
    predictions = torch.tensor([0, 1, 0, 1])
    labels = torch.tensor([0, 0, 1, 1])
    attributions = [_build_attr_output(torch.tensor([[0.1, -0.2, 0.3]])) for _ in samples]

    system_prompt, user_prompts, model_predictions = metric.construct_prompt(
        setting=AttrSim.prompt_types.E1_attribution_with_lp,
        interesting_samples=samples,
        corresponding_predictions=predictions,
        corresponding_labels=labels,
        nb_learning_samples=2,
        corresponding_attribution=attributions,
    )

    assert "Attributions:" in system_prompt
    assert len(user_prompts) == 2
    assert model_predictions == ["A", "B"]


def test_construct_prompt_with_contrastive_lp():
    """Contrastive setting should render both standard and contrastive attribution labels."""
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2", "s3"]
    predictions = torch.tensor([0, 1, 0, 1])
    labels = torch.tensor([0, 0, 1, 1])  # sample 1 is misclassified in LP
    attributions = [
        _build_attr_output(torch.tensor([[0.4, -0.1, 0.2], [0.1, 0.2, -0.3]])),
        _build_attr_output(torch.tensor([[0.2, -0.5, 0.1], [0.6, -0.1, -0.2]])),
        _build_attr_output(torch.tensor([[0.1, -0.3, 0.5], [-0.2, 0.7, 0.1]])),
        _build_attr_output(torch.tensor([[0.2, 0.2, -0.2], [0.3, -0.4, 0.1]])),
    ]

    system_prompt, user_prompts, model_predictions = metric.construct_prompt(
        setting=AttrSim.prompt_types.C1_contrastive_attribution_with_lp,
        interesting_samples=samples,
        corresponding_predictions=predictions,
        corresponding_labels=labels,
        nb_learning_samples=2,
        corresponding_attribution=attributions,
    )

    assert "Attributions for A" in system_prompt
    assert "Contrastive Attributions for supporting B rather than A" in system_prompt
    assert len(user_prompts) == 2
    assert model_predictions == ["A", "B"]


def test_construct_prompt_validates_lengths_and_lp_count():
    """Invalid lengths and invalid LP size should raise ValueError."""
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2"]
    predictions = torch.tensor([0, 1, 0])
    labels = torch.tensor([0, 1, 0])

    attributions_too_short = [_build_attr_output(torch.tensor([[0.1, -0.2, 0.3]])) for _ in range(2)]
    with pytest.raises(ValueError, match="same length"):
        metric.construct_prompt(
            setting=AttrSim.prompt_types.E1_attribution_with_lp,
            interesting_samples=samples,
            corresponding_predictions=predictions,
            corresponding_labels=labels,
            nb_learning_samples=1,
            corresponding_attribution=attributions_too_short,
        )

    attributions_ok = [_build_attr_output(torch.tensor([[0.1, -0.2, 0.3]])) for _ in samples]
    with pytest.raises(ValueError, match="nb_learning_samples"):
        metric.construct_prompt(
            setting=AttrSim.prompt_types.E1_attribution_with_lp,
            interesting_samples=samples,
            corresponding_predictions=predictions,
            corresponding_labels=labels,
            nb_learning_samples=3,
            corresponding_attribution=attributions_ok,
        )


def test_construct_prompt_contrastive_requires_classwise_for_miss():
    """Misclassified LP samples require class-wise attributions in contrastive mode."""
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2"]
    predictions = torch.tensor([0, 1, 0])
    labels = torch.tensor([0, 0, 1])

    attributions = [
        _build_attr_output(torch.tensor([[0.1, -0.2, 0.3]])),
        _build_attr_output(torch.tensor([[0.2, -0.5, 0.1]])),  # singleton axis for misclassified sample
        _build_attr_output(torch.tensor([[0.4, -0.1, 0.2]])),
    ]

    with pytest.raises(ValueError, match="class-wise attributions"):
        metric.construct_prompt(
            setting=AttrSim.prompt_types.C1_contrastive_attribution_with_lp,
            interesting_samples=samples,
            corresponding_predictions=predictions,
            corresponding_labels=labels,
            nb_learning_samples=2,
            corresponding_attribution=attributions,
        )


def test_prompt_setting_rejects_non_positive_top_k():
    """Prompt setting should reject non-positive top-k values."""
    metric = AttrSim(classes=["A", "B"])
    samples = ["s0", "s1", "s2"]
    predictions = torch.tensor([0, 1, 0])
    labels = torch.tensor([0, 0, 1])
    attributions = [_build_attr_output(torch.tensor([[0.1, -0.2, 0.3]])) for _ in samples]

    with pytest.raises(ValueError, match="attribution_top_k"):
        metric.construct_prompt(
            setting=PromptSetting(lp_samples=True, lp_attributions=True, attribution_top_k=0),
            interesting_samples=samples,
            corresponding_predictions=predictions,
            corresponding_labels=labels,
            nb_learning_samples=2,
            corresponding_attribution=attributions,
        )
