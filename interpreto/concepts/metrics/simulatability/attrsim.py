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

from enum import Enum
from typing import NamedTuple

import torch

from interpreto.attributions.base import AttributionOutput
from interpreto.concepts.metrics.simulatability.base import AutomatedSimulatability


class PromptSetting(NamedTuple):
    """
    Low-level configuration of a AttrSim prompt.

    Each flag enables one prompt block. `PromptTypes` exposes the common presets used in papers and
    tests, while direct `PromptSetting(...)` instances let advanced users define custom ablations.

    Attributes:
        lp_samples: bool
            Include learning-phase examples in the shared system prompt.
        lp_attributions: bool
            Add attribution explanations for each learning-phase example.
        lp_contrastive_attributions: bool
            Add contrastive attribution explanations for each learning-phase example.
            Contrastive are shown for errors and classic contributions for correct predictions.
            Incompatible with `lp_attributions`.
        anonymize_classes: bool
            Replace user-facing class names with `Class_i`.
            Preventing the LLM from using knowledge on classes names.
    """

    lp_samples: bool = True
    lp_attributions: bool = True
    lp_contrastive_attributions: bool = False
    anonymize_classes: bool = False

    def validate(
        self,
        *,
        labels: torch.Tensor | list[int] | None,
    ) -> None:
        """
        Validate internal consistency for a prompt setting.

        This method only checks setting-level constraints, such as mutually exclusive options or
        inputs required by a given prompt family. Tensor shape checks are handled separately in
        `AttrSim._check_input_settings_correspondence` so callers fail before any prompt text is
        rendered.

        Arguments:
            labels: torch.Tensor | list[int] | None
                Gold labels aligned with the selected samples. Required for contrastive prompts.

        Raises:
            ValueError:
                If the setting is inconsistent or requires missing inputs.
        """
        if self.lp_attributions and self.lp_contrastive_attributions:
            raise ValueError(
                "PromptSetting.lp_attributions and PromptSetting.lp_contrastive_attributions are mutually exclusive."
            )

        if not (self.lp_samples) and self.lp_attributions:
            raise ValueError("PromptSetting.lp_attributions requires `lp_samples=True`.")

        if not (self.lp_samples) and self.lp_contrastive_attributions:
            raise ValueError("PromptSetting.lp_contrastive_attributions requires `lp_samples=True`.")

        if self.lp_contrastive_attributions and labels is None:
            raise ValueError(
                "PromptSetting.lp_contrastive_attributions=True requires `labels` to be provided to AttrSim.construct_prompt()."
            )


class PromptTypes(Enum):
    """
    Named AttrSim prompt presets.

    Naming convention:
        - `L*`: baselines without attribution explanations.
        - `E*`: standard attribution explanations during learning phase.
        - `C*`: contrastive attribution explanations during learning phase.
        - `with_lp` / `without_lp`: whether learning-phase examples are included.

    Each enum value is a `PromptSetting`. Use the enum for standard experiments and direct
    `PromptSetting(...)` values for custom studies.
    """

    L1_baseline_without_lp = PromptSetting()
    L2_baseline_with_lp = PromptSetting(lp_samples=True)

    E1_attribution_with_lp = PromptSetting(lp_samples=True, lp_attributions=True)

    C1_contrastive_attribution_with_lp = PromptSetting(lp_samples=True, lp_contrastive_attributions=True)


class AttrSim(AutomatedSimulatability):
    """
    AttrSim prompt builder for attribution-based automated simulatability.

    AttrSim measures whether attribution explanations help a meta-predictor reproduce a classifier's
    outputs. In this module, `AttrSim` is responsible only for AttrSim-specific prompt
    design and validation. It does not compute model predictions, token/word/sentence-level attributions, or call the
    LLM on its own.

    Therefore, users need to compute model predictions and attribution explanations beforehand.

    Typical workflow:
        1. Instantiate `AttrSim(classes=...)`.
        2. Call `select_examples(...)` on precomputed inputs, labels, and model predictions.
        3. Use a fitted attribution explainer upstream to build the explanation artifacts required by
           the chosen setting.
        4. Call `construct_prompt(...)`.
        5. Run the prompts through your LLM interface outside this class.
        6. Compute responses with `llm_interface.batch_generate(...)`.
        7. Score the returned responses with `score_from_responses(...)`.

    Arguments:
        classes: list[str]
            Display names for class ids. Inherited from `AutomatedSimulatability`; `classes[i]`
            must match class id `i`.

    Attributes:
        classes: list[str]
            Display names for class ids.
        prompt_types: type[PromptTypes]
            Preset prompt configurations shipped with AttrSim.
            These are prompt settings that can be passed to `AttrSim.construct_prompt()`.
    """

    prompt_types: type[PromptTypes] = PromptTypes

    @staticmethod
    def _resolve_prompt_setting(setting: PromptTypes | PromptSetting) -> PromptSetting:
        return setting.value if isinstance(setting, PromptTypes) else setting

    @staticmethod
    def _format_attribution_for_pred(
        attribution_output: AttributionOutput,
        pred_index: int,
        top_k: int = 6,
    ) -> str:
        elements = attribution_output.elements
        if isinstance(elements, torch.Tensor):
            elements = [str(e.item()) for e in elements]
        else:
            elements = [str(e) for e in elements]

        attributions = attribution_output.attributions
        if attributions.ndim == 1:
            pred_attr = attributions
        elif attributions.ndim == 2 and attributions.shape[0] == 1:
            pred_attr = attributions[0]
        else:
            pred_attr = attributions[pred_index]

        top_k = min(top_k, pred_attr.shape[-1])
        top_indices = torch.topk(pred_attr.abs(), k=top_k).indices.tolist()

        pieces = []
        for idx in top_indices:
            token = elements[idx] if idx < len(elements) else f"tok_{idx}"
            pieces.append(f"{token}: {pred_attr[idx].item():+.3f}")
        return "{" + ", ".join(pieces) + "}"

    def construct_prompt(
        self,
        setting: PromptTypes | PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        *,
        corresponding_attribution: list[AttributionOutput],
    ) -> tuple[str, list[str], list[str]]:
        setting = self._resolve_prompt_setting(setting)
        self._check_input_settings_correspondence(
            setting=setting,
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            nb_learning_samples=nb_learning_samples,
            corresponding_attribution=corresponding_attribution,
        )

        classes_ids = sorted(corresponding_predictions.unique().tolist())
        classes = {class_id: self.classes[class_id] for class_id in classes_ids}

        if setting.anonymize_classes:
            classes = {i: f"Class_{i}" for i in classes.keys()}

        system_prompt_parts = [
            "You are a classifier. Predict the class for each evaluation sample.",
            "Use the provided learning examples and attribution explanations to infer the model behavior.",
            "Only return the class name, no additional text.",
            f"The classes are: [{', '.join(list(classes.values()))}]",
        ]

        if setting.lp_samples:
            lp_blocks = []
            for i in range(nb_learning_samples):
                pred_index = int(corresponding_predictions[i])
                lp_block = [
                    f"Sample_{i}:",
                    f"\tText: {interesting_samples[i]}",
                    f"\tLabel: {classes[pred_index]}",
                ]
                if setting.lp_attributions:
                    lp_block.append(
                        f"\tAttributions: {self._format_attribution_for_pred(corresponding_attribution[i], pred_index)}"
                    )
                if setting.lp_contrastive_attributions:
                    raise NotImplementedError("Contrastive attribution formatting is not implemented yet.")
                lp_blocks.append("\n".join(lp_block))

            system_prompt_parts.append("\n".join(lp_blocks))

        system_prompt = "\n\n".join(system_prompt_parts)

        user_prompts: list[str] = []
        model_predictions: list[str] = []
        for i in range(nb_learning_samples, len(interesting_samples)):
            user_prompts.append(f"Evaluation sample:\n\tText: {interesting_samples[i]}\n\tLabel: ")
            model_predictions.append(classes[int(corresponding_predictions[i])])

        return system_prompt, user_prompts, model_predictions

    def _check_input_settings_correspondence(
        self,
        setting: PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        corresponding_attribution: list[AttributionOutput] | None,
    ) -> None:
        setting.validate(labels=corresponding_labels)

        if len(corresponding_predictions) != len(interesting_samples):
            raise ValueError("`interesting_samples` and `corresponding_predictions` must have the same length.")

        if len(corresponding_labels) != len(interesting_samples):
            raise ValueError("`interesting_samples` and `corresponding_labels` must have the same length.")

        if nb_learning_samples >= len(interesting_samples):
            raise ValueError("`nb_learning_samples` must be smaller than number of provided samples.")

        if corresponding_attribution is None:
            if setting.lp_attributions or setting.lp_contrastive_attributions:
                raise ValueError(
                    "`corresponding_attribution` is required when using attribution-based learning prompts."
                )
            return

        if len(corresponding_attribution) != len(interesting_samples):
            raise ValueError("`interesting_samples` and `corresponding_attribution` must have the same length.")
