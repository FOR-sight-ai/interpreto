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
    """Low-level AttrSim prompt configuration."""

    lp_samples: bool = True
    lp_attributions: bool = True
    anonymize_classes: bool = False


class PromptTypes(Enum):
    """Named AttrSim prompt presets."""

    A1_attributions_with_lp = PromptSetting(lp_samples=True, lp_attributions=True)


class AttrSim(AutomatedSimulatability):
    """Attribution-based simulatability prompt builder.

    AttrSim mirrors ConSim but uses token-level attribution explanations instead of concept explanations.
    """

    prompt_types: type[PromptTypes] = PromptTypes

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
        if isinstance(setting, PromptTypes):
            setting = setting.value

        if len(interesting_samples) != len(corresponding_predictions):
            raise ValueError("`interesting_samples` and `corresponding_predictions` must have the same length.")
        if len(interesting_samples) != len(corresponding_labels):
            raise ValueError("`interesting_samples` and `corresponding_labels` must have the same length.")
        if len(interesting_samples) != len(corresponding_attribution):
            raise ValueError("`interesting_samples` and `corresponding_attribution` must have the same length.")
        if nb_learning_samples >= len(interesting_samples):
            raise ValueError("`nb_learning_samples` must be smaller than number of provided samples.")

        classes = {i: c for i, c in enumerate(self.classes)}
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
                lp_blocks.append("\n".join(lp_block))

            system_prompt_parts.append("\n".join(lp_blocks))

        system_prompt = "\n\n".join(system_prompt_parts)

        user_prompts: list[str] = []
        model_predictions: list[str] = []
        for i in range(nb_learning_samples, len(interesting_samples)):
            user_prompts.append(f"Evaluation sample:\n\tText: {interesting_samples[i]}\n\tLabel: ")
            model_predictions.append(classes[int(corresponding_predictions[i])])

        return system_prompt, user_prompts, model_predictions
