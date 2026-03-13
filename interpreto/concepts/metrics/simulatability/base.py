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

import warnings
from abc import abstractmethod
from typing import NamedTuple

import torch

from interpreto.model_wrapping.llm_interface import LLMInterface, Role


class AutomatedSimulatability:
    """
    ... TODO
    """

    def __init__(self, classes: list[str]):
        """
        ... TODO
        """
        self.classes: list[str] = classes

    @staticmethod
    def select_examples(
        inputs: list[str],
        labels: torch.Tensor,
        predictions: torch.Tensor,
        nb_samples: int = 20,
        seed: int = 0,
        classes_subset: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
        """
        Extract interesting elements from the inputs, labels, and predictions.
        It selects `nb_samples` samples from the inputs.
        The goal is to select uniformly between each class (with respect to the labels).
        There should be as many samples where the initial model prediction are good as miss.
        The samples are then randomly shuffled.
        Therefore, there is no guarantee on the repartition inside learning and evaluation phase.

        Arguments:
            inputs: list[str]
                The inputs to predict.
            labels: torch.Tensor
                The labels of the inputs.
            predictions: torch.Tensor
                The predictions of the model on the inputs.
            nb_lp_samples: int
                The number of samples to select.
            seed: int
                The seed to use for the random selection.
            classes_subset: list[int] | None
                Optional subset of class ids to sample from.

        Returns:
            interesting_samples: list[str]
                The interesting samples.
            labels: torch.Tensor
                The labels of the interesting samples.
            predictions: torch.Tensor
                The predictions of the model on the interesting samples.
        """
        # ------------------------------------------------------------------------------------------
        # Compute constants
        class_ids = classes_subset if classes_subset is not None else torch.unique(labels).tolist()

        nb_classes = len(class_ids)
        nb_good = nb_samples // 2
        nb_miss = nb_samples - nb_good

        if nb_classes > nb_good:
            raise ValueError(
                f"Not enough samples ({nb_samples}) to represent the {nb_classes} classes in the learning phase."
                "Please increase the number of to at least 2 times the number of classes, or take a subset of classes."
            )

        # ------------------------------------------------------------------------------------------
        # Extract the good and miss indices with the label in the classes_subset
        label_in_subset = torch.isin(labels, torch.tensor(class_ids))
        good_indices = torch.nonzero((predictions == labels) & label_in_subset)
        miss_indices = torch.nonzero((predictions != labels) & label_in_subset)

        if len(good_indices) < nb_good or len(miss_indices) < nb_miss:
            raise ValueError(
                f"Not enough good or miss predictions to select {nb_good} good and {nb_miss} miss."
                f"Either provide more inputs (currently {len(good_indices)} good and {len(miss_indices)} miss)"
                "or reduce the number of samples to select."
            )

        # select random indices
        torch.random.manual_seed(seed)
        good_indices = good_indices[torch.randperm(len(good_indices))]
        miss_indices = miss_indices[torch.randperm(len(miss_indices))]

        # ------------------------------------------------------------------------------------------
        # Get a set number of good and miss indices for each class
        nb_good_per_class = nb_good // nb_classes
        nb_miss_per_class = nb_miss // nb_classes

        if nb_good_per_class == 0 or nb_miss_per_class == 0:
            warnings.warn(
                f"Not enough good ({nb_good_per_class}) or miss ({nb_miss_per_class})"
                f" predictions to represent the {nb_classes} classes inb both good and miss."
                "The classes of interest will be selected randomly.",
                stacklevel=2,
            )
            nb_good_per_class = 1
            nb_miss_per_class = 1

        # select good and miss indices for each class (this also ensures the predictions are in the classes_subset)
        class_wise_good_indices = []
        class_wise_miss_indices = []
        for c in class_ids:
            class_wise_good_indices.append(good_indices[predictions[good_indices] == c])
            class_wise_miss_indices.append(miss_indices[predictions[miss_indices] == c])

        selected_good_indices = torch.cat([c[:nb_good_per_class] for c in class_wise_good_indices])[:nb_good]
        selected_miss_indices = torch.cat([c[:nb_miss_per_class] for c in class_wise_miss_indices])[:nb_miss]

        # ------------------------------------------------------------------------------------------
        # Fill in the remaining indices if the number of classes and required good are not multiples
        nb_good_remaining = nb_good - nb_good_per_class * nb_classes
        if nb_good_remaining:
            remaining_good_indices = torch.cat([c[nb_good_per_class:] for c in class_wise_good_indices])
            new_indices = torch.randint(len(remaining_good_indices), (nb_good_remaining,))
            additional_good_indices = remaining_good_indices[new_indices]
            selected_good_indices = torch.cat([selected_good_indices, additional_good_indices])

        nb_miss_remaining = nb_miss - nb_miss_per_class * nb_classes
        if nb_miss_remaining:
            remaining_miss_indices = torch.cat([c[nb_miss_per_class:] for c in class_wise_miss_indices])
            new_indices = torch.randint(len(remaining_miss_indices), (nb_miss_remaining,))
            additional_miss_indices = remaining_miss_indices[new_indices]
            selected_miss_indices = torch.cat([selected_miss_indices, additional_miss_indices])

        # ------------------------------------------------------------------------------------------
        # Concatenate, shuffle indices, and return indexed elements
        indices = torch.cat([selected_good_indices, selected_miss_indices])

        # shuffle the indices
        indices = indices[torch.randperm(len(indices))]

        interesting_samples = [inputs[i] for i in indices]

        return indices, interesting_samples, labels[indices], predictions[indices]

    @abstractmethod
    def construct_prompt(
        self,
        setting: NamedTuple,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        **kwargs,
    ) -> tuple[str, list[str], list[str]]:
        """
        ... TODO
        """
        raise NotImplementedError

    @staticmethod
    def score_from_prompts(
        llm_interface: LLMInterface,
        system_prompt: str,
        user_prompts: list[str],
        model_predictions: list[str],
    ):
        """
        ... TODO
        """

        if len(user_prompts) != len(model_predictions):
            raise ValueError(
                "The number of user prompts and model predictions must be the same. "
                f"Got {len(user_prompts)} user prompts and {len(model_predictions)} model predictions."
            )

        score = 0
        for user, pred in zip(user_prompts, model_predictions, strict=True):
            prompt = [
                (Role.SYSTEM, system_prompt),
                (Role.USER, user),
            ]
            response = llm_interface.generate(prompt)
            if response is not None and response.lower() == pred.lower():
                score += 1

        return score / len(user_prompts)
