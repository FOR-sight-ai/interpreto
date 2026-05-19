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


class AutomatedSimulatability:
    """
    Base class for prompt-based simulatability metrics.

    An automated simulatability metric turns model decisions and explanation artifacts into
    prompts for a meta-predictor, typically an LLM, then measures how often that meta-predictor
    reproduces the original model outputs.

    Architecture:
        - `select_examples` extracts a balanced subset from precomputed model outputs.
        - upstream code computes explanation artifacts for those samples.
        - subclasses implement `construct_prompt` to turn those samples and explanations into
          prompts.
        - caller code sends the prompts to an LLM interface or any other meta-predictor.
        - `score_from_responses` computes exact-match accuracy from the returned responses.

    Example:
        `AutomatedSimulatability` is abstract, so a minimal subclass is needed to show the
        end-to-end flow:

        >>> import torch
        >>> from typing import NamedTuple
        >>> from interpreto.concepts.metrics.simulatability.base import AutomatedSimulatability
        >>>
        >>> class ToySetting(NamedTuple):
        ...     pass
        >>>
        >>> class ToyMetric(AutomatedSimulatability):
        ...     def construct_prompt(
        ...         self,
        ...         setting,
        ...         interesting_samples,
        ...         corresponding_predictions,
        ...         corresponding_labels,
        ...         **kwargs,
        ...     ):
        ...         _ = (setting, corresponding_labels, kwargs)
        ...         system_prompt = "Return only the class name."
        ...         user_prompts = [f"Text: {sample}\\nLabel: " for sample in interesting_samples]
        ...         model_predictions = [
        ...             self.classes[int(prediction)] for prediction in corresponding_predictions.tolist()
        ...         ]
        ...         return system_prompt, user_prompts, model_predictions
        >>>
        >>> metric = ToyMetric(classes=["negative", "positive"])
        >>> inputs = ["awful movie", "great movie", "bad ending", "excellent ending"]
        >>> labels = torch.tensor([0, 1, 1, 0])
        >>> predictions = torch.tensor([0, 1, 0, 1])
        >>> _, interesting_samples, selected_labels, selected_predictions = metric.select_examples(
        ...     inputs=inputs,
        ...     labels=labels,
        ...     predictions=predictions,
        ...     nb_samples=4,
        ...     seed=0,
        ... )
        >>> system_prompt, user_prompts, model_predictions = metric.construct_prompt(
        ...     ToySetting(),
        ...     interesting_samples,
        ...     selected_predictions,
        ...     selected_labels,
        ... )
        >>> len(system_prompt) > 0 and len(user_prompts) == len(model_predictions)
        True
        >>> # In practice, call your LLM here:
        >>> # responses = llm_interface.batch_generate(system_prompt, user_prompts)
        >>> responses = list(model_predictions)
        >>> metric.score_from_responses(responses, model_predictions)
        1.0
    """

    def __init__(self, classes: list[str]):
        """
        Store the class names used by the metric.

        Arguments:
            classes: list[str]
                Class names indexed by model class id. `classes[i]` is the label expected for
                prediction `i`.
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
        Select a balanced subset of samples for a simulatability run.

        The method works on already computed `labels` and `predictions`. It tries to keep roughly
        half correct and half incorrect model predictions, while representing every requested class
        in both groups. The final subset is shuffled, so callers can later split it into learning
        and evaluation samples however they want.

        The goal is to find `interesting_samples` for the metric. With this, we over represent
        misclassifications and some classes. Therefore, the meta-predictor used after
        `construct_prompt` cannot shortcut the task by predicting real labels, otherwise it would
        obtain a score of 0.5.

        Arguments:
            inputs: list[str]
                Raw inputs, length `all_samples`.
            labels: torch.Tensor
                Gold labels aligned with `inputs`, shape `(all_samples,)`.
            predictions: torch.Tensor
                Model predictions aligned with `inputs`, shape `(all_samples,)`.
            nb_samples: int
                Number of samples to keep in the returned subset.
            seed: int
                Seed used for the random sampling and final shuffle.
            classes_subset: list[int] | None
                Optional subset of class ids to select samples from.
                When provided, only samples with prediction and label in that subset are kept.

        Returns:
            indices: torch.Tensor
                Indices of the selected samples in the original inputs, shape `(nb_samples,)`.
            interesting_samples: list[str]
                Selected inputs, length `nb_samples`.
            labels: torch.Tensor
                Selected labels, shape `(nb_samples,)`.
            predictions: torch.Tensor
                Selected predictions, shape `(nb_samples,)`.

        Raises:
            ValueError:
                If `nb_samples` is too small to cover the requested classes, or if there are not
                enough correct / incorrect predictions to build the balanced subset.
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
        Build prompts for a concrete simulatability metric.

        This is the customization point for subclasses.

        A typical implementation:
        - validates metric-specific inputs,
        - decides which selected samples to use in the learning phase or evaluation phase,
        - and returns a shared system prompt plus one user prompt per evaluation sample.

        A prompt have:
        - A task description, adapted to the type of explanations that will be provided.
        - The classes names list, potentially anonymized.
        - Optional global concept summaries.
        - Optional learning-phase examples with their predictions.
        - One user prompt per evaluation sample.

        Arguments:
            setting: NamedTuple
                Metric-specific configuration describing which prompt blocks to include.
            interesting_samples: list[str]
                Selected inputs, shape `(n_samples,)`.
            corresponding_predictions: torch.Tensor
                Model predictions aligned with `interesting_samples`, shape `(n_samples,)`.
            corresponding_labels: torch.Tensor
                Reference labels aligned with `interesting_samples`, shape `(n_samples,)`.
            **kwargs:
                Metric-specific explanation artifacts required to render the prompt.

        Returns:
            system_prompt: str
                Shared instructions and context reused for every evaluation request.
            user_prompts: list[str]
                One prompt per evaluation sample.
            model_predictions: list[str]
                Expected class names aligned with `user_prompts`.

        Notes:
            `score_from_responses` assumes `len(user_prompts) == len(model_predictions)` and that
            downstream generation returns one response per prompt.
        """
        raise NotImplementedError

    @staticmethod
    def score_from_responses(
        responses: list[str | None],
        model_predictions: list[str],
    ):
        """
        Score precomputed responses with exact-match accuracy.

        Each response is compared case-insensitively to the expected class name in
        `model_predictions`.

        Keeping scoring here lets subclasses focus on prompt construction while callers remain
        free to cache prompts, swap LLM backends, or replace the scorer entirely if they need a
        more elaborate protocol.

        Arguments:
            responses: list[str | None]
                One generated response per evaluation prompt.
            model_predictions: list[str]
                Expected class names aligned with `responses`.

        Returns:
            simulatability_score: float
                Mean exact-match accuracy in `[0, 1]`.

        Raises:
            ValueError:
                If `responses` and `model_predictions` do not have the same length.
        """

        if len(responses) != len(model_predictions):
            raise ValueError(
                "The number of responses and model predictions must be the same. "
                f"Got {len(responses)} responses and {len(model_predictions)} model predictions."
            )

        score = 0
        for llm_pred, ref_pred in zip(responses, model_predictions, strict=True):
            if llm_pred is None:
                continue

            if llm_pred.split(" ")[0].lower() == ref_pred.lower():
                score += 1

        return score / len(responses)
