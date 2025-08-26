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

"""
Deletion and Insertion metrics

This module contains classes for the Insertion and Deletion metrics, which are perturbation-based metrics where we
iteratively include/occlude more tokens from the input text based on the importance of the attributions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from interpreto.attributions.aggregations.insertion_deletion_aggregation import InsertionDeletionAggregator
from interpreto.attributions.base import AttributionExplainer, AttributionOutput, MultitaskExplainerMixin
from interpreto.attributions.perturbations.insertion_deletion_perturbation import (
    DeletionPerturbator,
    InsertionPerturbator,
)
from interpreto.commons.generator_tools import split_iterator
from interpreto.commons.granularity import Granularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes
from interpreto.typing import TensorMapping


class InsertionDeletionBase(MultitaskExplainerMixin, AttributionExplainer):
    """
    Abstract base class for Insertion and Deletion metrics. Only the perturbator class is different between the two
    metrics.

    This class implements the core logic for insertion and deletion metrics, where tokens are either inserted or
    deleted from the input text based on their importance scores. The perturbator class is responsible for
    handling the specific perturbation logic (insertion or deletion).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        granularity: Granularity = Granularity.TOKEN,
        device: torch.device | None = None,
        n_perturbations: int = 100,
        max_percentage_perturbed: float = 1.0,
    ):
        """
        Initializes the metric.

        Args:
            model (PreTrainedModel): model used to generate explanations
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            batch_size (int): batch size for the inference of the metric
            granularity (Granularity): granularity level of the perturbations (token, word, sentence, etc.)
            device (torch.device): device on which the attribution method will be run
            n_perturbations (int): number of perturbations from which the metric will be computed (i.e. the number of
                steps from which the AUC will be computed).
            max_percentage_perturbed (float): maximum percentage of elements in the sequence to be perturbed. Defaults
                to 1.0, meaning that all elements can be perturbed. If set to 0.5, only half of the elements will be
                perturbed (i.e. only the 50% most important). This is useful to avoid perturbing too many elements with
                low scores in long sequences.
        """
        model, replace_token_id = self._set_tokenizer(model, tokenizer)

        super().__init__(
            model=model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            device=device,
            perturbator=self._perturbator_class(
                tokenizer=self.tokenizer,
                granularity=granularity,
                n_perturbations=n_perturbations,
                max_percentage_perturbed=max_percentage_perturbed,
                replace_token_id=replace_token_id,
            ),
            aggregator=InsertionDeletionAggregator(),
            granularity=granularity,
            inference_mode=InferenceModes.SOFTMAX,
            use_gradient=False,
        )

    @property
    @abstractmethod
    def _perturbator_class(self) -> Callable:
        """
        Returns the perturbator class used for insertion or deletion.

        This method should be overridden in subclasses to return the appropriate perturbator.
        """
        raise NotImplementedError()

    def explain(*args, **kwargs):  # type: ignore (incoherent override)
        """This method, inherited from AttributionExplainer, is not implemented for Insertion and Deletion metrics."""

        raise NotImplementedError(
            "Insertion and Deletion metrics do not support `explain` method. "
            "Use the `evaluate` method to evaluate the deletion metric."
        )

    def evaluate(self, attributions_outputs: Iterable[AttributionOutput]) -> tuple[float, list[np.ndarray]]:
        """
        Evaluates the insertion or deletion metric based on the provided attributions.

        Args:
            attributions_outputs (Iterable[AttributionOutput]): Outputs from the attribution method.

        Returns:
            tuple: A tuple containing:
                - the average AUC score across all attributions
                - a list of scores for each sequence, where each element is a numpy array of scores.
        """

        # Assert that the granularity of all attributions is the same as the granularity of the deletion.
        grans = [a.granularity for a in attributions_outputs]
        if not all(g == self.granularity for g in grans):
            raise ValueError(
                f"All attributions must have the same granularity as the insertion/deletion metric. "
                f"Expected {self.granularity} for all attributions."
                "If you need to use attributions with different granularities, please raise an issue on GitHub."
            )

        # Create perturbation masks and perturb inputs based on the masks.
        pert_generator: Iterable[TensorMapping]
        mask_generator: Iterable[torch.Tensor | None]
        pert_generator, mask_generator = split_iterator(
            self.perturbator.perturb(a.model_inputs_to_explain, attributions=a.attributions)
            for a in attributions_outputs
        )

        # Compute the score on perturbed inputs:
        scores: Iterable[torch.Tensor] = self.get_scores(
            pert_generator, (a.targets.to(self.device) for a in attributions_outputs)
        )

        # Aggregate the scores using the aggregator function and the perturbation masks.
        cleaned_scores = (
            self.aggregator(score.detach(), mask.to(self.device) if mask is not None else None).squeeze(0)
            for score, mask in zip(scores, mask_generator, strict=True)
        )

        # Compute AUC using trapezoidal rule
        cleaned_scores = [s.cpu().numpy() for s in cleaned_scores]
        aucs = []
        for s in cleaned_scores:
            auc = ((s.sum(axis=0) - 0.5 * (s[0] + s[-1])) / (len(s) - 1)).mean()  # Average AUC across targets
            aucs.append(auc)
        auc = float(np.mean(aucs))

        return auc, cleaned_scores


class Insertion(InsertionDeletionBase):
    """
    The insertion metric measures the quality of an attribution method by evaluating how the prediction score of a
    model improves when the most important elements of a sequence are gradually added. The importance of the elements
    is determined by the attribution-based method.

    A curve is built by computing the prediction score while iteratively inserting the most important elements,
    starting from a masked sequence. The area under this curve (AUC) is then computed to quantify the quality of the
    attribution method. The `evaluate` method returns both:

    - the average AUC across all sequences and targets,
    - for each sequence-target pair, the scores associated to the successive insertions.

    An attribution method is considered good if the AUC is high, meaning that the model's prediction score increases
    significantly as the most important elements are added back to the sequence. Conversely, a low AUC indicates that
    the attribution method is not effective in identifying the most important elements for the model's prediction.

    This metric only evaluates the order of importance of the elements in the sequence, not their actual values.

    Examples:
        >>> from interpreto.attributions.metrics import Insertion
        >>>
        >>> # Get explanations from an attribution method
        >>> explainer = Method(model, tokenizer, kwargs)
        >>> explanations = explainer(inputs, targets)
        >>>
        >>> # Run the insertion metric
        >>> metric = Insertion(model, tokenizer, n_perturbations=100)
        >>> auc, metric_scores = metric.evaluate(explanations)

    Args:
        model (PreTrainedModel): model used to generate explanations
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
        batch_size (int): batch size for the inference of the metric
        granularity (Granularity): granularity level of the perturbations (token, word, sentence, etc.)
        device (torch.device): device on which the attribution method will be run
        n_perturbations (int): number of perturbations from which the metric will be computed (i.e. the number of
            steps from which the AUC will be computed).
        max_percentage_perturbed (float): maximum percentage of elements in the sequence to be perturbed. Defaults
            to 1.0, meaning that all elements can be perturbed. If set to 0.5, only half of the elements will be
            perturbed (i.e. only the 50% most important). This is useful to avoid perturbing too many elements with
            low scores in long sequences.
    """

    @property
    def _perturbator_class(self) -> Callable:
        """Return the perturbator class used for insertion."""
        return InsertionPerturbator


class Deletion(InsertionDeletionBase):
    """
    The deletion metric measures the quality of an attribution method by evaluating how the prediction score of a
    model drops when the most important elements of a sequence are gradually removed. The importance of the elements
    is determined by the attribution-based method.

    A curve is built by computing the prediction score while iteratively masking the most important elements,
    starting from the whole sequence. The area under this curve (AUC) is then computed to quantify the quality of the
    attribution method. The `evaluate` method returns both:

    - the average AUC across all sequences and targets,
    - for each sequence-target pair, the scores associated to the successive insertions.

    An attribution method is considered good if the AUC is low, meaning that the model's prediction score decreases
    significantly as the most important elements are removed from the sequence. Conversely, a high AUC indicates that
    the attribution method is not effective in identifying the most important elements for the model's prediction.

    This metric only evaluates the order of importance of the elements in the sequence, not their actual values.

    Examples:
        >>> from interpreto.attributions.metrics import Deletion
        >>>
        >>> # Get explanations from an attribution method
        >>> explainer = Method(model, tokenizer, kwargs)
        >>> explanations = explainer(inputs, targets)
        >>>
        >>> # Run the deletion metric
        >>> metric = Deletion(model, tokenizer, n_perturbations=100)
        >>> auc, metric_scores = metric.evaluate(explanations)

    Args:
        model (PreTrainedModel): model used to generate explanations
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
        batch_size (int): batch size for the inference of the metric
        granularity (Granularity): granularity level of the perturbations (token, word, sentence, etc.)
        device (torch.device): device on which the attribution method will be run
        n_perturbations (int): number of perturbations from which the metric will be computed (i.e. the number of
            steps from which the AUC will be computed).
        max_percentage_perturbed (float): maximum percentage of elements in the sequence to be perturbed. Defaults
            to 1.0, meaning that all elements can be perturbed. If set to 0.5, only half of the elements will be
            perturbed (i.e. only the 50% most important). This is useful to avoid perturbing too many elements with
            low scores in long sequences.
    """

    @property
    def _perturbator_class(self) -> Callable:
        """Return the perturbator class used for deletion."""
        return DeletionPerturbator
