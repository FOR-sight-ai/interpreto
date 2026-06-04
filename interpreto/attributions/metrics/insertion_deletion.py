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

import itertools
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.base import AttributionOutput, setup_token_ids
from interpreto.attributions.perturbations.insertion_deletion_perturbation import (
    DeletionPerturbator,
    InsertionDeletionPerturbator,
    InsertionPerturbator,
)
from interpreto.commons.generator_tools import split_iterator
from interpreto.commons.granularity import GranularityAggregationStrategy
from interpreto.attributions.inference_wrappers.classification_inference_wrapper import ClassificationInferenceWrapper
from interpreto.attributions.inference_wrappers.generation_inference_wrapper import GenerationInferenceWrapper
from interpreto.attributions.inference_wrappers.inference_wrapper import (
    InferenceModes,
    InferenceWrapper,
)
from interpreto.typing import SingleAttribution


class InsertionDeletionBase:
    """
    Abstract base class for Insertion and Deletion metrics. Only the perturbator class is different between the two
    metrics.

    This class implements the core logic for insertion and deletion metrics, where tokens are either inserted or
    deleted from the input text based on their importance scores. The perturbator class is responsible for
    handling the specific perturbation logic (insertion or deletion).
    """

    _associated_inference_wrapper = InferenceWrapper

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
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
            device (torch.device): device on which the attribution method will be run
            n_perturbations (int): number of perturbations from which the metric will be computed (i.e. the number of
                steps from which the AUC will be computed).
            max_percentage_perturbed (float): maximum percentage of elements in the sequence to be perturbed. Defaults
                to 1.0, meaning that all elements can be perturbed. If set to 0.5, only half of the elements will be
                perturbed (i.e. only the 50% most important). This is useful to avoid perturbing too many elements with
                low scores in long sequences.
        """
        self.tokenizer = tokenizer
        replace_token_id = setup_token_ids(model, self.tokenizer)

        # perturbator
        self.perturbator = self._perturbator_class(
            tokenizer=self.tokenizer,
            n_perturbations=n_perturbations,
            max_percentage_perturbed=max_percentage_perturbed,
            replace_token_id=replace_token_id,
        )

        # inference wrapper
        self.inference_wrapper = self._associated_inference_wrapper(
            model, batch_size=batch_size, device=device, mode=InferenceModes.SOFTMAX
        )  # type: ignore

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model is located.
        """
        return self.inference_wrapper.device

    @device.setter
    def device(self, device: torch.device) -> None:
        """
        Sets the device on which the model is located.
        """
        self.inference_wrapper.device = device

    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): The device to which the model should be moved.
        """
        self.inference_wrapper.to(device)

    def cpu(self):
        """
        Move the model to the CPU.
        """
        self.device = torch.device("cpu")

    def cuda(self):
        """
        Move the model to the GPU.
        """
        self.device = torch.device("cuda")

    @property
    @abstractmethod
    def _perturbator_class(self) -> type[InsertionDeletionPerturbator]:
        """
        Returns the perturbator class used for insertion or deletion.

        This method should be overridden in subclasses to return the appropriate perturbator.
        """
        raise NotImplementedError()

    @abstractmethod
    def perturbation_generator(
        self, attributions_outputs: Iterable[AttributionOutput]
    ) -> Iterable[tuple[int, BatchEncoding, torch.Tensor]]:
        """
        Returns a generator that yields perturbed inputs and their corresponding attributions.
        """
        raise NotImplementedError()

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def aggregate(
        scores_generator: Iterable[Float[torch.Tensor, "p tg"]],
        sample_indices_generator: Iterable[int],
        granularity_aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
    ) -> tuple[float, list[list[Float[torch.Tensor, "p"]]]]:
        """
        Aggregate the scores for Insertion and Deletion metrics.

        For a given sample-target pair, the scores are the softmax values of the model outputs.
        There is one score per perturbation. (For generation, this is multiplied by the number of tokens in the target.)
        The metric is the mean auc under the perturbation curve.
        This aggregation thus as 4 steps:
        - aggregate the target tokens scores for each perturbation using the granularity aggregation strategy (only for generation)
        - aggregate the scores for each perturbation using the trapezoidal rule
        - aggregate the scores for each target by averaging them
        - aggregate the scores for each sample by averaging them

        This function also return the scores for each sample, target, and perturbation.

        Args:
            scores_generator: Iterable[Float[torch.Tensor, "p tg"]]
                Generator of scores. It contains $n * t$ elements of shape $(p, tg)$
                $n$ is the number of samples, $t$ the number of attributions per sample
                (depends on the target, thus classes for classification, and outputs for generation).
                $p$ is the number of perturbations, $tg$ is 1 for classification and
                the number of token in the targeted element of granularity in the case of generation.
                In classification, $t$ and $p$ are constant.

            sample_indices_generator: Iterable[int]
                Keeps track of the sample indices , thus the $n$ in the $n * t$.

            granularity_aggregation_strategy: GranularityAggregationStrategy
                The aggregation method to use.
                It should be an attribute of `GranularityAggregationStrategy`. Choices are:
                    - `MEAN`: average of contribution
                    - `MAX`: maximum contribution
                    - `MIN`: minimum contribution
                    - `SUM`: sum of contribution
                    - `SIGNED_MAX`: contribution with the largest absolute value, preserving its sign

        Returns:
            auc: float
                The average AUC across all samples and targets.
                Each sample has the same weight.
                For each sample, each target has the same weight.

            all_scores: list[list[Float[torch.Tensor, "p"]]]
                The scores for each sample, targets, and perturbations.
                In generation, the number of perturbations depends on the sample and target.
                Thus it cannot be stored as a tensor.
        """
        # iterate over scores grouped by samples
        sample_aucs: list[float] = []  # there will be n elements
        sample_scores: Iterable[tuple[int, Float[torch.Tensor, "p tg"]]]  # there is t elements
        all_scores: list[list[Float[torch.Tensor, "p"]]] = []
        for _, sample_scores in itertools.groupby(
            zip(sample_indices_generator, scores_generator, strict=True), key=lambda x: x[0]
        ):
            all_scores.append([])
            target_aucs: list[float] = []  # there will be t elements
            for _, target_scores in sample_scores:
                # aggregate along the output granular element (correspond to a squeeze for classification)
                all_scores[-1].append(
                    granularity_aggregation_strategy.aggregate(target_scores, dim=-1).squeeze(dim=-1)
                )

                # aggregate along the perturbation dimension with trapezoidal rule
                # $(sum_1^n (x_i + x_{i-1})/2) / (n - 1)$
                target_aucs.append(
                    torch.trapezoid(all_scores[-1][-1], dim=0).item() / (all_scores[-1][-1].shape[0] - 1)
                )

            # aggregate along the target dimension
            sample_aucs.append(sum(target_aucs) / len(target_aucs))

        # aggregate along the sample dimension
        auc: float = sum(sample_aucs) / len(sample_aucs)
        return auc, all_scores

    def __verify_and_set_granularity(
        self, attributions_outputs: Iterable[AttributionOutput]
    ) -> GranularityAggregationStrategy:
        """
        Verify that all attributions have the same granularity and granularity aggregation strategy.

        Then set or return the granularity and granularity aggregation strategy.

        Args:
            attributions_outputs (Iterable[AttributionOutput]): Outputs from the attribution method.

        Returns:
            GranularityAggregationStrategy: The granularity aggregation strategy.
        """
        # Granularity
        grans = [a.granularity for a in attributions_outputs]
        if not all(g == grans[0] for g in grans):
            raise ValueError("All attributions must have the same granularity.")
        self.granularity = grans[0]
        self.perturbator.granularity = grans[0]

        # Granularity Aggregation Strategy
        gas = [a.granularity_aggregation_strategy for a in attributions_outputs]
        if not all(g == gas[0] for g in gas):
            raise ValueError("All attributions must have the same granularity aggregation strategy.")

        return gas[0]

    def evaluate(
        self, attributions_outputs: Iterable[AttributionOutput]
    ) -> tuple[float, list[list[Float[torch.Tensor, "p"]]]]:
        """
        Evaluates the insertion or deletion metric based on the provided attributions.

        Args:
            attributions_outputs (Iterable[AttributionOutput]): Outputs from the attribution method.

        Returns:
            auc: float
                The average AUC across all samples and targets.
                Each sample has the same weight.
                For each sample, each target has the same weight.

            all_scores: list[list[Float[torch.Tensor, "p"]]]
                The scores for each sample, targets, and perturbations.
                In generation, the number of perturbations depends on the sample and target.
                Thus it cannot be stored as a tensor.
        """

        # Verify that all attributions have the same granularity and granularity aggregation strategy.
        granularity_aggregation_strategy = self.__verify_and_set_granularity(attributions_outputs)

        # Perturb the inputs
        sample_indices_generator: Iterator[int]
        pert_generator: Iterator[BatchEncoding]
        target_generator: Iterator[torch.Tensor | None]
        sample_indices_generator, pert_generator, target_generator = split_iterator(
            self.perturbation_generator(attributions_outputs)  # type: ignore
        )

        # Compute the score on perturbed inputs
        scores: Iterator[torch.Tensor] = self.inference_wrapper(pert_generator, target_generator)

        # Aggregate scores
        auc, grouped_scores = InsertionDeletionBase.aggregate(
            scores_generator=scores,
            sample_indices_generator=sample_indices_generator,
            granularity_aggregation_strategy=granularity_aggregation_strategy,
        )

        return auc, grouped_scores

    def __call__(
        self, attributions_outputs: Iterable[AttributionOutput]
    ) -> tuple[float, list[list[Float[torch.Tensor, "p"]]]]:
        """alias for evaluate"""
        return self.evaluate(attributions_outputs)


class ClassificationInsertionDeletionBase(InsertionDeletionBase):
    """
    Base class for the insertion and deletion metrics for classification models.

    The difference with generation is what is explained, classes for classification,
    and generated tokens for generation. As the perturbation depends on the explanation,
    different subject of the explanations leads to different structures of perturbations.

    In the case of classification, for a given sample, explanations have the shape $(t, g)$,
    with $g$ the number of input granular elements (tokens, words, etc. depending on the granularity)
    and $t$ the number of targets (classes).

    In most cases, a single class is explained for each sample, so the number of targets is 1.

    In any case, for each sample, there is $t$ explanations.
    Thus the insertion/deletion metric should be computed for each one of them.
    Hence, for classification, we iterate first on the samples, then on the explanations.
    The perturbations are computed for each sample-explanation pair.
    """

    _associated_inference_wrapper = ClassificationInferenceWrapper
    inference_wrapper: ClassificationInferenceWrapper

    def perturbation_generator(
        self, attributions_outputs: Iterable[AttributionOutput]
    ) -> Iterable[tuple[int, BatchEncoding, torch.Tensor]]:
        # iterate over the samples
        for i, a in enumerate(attributions_outputs):
            all_inputs: BatchEncoding = a.model_inputs_to_explain  # type: ignore
            granularity_indices = self.granularity.get_indices(all_inputs, self.tokenizer)  # type: ignore

            # for a sample, iterate over attributions (thus targets)
            attrib: SingleAttribution
            for target, attrib in zip(a.targets, a.attributions, strict=True):
                pert = self.perturbator.perturb(
                    all_inputs, attributions=attrib, granularity_indices=granularity_indices
                )
                yield i, pert, target.unsqueeze(0)


class GenerationInsertionDeletionBase(InsertionDeletionBase):
    """
    Base class for the insertion and deletion metrics for generation models.

    The difference with generation is what is explained, classes for classification,
    and generated tokens for generation. As the perturbation depends on the explanation,
    different subject of the explanations leads to different structures of perturbations.

    In the case of generation, for a given sample, for each of the $t$ predicted granular element,
    the attributions are of length $g$. $g$ corresponds to the number of granular elements
    (tokens, words, ect. depending on the granularity) preceding the predicted granular element.
    In practice, the explanations are stored in a matrix of shape $(t_max, g_max)$ with NaN, but it has no importance here.

    In any case, for each sample, there are $t$ explanations.
    Thus the insertion/deletion metric should be computed for each one of them.
    Hence, for generation, we iterate first on the samples, then on the explanations.
    The perturbations are computed for each sample-explanation pair.

    The problem is that for a given attribution, thus a given sample-explanation pair,
    the insertion/deletion perturbation should only be done on the inputs used to predict the explained predicted granular element.
    We could do perturbation on all elements, but in the worst case it makes 2 times more inferences and provides less precise values.

    Hence, we iteratively include one more granular element at a time and set this element as the target.
    To know the link between granular elements and tokens, we use the granularity indices.

    Example: (keep in mind that for generation, the targets are included in the input to explain)
    initial input: "A BC", initial target "DEF GHIJ KLMNOP"
    pert 1 input: "A BC DEF", pert 1 target "DEF"
    pert 2 input: "A BC DEF GHIJ", pert 2 target "GHIJ"
    pert 3 input: "A BC DEF GHIJ KLMNOP", pert 3 target "KLMNOP"
    """

    _associated_inference_wrapper = GenerationInferenceWrapper
    inference_wrapper: GenerationInferenceWrapper

    def perturbation_generator(
        self, attributions_outputs: Iterable[AttributionOutput]
    ) -> Iterable[tuple[int, BatchEncoding, torch.Tensor]]:
        # iterate over the samples
        for sample_index, a in enumerate(attributions_outputs):
            # get the granularity indices from input tokens
            # there is always a single input, we create an `AttributionOutput` for each input
            model_inputs: BatchEncoding = a.model_inputs_to_explain  # type: ignore
            granularity_indices = self.granularity.get_indices(model_inputs, self.tokenizer)  # type: ignore

            if len(granularity_indices[0]) != a.attributions.shape[1]:
                raise ValueError(
                    "The number of granularity elements does not match the number of attributions."
                    f"Got {len(granularity_indices[0])} granularity elements and {a.attributions.shape[1]} attributions."
                )
            nb_granular_elements = a.attributions.shape[1]
            nb_initial_input_granular_elements = nb_granular_elements - a.attributions.shape[0]

            # for a sample, iterate over attributions (thus targets)
            # a each new attribution, a new granularity element is added to the current inputs
            for attrib_index, attrib in enumerate(a.attributions):
                # index of the target granular element (which should be included in the inputs)
                nb_current_input_granular_elements = nb_initial_input_granular_elements + attrib_index + 1
                current_indices = [granularity_indices[0][:nb_current_input_granular_elements]]

                # index of the token to cut the inputs at (obtained via the granularity indices)
                nb_current_inputs = len(
                    [tt for t in granularity_indices[0][:nb_current_input_granular_elements] for tt in t]
                )

                # cut the inputs at the right index
                # we can convert to BatchEncoding like this because we do not need all the other information
                current_inputs = BatchEncoding({k: v[:, :nb_current_inputs] for k, v in model_inputs.items()})

                # cut attributions to keep only input elements
                current_attrib: SingleAttribution = attrib[:nb_current_input_granular_elements]

                # perturb the inputs
                pert: BatchEncoding = self.perturbator.perturb(
                    current_inputs, attributions=current_attrib, granularity_indices=current_indices
                )

                # targets correspond to the current granular element
                targets = model_inputs["input_ids"][0, granularity_indices[0][nb_current_input_granular_elements - 1]]  # type: ignore

                yield (sample_index, pert, targets)


class FactoryGeneratedMeta(type):
    """
    Metaclass to distinguish classes generated by the MultitaskExplainerMixin.
    """


class MultitaskMetricMixin(InsertionDeletionBase):
    """
    Mixin class to generate the appropriate Explainer based on the model type.
    """

    def __new__(cls, model: PreTrainedModel, *args: Any, **kwargs: Any) -> InsertionDeletionBase:
        if isinstance(cls, FactoryGeneratedMeta):
            return super().__new__(cls)  # type: ignore
        if model.__class__.__name__.endswith("ForSequenceClassification"):
            t = FactoryGeneratedMeta("Classification" + cls.__name__, (cls, ClassificationInsertionDeletionBase), {})
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
            t = FactoryGeneratedMeta("Generation" + cls.__name__, (cls, GenerationInsertionDeletionBase), {})
            return t.__new__(t, model, *args, **kwargs)  # type: ignore
        raise NotImplementedError(
            "Model type not supported for Explainer. Use a ModelForSequenceClassification, a ModelForCausalLM model or a LMHeadModel model."
        )


class Insertion(MultitaskMetricMixin, InsertionDeletionBase):
    """
    The insertion metric measures the faithfulness of an attribution method by evaluating how the prediction score of a
    model improves when the most important elements of a sequence are gradually added. The importance of the elements
    is determined by the attribution-based method.

    Insertion was introduced by Petsiuk et al. (2017) for image classification.
    This is our adaptation to the metric for text and in particular for generation (multi-predictions).

    A curve is built by computing the prediction score while iteratively inserting the most important elements,
    starting from a masked sequence. The scores are the softmax outputs, between 0 and 1. The area under this curve
    (AUC) is then computed to quantify the faithfulness of the attribution method. A higher AUC is better.

    The `evaluate` method returns both:

    - the average AUC across all sequences and targets,
    - for each sequence-target pair, the softmax scores associated to the successive insertions. The softmax scores are
        preferred over logits as they are bounded between 0 and 1, which makes the AUC more interpretable.

    An attribution method is considered good if the Insertion AUC is high, meaning that the model's prediction score increases
    significantly as the most important elements are added back to the sequence. Conversely, a low AUC indicates that
    the attribution method is not effective in identifying the most important elements for the model's prediction.

    This metric only evaluates the order of importance of the elements in the sequence, not their actual values.

    **Reference:**
    Vitali Petsiuk, Abir Das, and Kate Saenko. (2017). *RISE: Randomized Input Sampling for Explanation of Black-box Models*
    [Paper](https://arxiv.org/abs/1806.07421)

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
    def _perturbator_class(self) -> type[InsertionPerturbator]:
        """Return the perturbator class used for insertion."""
        return InsertionPerturbator


class Deletion(MultitaskMetricMixin, InsertionDeletionBase):
    """
    The deletion metric measures the faithfulness of an attribution method by evaluating how the prediction score of a
    model drops when the most important elements of a sequence are gradually removed. The importance of the elements
    is determined by the attribution-based method.

    Deletion was introduced by Petsiuk et al. (2017) for image classification.
    This is our adaptation to the metric for text and in particular for generation (multi-predictions).

    A curve is built by computing the prediction score while iteratively masking the most important elements,
    starting from the whole sequence. The scores are the softmax outputs, between 0 and 1. The area under this curve
    (AUC) is then computed to quantify the faithfulness of the attribution method. A lower AUC is better.

    The `evaluate` method returns both:

    - the average AUC across all sequences and targets,
    - for each sequence-target pair, the softmax scores associated to the successive deletions. The softmax scores are
        preferred over logits as they are bounded between 0 and 1, which makes the AUC more interpretable.

    An attribution method is considered good if the Deletion AUC is low, meaning that the model's prediction score decreases
    significantly as the most important elements are removed from the sequence. Conversely, a high AUC indicates that
    the attribution method is not effective in identifying the most important elements for the model's prediction.

    This metric only evaluates the order of importance of the elements in the sequence, not their actual values.

    **Reference:**
    Vitali Petsiuk, Abir Das, and Kate Saenko. (2017). *RISE: Randomized Input Sampling for Explanation of Black-box Models*
    [Paper](https://arxiv.org/abs/1806.07421)

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
    def _perturbator_class(self) -> type[DeletionPerturbator]:
        """Return the perturbator class used for deletion."""
        return DeletionPerturbator
