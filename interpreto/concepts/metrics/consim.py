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
from enum import Enum
from typing import NamedTuple

import torch
from tqdm import tqdm

from interpreto import ModelWithSplitPoints
from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.model_wrapping.llm_interface import LLMInterface, Role
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity


class PromptSetting(NamedTuple):
    """
    Configuration of the ConSim prompts.
    It says which elements should be included in the prompt.

    This is used to define the different `PromptTypes` available.
    """

    # initial phase
    concepts_global_importances: bool = False
    global_contrastive_importances: bool = False

    # learning phase
    lp_samples: bool = False
    lp_concepts_local_contributions: bool = False
    lp_local_contrastive_importance: bool = False

    # anonymization
    anonymize_classes: bool = False

    # prediction
    # pred_concepts: bool = False

    def validate(
        self,
        *,
        concepts_interpretation: dict[int, str],
        gold_labels: torch.Tensor | list[int] | None,
    ) -> None:
        """
        Validate the prompt setting against required inputs and incompatible options.
        """
        _ = concepts_interpretation
        if self.concepts_global_importances and self.global_contrastive_importances:
            raise ValueError(
                "PromptSetting.concepts_global_importances and PromptSetting.global_contrastive_importances are mutually exclusive."
            )

        if self.lp_concepts_local_contributions and self.lp_local_contrastive_importance:
            raise ValueError(
                "PromptSetting.lp_concepts_local_contributions and PromptSetting.lp_local_contrastive_importance are mutually exclusive."
            )

        if self.lp_local_contrastive_importance and gold_labels is None:
            raise ValueError(
                "PromptSetting.lp_local_contrastive_importance=True requires `gold_labels` to be provided to ConSim.evaluate()."
            )


class PromptTypes(Enum):
    """
    There are six types of prompts, including two baselines and an upper bond:

    Attributes:
        `L1_baseline_without_lp`:
            IP.1 and EP.1 are included in the prompt.
            Only the task description, but explanations or learning phase.

        `E1_global_concepts_without_lp`:
            IP.1, IP.2, and EP.1 are included in the prompt.
            Only task description and global concepts explanation, but no learning phase.

        `L2_baseline_with_lp`:
            IP.1, LP.1, and EP.1 are included in the prompt.
            Task description and learning phase, but no explanations.

        `E2_global_concepts_with_lp`:
            IP.1, IP.2, LP.1, and EP.1 are included in the prompt.
            Task description, global concepts explanation, and learning phase. But no local concepts explanation.

        `E3_global_and_local_concepts_with_lp`:
            IP.1, IP.2, LP.1, LP.2, and EP.1 are included in the prompt.
            Task description, learning phase, and both global and local concepts explanation.
    """

    L1_baseline_without_lp = PromptSetting()
    E1_global_concepts_without_lp = PromptSetting(concepts_global_importances=True)
    L2_baseline_with_lp = PromptSetting(lp_samples=True)
    E2_global_concepts_with_lp = PromptSetting(concepts_global_importances=True, lp_samples=True)
    E3_global_and_local_concepts_with_lp = PromptSetting(
        concepts_global_importances=True,
        lp_samples=True,
        lp_concepts_local_contributions=True,
    )
    C1_contrastive_global_concepts_without_lp = PromptSetting(
        global_contrastive_importances=True,
    )
    C2_contrastive_global_concepts_with_lp = PromptSetting(
        global_contrastive_importances=True,
        lp_samples=True,
    )
    C3_contrastive_global_and_local_concepts_with_lp = PromptSetting(
        global_contrastive_importances=True,
        lp_samples=True,
        lp_local_contrastive_importance=True,
    )
    C4_contrastive_local_concepts = PromptSetting(
        concepts_global_importances=True,
        lp_samples=True,
        lp_local_contrastive_importance=True,
    )
    C5_contrastive_local_only = PromptSetting(
        lp_samples=True,
        lp_local_contrastive_importance=True,
    )


class ConSim:
    """Code: [:octicons-mark-github-24: `concepts/metrics/consim.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/metrics/consim.py)

    ConSim for Concept-based Simulatability. Was introduced by Poché et al. in 2025[^1].

    It evaluates all three components of the concept-based explanation:

    - the concepts space

    - the concepts interpretation

    - the concepts importance

    To evaluate explanations on a given model $f$, ConSim evaluates to which extent explanations
    help a meta-predictor $\\Psi$ to simulate the predictions of the model $f$.

    In our case, the role of the meta-predictor will be played by `user_llm`, and interface calling
    a model either from local, or from a remote API, such as OpenAI or HuggingFace.
    Therefore, most of the code correspond to building the prompts for the LLM.

    There are three steps to ConSim:

    - Step 0:
        Instantiate the ConSim metric
        with the `model_with_split_points` ($f$) and the `user_llm` ($\\Psi$).

    - Step 1:
        Select interesting examples for ConSim with the `select_examples` method.
        Samples are selected to see how well $\\Psi$ can simulate $f$.
        Thus there are samples for every classes and many initial errors from $f$.

    - Step 2:
        Evaluate the ConSim score with the `evaluate` method. It is an accuracy score between $\\Psi$ and $f$ predictions.
        But we selected interesting examples, so it cannot be compared to a natural accuracy on the dataset.
        Therefore, we need to compare it to a baseline ().

    Tip:
        We highly recommend to do the steps 1 and 2 several times with different seeds to get more statistically significant results.
        The initial papers[^1] used five different seeds..

    [^1]:
        A. Poché, A. Jacovi, A.M. Picard, V. Boutin, and F. Jourdan.
        [ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](https://aclanthology.org/2025.acl-long.279/).
        In the Proceedings of the 2025 Association for Computational Linguistics (ACL).

    Arguments:
        model_with_split_points: ModelWithSplitPoints
            The model to explain. Is is a wrapper around a model and a tokenizer to easily get activations.

        user_llm: LLMInterface | None
            The LLM interface that will serve as the meta-predictor.
            If not provided the user will have to call the ConSim prompts manually.
            If your preferred LLM API is not supported, you can implement your own LLM interface.
            You just have to implement the `generate` method.

            The format of the prompt is:

            `[(Role.SYSTEM, "system prompt"), (Role.USER, "user prompt"), (Role.ASSISTANT, "assistant prompt")]`

        activation_granularity: ActivationGranularity
            The granularity of the activations to use for the explanations.

        classes: list[str] | None
            The names of classes of the dataset.

        split_point: str
            Where to split the model to explain.

    Attributes:
        classes: list[str] | None
            The names of classes of the dataset.

        prompt_types: PromptTypes
            Enum of the possible prompts types to use.

        model_with_split_points: ModelWithSplitPoints
            The model to explain. Is is a wrapper around a model and a tokenizer to easily get activations.

        split_point: str
            Where to split the model to explain.

        user_llm: LLMInterface | None
            The LLM interface that will serve as the meta-predictor.
            If your preferred LLM API is not supported, you can implement your own LLM interface.
            You just have to implement the `generate` method.

            The format of the prompt is:

            `[(Role.SYSTEM, "system prompt"), (Role.USER, "user prompt"), (Role.ASSISTANT, "assistant prompt")]`

    TODO:
        validate example in practice

    Examples:
        Preamble to a metric, fit a concept explainer:
        >>> import datasets
        >>> import torch
        >>> from interpreto import ConSim, ModelWithSplitPoints, ICAConcepts, OpenAILLM
        >>>
        >>> # ------------------------
        >>> # Load a model and wrap it
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     "textattack/bert-base-uncased-ag-news",
        ...     split_points=["bert.encoder.layer.10.output"],
        ...     model_autoclass=AutoModelForSequenceClassification,  # type: ignore
        ...     batch_size=4,
        ... )
        >>>
        >>> # --------------------------------------
        >>> # Load a dataset and compute activations
        >>> dataset = datasets.load_dataset("fancyzhx/ag_news")
        >>> classes = ["World", "Sports", "Business", "Sci/Tech"]
        >>> activations = model_with_split_points.get_activations(dataset["train"]["text"])
        >>>
        >>> # -------------------------
        >>> # Fit the concept explainer
        >>> concept_explainer_1 = ICAConcepts(model_with_split_points, nb_concepts=50)
        >>> concept_explainer_1.fit(activations)

        The two steps of ConSim:
        >>> # ------------------------------------------------------------------
        >>> # Step 0: Define the User-LLM and instantiate the ConSim metric
        >>> user_llm = OpenAILLM(api_key="YOUR_OPENAI_API_KEY", model="gpt-4.1-nano")
        >>> consim = ConSim(
        ...     model_with_split_points,
        ...     user_llm,
        ...     activation_granularities=ModelWithSplitPoints.activation_granularities.TOKEN,
        ...     classes=classes,
        ... )
        >>>
        >>> # ----------------------------------------------
        >>> # Step 1: Select interesting examples for ConSim
        >>> samples, labels, predictions = consim.select_examples(
        ...     dataset["train"]["text"], dataset["train"]["label"],
        ... )
        >>>
        >>> # -------------------------------------------------------------
        >>> # Step 2: Evaluate the ConSim score, do not forget the baseline
        >>> concepts_interpretation = {0: "example concept"}
        >>> global_importances = torch.zeros(len(classes), 1)
        >>> baseline = consim.evaluate(
        ...     interesting_samples=samples,
        ...     predictions=predictions,
        ...     concepts_interpretation=concepts_interpretation,
        ...     global_importances=global_importances,
        ...     prompt_type=PromptTypes.L2_baseline_with_lp,
        ... )
        >>> consim_score = consim.evaluate(
        ...     interesting_samples=samples,
        ...     predictions=predictions,
        ...     concept_explainer=concept_explainer_1,
        ...     concepts_interpretation=concepts_interpretation,
        ...     global_importances=global_importances,
        ...     prompt_type=PromptTypes.E3_global_and_local_concepts_with_lp,
        ... )
    """

    prompt_types: type[PromptTypes] = PromptTypes

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        activation_granularity: ActivationGranularity,
        classes: list[str],
        user_llm: LLMInterface | None = None,
        split_point: str | None = None,
    ):
        """
        Initialize the ConSim metric.
        """
        self.model_with_split_points = model_with_split_points
        if split_point is None:
            if len(self.model_with_split_points.split_points) > 1:
                raise ValueError(
                    "If the model has more than one split point, a split point for fitting the concept model should "
                    f"be specified. Got split point: '{split_point}' with model split points: "
                    f"{', '.join(self.model_with_split_points.split_points)}."
                )
            split_point = self.model_with_split_points.split_points[0]

        if split_point not in self.model_with_split_points.split_points:
            raise ValueError(
                f"Split point '{split_point}' not found in model split points: {', '.join(self.model_with_split_points.split_points)}."
            )

        self.split_point: str = split_point
        self.activation_granularity: ActivationGranularity = activation_granularity
        self.user_llm: LLMInterface | None = user_llm
        self.classes: list[str] = classes

    def _get_predictions(
        self, inputs: list[str], batch_size: int = 64, device: torch.device | str | None = None, tqdm_bar: bool = False
    ) -> torch.Tensor:
        """
        Get the predictions of the model on a list of inputs.
        Called by `select_examples`.

        Arguments:
            inputs: list[str]
                The inputs to predict.
            batch_size: int
                The batch size to use for the predictions.
            device: torch.device | str
                The device to use for the predictions.
            tqdm_bar: bool
                Whether to show a tqdm bar.

        Returns:
            predictions: torch.Tensor
                The predictions of the model on the inputs.
        """
        if getattr(self.model_with_split_points, "dispatched", True) is False:
            # ConSim forwards through self.model_with_split_points._model directly, so ensure weights are loaded.
            # This is not what NNsight expects, but we do not want to load it a second time nor look into now.
            self.model_with_split_points.dispatch()

        device = device if device is not None else self.model_with_split_points.device
        all_predictions = []
        for batch_index in tqdm(
            range(0, len(inputs), batch_size),
            desc="Computing predictions",
            unit="batch",
            total=len(inputs),
            disable=not tqdm_bar,
        ):
            batch_inputs = inputs[batch_index : batch_index + batch_size]
            batch_tokens = self.model_with_split_points.tokenizer(
                batch_inputs, return_tensors="pt", padding=True, truncation=True
            ).to(device)  # type: ignore
            logits = self.model_with_split_points._model(
                batch_tokens["input_ids"], batch_tokens["attention_mask"]
            ).logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(predictions)
        return torch.cat(all_predictions)

    def _extract_interesting_elements(
        self,
        inputs: list[str],
        labels: torch.Tensor,
        predictions: torch.Tensor,
        nb_lp_samples: int = 20,
        nb_ep_samples: int = 20,
        seed: int = 0,
        classes_subset: list[int] | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Extract interesting elements from the inputs, labels, and predictions.
        It selects `nb_lp_samples` + `nb_ep_samples` samples from the inputs.
        The goal is to select uniformly between each class (with respect to the labels).
        There should be as many samples where the initial model prediction are correct as incorrect.
        The samples are then randomly shuffled.

        The first `nb_lp_samples` samples are selected for the learning phase.
        The last `nb_ep_samples` samples are selected for the evaluation phase.

        Therefore, there is no guarantee on the repartition inside learning and evaluation phase.

        Called by `select_examples`.

        Arguments:
            inputs: list[str]
                The inputs to predict.
            labels: torch.Tensor
                The labels of the inputs.
            predictions: torch.Tensor
                The predictions of the model on the inputs.
            nb_lp_samples: int
                The number of samples to select for the learning phase.
            nb_ep_samples: int
                The number of samples to select for the evaluation phase.
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
        if classes_subset is None:
            class_ids = list(range(len(self.classes))) if self.classes is not None else torch.unique(labels).tolist()
        else:
            class_ids = classes_subset

        nb_classes = len(class_ids)
        nb_correct = (nb_lp_samples + nb_ep_samples) // 2
        nb_mistakes = nb_lp_samples + nb_ep_samples - nb_correct

        if nb_classes > nb_lp_samples:
            raise ValueError(
                f"Not enough samples ({nb_lp_samples}) to represent the {nb_classes} classes in the learning phase."
                "Please increase the number of learning phase samples or take a subset of classes."
            )

        # Find the correct and incorrect indices
        is_prediction_correct = predictions == labels
        correct_indices = torch.nonzero(is_prediction_correct)
        incorrect_indices = torch.nonzero(~is_prediction_correct)
        del is_prediction_correct

        if len(correct_indices) < nb_correct or len(incorrect_indices) < nb_mistakes:
            raise ValueError(
                f"Not enough correct or incorrect predictions to select {nb_correct} correct and {nb_mistakes} incorrect."
                f"Either provide more inputs (currently {len(correct_indices)} correct and {len(incorrect_indices)} incorrect)"
                "or reduce the number of samples to select."
            )

        # select random indices
        torch.random.manual_seed(seed)
        correct_indices = correct_indices[torch.randperm(len(correct_indices))]
        incorrect_indices = incorrect_indices[torch.randperm(len(incorrect_indices))]

        # select the first nb_correct and nb_mistakes indices, each class should be represented
        nb_correct_elements_per_class = nb_correct // nb_classes
        nb_mistakes_elements_per_class = nb_mistakes // nb_classes

        if nb_correct_elements_per_class == 0 or nb_mistakes_elements_per_class == 0:
            warnings.warn(
                f"Not enough correct ({nb_correct_elements_per_class}) or incorrect ({nb_mistakes_elements_per_class})"
                f" predictions to represent the {nb_classes} classes inb both correct and incorrect."
                "The classes of interest will be selected randomly.",
                stacklevel=2,
            )
            nb_correct_elements_per_class = 1
            nb_mistakes_elements_per_class = 1

        # select correct and incorrect indices for each class
        class_wise_correct_indices = []
        class_wise_incorrect_indices = []
        for c in class_ids:
            class_wise_correct_indices.append(correct_indices[labels[correct_indices] == c])
            class_wise_incorrect_indices.append(incorrect_indices[labels[incorrect_indices] == c])

        selected_correct_indices = torch.cat([c[:nb_correct_elements_per_class] for c in class_wise_correct_indices])[
            :nb_correct
        ]
        selected_incorrect_indices = torch.cat(
            [c[:nb_mistakes_elements_per_class] for c in class_wise_incorrect_indices]
        )[:nb_mistakes]

        # in case the number of correct or incorrect is not a multiple of the number of classes
        nb_correct_remaining = nb_correct - nb_correct_elements_per_class * nb_classes
        if nb_correct_remaining:
            additional_possible_correct_indices = torch.cat(
                [c[nb_correct_elements_per_class:] for c in class_wise_correct_indices]
            )
            new_indices = torch.randint(len(additional_possible_correct_indices), (nb_correct_remaining,))
            additional_correct_indices = additional_possible_correct_indices[new_indices]
            selected_correct_indices = torch.cat([selected_correct_indices, additional_correct_indices])

        nb_mistakes_remaining = nb_mistakes - nb_mistakes_elements_per_class * nb_classes
        if nb_mistakes_remaining:
            additional_possible_incorrect_indices = torch.cat(
                [c[nb_mistakes_elements_per_class:] for c in class_wise_incorrect_indices]
            )
            new_indices = torch.randint(len(additional_possible_incorrect_indices), (nb_mistakes_remaining,))
            additional_incorrect_indices = additional_possible_incorrect_indices[new_indices]
            selected_incorrect_indices = torch.cat([selected_incorrect_indices, additional_incorrect_indices])

        indices = torch.cat([selected_correct_indices, selected_incorrect_indices])

        # shuffle the indices
        indices = indices[torch.randperm(len(indices))]

        interesting_samples = [inputs[i] for i in indices]

        return interesting_samples, labels[indices], predictions[indices]

    def select_examples(
        self,
        inputs: list[str],
        labels: torch.Tensor,
        predictions: torch.Tensor | None = None,
        nb_lp_samples: int = 20,
        nb_ep_samples: int = 20,
        seed: int = 0,
        batch_size: int = 64,
        device: torch.device | str | None = None,
        classes_subset: list[int] | list[str] | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Select examples for the ConSim metric. It first computes the models' predictions on the inputs.
        Then, it selects `nb_lp_samples` + `nb_ep_samples` samples from the inputs.
        The goal is to select uniformly between each class (with respect to the labels).
        There should be as many samples where the initial model prediction are correct as incorrect.
        The samples are then randomly shuffled.

        The first `nb_lp_samples` samples are selected for the learning phase.
        The last `nb_ep_samples` samples are selected for the evaluation phase.

        Therefore, there is no guarantee on the repartition inside learning and evaluation phase.

        Arguments:
            inputs: list[str]
                The inputs to predict.
            labels: torch.Tensor
                The labels of the inputs.
            predictions: torch.Tensor | None
                The predictions of the model on the inputs.
                If not provided, the predictions are computed by calling `self._get_predictions`.
            nb_lp_samples: int
                The number of samples to select for the learning phase.
            nb_ep_samples: int
                The number of samples to select for the evaluation phase.
            seed: int
                The seed to use for the random selection.
            batch_size: int
                The batch size to use for the predictions.
            device: torch.device | str | None
                The device to use for the predictions.
            classes_subset: list[int] | None
                Optional subset of classes to sample from. When provided, only samples whose labels
                and predictions are both in this subset are considered.

        Returns:
            interesting_samples: list[str]
                The interesting samples.
            labels: torch.Tensor
                The labels of the interesting samples.
            predictions: torch.Tensor
                The predictions of the model on the interesting samples.
        """
        # compute predictions if not provided
        if predictions is None:
            predictions = self._get_predictions(inputs, batch_size=batch_size, device=device)

        class_ids = None
        if classes_subset is not None:
            if not isinstance(classes_subset, (list, tuple)):
                raise TypeError("`classes_subset` must be a list or tuple of class ids or class names.")
            if len(classes_subset) < 2:
                raise ValueError("`classes_subset` must contain at least two classes.")

            if not all(isinstance(c, int) for c in classes_subset):
                raise TypeError("`classes_subset` must be a list of class ids.")

            class_ids = [int(c) for c in classes_subset]

            seen = set()
            class_ids = [c for c in class_ids if not (c in seen or seen.add(c))]

            if predictions.device != labels.device:
                predictions = predictions.to(labels.device)

            class_ids_tensor = torch.tensor(class_ids, device=labels.device, dtype=labels.dtype)
            label_mask = (labels.unsqueeze(1) == class_ids_tensor).any(dim=1)
            pred_mask = (predictions.unsqueeze(1) == class_ids_tensor).any(dim=1)
            subset_mask = label_mask & pred_mask

            if not torch.any(subset_mask):
                raise ValueError(
                    "No samples found where both labels and predictions are in the requested `classes_subset`."
                )

            subset_indices = torch.nonzero(subset_mask, as_tuple=False).squeeze(-1)
            inputs = [inputs[i] for i in subset_indices.tolist()]
            labels = labels[subset_indices]
            predictions = predictions[subset_indices]

        return self._extract_interesting_elements(
            inputs=inputs,
            labels=labels,
            predictions=predictions.cpu(),
            nb_lp_samples=nb_lp_samples,
            nb_ep_samples=nb_ep_samples,
            seed=seed,
            classes_subset=class_ids,
        )

    @staticmethod
    def _quantize_importances(importance: float, threshold: float = 0.05) -> str | None:
        """
        Convert the normalized importances to literals.
        The literals and ranges (values for the default threshold value of 0.05) are:
        - "++" for values above 0.3
        - "+" for values between 0.05 and 0.3
        - "-" for values between -0.05 and -0.3
        - "--" for values below -0.3

        Arguments:
            importance: float
                The importance to convert.
            threshold: float
                The threshold to select the most important concepts for each class.

        Returns:
            literals: str | None
                The literals corresponding to the importances.
                None if the importance is below the threshold. This should be filtered afterwards.
        """
        if importance <= -6 * threshold:
            return "--"

        if importance <= -threshold:
            return "-"

        if importance >= 6 * threshold:
            return "++"

        if importance >= threshold:
            return "+"

        return None

    @staticmethod
    def _resolve_prompt_setting(prompt_type: PromptTypes | PromptSetting) -> PromptSetting:
        """
        Resolve a PromptSetting from either a PromptTypes enum or a direct PromptSetting.
        """
        if isinstance(prompt_type, PromptTypes):
            return prompt_type.value
        return prompt_type

    @staticmethod
    def _concept_descriptor(
        concept_id: int,
        concepts_interpretation: dict[int, str],
    ) -> str:
        """
        Produce a display string for a concept id including its label.

        Concept descriptions are limited to 100 characters.
        """
        if concept_id not in concepts_interpretation:
            raise ValueError(f"Missing label for concept id {concept_id} in `concepts_interpretation`.")

        return f"C{concept_id} ({concepts_interpretation[concept_id][:100]})"

    @staticmethod
    def _concepts_to_string(
        importances: torch.Tensor,
        concepts_interpretation: dict[int, str],
        top_k: int = 5,
        threshold: float = 0.05,
    ) -> str:
        """
        Format the top-k concepts into a printable string.
        Concepts are sorted by absolute importance.
        Importances are assumed to be normalized.
        For contrastive settings, provide `foil_importances` to compute fact - foil.
        """
        if not isinstance(importances, torch.Tensor) or importances.ndim != 1:
            raise ValueError("`importances` must be a 1D torch.Tensor of concept importances.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        k = min(top_k, importances.numel())
        top_indices = torch.topk(importances.abs(), k=k).indices.tolist()

        concepts_key_value = [
            (
                ConSim._concept_descriptor(concept_id, concepts_interpretation),
                ConSim._quantize_importances(importances[concept_id].item(), threshold=threshold),
            )
            for concept_id in top_indices
        ]

        return (
            "{"
            + ", ".join(
                [
                    f"{concept_interpretation}: {concept_importance}"
                    for concept_interpretation, concept_importance in concepts_key_value
                    if concept_importance is not None
                ]
            )
            + "}"
        )

    @staticmethod
    def _select_concept_ids(
        importances: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.05,
    ) -> set[int]:
        """
        Select concept ids that would appear in a rendered concept string.
        """
        if not isinstance(importances, torch.Tensor) or importances.ndim != 1:
            raise ValueError("`importances` must be a 1D torch.Tensor of concept importances.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        k = min(top_k, importances.numel())
        if k == 0:
            return set()

        top_indices = torch.topk(importances.abs(), k=k).indices.tolist()
        return {
            int(concept_id)
            for concept_id in top_indices
            if ConSim._quantize_importances(importances[concept_id].item(), threshold) is not None
        }

    @staticmethod
    def _setting_to_prompt(  # noqa: PLR0912  # ignore too many branches  # too many special cases
        setting: PromptSetting,
        sentences: list[str],
        predictions: torch.Tensor,
        classes: dict[int, str],
        concepts_interpretation: dict[int, str],
        global_importances: dict[int, torch.Tensor],
        local_importances: list[torch.Tensor] | None,
        gold_labels: list[int] | None,
        top_k: int = 5,
        importance_threshold: float = 0.05,
        contrastive_pairs: list[tuple[int, int]] | None = None,
    ) -> tuple[str, str, list[str]]:
        """
        Create a prompt for the LLM model by integrating the different elements.
        The text is adapted to the particular setting to cover all possibilities.

        Many possibilities are not explored through the `PromptTypes` enum, because they do not make sense.

        Arguments:
            setting: PromptSetting
                Configuration, it says which elements should be included in the prompt.
            sentences: list[str]
                The sentences, the first half serve as examples and the second half is to be classified.
            predictions: torch.Tensor
                The predictions of the model on the sentences.
            classes: list[str]
                The classes of the dataset.
            concepts_interpretation: dict[int, str]
                Dictionary with concepts interpretations.
            global_importances: dict[int, torch.Tensor]
                The importance of the concepts for each class.
                A dictionary with the classes ids as keys and a vector of importances as values.
            local_importances: list[dict[int, str]] | None
                The importance of concepts for each sentence.
                A list with each element corresponding to one sentence.
                Each element of the list if a dictionary with an importance associated to a concept id.
            gold_labels: list[int] | None
                Gold labels for samples, required when contrastive local explanations are enabled.
            contrastive_pairs: list[tuple[int, int]] | None
                Contrastive pairs of classes, required when contrastive global explanations are enabled.

        Returns:
            system_prompt: str
                The system prompt for the LLM. All instructions, the initial and learning phases.
            user_prompt: str
                The user prompt for the LLM. The examples on with the user-llm should predict, thus the evaluation phase.
        """
        system_prompt_parts = []

        # ==============================================================================================
        # Global
        # ----------------
        # task description

        task_description_prompt = "You are a classifier. For each sample, you have to predict the class. "
        if setting.global_contrastive_importances:
            task_description_prompt += "To complete the task, you will be given the concepts and their contrastive importance for each class. "
        elif setting.concepts_global_importances:
            task_description_prompt += (
                "To complete the task, you will be given the most important concepts for each class. "
            )
        if setting.lp_samples:
            if setting.lp_concepts_local_contributions:
                task_description_prompt += "You will have examples of samples, labels, and concepts contributions to labels as reference for the task. "
            elif setting.lp_local_contrastive_importance:
                task_description_prompt += "You will have examples of samples, labels, and contrastive concepts contributions (why predict this class and not the other) as reference for the task. "
            else:
                task_description_prompt += "You will have examples of samples and labels as reference for the task. "
        task_description_prompt += "Each sample class prediction should be in the format: ```\nSample_0: {label_0}\nSample_1: {label_1}\n...\n``` with {label_i} being the class associated to the sample."
        # TODO: for concepts say how to interpret the values ++ + - --... what is negative and positive
        assert len(task_description_prompt) > 0
        system_prompt_parts.append(task_description_prompt)

        # -------
        # classes
        # if setting.pred_concepts:
        #     # show the concepts that could be predicted
        #     classes_prompt = f"The concepts are: [{', '.join(concepts_interpretation.keys())}]"
        if setting.anonymize_classes:
            classes = {i: f"Class_{i}" for i in classes.keys()}
        system_prompt_parts.append(f"The classes are: [{', '.join(list(classes.values()))}]")

        # ---------------------------
        # classes concepts importance
        if setting.concepts_global_importances:
            # for each anonymized class, show the top k concepts
            classes_concepts_prompt = (
                "The most important concepts and their importance for each class are:\n"
                + "\n".join(
                    [
                        f"\t{class_name}: {
                            ConSim._concepts_to_string(
                                global_importances[class_index],
                                concepts_interpretation,
                                top_k=top_k,
                                threshold=importance_threshold,
                            )
                        }"
                        for class_index, class_name in classes.items()
                    ]
                )
            )
            system_prompt_parts.append(classes_concepts_prompt)

        # global contrastive explanation
        if setting.global_contrastive_importances:
            if contrastive_pairs is None:
                raise ValueError(
                    "PromptSetting.global_contrastive_importances=True requires `contrastive_pairs` to be provided to ConSim.evaluate()."
                )

            contrastive_prompt_parts = []
            # for each contrastive pair, show the concept for fact - foil
            for pair in contrastive_pairs:
                contrastive_importance = global_importances[pair[0]] - global_importances[pair[1]]
                str_concept = ConSim._concepts_to_string(
                    contrastive_importance, concepts_interpretation, top_k=top_k, threshold=importance_threshold
                )
                contrastive_prompt_parts.append(f"\tfact: {classes[pair[0]]}, foil: {classes[pair[1]]}: {str_concept}")

            contrastive_global_prompt = (
                "The contrastively important concepts to choose fact over foil are:\n"
                + "\n".join(contrastive_prompt_parts)
            )
            system_prompt_parts.append(contrastive_global_prompt)

        # ==============================================================================================
        # Learning phase
        mid_index = len(sentences) // 2
        if setting.lp_samples:
            learning_phase_blocks = []
            for i in range(mid_index):
                # ----------------------------------------
                # samples text and label (predicted class)
                block = [f"Sample_{i}:", f"\tText: {sentences[i]}", f"\tLabel: {classes[int(predictions[i])]}"]

                # ----------------------------
                # concepts local contributions
                if setting.lp_concepts_local_contributions:
                    str_importances = ConSim._concepts_to_string(
                        local_importances[i][predictions[i].to(torch.int32).item()],  # type: ignore
                        concepts_interpretation,
                        top_k=top_k,
                        threshold=importance_threshold,
                    )
                    block.append(f"\tConcepts contributions: {str_importances}")  # type: ignore

                # ----------------------------------------
                # contrastive local concepts contributions
                if setting.lp_local_contrastive_importance:
                    pred_index = int(predictions[i].item())
                    gold_index = gold_labels[i]  # type: ignore

                    # show the contrastive only for misclassified samples
                    if pred_index == gold_index:
                        text = "Concepts contributions"
                        importances = local_importances[i][pred_index]  # type: ignore
                    else:
                        text = f"Concepts contributions supporting {classes[pred_index]} rather than {classes[gold_index]}"
                        importances = local_importances[i][pred_index] - local_importances[i][gold_index]  # type: ignore

                    # convert the importances to a string
                    str_importances = ConSim._concepts_to_string(
                        importances,
                        concepts_interpretation,
                        top_k=top_k,
                        threshold=importance_threshold,
                    )
                    block.append(f"\t{text}: {str_importances}")  # type: ignore

                learning_phase_blocks.append("\n".join(block))
            system_prompt_parts.append("\n".join(learning_phase_blocks))

        # ==============================================================================================
        # Inference
        # ----------------
        # show the samples
        ep_local_prompt = "\n".join([f"Sample_{i}: {sentences[i]}" for i in range(mid_index, 2 * mid_index)])

        # -----------------
        # model predictions (not included in the prompt, but returned to compute accuracy)
        literal_model_predictions = [classes[int(predictions[i])] for i in range(mid_index)]

        # concatenate prompts parts
        system_prompt = "\n\n".join(system_prompt_parts)
        user_prompt = ep_local_prompt

        return system_prompt, user_prompt, literal_model_predictions

    @staticmethod
    def _check_input_settings_correspondence(
        sentences: list[str],
        nb_classes: int,
        concepts_interpretation: dict[int, str],
        global_importances: torch.Tensor,
        local_importances: list[torch.Tensor] | None,
        gold_labels: list[int] | None,
        prompt_type: PromptTypes | PromptSetting = PromptTypes.E3_global_and_local_concepts_with_lp,
    ):
        """
        Create prompts for the user-llm or meta-predictor.

        First the different elements are processed so that they can be included in the prompt via `ConSim._generate_prompt`.
        Then the elements are integrated into a prompt via `ConSim._setting_to_prompt`.

        Arguments:
            sentences: list[str]
                The sentences, the first half serve as examples and the second half is to be classified.
            predictions: torch.Tensor
                The predictions of the model on the sentences.
            nb_classes: int
                The number of classes.
            concepts_interpretation: dict[int, str]
                The interpretation of the concepts, concepts are the keys.
                For example, an interpretation could be the topk words that activates the most a given concepts.
            global_importances: torch.Tensor
                The importance of the concepts for each class.
                Shape must be (nb_classes, nb_concepts).
            local_importances: list[torch.Tensor] | None
                Local concepts importances for each sentence.
                Each element must have shape (nb_classes, nb_concepts).
            gold_labels: list[int] | None
                Gold labels for learning phase samples, required when contrastive local explanations are enabled.
            prompt_type: PromptTypes | PromptSetting
                The type of prompt to use. Possible values are:

                - `PromptTypes.L1_baseline_without_lp`: baseline without learning phase.

                - `PromptTypes.E1_global_concepts_without_lp`: global concepts without learning phase.

                - `PromptTypes.L2_baseline_with_lp`: baseline with learning phase.

                - `PromptTypes.E2_global_concepts_with_lp`: global concepts with learning phase.

                - `PromptTypes.E3_global_and_local_concepts_with_lp`: global and local concepts with learning phase.

                - `PromptTypes.U1_upper_bound_concepts_at_ep`: upper bound - concepts at evaluation phase.

            importance_threshold: float
                The threshold to quantize concept importances.

        Returns:
            prompt: list[tuple[Role, str]]
                The prompts for the LLM, the format matches the `LLMInterface` API.
            literal_model_predictions: list[str]
                The model predictions as a list of strings, it allows easier comparison with the `user_llm` answers.
        """
        setting = ConSim._resolve_prompt_setting(prompt_type)
        setting.validate(concepts_interpretation=concepts_interpretation, gold_labels=gold_labels)

        mid_index = len(sentences) // 2

        # ==========================================================================================
        # Extensive checks
        if not isinstance(global_importances, torch.Tensor) or global_importances.ndim != 2:
            raise ValueError("`global_importances` must be a torch.Tensor with shape (nb_classes, nb_concepts).")
        if global_importances.shape[0] != nb_classes:
            raise ValueError(
                "`global_importances` must have shape (nb_classes, nb_concepts) with nb_classes matching `classes`."
            )

        if local_importances is not None:
            if not isinstance(local_importances, list) or not local_importances:
                raise ValueError(
                    "`local_importances` must be a non-empty list of torch.Tensor with shape (nb_classes, nb_concepts)."
                )
            for sample_importances in local_importances:
                if not isinstance(sample_importances, torch.Tensor) or sample_importances.ndim != 2:
                    raise ValueError(
                        "`local_importances` must be a list of torch.Tensor with shape (nb_classes, nb_concepts)."
                    )
                if sample_importances.shape[0] != nb_classes:
                    raise ValueError(
                        "`local_importances` entries must have shape (nb_classes, nb_concepts) with nb_classes matching `classes`."
                    )
            local_importances_count = len(local_importances)
            if local_importances_count not in {mid_index, len(sentences)}:
                raise ValueError(
                    "`local_importances` must have nb_samples entries matching the learning phase size "
                    "or the full number of sentences. "
                    f"Got nb_samples={local_importances_count} and nb_sentences={len(sentences)}."
                )

        if setting.lp_concepts_local_contributions or setting.lp_local_contrastive_importance:
            if local_importances is None:
                raise ValueError(
                    "PromptSetting.lp_concepts_local_contributions or PromptSetting.lp_local_contrastive_importance "
                    "requires `local_importances` to be provided to ConSim.evaluate()."
                )
            if setting.lp_local_contrastive_importance:
                if gold_labels is None or len(gold_labels) < mid_index:
                    raise ValueError(
                        "PromptSetting.lp_local_contrastive_importance=True requires `gold_labels` "
                        "to be provided for the learning phase samples."
                    )

    @staticmethod
    def _extract_predictions_from_response(response: str | None, expected_length: int) -> list[str] | None:
        """
        Extract the model predictions from the response.
        The response is expected to be a list of predictions for each sample.
        The predictions are expected to be separated by "\n".

        Example of a response:

        ```
        Sample_0: physician
        Sample_1: surgeon
        Sample_2: nurse
        ```

        Arguments:
            response: str
                The response from the user-llm or meta-predictor.
            expected_length: int
                The expected length of the response.

        Returns:
            predictions: list[str] | None
                The model predictions.
                If the response is empty or the expected length is not respected, returns None.
        """
        if response == "" or response is None:
            return None

        sentences = response.split("\n")

        while sentences[-1] == "":
            sentences = sentences[:-1]

        if len(sentences) != expected_length:
            return None

        # format not respected, expecting predictions to be only separated by "\n"
        if ":" not in sentences[0]:
            return sentences

        # expected format: Sample_0: physician\nSample_1: surgeon\nSample_2: nurse
        predictions = [
            sentence.split(": ")[1].strip().lower().split(" ")[0]
            for sentence in sentences
            if (sentence[:10] == "Prediction" or sentence[:8] == "Sentence" or sentence[:6] == "Sample")
        ]
        return predictions

    @staticmethod
    def _predictions_accuracy(model_predictions: list[str], user_llm_predictions: list[str]) -> float | None:
        """
        Compute the accuracy of the model predictions.

        Arguments:
            model_predictions: list[str]
                The model predictions.
            user_llm_predictions: list[str]
                The user-llm predictions.

        Returns:
            accuracy: float | None
                The accuracy of the model predictions.
                If the model predictions are empty or the user-llm predictions are empty, returns None.

        Raises:
            ValueError
                If the model predictions and the user-llm predictions have different lengths.
        """
        if len(model_predictions) != len(user_llm_predictions):
            warnings.warn(
                "Predictions between model and user-llm have different lengths, returning None"
                f"respectively: {len(model_predictions)} and {len(user_llm_predictions)}.",
                stacklevel=2,
            )
            return None

        n_correct = len(
            [
                1
                for pred1, pred2 in zip(model_predictions, user_llm_predictions, strict=True)
                if pred1.lower() == pred2.lower()
            ]
        )

        return n_correct / len(model_predictions)

    @staticmethod
    def _compute_score(
        user_llm_response: str | None,
        literal_model_predictions: list[str],
    ) -> float | None:
        """
        Compute the score of the ConSim metric,
        thus the accuracy of the `user_llm` predictions with respect to the model predictions.

        Responses from the `user_llm` are expected to be in the format:

        ```
        Sample_0: physician
        Sample_1: surgeon
        Sample_2: nurse
        ```

        Arguments:
            user_llm_response: str
                The response from the user-llm or meta-predictor.
            literal_model_predictions: list[str]
                The model predictions.

        Returns:
            score: float | None
                The score of the ConSim metric.
                If the model predictions are empty or the user-llm predictions are empty, returns None.

        Raises:
            ValueError
                If the model predictions and the user-llm predictions have different lengths.
        """
        # extract the model predictions
        literal_meta_predictions = ConSim._extract_predictions_from_response(
            response=user_llm_response, expected_length=len(literal_model_predictions)
        )

        if literal_meta_predictions is None or len(literal_meta_predictions) == 0:
            warnings.warn(
                "The user-llm responses are empty or the format is not respected. Returning None. "
                f"The response was: '{user_llm_response}'",
                stacklevel=2,
            )
            return None

        # compute the accuracy
        return ConSim._predictions_accuracy(literal_model_predictions, literal_meta_predictions)

    def evaluate(
        self,
        interesting_samples: list[str],
        predictions: torch.Tensor,
        concept_explainer: ConceptAutoEncoderExplainer | None = None,
        *,
        concepts_interpretation: dict[int, str],
        global_importances: torch.Tensor,
        local_importances: list[torch.Tensor] | None = None,
        prompt_type: PromptTypes | PromptSetting = PromptTypes.E3_global_and_local_concepts_with_lp,
        top_k: int = 5,
        importance_threshold: float = 0.05,
        gold_labels: list[int] | None = None,
        contrastive_pairs: list[tuple[int, int]] | None = None,
    ) -> float | None | tuple[list[tuple[Role, str]], list[str]]:
        """
        Evaluate the ConSim metric, thus the accuracy of the `user_llm` predictions with respect to the model predictions.

        First local concepts importances are computed via the `concept_explainer`.
        Then a prompt is constructed by integrating all the different elements and following the `prompt_type`.
        The prompt is sent to the `user_llm` and the model predictions are extracted from the response.
        Finally, the score is computed by comparing the model predictions with the `user_llm` predictions.

        The prompts have five parts:

        - Initial Phase (IP.1) the first part is the task description, which is a list of questions to ask the LLM.

        - Initial Phase (IP.2) the second is a global concepts explanation on $f$. Listing the important concepts for each class.

        - Learning Phase (LP.1) the third gives examples of samples and predictions from the model $f$.

        - Learning Phase (LP.2) the fourth is a local concepts explanation on $f$. Listing the important concepts in each example.

        - Evaluation Phase (EP.1) the fifth is a list of samples on which the meta-predictor $\\Psi$ will be asked to predict the model $f$ predictions.

        The answer of the LLM will be a list of predictions for each sample. ConSim compares these predictions to the
        model $f$ predictions and computes the accuracy of the explanations.

        Arguments:
            interesting_samples: list[str]
                The interesting samples.

            predictions: torch.Tensor
                The predictions of the model on the interesting samples.

            concept_explainer: ConceptAutoEncoderExplainer | None
                The concept explainer. Can be None for the baseline.

            concepts_interpretation: dict[int, str]
                The concepts interpretation labels, keyed by concept id.

            global_importances: torch.Tensor
                The importance of the concepts for each class.
                Shape must be (nb_classes, nb_concepts).

            local_importances: list[torch.Tensor] | None
                The importance of concepts for each sentence.
                Each element must have shape (nb_classes, nb_concepts).
                This is computed automatically if not provided.

            prompt_type: PromptTypes | PromptSetting
                The type of prompt to use. Possible values are:

                - `PromptTypes.L1_baseline_without_lp`: baseline without learning phase.

                - `PromptTypes.E1_global_concepts_without_lp`: global concepts without learning phase.

                - `PromptTypes.L2_baseline_with_lp`: baseline with learning phase.

                - `PromptTypes.E2_global_concepts_with_lp`: global concepts with learning phase.

                - `PromptTypes.E3_global_and_local_concepts_with_lp`: global and local concepts with learning phase.

                - `PromptTypes.U1_upper_bound_concepts_at_ep`: upper bound - concepts at evaluation phase.

            top_k: int
                The number of top concepts to show for each class / sample.

            importance_threshold: float
                The threshold to quantize concept importances.

            gold_labels: list[int] | None
                Gold labels for learning phase samples, required when contrastive local explanations are enabled.

            contrastive_pairs: list[tuple[int, int]] | None
                Contrastive pairs of classes, required when contrastive global explanations are enabled.


        Returns:
            score or prompts and model predictions: float | None | tuple[list[tuple[Role, str]], list[str]]
                Possible outputs:

                - score (float): The score of the ConSim metric. (The nominal behavior)
                - None: If the model predictions are empty or the user-llm predictions are empty.
                    It was chosen to return None,
                    because ConSim should be called a lot of times for statistically significant results.
                    Therefore, having a None score once in a while is better than the script crashing.
                - prompts and model predictions (tuple[list[tuple[Role, str]], list[str]]):
                    If no user_llm is provided, returns the prompts and the model predictions.
                    The prompt is the first element of the tuple (list[tuple[Role, str]]).
                    The predictions are the second element of the tuple (list[str]).
                    The user will have to call the ConSim prompts manually.
                    The response of the LLM on the prompts should be compared to the model predictions.

        Raises:
            ValueError
                If the model predictions and the user-llm predictions have different lengths.
            Warnings
                If the user-llm response is empty or the format is not respected.
        """
        setting = ConSim._resolve_prompt_setting(prompt_type)

        needs_local_importances = setting.lp_concepts_local_contributions or setting.lp_local_contrastive_importance
        if local_importances is None and concept_explainer is not None and needs_local_importances:
            # Ensure the mwsp of the explainer is the same as the one used in the provided concept_explainer
            if concept_explainer.split_point not in self.model_with_split_points.split_points:
                raise ValueError(
                    "The split point used in the provided `concept_explainer` should be one of the `model_with_split_points` ones."
                    f"Got split point: '{concept_explainer.split_point}' with model split points: "
                    f"{', '.join(self.model_with_split_points.split_points)}."
                )
            if (
                concept_explainer.model_with_split_points._model.config.name_or_path
                != self.model_with_split_points._model.config.name_or_path
            ):
                raise ValueError(
                    "The model used in the provided `concept_explainer` should be the same as the one used in the `model_with_split_points`."
                    f"Got (concept_explainer) model name or path: '{concept_explainer.model_with_split_points._model.config.name_or_path}'"
                    f"and (model_with_split_points) model name or path: '{self.model_with_split_points._model.config.name_or_path}'."
                )

            # compute concepts importance  # TODO: when first layers can be skipped pass the concept activations
            # For now we force gradient-input

            local_importances = concept_explainer.concept_output_gradient(
                inputs=interesting_samples[: len(interesting_samples) // 2],
                split_point=self.split_point,
                activation_granularity=self.activation_granularity,
                concepts_x_gradients=True,
                tqdm_bar=False,
            )
            local_importances = [sample_importance.squeeze(1) for sample_importance in local_importances]

        # check inputs with respect to setting
        ConSim._check_input_settings_correspondence(
            sentences=interesting_samples,
            nb_classes=len(self.classes),
            concepts_interpretation=concepts_interpretation,
            global_importances=global_importances,
            local_importances=local_importances,
            gold_labels=gold_labels,
            prompt_type=setting,
        )

        # extract the classes present in the predictions or the gold labels
        classes_ids = sorted(set(predictions.tolist()) | set(gold_labels or []))
        classes = {class_id: self.classes[class_id] for class_id in classes_ids}

        # filter based on classes subset
        global_importances_dict = {class_id: global_importances[class_id] for class_id in classes_ids}
        if contrastive_pairs is not None:
            contrastive_pairs = [
                pair for pair in contrastive_pairs if (pair[0] in classes_ids and pair[1] in classes_ids)
            ]  # type: ignore

        # integrate the different elements into a prompt
        system_prompt, user_prompt, literal_model_predictions = ConSim._setting_to_prompt(
            setting=setting,
            sentences=interesting_samples,
            predictions=predictions,
            classes=classes,
            concepts_interpretation=concepts_interpretation,
            global_importances=global_importances_dict,
            local_importances=local_importances,
            gold_labels=gold_labels,
            top_k=top_k,
            importance_threshold=importance_threshold,
            contrastive_pairs=contrastive_pairs,
        )

        # convert the prompt to match the `LLMInterface` API
        prompts: list[tuple[Role, str]] = [
            (Role.SYSTEM, system_prompt),
            (Role.USER, user_prompt),
            (Role.ASSISTANT, ""),
        ]

        # if no user_llm is provided, we return the prompts and the model predictions
        if self.user_llm is None:
            return prompts, literal_model_predictions

        user_llm_response = self.user_llm.generate(prompts)

        # raise warnings if the response is empty or the format is not respected
        return self._compute_score(
            user_llm_response=user_llm_response,
            literal_model_predictions=literal_model_predictions,
        )
