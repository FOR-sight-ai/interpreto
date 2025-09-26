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

    # global
    concepts_interpretation: bool = False

    # initial phase
    concepts_global_importances: bool = False

    # learning phase
    lp_samples: bool = False
    lp_concepts_local_contributions: bool = False
    lp_labels: bool = False

    # evaluation phase
    ep_samples: bool = True
    ep_concepts_local_contributions: bool = False

    # prediction
    # pred_concepts: bool = False


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

        `U1_upper_bound_concepts_at_ep`:
            IP.1, IP.2, LP.1, LP.2, EP.1, and EP.2 are included in the prompt.
            Same as `E3_global_and_local_concepts_with_lp`, but local explanations are also given at evaluation phase.
            This has a very high probability to leak the initial model predictions via EP local explanations.
            Warning, this should not be considered as a ConSim score.
            But it gives an upper bound to the ConSim scores.
    """

    L1_baseline_without_lp = PromptSetting()
    E1_global_concepts_without_lp = PromptSetting(concepts_interpretation=True, concepts_global_importances=True)
    L2_baseline_with_lp = PromptSetting(lp_samples=True, lp_labels=True)
    E2_global_concepts_with_lp = PromptSetting(
        concepts_interpretation=True, concepts_global_importances=True, lp_samples=True, lp_labels=True
    )
    E3_global_and_local_concepts_with_lp = PromptSetting(
        concepts_interpretation=True,
        concepts_global_importances=True,
        lp_samples=True,
        lp_concepts_local_contributions=True,
        lp_labels=True,
    )
    U1_upper_bound_concepts_at_ep = PromptSetting(
        concepts_interpretation=True,
        concepts_global_importances=True,
        lp_samples=True,
        lp_concepts_local_contributions=True,
        lp_labels=True,
        ep_concepts_local_contributions=True,
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
        >>> concept_explainer.fit(activations)

        The two steps of ConSim:
        >>> # ------------------------------------------------------------------
        >>> # Step 0: Define the User-LLM and instantiate the ConSim metric
        >>> user_llm = OpenAILLM(api_key="YOUR_OPENAI_API_KEY", model="gpt4o-mini")
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
        >>> baseline = consim.evaluate(samples, labels, predictions, prompt_type=PromptTypes.L2_baseline_with_lp)
        >>> consim_score = consim.evaluate(samples, labels, predictions, concept_explainer_1, prompt_type=PromptTypes.E3_global_and_local_concepts_with_lp)
    """

    prompt_types: type[PromptTypes] = PromptTypes

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        user_llm: LLMInterface | None,
        activation_granularity: ActivationGranularity,
        classes: list[str] | None = None,
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
        self.classes: list[str] | None = classes

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

        Returns:
            interesting_samples: list[str]
                The interesting samples.
            labels: torch.Tensor
                The labels of the interesting samples.
            predictions: torch.Tensor
                The predictions of the model on the interesting samples.
        """
        nb_classes = len(self.classes) if self.classes is not None else len(torch.unique(labels))
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
        for c in range(nb_classes):
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
        nb_lp_samples: int = 20,
        nb_ep_samples: int = 20,
        seed: int = 0,
        batch_size: int = 64,
        device: torch.device | str = "cpu",
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
            nb_lp_samples: int
                The number of samples to select for the learning phase.
            nb_ep_samples: int
                The number of samples to select for the evaluation phase.
            seed: int
                The seed to use for the random selection.
            batch_size: int
                The batch size to use for the predictions.
            device: torch.device | str
                The device to use for the predictions.

        Returns:
            interesting_samples: list[str]
                The interesting samples.
            labels: torch.Tensor
                The labels of the interesting samples.
            predictions: torch.Tensor
                The predictions of the model on the interesting samples.
        """
        predictions = self._get_predictions(inputs, batch_size=batch_size, device=device)
        return self._extract_interesting_elements(
            inputs=inputs,
            labels=labels,
            predictions=predictions,
            nb_lp_samples=nb_lp_samples,
            nb_ep_samples=nb_ep_samples,
            seed=seed,
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
    def _filter_and_quantize_concepts_importances(
        concepts_interpretation: dict[str, str],
        global_importances: dict[str, dict[str, float]],
        local_importances: torch.Tensor | None,
        importance_threshold: float = 0.05,
    ) -> tuple[dict[str, str], dict[str, dict[str, str]], list[dict[str, str]] | None]:
        """
        Filter the concepts importance and quantize the values.

        The concepts with normalized global importances under the threshold are removed.
        Similarly, the concepts with normalized local importances under the threshold are removed.

        Only the intersection between the two are kept.

        The concepts importance are quantized to literals.
        The corresponding literals and ranges (values for the default threshold value of 0.05) are:

        - "++" for values above 0.3

        - "+" for values between 0.05 and 0.3

        - "-" for values between -0.05 and -0.3

        - "--" for values below -0.3

        Note that this values correspond to the normalized importance of the concepts.
        The normalization is done by dividing the importances by the sum of the absolute values of the importances.

        Arguments:
            concepts_interpretation: dict[str, str]
                The words that activate the concepts the most and the least.
                A dictionary with the concepts as keys and another dictionary as values.
                The inner dictionary has the words as keys and the activations as values.
            global_importances: dict[str, str]
                The importance of the concepts for each class.
                A dictionary with the classes as keys and another dictionary as values.
                The inner dictionary has the concepts as keys and the importance as values.
            local_importances: torch.Tensor | None
                Matrix of concept importances for each sentence. Shape (n_sentences, n_concepts)
            importance_threshold: float
                The threshold to select the most important concepts for each class.
                The threshold correspond to the cumulative importance of the concepts to keep.

        Returns:
            concepts_interpretation: dict[str, str]
                The words that activate the concepts the most and the least.
                A dictionary with the concepts as keys and another dictionary as values.
                The inner dictionary has the words as keys and the activations as values.
                Filtered to keep only the important concepts.
            filtered_global_importances: dict[str, dict[str, float]]
                The importance of the concepts for each class.
                A dictionary with the classes as keys and another dictionary as values.
                The inner dictionary has the concepts as keys and the importance as values.
                Filtered to keep only the important concepts.
            filtered_local_importances: list[dict[str, str]] | None
                The importance of concepts for each sentence.
                A list with each element corresponding to one sentence.
                Each element of the list if a dictionary with an importance associated to a concept id.
                Filtered to keep only the important concepts.
        """
        # ------------------------------------------------------------------------------------------
        # filter concepts which are important for at least one class
        concepts_to_keep = []
        while len(concepts_to_keep) == 0:
            for concepts_importance in global_importances.values():
                if len(concepts_importance) == 0:
                    continue

                # normalize the importances
                importances = torch.abs(torch.Tensor(list(concepts_importance.values())))
                normalized_importances = importances / importances.sum()

                # select the important concepts
                added_concepts = torch.where(normalized_importances > importance_threshold)[0]
                concepts_to_keep.extend(added_concepts)
            if len(concepts_to_keep) == 0:
                importance_threshold /= 2

        concepts_to_show = torch.unique(torch.stack(concepts_to_keep))
        interpretation_concepts_ids = [int(cpt.split("_")[-1]) for cpt in concepts_interpretation.keys()]
        concepts_to_show = torch.Tensor([cpt for cpt in concepts_to_show if cpt in interpretation_concepts_ids]).to(
            dtype=torch.int64
        )

        # ------------------------------------------------------------------------------------------
        # filter the concepts activating words
        concepts_interpretation = {f"concept_{c}": concepts_interpretation[f"concept_{c}"] for c in concepts_to_show}

        # ------------------------------------------------------------------------------------------
        # filter the concepts importance
        quantized_global_importances: dict[str, dict[str, str]] = {}
        # iterate over classes
        for class_name, concepts_importance in global_importances.items():
            quantized_global_importances[class_name] = {}
            # iterate over concepts
            for c, importance in concepts_importance.items():
                # quantize the importance and pass it to string
                quantized_importance: str | None = ConSim._quantize_importances(importance, importance_threshold)
                # keep only the concepts that should be shown and are important enough
                if int(c.split("_")[-1]) in concepts_to_show and quantized_importance is not None:
                    quantized_global_importances[class_name][c] = quantized_importance

        if local_importances is None:
            return concepts_interpretation, quantized_global_importances, None

        # normalize sentences concepts importances
        local_importances = local_importances / local_importances.abs().sum(dim=1, keepdim=True)

        # ------------------------------------------------------------------------------------------
        # clean elements to leave only the important concepts and quantize values to literals
        filtered_local_importances: list[dict[str, str]] = []
        # iterate over sentences
        for sentence_concepts_importances in local_importances:
            filtered_sentence_importances: dict[str, str] = {}
            # iterate over concepts
            for cpt, importance in enumerate(sentence_concepts_importances):
                # quantize the importance and pass it to string
                quantized_importance: str | None = ConSim._quantize_importances(
                    importance.item(), importance_threshold
                )
                # keep only the concepts that should be shown and are important enough
                if cpt in concepts_to_show and quantized_importance is not None:
                    filtered_sentence_importances[f"concept_{cpt}"] = quantized_importance
            filtered_local_importances.append(filtered_sentence_importances)

        return concepts_interpretation, quantized_global_importances, filtered_local_importances

    @staticmethod
    def _setting_to_prompt(  # noqa: PLR0912  # ignore too many branches  # too many special cases
        setting: PromptSetting,
        anonymize_classes: bool,
        sentences: list[str],
        predictions: torch.Tensor,
        classes: list[str],
        concepts_interpretation: dict[str, str] | None,
        global_importances: dict[str, dict[str, str]] | None,
        local_importances: list[dict[str, str]] | None,
    ) -> tuple[str, str, list[str]]:
        """
        Create a prompt for the LLM model by integrating the different elements.
        The text is adapted to the particular setting to cover all possibilities.

        Many possibilities are not explored through the `PromptTypes` enum, because they do not make sense.

        Arguments:
            setting: PromptSetting
                Configuration, it says which elements should be included in the prompt.
            anonymize_classes: bool
                Whether to anonymize the classes.
            sentences: list[str]
                The sentences, the first half serve as examples and the second half is to be classified.
            predictions: torch.Tensor
                The predictions of the model on the sentences.
            classes: list[str]
                The classes of the dataset.
            concepts_interpretation: dict[str, str]
                The words that activate the concepts the most and the least.
                A dictionary with the concepts as keys and another dictionary as values.
                The inner dictionary has the words as keys and the activations as values.
            global_importances: dict[str, dict[str, str]]
                The importance of the concepts for each class.
                A dictionary with the classes as keys and another dictionary as values.
                The inner dictionary has the concepts as keys and the importance as values.
            local_importances: list[dict[str, str]]
                The importance of concepts for each sentence.
                A list with each element corresponding to one sentence.
                Each element of the list if a dictionary with an importance associated to a concept id.

        Returns:
            system_prompt: str
                The system prompt for the LLM. All instructions, the initial and learning phases.
            user_prompt: str
                The user prompt for the LLM. The examples on with the user-llm should predict, thus the evaluation phase.
        """
        system_prompt_parts = []
        user_prompt_parts = []

        # ==============================================================================================
        # Global
        # ----------------
        # task description

        task_description_prompt = "You are a classifier. For each sample, you have to predict the class. "
        if setting.concepts_interpretation or setting.concepts_global_importances:
            task_description_prompt += (
                "To complete the task, you will be given the concepts and their importance for each class. "
            )
        if setting.lp_samples and setting.lp_labels:
            if setting.lp_concepts_local_contributions:
                task_description_prompt += "You will have examples of samples, labels, and concepts contributions to labels as reference for the task. "
            else:
                task_description_prompt += "You will have examples of samples and labels as reference for the task. "
        if setting.ep_concepts_local_contributions:
            task_description_prompt += "At inference time, you will have concepts contributions to labels. "
        task_description_prompt += (
            "Each sample class prediction should be in the format: 'Sample_{i}: {predicted_class}'."
        )

        assert len(task_description_prompt) > 0
        system_prompt_parts.append(task_description_prompt)

        # -------
        # classes
        # if setting.pred_concepts:
        #     # show the concepts that could be predicted
        #     classes_prompt = f"The concepts are: [{', '.join(concepts_interpretation.keys())}]"
        if anonymize_classes:
            # show the classes without their names
            anonym_classes = {class_name: f"Class_{i}" for i, class_name in enumerate(classes)}
            classes_prompt = f"The classes are: [{', '.join(anonym_classes.values())}]"
        else:
            # show the classes
            anonym_classes = {class_name: class_name for class_name in classes}  # placeholder for type checker
            classes_prompt = f"The classes are: [{', '.join(classes)}]"
        system_prompt_parts.append(classes_prompt)

        # -------------------------
        # concepts activating words
        if setting.concepts_interpretation:
            if concepts_interpretation is None:
                raise ValueError(
                    "Concepts interpretation must be provided if prompt_type is not a baseline."
                    "`concepts_interpretation` is an argument of the `ConSim.evaluate()` method, but it is None."
                    "It can be computed via `concept_explainer.interpret`."
                )

            # for each concept, show 10 words, 5 that aligns the most and 5 that are the most opposed
            concepts_interpretation_prompt = (
                "For each concept, the most aligned words or descriptions are:\n"
                + "\n".join(
                    [
                        f"{concept_id}: {interpretation}"
                        for concept_id, interpretation in concepts_interpretation.items()
                    ]
                )
            )
            system_prompt_parts.append(concepts_interpretation_prompt)

        # ---------------------------
        # classes concepts importance
        if setting.concepts_global_importances:
            if global_importances is None:
                raise ValueError(
                    "Global concepts importances must be provided if prompt_type is not a baseline."
                    "`global_importances` is an argument of the `ConSim.evaluate()` method, but it is None."
                    "It can be computed via `concept_explainer.concept_output_gradient` then averaging for each class."
                )

            # show the importance of the concepts for each class
            if anonymize_classes:
                classes_concepts_prompt = (
                    "The most important concepts and their importance for each class are:\n"
                    + "\n".join(
                        [f"{anonym_classes[class_name]}: {value}" for class_name, value in global_importances.items()]
                    )
                )
            else:
                classes_concepts_prompt = (
                    "The most important concepts and their importance for each class are:\n"
                    + "\n".join([f"{key}: {value}" for key, value in global_importances.items()])
                )
            system_prompt_parts.append(classes_concepts_prompt)

        # ==============================================================================================
        # Learning phase
        mid_index = len(sentences) // 2
        # -------
        # samples
        if setting.lp_samples:
            # show the samples
            lp_local_prompt = "\n".join([f"Sample_{i}: {sentences[i]}" for i in range(mid_index)])
            system_prompt_parts.append(lp_local_prompt)

        # ----------------------------
        # concepts local contributions
        if setting.lp_concepts_local_contributions:
            if local_importances is None:
                raise ValueError(
                    "Local concepts importances must be provided if prompt_type is E3 or U1. "
                    "`local_importances` are computed via `concept_explainer.concept_output_gradient`. "
                    "Consider using the `ConSim.evaluate()` method, it includes the computation of the local importances."
                )

            # show the concepts contributions to the samples
            lp_concepts_local_contributions_prompt = "\n".join(
                [f"Concepts contributions for Sample_{i}: {local_importances[i]}" for i in range(mid_index)]
            )
            system_prompt_parts.append(lp_concepts_local_contributions_prompt)

        # ------
        # labels
        if setting.lp_labels:
            # show the labels
            if anonymize_classes:
                lp_labels_prompt = "\n".join(
                    [f"Sample_{i}: {anonym_classes[classes[predictions[i]]]}" for i in range(mid_index)]
                )
            else:
                lp_labels_prompt = "\n".join([f"Sample_{i}: {classes[predictions[i]]}" for i in range(mid_index)])
            system_prompt_parts.append(lp_labels_prompt)

        # ==============================================================================================
        # Inference
        # -------
        # samples
        if setting.ep_samples:
            # show the samples
            ep_local_prompt = "\n".join([f"Sample_{i}: {sentences[i]}" for i in range(mid_index, 2 * mid_index)])
            user_prompt_parts.append(ep_local_prompt)

        # ----------------------------
        # concepts local contributions
        if setting.ep_concepts_local_contributions:
            if local_importances is None:
                raise ValueError(
                    "Local concepts importances must be provided if prompt_type is U1. "
                    "`local_importances` are computed via `concept_explainer.concept_output_gradient`. "
                    "Consider using the `ConSim.evaluate()` method, it includes the computation of the local importances."
                )

            # show the concepts contributions to the samples
            ep_concepts_local_contributions_prompt = "\n".join(
                [
                    f"Concepts contributions for Sample_{i}: {local_importances[i]}"
                    for i in range(mid_index, 2 * mid_index)
                ]
            )
            user_prompt_parts.append(ep_concepts_local_contributions_prompt)

        # -----------------
        # model predictions (not included in the prompt, but returned to compute accuracy)
        literal_model_predictions = [classes[predictions[i]] for i in range(mid_index)]
        if anonymize_classes:
            literal_model_predictions = [anonym_classes[class_name] for class_name in literal_model_predictions]

        # concatenate prompts parts
        system_prompt = "\n\n".join(system_prompt_parts)
        user_prompt = "\n\n".join(user_prompt_parts)

        return system_prompt, user_prompt, literal_model_predictions

    @staticmethod
    def _generate_prompt(
        sentences: list[str],
        predictions: torch.Tensor,
        classes: list[str] | None,
        concepts_interpretation: dict[str, str] | None,
        global_importances: dict[str, dict[str, float]] | None,
        local_importances: torch.Tensor | None,
        prompt_type: PromptTypes = PromptTypes.E3_global_and_local_concepts_with_lp,
        anonymize_classes: bool = False,
        importance_threshold: float = 0.05,
    ) -> tuple[list[tuple[Role, str]], list[str]]:
        """
        Create prompts for the user-llm or meta-predictor.

        First the different elements are processed so that they can be included in the prompt via `ConSim._generate_prompt`.
        Then the elements are integrated into a prompt via `ConSim._setting_to_prompt`.

        Arguments:
            sentences: list[str]
                The sentences, the first half serve as examples and the second half is to be classified.
            predictions: list[float]
                The predictions of the model on the sentences.
            classes: list[str] | None
                The classes of the dataset.
            concepts_interpretation: dict[str, str] | None
                The interpretation of the concepts, concepts are the keys.
                For example, an interpretation could be the topk words that activates the most a given concepts.
            global_importances: dict[str, dict[str, float]] | None
                The importance of the concepts for each class.
                A dictionary with the classes as keys and another dictionary as values.
                The inner dictionary has the concepts as keys and the importance as values.
            local_importances: torch.Tensor | None
                Local concepts importances for each sentence.
                A list of tensors with shape (n_concepts,).
            prompt_type: PromptTypes
                The type of prompt to use. Possible values are:

                - `PromptTypes.L1_baseline_without_lp`: baseline without learning phase.

                - `PromptTypes.E1_global_concepts_without_lp`: global concepts without learning phase.

                - `PromptTypes.L2_baseline_with_lp`: baseline with learning phase.

                - `PromptTypes.E2_global_concepts_with_lp`: global concepts with learning phase.

                - `PromptTypes.E3_global_and_local_concepts_with_lp`: global and local concepts with learning phase.

                - `PromptTypes.U1_upper_bound_concepts_at_ep`: upper bound - concepts at evaluation phase.

            anonymize_classes: bool
                Whether to anonymize the classes. Class names will be replaced by "Class_i" where i is the index of the class.
                It prevents the user-llm to solve the task by simply guessing the class.
            importance_threshold: float
                The threshold to select the most important concepts for each class.
                The threshold correspond to the cumulative importance of the concepts to keep.

        Returns:
            prompt: list[tuple[Role, str]]
                The prompts for the LLM, the format matches the `LLMInterface` API.
            literal_model_predictions: list[str]
                The model predictions as a list of strings, it allows easier comparison with the `user_llm` answers.
        """
        if classes is None and anonymize_classes is False:
            raise ValueError(
                "Classes must be provided if anonymize_classes is False."
                "`classes` is an attribute of the `ConSim` class, but it is None."
                "It can be set a initialization or with `consim_metric.classes = ['cat', 'dog', 'frog']`."
            )

        # guessing the classes if not provided
        if classes is None:
            classes = ["Class_" + str(i) for i in range(int(predictions.max().item()) + 1)]

        # Catch non provided but required elements
        if prompt_type.value.concepts_interpretation and concepts_interpretation is None:
            raise ValueError(
                "Concepts interpretation must be provided if prompt_type is not a baseline."
                "`concepts_interpretation` is an argument of the `ConSim.evaluate()` method, but it is None."
                "It can be computed via `concept_explainer.interpret`."
            )

        if prompt_type.value.concepts_global_importances and global_importances is None:
            raise ValueError(
                "Global concepts importances must be provided if prompt_type is not a baseline."
                "`global_importances` is an argument of the `ConSim.evaluate()` method, but it is None."
                "It can be computed via `concept_explainer.concept_output_gradient` then averaging for each class."
            )

        if prompt_type.value.lp_concepts_local_contributions and local_importances is None:
            raise ValueError(
                "Local concepts importances must be provided if prompt_type is E3 or U1. "
                "`local_importances` are computed via `concept_explainer.concept_output_gradient`. "
                "Consider using the `ConSim.evaluate()` method, it includes the computation of the local importances."
            )

        if prompt_type.value.ep_concepts_local_contributions and local_importances is None:
            raise ValueError(
                "Local concepts importances must be provided if prompt_type is U1. "
                "`local_importances` are computed via `concept_explainer.concept_output_gradient`. "
                "Consider using the `ConSim.evaluate()` method, it includes the computation of the local importances."
            )

        # filter and quantize the concepts importances
        if prompt_type in [PromptTypes.L1_baseline_without_lp, PromptTypes.L2_baseline_with_lp]:
            concepts_interpretation = None
            processed_global_importances = None
            processed_local_importances = None
        else:
            concepts_interpretation, processed_global_importances, processed_local_importances = (
                ConSim._filter_and_quantize_concepts_importances(
                    concepts_interpretation=concepts_interpretation,  # type: ignore
                    global_importances=global_importances,  # type: ignore
                    local_importances=local_importances,  # type: ignore
                    importance_threshold=importance_threshold,
                )
            )

        # integrate the different elements into a prompt
        system_prompt, user_prompt, literal_model_predictions = ConSim._setting_to_prompt(
            setting=prompt_type.value,
            anonymize_classes=anonymize_classes,
            sentences=sentences,
            predictions=predictions,
            classes=classes,
            concepts_interpretation=concepts_interpretation,
            global_importances=processed_global_importances,
            local_importances=processed_local_importances,
        )

        # convert the prompt to match the `LLMInterface` API
        prompt: list[tuple[Role, str]] = [
            (Role.SYSTEM, system_prompt),
            (Role.USER, user_prompt),
            (Role.ASSISTANT, ""),
        ]

        return prompt, literal_model_predictions

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
        concept_explainer: ConceptAutoEncoderExplainer,
        concepts_interpretation: dict[str, str],
        global_importances: dict[str, dict[str, float]],
        prompt_type: PromptTypes = PromptTypes.E3_global_and_local_concepts_with_lp,
        anonymize_classes: bool = False,
        importance_threshold: float = 0.05,
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

            concept_explainer: ConceptAutoEncoderExplainer
                The concept explainer.

            concepts_interpretation: dict[str, str]
                The words that activate the concepts the most and the least.
                A dictionary with the concepts as keys and another dictionary as values.
                The inner dictionary has the words as keys and the activations as values.

            global_importances: dict[str, dict[str, float]]
                The importance of the concepts for each class.
                A dictionary with the classes as keys and another dictionary as values.
                The inner dictionary has the concepts as keys and the importance as values.

            prompt_type: PromptTypes
                The type of prompt to use. Possible values are:

                - `PromptTypes.L1_baseline_without_lp`: baseline without learning phase.

                - `PromptTypes.E1_global_concepts_without_lp`: global concepts without learning phase.

                - `PromptTypes.L2_baseline_with_lp`: baseline with learning phase.

                - `PromptTypes.E2_global_concepts_with_lp`: global concepts with learning phase.

                - `PromptTypes.E3_global_and_local_concepts_with_lp`: global and local concepts with learning phase.

                - `PromptTypes.U1_upper_bound_concepts_at_ep`: upper bound - concepts at evaluation phase.

            anonymize_classes: bool
                Whether to anonymize the classes. Class names will be replaced by "Class_i" where i is the index of the class.
                It prevents the user-llm to solve the task by simply guessing the class.

            importance_threshold: float
                The threshold to select the most important concepts for each class.
                The threshold correspond to the cumulative importance of the concepts to keep.

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
        # TODO: precise shapes with jaxtyping
        if prompt_type in [
            PromptTypes.E3_global_and_local_concepts_with_lp,
            PromptTypes.U1_upper_bound_concepts_at_ep,
        ]:
            if prompt_type is PromptTypes.E3_global_and_local_concepts_with_lp:
                samples_to_explain = interesting_samples[: len(interesting_samples) // 2]
            else:
                samples_to_explain = interesting_samples
            local_importances_list = concept_explainer.concept_output_gradient(
                inputs=samples_to_explain,
                split_point=self.split_point,
                activation_granularity=self.activation_granularity,
                concepts_x_gradients=True,
                tqdm_bar=False,
            )
            local_importances: torch.Tensor | None = torch.stack(local_importances_list)
        else:
            local_importances = None

        # generate the prompt
        prompts, literal_model_predictions = ConSim._generate_prompt(
            sentences=interesting_samples,
            predictions=predictions,
            classes=self.classes,
            concepts_interpretation=concepts_interpretation,
            global_importances=global_importances,
            local_importances=local_importances,
            prompt_type=prompt_type,
            anonymize_classes=anonymize_classes,
            importance_threshold=importance_threshold,
        )

        # if no user_llm is provided, we return the prompts and the model predictions
        if self.user_llm is None:
            return prompts, literal_model_predictions

        user_llm_response = self.user_llm.generate(prompts)

        # raise warnings if the response is empty or the format is not respected
        return self._compute_score(
            user_llm_response=user_llm_response,
            literal_model_predictions=literal_model_predictions,
        )
