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

import torch
from tqdm import tqdm

from interpreto import ModelWithSplitPoints
from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.model_wrapping.llm_interface import LLMInterface, Role
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity
from interpreto.typing import ConceptsActivations, LatentActivations


class PromptSetting(Namedtuple):
    # global
    anonymous_classes: bool = False
    concepts_interpretation: bool = False
    concepts_global_importances: bool = False

    # learning phase
    lr_samples: bool = False
    lr_concepts_local_contributions: bool = False
    lr_labels: bool = False

    # inference
    inf_samples: bool = True
    inf_concepts_local_contributions: bool = False

    # prediction
    # pred_concepts: bool = False


def setting_to_prompt(
    setting: PromptSetting,
    sentences: list[str],
    predictions: list[float],
    classes: list[str],
    concepts_interpretation: dict[str, str],
    classes_concepts_importance: dict[str, dict[str, float]],
    samples_concepts_importances: list[dict[str, float]],
) -> str:
    """
    Create a prompt for the LLM model.
    It adapts to the setting to cover all possibilities

    Parameter
    ---------
    setting: PromptSetting
        Configuration, it says which elements should be included in the prompt.
    sentences: list[str]
        The sentences, the first half serve as examples and the second half is to be classified.
    predictions: list[float]
        The predictions of the model on the sentences.
    classes: list[str]
        The classes of the dataset.
    concepts_interpretation: dict[str, str]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: dict[str, dict[str, float]]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    samples_concepts_importances
        The importance of concepts for each sentence.
        A list with each element corresponding to one sentence.
        Each element of the list if a dictionary with an importance associated to a concept id.

    Returns
    -------
    prompt: str
        The prompt for the LLM.
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
    if setting.lr_samples and setting.lr_labels:
        if setting.lr_concepts_local_contributions:
            task_description_prompt += "You will have examples of samples, labels, and concepts contributions to labels as reference for the task. "
        else:
            task_description_prompt += "You will have examples of samples and labels as reference for the task. "
    if setting.inf_concepts_local_contributions:
        task_description_prompt += "At inference time, you will have concepts contributions to labels. "
    task_description_prompt += "Each sample class prediction should be in the format: 'Sample_{i}: {predicted_class}'."

    assert len(task_description_prompt) > 0
    system_prompt_parts.append(task_description_prompt)

    # -------
    # classes
    # if setting.pred_concepts:
    #     # show the concepts that could be predicted
    #     classes_prompt = f"The concepts are: [{', '.join(concepts_interpretation.keys())}]"
    if setting.anonymous_classes:
        # show the classes without their names
        anonym_classes = {class_name: f"Class_{i}" for i, class_name in enumerate(classes)}
        classes_prompt = f"The classes are: [{', '.join(anonym_classes.values())}]"
    else:
        # show the classes
        classes_prompt = f"The classes are: [{', '.join(classes)}]"
    system_prompt_parts.append(classes_prompt)

    # -------------------------
    # concepts activating words
    if setting.concepts_interpretation:
        # for each concept, show 10 words, 5 that aligns the most and 5 that are the most opposed
        concepts_interpretation_prompt = "For each concept, the most aligned and opposed words are:\n" + "\n".join(
            [
                f"{concept_id}: aligned: {list(words['aligned'].keys())}] opposed: {list(words['opposed'].keys())}"
                if len(words["opposed"])
                else f"{concept_id}: aligned: {list(words['aligned'].keys())}"
                for concept_id, words in concepts_interpretation.items()
            ]
        )
        system_prompt_parts.append(concepts_interpretation_prompt)

    # ---------------------------
    # classes concepts importance
    if setting.concepts_global_importances:
        # show the importance of the concepts for each class
        if setting.anonymous_classes:
            classes_concepts_prompt = (
                "The most important concepts and their importance for each class are:\n"
                + "\n".join(
                    [
                        f"{anonym_classes[class_name]}: {value}"
                        for class_name, value in classes_concepts_importance.items()
                    ]
                )
            )
        else:
            classes_concepts_prompt = (
                "The most important concepts and their importance for each class are:\n"
                + "\n".join([f"{key}: {value}" for key, value in classes_concepts_importance.items()])
            )
        system_prompt_parts.append(classes_concepts_prompt)

    # ==============================================================================================
    # Learning phase
    mid_index = len(sentences) // 2
    # -------
    # samples
    if setting.lr_samples:
        # show the samples
        lr_samples_prompt = "\n".join([f"Sample_{i}: {sentences[i]}" for i in range(mid_index)])
        system_prompt_parts.append(lr_samples_prompt)

    # ----------------------------
    # concepts local contributions
    if setting.lr_concepts_local_contributions:
        # show the concepts contributions to the samples
        lr_concepts_local_contributions_prompt = "\n".join(
            [f"Concepts contributions for Sample_{i}: {samples_concepts_importances[i]}" for i in range(mid_index)]
        )
        system_prompt_parts.append(lr_concepts_local_contributions_prompt)

    # ------
    # labels
    if setting.lr_labels:
        # show the labels
        if setting.anonymous_classes:
            lr_labels_prompt = "\n".join(
                [f"Sample_{i}: {anonym_classes[classes[predictions[i]]]}" for i in range(mid_index)]
            )
        else:
            lr_labels_prompt = "\n".join([f"Sample_{i}: {classes[predictions[i]]}" for i in range(mid_index)])
        system_prompt_parts.append(lr_labels_prompt)

    # ==============================================================================================
    # Inference
    # -------
    # samples
    if setting.inf_samples:
        # show the samples
        inf_samples_prompt = "\n".join([f"Sample_{i}: {sentences[i]}" for i in range(mid_index, 2 * mid_index)])
        user_prompt_parts.append(inf_samples_prompt)

    # ----------------------------
    # concepts local contributions
    if setting.inf_concepts_local_contributions:
        # show the concepts contributions to the samples
        inf_concepts_local_contributions_prompt = "\n".join(
            [
                f"Concepts contributions for Sample_{i}: {samples_concepts_importances[i]}"
                for i in range(mid_index, 2 * mid_index)
            ]
        )
        user_prompt_parts.append(inf_concepts_local_contributions_prompt)

    # concatenate prompts parts
    system_prompt = "\n\n".join(system_prompt_parts)
    user_prompt = "\n\n".join(user_prompt_parts)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return prompt


def quantize_importances(importance: float, threshold: float = 0.05) -> str:
    """
    Convert the normalized importances to literals.
    The literals are:
    - "++" for values above 0.3
    - "+" for values between 0.05 and 0.3
    - "-" for values between -0.05 and -0.3
    - "--" for values below -0.3

    Parameters
    ----------
    importance: float
        The importance to convert.
    threshold: float
        The threshold to select the most important concepts for each class.

    Returns
    -------
    literals: str
        The literals corresponding to the importances.
    """
    if importance <= -6 * threshold:
        return "--"

    if importance <= -threshold:
        return "-"

    if importance >= threshold:
        return "+"

    if importance >= 6 * threshold:
        return "++"

    raise ValueError(f"Quantization of importance {importance} failed.")


def filter_and_quantize_concepts_importances(
    concepts_interpretation: dict[str, str],
    classes_concepts_importance: dict[str, dict[str, float]],
    samples_concepts_importances: torch.Tensor,
    importance_threshold: float = 0.05,
) -> tuple[dict[str, str], dict[str, str], torch.Tensor]:
    """
    Filter the concepts importance and quantize the values.

    Parameters
    ----------
    concepts_interpretation: dict[str, str]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: dict[str, str]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    samples_concepts_importances: torch.Tensor
        Matrix of concept importances for each sentence. Shape (n_sentences, n_concepts)
    importance_threshold: float
        The threshold to select the most important concepts for each class.
        The threshold correspond to the cumulative importance of the concepts to keep.

    Returns
    -------
    concepts_interpretation: dict[str, str]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: dict[str, dict[str, float]]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    samples_concepts_importances: torch.Tensor
        Matrix of concept importances for each sentence. Shape (n_sentences, n_concepts)
    """

    # filter concepts which are important for at least one class
    concepts_to_keep = []
    while len(concepts_to_keep) == 0:
        for concepts_importance in classes_concepts_importance.values():
            if len(concepts_importance) == 0:
                continue

            # normalize the importances
            importances = np.abs(np.array(list(concepts_importance.values())))
            normalized_importances = importances / importances.sum()

            # select the important concepts
            added_concepts = np.where(normalized_importances > importance_threshold)[0]
            concepts_to_keep.extend(added_concepts)
        if len(concepts_to_keep) == 0:
            importance_threshold /= 2

    concepts_to_show = np.unique(concepts_to_keep)
    interpretation_concepts_ids = np.array(
        [int(cpt.split("_")[-1]) for cpt in concepts_interpretation.keys()], dtype=int
    )
    concepts_to_show = np.intersect1d(concepts_to_show, interpretation_concepts_ids)

    # filter the concepts activating words
    concepts_interpretation = {f"concept_{c}": concepts_interpretation[f"concept_{c}"] for c in concepts_to_show}

    # filter the concepts importance
    classes_concepts_importance = {
        class_name: {
            c: quantize_importances(importance, importance_threshold)
            for c, importance in concepts_importance.items()
            if int(c.split("_")[-1]) in concepts_to_show
            and quantize_importances(importance, importance_threshold) is not None
        }
        for class_name, concepts_importance in classes_concepts_importance.items()
    }

    # normalize sentences concepts importances
    samples_concepts_importances = (
        samples_concepts_importances / np.abs(samples_concepts_importances).sum(axis=1)[:, np.newaxis]
    )

    # clean elements to leave only the important concepts and quantize values to literals
    filtered_samples_concepts_contributions = [
        {
            f"concept_{c}": quantize_importances(importance)
            for c, importance in enumerate(sentence_concepts_importances)
            if c in concepts_to_show and quantize_importances(importance) is not None
        }
        for sentence_concepts_importances in samples_concepts_importances
    ]

    return concepts_interpretation, classes_concepts_importance, filtered_samples_concepts_contributions


def make_prompts(
    sentences: list[str],
    predictions: list[float],
    classes: list[str],
    concepts_interpretation: dict[str, str],
    classes_concepts_importance: dict[str, dict[str, float]],
    samples_concepts_importances: torch.Tensor,
    importance_threshold: float = 0.05,
) -> dict[str, str]:
    """
    Create prompts for the LLM model.
    There are three types of prompts:
    - without_explanation: The LLM has to predict the model's prediction for the next sentences.
    - with_concepts_explanations: The LLM has to predict the model's prediction for the next sentences.
    - with_concepts_explanations_and_predictions:
    The LLM has to predict the activations of the concepts and the model's prediction for the next sentences.

    Parameters
    ----------
    sentences: list[str]
        The sentences, the first half serve as examples and the second half is to be classified.
    predictions: list[float]
        The predictions of the model on the sentences.
    classes: list[str]
        The classes of the dataset.
    concepts_interpretation: dict[str, dict[str, float]]
        The words that activate the concepts the most and the least.
        A dictionary with the concepts as keys and another dictionary as values.
        The inner dictionary has the words as keys and the activations as values.
    classes_concepts_importance: dict[str, dict[str, float]]
        The importance of the concepts for each class.
        A dictionary with the classes as keys and another dictionary as values.
        The inner dictionary has the concepts as keys and the importance as values.
    samples_concepts_importances: torch.Tensor
        Matrix of concept importances for each sentence. Shape (n_sentences, n_concepts)
    importance_threshold: float
        The threshold to select the most important concepts for each class.
        The threshold correspond to the cumulative importance of the concepts to keep.

    Returns
    -------
    prompts_and_inputs: dict[str, str]
        The prompts for the LLM, the inputs and expected outputs.
    """

    # filter and quantize the concepts importances
    concepts_interpretation, classes_concepts_importance, filtered_samples_concepts_contributions = (
        filter_and_quantize_concepts_importances(
            concepts_interpretation,
            classes_concepts_importance,
            samples_concepts_importances,
            importance_threshold,
        )
    )

    # regroup prompting kwargs
    kwargs = {
        "sentences": sentences,
        "predictions": predictions,
        "classes": classes,
        "concepts_interpretation": concepts_interpretation,
        "classes_concepts_importance": classes_concepts_importance,
        "samples_concepts_importances": filtered_samples_concepts_contributions,
    }

    # list settings
    experiments_settings = {
        # inputs to outputs
        "L1: no LR baseline": PromptSetting(),
        "E1: concepts without LR": PromptSetting(concepts_interpretation=True, concepts_global_importances=True),
        "L2: with LR baseline": PromptSetting(lr_samples=True, lr_labels=True),
        "E2: concepts with LR": PromptSetting(
            concepts_interpretation=True, concepts_global_importances=True, lr_samples=True, lr_labels=True
        ),
        "E3: concepts with contributions at LR": PromptSetting(
            concepts_interpretation=True,
            concepts_global_importances=True,
            lr_samples=True,
            lr_concepts_local_contributions=True,
            lr_labels=True,
        ),
        # inputs and concepts to outputs
        "U1: concepts with contributions at LR and inf": PromptSetting(
            concepts_interpretation=True,
            concepts_global_importances=True,
            lr_samples=True,
            lr_concepts_local_contributions=True,
            lr_labels=True,
            inf_concepts_local_contributions=True,
        ),
    }

    experiments_settings.update(
        {
            "-a:".join(xp_name.split(":")): PromptSetting(**setting._asdict())._replace(anonymous_classes=True)
            for xp_name, setting in experiments_settings.items()
        }
    )

    # create the prompt for each setting
    prompts = {xp_name: setting_to_prompt(setting, **kwargs) for xp_name, setting in experiments_settings.items()}

    return prompts


class ConSim:
    """Code: [:octicons-mark-github-24: `concepts/metrics/con_sim.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/metrics/con_sim.py)

    ConSim for Concept-based Simulatability. Was introduced by Poché et al. in 2025 [^1].

    It evaluates all three components of the concept-based explanation:

    - the concepts space

    - the concepts interpretation

    - the concepts importance

    To evaluate explanations on a given model $f$, ConSim evaluates to which extent explanations
    help a meta-predictor $\Psi$ to simulate the predictions of the model $f$.

    In our case, the role of the meta-predictor will be played by `judge_llm`, and interface calling
    a model either from local, or from a remote API, such as OpenAI or HuggingFace.
    Therefore, most of the code correspond to building the prompts for the LLM.

    The prompts have five parts:

    - the first part is the task description, which is a list of questions to ask the LLM.

    - the second is a global concepts explanation on $f$. Listing the important concepts for each class.

    - the third gives examples of samples and predictions from the model $f$.

    - the fourth is a local concepts explanation on $f$. Listing the important concepts in each example.

    - the fifth is a list of samples on which the meta-predictor $\Psi$ will be asked to predict the model $f$ predictions.

    The answer of the LLM will be a list of predictions for each sample. ConSim compares these predictions to the
    model $f$ predictions and computes the accuracy of the explanations.

    Attributes
    ----------
    model_with_split_points: ModelWithSplitPoints
        The model to explain. Is is a wrapper around a model and a tokenizer to easily get activations.
    judge_llm: LLMInterface
        The LLM interface that will serve as the meta-predictor.
    split_point: str
        Where to split the model to explain.
    activation_granularity: ActivationGranularity
        The granularity of the activations to use for the explanations.

    References
    ----------
    [^1]:
        A. Poch{\'e}, A. Jacovi, A.M. Picard, V. Boutin, and F. Jourdan.
        ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability.
        In the Proceedings of the 2025 Association for Computational Linguistics (ACL).

    Examples
    --------

    >>> import datasets
    >>> from interpreto import ConSim, ModelWithSplitPoints, ICAConcepts, OpenAILLM
    >>>
    >>> # Load a model and wrap it
    >>> model_with_split_points = ModelWithSplitPoints(
    ...     "textattack/bert-base-uncased-ag-news",
    ...     split_points=["bert.encoder.layer.10.output"],
    ...     model_autoclass=AutoModelForSequenceClassification,  # type: ignore
    ...     batch_size=4,
    ... )
    >>>
    >>> # Load a dataset and compute activations
    >>> dataset = datasets.load_dataset("fancyzhx/ag_news")
    >>> activations = model_with_split_points.get_activations(dataset["train"]["text"])
    >>>
    >>> # Fit the concept explainer
    >>> concept_explainer_1 = ICAConcepts(model_with_split_points, nb_concepts=50)
    >>> concept_explainer.fit(activations)
    >>>
    >>> # Define the judge-LLM
    >>> judge_llm = OpenAILLM(api_key="YOUR_OPENAI_API_KEY", model="gpt4o-mini")
    >>>
    >>> con_sim = ConSim(model_with_split_points, judge_llm)
    >>> samples, labels, predictions = con_sim.select_examples(
    ...     dataset["train"]["text"], dataset["train"]["label"], nb_classes=4
    ... )
    >>> score_1 = con_sim.evaluate(samples, labels, predictions, concept_explainer_1)
    >>> score_2 = con_sim.evaluate(samples, labels, predictions, concept_explainer_2)
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        judge_llm: LLMInterface,
        split_point: str | None = None,
        activation_granularity: ActivationGranularity = ActivationGranularity.TOKEN,
    ):
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

    def get_predictions(
        self, inputs: list[str], batch_size: int = 64, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        all_predictions = []
        for batch_index in tqdm(
            range(0, len(inputs), batch_size),
            desc="Computing predictions",
            unit="batch",
            total=len(inputs),
            disable=not tqdm_bar,
        ):
            batch_inputs = inputs[batch_index : batch_index + batch_size]
            batch_tokens = self.model_with_split_points.tokenizer(batch_inputs, return_tensors="pt").to(device)
            logits = self.model_with_split_points.model(batch_tokens["input_ids"], batch_tokens["attention_mask"])
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(predictions)
        return torch.cat(all_predictions)

    def extract_interesting_elements(
        self,
        inputs: list[str],
        labels: torch.Tensor,
        predictions: torch.Tensor,
        nb_classes: int,
        nb_lp_samples: int = 20,
        nb_ep_samples: int = 20,
        seed: int = 0,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        nb_correct = (nb_lp_samples + nb_ep_samples) // 2
        nb_mistakes = nb_lp_samples + nb_ep_samples - nb_correct

        # Find the correct and incorrect indices
        is_prediction_correct = predictions == labels
        correct_indices = torch.nonzero(is_prediction_correct)
        incorrect_indices = torch.nonzero(~is_prediction_correct)
        del is_prediction_correct

        if len(correct_indices) < nb_correct or len(incorrect_indices) < nb_mistakes:
            raise ValueError(
                f"Not enough correct or incorrect predictions to select {nb_correct} correct and {nb_mistakes} incorrect."
                f"Either provide more inputs (actually {len(correct_indices)} correct and {len(incorrect_indices)} incorrect)"
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
        labels: list[int],
        nb_classes: int,
        nb_lp_samples: int = 20,
        nb_ep_samples: int = 20,
        seed: int = 0,
        batch_size: int = 64,
        device: torch.device | str = "cpu",
    ) -> tuple[list[str], list[int], list[int]]:
        predictions = self.get_predictions(inputs, batch_size=batch_size, device=device)
        return self.extract_interesting_elements(
            inputs,
            labels,
            predictions,
            nb_classes,
            nb_lp_samples,
            nb_ep_samples,
            seed,
        )

    def compute_elements(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
        interesting_samples: list[str],
        interesting_indices: list[int] | None = None,
    ) -> tuple[LatentActivations, ConceptsActivations, torch.Tensor]:
        # compute latent activations
        latent_activations_dict: dict[str, LatentActivations] = self.model_with_split_points.get_activations(
            interesting_samples,
            activation_granularity=self.activation_granularity,
        )
        latent_activations = self.model_with_split_points.get_split_activations(
            latent_activations_dict, split_point=self.split_point
        )

        # compute concepts activations
        concepts_activations: ConceptsActivations = concept_explainer.encode_activations(latent_activations)

        # compute concepts importance  # TODO: when first layers can be skipped pass the concept activations
        concepts_importance: torch.Tensor = concept_explainer.concept_output_gradient(
            inputs=interesting_samples,
            targets=interesting_indices,
            split_point=self.split_point,
            activation_granularity=self.activation_granularity,
            concepts_x_gradients=True,
            tqdm_bar=False,
        )
        return latent_activations, concepts_activations, concepts_importance

    def generate_prompts(
        self,
        interesting_samples: list[str],
        labels: list[int],
        predictions: list[int],
    ) -> dict[str, str]: ...  # TODO

    def evaluate(
        self,
        interesting_samples: list[str],
        labels: list[int],
        predictions: list[int],
        concept_explainer: ConceptAutoEncoderExplainer,
    ) -> float:
        concept_activations, concepts_activations, concepts_importance = self.compute_elements(
            concept_explainer, interesting_samples
        )

        prompts = self.generate_prompts(
            interesting_samples=interesting_samples,
            labels=labels,
            predictions=predictions,
        )

        meta_predictions = self.judge_llm.generate(prompts)

        return self.compute_score(meta_predictions, predictions, concept_activations, concepts_activations)
