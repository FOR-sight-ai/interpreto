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
This file tests the ConSim metric.

The ConSim metric has many methods, most of them will be tested one by one:
    ConSim.__init__
    ConSim.select_examples
    ConSim.quantize_importances
    ConSim._concepts_to_string
    ConSim._setting_to_prompt
    ConSim.construct_prompt
    AutomatedSimulatability.score_from_responses

In the unit tests listed above some configurations will be common:
- the `ModelWithSplitPoints` will be used around a Bert model,
- the `ConceptAutoEncoderExplainers` will be a `NeuronAsConcepts` explainer,
- the `LLMInterface` will be replaced by a place holder predicting the classes specified randomly,

Then the ConSim metric will be tested with different `ConceptAutoEncoderExplainers`.

Finally, an end to end test will include a call to the `OpenAILLM` if an API key is available.
"""

from __future__ import annotations

import importlib.util
import os
from abc import abstractmethod

import pytest
import torch

from interpreto import ModelWithSplitPoints
from interpreto.concepts.metrics import ConSim
from interpreto.concepts.metrics.simulatability.base import AutomatedSimulatability
from interpreto.model_wrapping.llm_interface import LLMInterface

PromptTypes = ConSim.prompt_types
AG = ModelWithSplitPoints.activation_granularities

PROMPT_TYPES = [
    PromptTypes.L1_baseline_without_lp,
    PromptTypes.E1_global_concepts_without_lp,
    PromptTypes.L2_baseline_with_lp,
    PromptTypes.E2_global_concepts_with_lp,
    PromptTypes.E3_global_and_local_concepts_with_lp,
    PromptTypes.C1_contrastive_global_concepts_without_lp,
    PromptTypes.C2_contrastive_global_concepts_with_lp,
    PromptTypes.C3_contrastive_global_and_local_concepts_with_lp,
    PromptTypes.C4_contrastive_local_concepts,
    PromptTypes.C5_contrastive_local_only,
]


class LLMInterfaceParent(LLMInterface):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        pass

    def batch_generate(self, system_prompt: str, user_prompts: list[str], **generation_kwargs) -> list[str | None]:
        return [self.generate(system_prompt, p, **generation_kwargs) for p in user_prompts]


class LLMInterfacePlaceholder(LLMInterfaceParent):
    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        # extract the classes from the system prompt
        classes_str = system_prompt.split("The classes are: [")[1].split("]", maxsplit=1)[0]
        classes = classes_str.split(", ")

        # extract the sample indices from the user prompt
        ep_samples_str = user_prompt.split("\n\nConcepts contributions for Sample_", maxsplit=1)[0]
        # Format:
        #     Sample_0: "this is the first sample"
        #     Sample_1: "this is the second sample"
        ep_sample_indices = [int(s.split(":")[0][7:]) for s in ep_samples_str.split("\n") if s]

        # generate the response, give a random class for each sample
        response = "\n".join([f"Sample_{i}: {classes[i % len(classes)]}" for i in ep_sample_indices])
        return response


class WrongNumberOfAnswers(LLMInterfaceParent):
    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        # extract the classes from the system prompt
        classes_str = system_prompt.split("The classes are: [")[1].split("]", maxsplit=1)[0]
        classes = classes_str.split(", ")

        # extract the sample indices from the user prompt
        ep_samples_str = user_prompt.split("\n\nConcepts contributions for Sample_", maxsplit=1)[0]
        # Format:
        #     Sample_0: "this is the first sample"
        #     Sample_1: "this is the second sample"
        ep_sample_indices = [int(s.split(":")[0][7:]) for s in ep_samples_str.split("\n") if s]

        # remove the last sample to have a wrong number of answers
        ep_sample_indices = ep_sample_indices[:-1]

        # generate the response, give a random class for each sample
        response = "\n".join([f"Sample_{i}: {classes[i % len(classes)]}" for i in ep_sample_indices])
        return response


class WrongFormat(LLMInterfaceParent):
    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        return "The predictions are: {class_0, class_1, class_2}"


class EmptyResponse(LLMInterfaceParent):
    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        return ""


def test_consim_select_examples():
    """
    Test the `select_examples` method of the ConSim metric.
    """
    classes = ["0", "1"]
    consim = ConSim(classes=classes)

    inputs = [f"sentence {i}" for i in range(6)]
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    predictions = torch.tensor([0, 0, 0, 1, 1, 1])

    # 2 correct and 2 incorrect elements should be returned
    indices, samples, labels, predictions = consim.select_examples(inputs, labels, predictions, nb_samples=4, seed=0)

    # ensure samples, labels, and predictions all have 4 elements
    assert len(indices) == 4, "number of indices from `consim.select_examples` should match nb_samples"
    assert len(samples) == 4, "number of samples from `consim.select_examples` should match nb_samples"
    assert labels.shape == (4,), "number of labels from `consim.select_examples` should match nb_samples"
    assert predictions.shape == (4,), "number of predictions from `consim.select_examples` should match nb_samples"

    # ensure exactly half of the predictions match the labels
    assert torch.sum(labels == predictions) == 2, "exactly half of the predictions should match the labels"

    # ensure each class represents half of the predictions and labels
    assert torch.sum(labels == 0) == 2, "exactly half of the labels should be of each class"
    assert torch.sum(predictions == 0) == 2, "exactly half of the predictions should be of each class"

    # not enough correct predictions should raise a value error
    with pytest.raises(ValueError):
        consim.select_examples(inputs[:2], labels[:2], predictions[:2], nb_samples=4)


def test_consim_select_examples_subset_classes():
    """
    Test the `select_examples` method with a subset of classes.
    """
    classes_names = ["A", "B", "C"]
    classes_subset = [0, 2]
    consim = ConSim(classes=classes_names)

    inputs = [f"sample {i}" for i in range(9)]
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    predictions = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    indices, samples, selected_labels, selected_predictions = consim.select_examples(
        inputs=inputs,
        labels=labels,
        predictions=predictions,
        nb_samples=4,
        seed=0,
        classes_subset=classes_subset,
    )

    assert len(indices) == 4, "number of indices should match nb_samples"
    assert len(samples) == 4, "number of samples should match nb_samples"
    assert selected_labels.shape == (4,), "number of labels should match nb_samples"
    assert selected_predictions.shape == (4,), "number of predictions should match nb_samples"

    assert all(int(label) in classes_subset for label in selected_labels), "labels should be restricted to the subset"
    assert all(int(pred) in classes_subset for pred in selected_predictions), (
        "predictions should be restricted to the subset"
    )


def test_consim_quantize_importances():
    """
    Test the `quantize_importances` method of the ConSim metric.
    """
    thr = 0.05
    assert ConSim._quantize_importances(0.4, thr) == "Highly supportive", (
        "quantize_importances should return Highly supportive for values > 0.3"
    )
    assert ConSim._quantize_importances(0.1, thr) == "Supportive", (
        "quantize_importances should return Supportive for values between 0.05 and 0.3"
    )
    assert ConSim._quantize_importances(-0.1, thr) == "Opposed", (
        "quantize_importances should return - for values between -0.3 and -0.05"
    )
    assert ConSim._quantize_importances(-0.5, thr) == "Very opposed", (
        "quantize_importances should return Very opposed for values < -0.3"
    )
    assert ConSim._quantize_importances(0.04, thr) is None, (
        "quantize_importances should return None for values between -0.05 and 0.05"
    )
    assert ConSim._quantize_importances(-0.04, thr) is None, (
        "quantize_importances should return None for values between -0.05 and 0.05"
    )


def test_consim_quantize_concepts_importances():
    """
    Test the `_concepts_to_string` method of the ConSim metric.
    """
    concepts_interpretation = {
        0: "w0",
        1: "w1",
        2: "w2",
        3: "w3",
    }
    importances = torch.tensor([0.5, -0.06, 0.02, -0.4])

    rendered = ConSim._concepts_to_string(
        importances=importances,
        concepts_interpretation=concepts_interpretation,
        top_k=3,
        threshold=0.05,
    )

    assert rendered == "{C0 (w0): Highly supportive, C3 (w3): Very opposed, C1 (w1): Opposed}", (
        "wrong concept string rendering"
    )


def test_score_from_responses():
    """
    Test the `score_from_responses` helper on precomputed responses.
    """
    responses = ["A trailing text", "b", "wrong", None]
    model_predictions = ["A", "B", "C", "D"]

    score = AutomatedSimulatability.score_from_responses(responses, model_predictions)

    assert score == 0.5, "two responses should match exactly after normalization"


def test_score_from_responses_length_mismatch():
    """
    Test `score_from_responses` length validation.
    """
    with pytest.raises(ValueError):
        AutomatedSimulatability.score_from_responses(["A"], ["A", "B"])


@pytest.mark.parametrize("prompt_type", PROMPT_TYPES)
@pytest.mark.parametrize("anonymize_classes", [False, True])
def test_consim_setting_to_prompt(prompt_type: PromptTypes, anonymize_classes: bool):  # noqa: PLR0912  # ignore too many branches  # too many special cases
    """
    Test the `_setting_to_prompt` method of the ConSim metric.
    """
    # prepare fake method so we only test the logic of setting_to_prompt
    sentences = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 1, 0])
    labels = torch.tensor([0, 0, 1, 1])
    classes = ["A", "B"]
    interp = {0: "word"}
    glob = torch.tensor([[0.6], [-0.2]])  # {"A": {0: "Highly supportive"}, "B": {0: "Opposed"}}
    loc = [  # [{0: "Supportive"}, {0: "Opposed"}, {0: "Supportive"}, {0: "Opposed"}]
        torch.tensor([[0.2], [-0.2]]),
        torch.tensor([[0.2], [-0.1]]),
        torch.tensor([[-0.1], [0.2]]),
        torch.tensor([[-0.05], [0.1]]),
    ]
    contrastive_pairs = [(1, 0)]

    # convert the prompt type to settings
    prompt_settings = prompt_type.value._replace(anonymize_classes=anonymize_classes)

    # ==============================================================================================
    # Verify the settings based on the prompt type
    prompt_type_name = str(prompt_type)
    if "baseline" in prompt_type_name:
        assert prompt_settings.concepts_global_importances is False, "wrong prompt settings for baseline"
        assert prompt_settings.lp_concepts_local_contributions is False, "wrong prompt settings for baseline"
    if "without_lp" in prompt_type_name:
        assert prompt_settings.lp_samples is False, "wrong prompt settings for without_lp"
        assert prompt_settings.lp_concepts_local_contributions is False, "wrong prompt settings for without_lp"
    if "with_lp" in prompt_type_name:
        assert prompt_settings.lp_samples, "wrong prompt settings for with_lp"
    if "global" in prompt_type_name and "contrastive" not in prompt_type_name:
        assert prompt_settings.concepts_global_importances, "wrong prompt settings for global"
    if "local" in prompt_type_name and "contrastive" not in prompt_type_name:
        assert prompt_settings.lp_concepts_local_contributions, "wrong prompt settings for local"
    if "contrastive" in prompt_type_name:
        if "global" in prompt_type_name:
            assert prompt_settings.global_contrastive_importances, "wrong prompt settings for contrastive global"
        if "local" in prompt_type_name:
            assert prompt_settings.lp_samples, "wrong prompt settings for contrastive local"
            assert prompt_settings.lp_local_contrastive_importance, "wrong prompt settings for contrastive local"

    # convert the settings to prompt
    system, user, literal = ConSim._setting_to_prompt(
        setting=prompt_settings,
        interesting_samples=sentences,
        corresponding_predictions=preds,
        corresponding_labels=labels,
        nb_learning_samples=2,
        classes={i: classes[i] for i in range(len(classes))},
        concepts_interpretation=interp,
        global_importances=dict(enumerate(glob)),
        local_importances=loc,
        contrastive_pairs=contrastive_pairs,
    )

    # ==============================================================================================
    # Verify the generated prompts elements

    # -------------
    # Initial Phase
    # task description
    assert "You are a classifier." in system, "system prompt should contain the task description"
    assert (
        "User's prompt will contain an evaluation sample on which you should predict the class. Only return the class name, no other text."
        in system
    ), "system prompt should contain the expected response format"
    if anonymize_classes:
        assert "The classes are: [Class_0, Class_1]" in system, "system prompt should contain the anonymized classes"
    else:
        assert "The classes are: [A, B]" in system, "system prompt should contain the classes"

    # global concepts explanation
    if prompt_settings.concepts_global_importances:
        if anonymize_classes:
            assert "Class_0: {C0 (word): Highly supportive}" in system, (
                "system prompt should contain the anonymized global concepts importance"
            )
        else:
            assert "B: {C0 (word): Opposed}" in system, "system prompt should contain the global concepts importance"

    # contrastive global explanation
    if prompt_settings.global_contrastive_importances:
        if anonymize_classes:
            assert "\tfact: Class_1, foil: Class_0: {C0 (word): Very opposed}" in system, (
                "system prompt should contain the anonymized global contrastive concepts importance"
            )
        else:
            assert "\tfact: B, foil: A: {C0 (word): Very opposed}" in system, (
                "system prompt should contain the global contrastive concepts importance"
            )

    # --------------
    # Learning Phase
    # examples
    if prompt_settings.lp_samples:
        # samples, only the 2 first samples for lp
        assert "Sample_0:\n\tText: s0" in system, "system prompt should contain the lp samples"
        assert "Sample_1:\n\tText: s1" in system, "system prompt should contain the lp samples"
        assert "Sample_2" not in system and "Sample_3" not in system, "system prompt should not contain the ep samples"
        # labels
        if anonymize_classes:
            assert "\n\tLabel: Class_0" in system, "system prompt should contain the anonymized predictions"
            assert "\n\tLabel: Class_1" in system, "system prompt should contain the anonymized predictions"
        else:
            assert "\n\tLabel: A" in system, "system prompt should contain the anonymized predictions"
            assert "\n\tLabel: B" in system, "system prompt should contain the anonymized predictions"

    # local concepts explanation
    if prompt_settings.lp_concepts_local_contributions:
        assert "\n\tConcepts contributions: {C0 (word): Supportive}" in system, (
            "system prompt should contain the lp concepts contributions"
        )
        assert "\n\tConcepts contributions: {C0 (word): Opposed}" in system, (
            "system prompt should contain the lp concepts contributions"
        )

    # contrastive local explanation
    if prompt_settings.lp_local_contrastive_importance:
        assert "\n\tConcepts contributions: {C0 (word): Supportive}" in system, (
            "system prompt should contain the lp concepts contributions"
        )
        if anonymize_classes:
            assert (
                "\n\tConcepts contributions supporting Class_1 rather than Class_0: {C0 (word): Very opposed}"
                in system
            ), "system prompt should contain the anonymized contrastive local concepts importance"
        else:
            assert "\n\tConcepts contributions supporting B rather than A: {C0 (word): Very opposed}" in system, (
                "system prompt should contain the anonymized contrastive local concepts importance"
            )
    # ----------------
    # Evaluation Phase
    # samples, only the 2 last samples for ep
    assert user[0] == "Evaluation sample:\n\tText: s2\n\tLabel: ", "user prompt should contain the ep samples"
    assert user[1] == "Evaluation sample:\n\tText: s3\n\tLabel: ", "user prompt should contain the ep samples"


def test_consim_generate_prompt():
    """
    Test the `_generate_prompt` method of the ConSim metric.
    """
    sentences = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 0, 1])
    labels = torch.tensor([0, 0, 1, 1])
    nb_samples = len(sentences)
    nb_learning_samples = 2
    classes = ["A", "B"]
    interp = {0: "word", 1: "test"}
    glob = torch.tensor([[0.6, 0.1], [-0.6, -0.2]])
    loc = [
        torch.tensor([[0.4, 0.1], [0.1, -0.2]]),
        torch.tensor([[0.1, 0.4], [-0.4, -0.1]]),
        torch.tensor([[0.5, 0.01], [0.2, -0.02]]),
        torch.tensor([[-0.2, 0.02], [-0.5, -0.01]]),
    ]

    system, user, literal = ConSim._setting_to_prompt(
        setting=PromptTypes.E3_global_and_local_concepts_with_lp.value,
        interesting_samples=sentences,
        corresponding_predictions=preds,
        corresponding_labels=labels,
        nb_learning_samples=nb_learning_samples,
        classes={i: classes[i] for i in range(len(classes))},
        concepts_interpretation=interp,
        global_importances=dict(enumerate(glob)),
        local_importances=loc,
    )

    # test prompt format
    assert isinstance(system, str), "system prompt should be a string"
    assert isinstance(user, list) and all(isinstance(s, str) for s in user), "user prompt should be a list of strings"
    assert isinstance(literal, list) and all(isinstance(s, str) for s in literal), (
        "literal predictions should be a list of strings"
    )

    # verify lengths
    assert len(user) == nb_samples - nb_learning_samples, "user prompt should have the correct length"
    assert len(literal) == nb_samples - nb_learning_samples, "literal predictions should have the correct length"

    # verify literal predictions
    assert literal == ["A", "B"], "literal predictions should match the expected"

    # global importance should have been converted to:
    # glob = {"A": {0: 0.6}, "B": {0: -0.6}}
    # with the concept_1 being removed
    # assert 1 not in system, "concepts with low global importance should be removed from the system prompt"
    assert "C0 (word)" in system, (
        "high global importance concepts should be in the system prompt"
    )  # E3 includes the concepts interpretation
    assert "A: {C0 (word): Highly supportive, C1 (test): Supportive}" in system, (
        "high global importance concepts should be in the system prompt"
    )  # E3 includes the global concepts importance
    assert "B: {C0 (word): Very opposed, C1 (test): Opposed}" in system, (
        "high global importance concepts should be in the system prompt"
    )  # E3 includes the global concepts importance
    assert "Sample_0:\n\tText: s0" in system, "system prompt should contain the lp samples"
    assert "Sample_1:\n\tText: s1" in system, "system prompt should contain the lp samples"
    assert "Sample_2" not in system and "Sample_3" not in system, "system prompt should not contain the ep samples"
    assert "\n\tLabel: A" in system, "system prompt should contain the anonymized predictions"
    assert "\n\tLabel: B" in system, "system prompt should contain the anonymized predictions"
    assert "\n\tConcepts contributions: {C0 (word): Highly supportive, C1 (test): Supportive}" in system, (
        "system prompt should contain the lp concepts contributions"
    )
    assert "\n\tConcepts contributions: {C0 (word): Very opposed, C1 (test): Opposed}" in system, (
        "system prompt should contain the lp concepts contributions"
    )
    assert user[0] == "Evaluation sample:\n\tText: s2\n\tLabel: ", "user prompt should contain the ep samples"
    assert user[1] == "Evaluation sample:\n\tText: s3\n\tLabel: ", "user prompt should contain the ep samples"


# TODO: outdated tests, might be useful if we do a consim manager
# @pytest.mark.parametrize("prompt_type", PROMPT_TYPES)
# def test_consim_evaluate(splitted_encoder_ml: ModelWithSplitPoints, prompt_type: PromptTypes):
#     """
#     Test the `evaluate` method of the ConSim metric.

#     Parameters
#     ----------
#     splitted_encoder_ml: ModelWithSplitPoints
#         The model to explain. Is is a wrapper around a model and a tokenizer to easily get activations.
#         Here a Bert model, but this is not used in this test apart from initializing the ConSim metric.
#     llm_placeholder: LLMInterface
#         The LLM interface that will serve as the meta-predictor.
#         It randomly predicts the classes specified in the prompt.
#     """
#     classes = ["A", "B", "C", "D"]
#     samples = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
#     preds = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
#     labels = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3]
#     contrastive_pairs = [(0, 1), (2, 3)]

#     # creating a dummy explainer that will return a gradient of ones
#     class DummyExplainer(ConceptAutoEncoderExplainer):
#         fitted = True
#         _split_point = splitted_encoder_ml.split_points[0]

#         def __init__(self, model_with_split_points: ModelWithSplitPoints):  # type: ignore
#             self.model_with_split_points = model_with_split_points

#         def concept_output_gradient(self, inputs, *args, **kwargs):
#             """
#             consim.evaluate calls this method
#             """
#             # Therefore we ensure that it is called only when necessary.
#             assert prompt_type in [
#                 PromptTypes.E3_global_and_local_concepts_with_lp,
#                 PromptTypes.C3_contrastive_global_and_local_concepts_with_lp,
#                 PromptTypes.C4_contrastive_local_concepts,
#                 PromptTypes.C5_contrastive_local_only,
#             ]
#             # Furthermore, we ensure it is called only with the necessary elements.
#             assert inputs == ["s0", "s1", "s2", "s3", "s4"]

#             return [torch.ones(len(classes), 1, 1)] * len(inputs)

#         def fit(self, *args, **kwargs):
#             pass

#     explainer = DummyExplainer(splitted_encoder_ml)

#     # -----------------------------------
#     # Test consim with a valid LLM output

#     llm = LLMInterfacePlaceholder()
#     consim = ConSim(splitted_encoder_ml, user_llm=llm, activation_granularity=AG.TOKEN, classes=classes)

#     # evaluate the ConSim metric
#     score: float | None = consim.evaluate(  # type: ignore
#         interesting_samples=samples,
#         predictions=preds,
#         labels=labels,
#         concept_explainer=explainer,
#         concepts_interpretation={0: "word"},
#         global_importances=torch.ones(len(classes), 1),
#         prompt_type=prompt_type,
#         contrastive_pairs=contrastive_pairs,
#     )

#     # None is allowed in the typing, but it should not happen in this case
#     # because the llm placeholder always predicts in the expected format
#     assert score is not None, "consim should not return None with a valid llm response"
#     assert 0.0 <= score <= 1.0, "consim score should be between 0 and 1"

#     # ---------------------------------
#     # Test weird LLM outputs management

#     # empty response should return None
#     consim.user_llm = EmptyResponse()
#     score: float | None = consim.evaluate(  # type: ignore
#         interesting_samples=samples,
#         predictions=preds,
#         labels=labels,
#         concept_explainer=explainer,
#         concepts_interpretation={0: "word"},
#         global_importances=torch.ones(len(classes), 1),
#         prompt_type=prompt_type,
#         contrastive_pairs=contrastive_pairs,
#     )
#     assert score is None, "consim should return None on empty llm response"

#     # wrong format should return None
#     consim.user_llm = WrongFormat()
#     score: float | None = consim.evaluate(  # type: ignore
#         interesting_samples=samples,
#         predictions=preds,
#         labels=labels,
#         concept_explainer=explainer,
#         concepts_interpretation={0: "word"},
#         global_importances=torch.ones(len(classes), 1),
#         prompt_type=prompt_type,
#         contrastive_pairs=contrastive_pairs,
#     )
#     assert score is None, "consim should return None on wrong format llm response"

#     # wrong number of answers should return None
#     consim.user_llm = WrongNumberOfAnswers()
#     score: float | None = consim.evaluate(  # type: ignore
#         interesting_samples=samples,
#         predictions=preds,
#         labels=labels,
#         concept_explainer=explainer,
#         concepts_interpretation={0: "word"},
#         global_importances=torch.ones(len(classes), 1),
#         prompt_type=prompt_type,
#         contrastive_pairs=contrastive_pairs,
#     )
#     assert score is None, "consim should return None on wrong number of answers in llm response"


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None or importlib.util.find_spec("openai") is None,
    reason="No OpenAI API key available.",
)
@pytest.mark.slow
def test_consim_evaluate_with_openai(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `evaluate` method of the ConSim metric with OpenAI API.
    """
    # lazy import to avoid importing openai
    from interpreto.model_wrapping.llm_interface import (  # noqa: PLC0415  # ruff: disable=import-outside-toplevel
        OpenAILLM,
    )


#     open_ai_llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-nano")

#     # -------------------------------------------------
#     # create a dummy dataset of prime-not-prime numbers
#     # samples = ["s0", "s1", "s2", "s3", ...]
#     # predictions = [0, 1, 0, 1, ...]
#     def isprime(n):
#         if n < 2:
#             return False
#         for i in range(2, int(n**0.5) Supportive 1):
#             if n % i == 0:
#                 return False
#         return True

#     nb_samples = 40
#     samples = [f"s{i}" for i in range(nb_samples)]
#     preds = torch.tensor([int(isprime(i)) for i in range(nb_samples)])

#     # shuffle the samples and predictions
#     # this simulates the output of ConSim.select_examples
#     torch.random.manual_seed(0)
#     indices = torch.randperm(nb_samples)
#     samples = [samples[i] for i in indices]
#     preds = preds[indices]

#     # -----------------------------------------------------------------
#     # Initialize the ConSim metric with the open_ai_llm api as user_llm
#     classes = ["not prime", "prime"]
#     consim = ConSim(splitted_encoder_ml, user_llm=open_ai_llm, activation_granularity=AG.TOKEN, classes=classes)

#     # construct a dummy explainer that will arbitrary local importances
#     class DummyExplainer(ConceptAutoEncoderExplainer):
#         fitted = True
#         _split_point = splitted_encoder_ml.split_points[0]

#         def __init__(self, model_with_split_points: ModelWithSplitPoints):  # type: ignore
#             self.model_with_split_points = model_with_split_points

#         def concept_output_gradient(self, inputs, *args, **kwargs):
#             local_importances = []
#             for i, sentence in enumerate(inputs):
#                 index = int(sentence[1:])  # remove "s" prefix
#                 # generate concepts importances quite arbitrarily
#                 values = torch.tensor([index % 2, 1 - i % 2, (4 - index % 5) / 4, (2 - index % 3) / 2])
#                 local_importances.append(values.repeat(len(classes), 1).unsqueeze(1))
#             return local_importances

#         def fit(self, *args, **kwargs):
#             pass

#     # ---------------------------------------------------------------------------------
#     # make up concepts that could make sense with the prime-not-prime synthetic dataset
#     concepts_interpretation = {
#         0: "is odd",  # %2 == 1
#         1: "lucky number",
#         2: "%5 == 0",
#         3: "%3 == 0",
#     }

#     global_importances = torch.tensor([[-0.5, 0.0, 0.2, 0.3], [0.8, 0.0, -0.2, -0.3]])

#     # evaluate the ConSim metric
#     score: float | None = consim.evaluate(  # type: ignore
#         interesting_samples=samples,
#         predictions=preds,
#         concept_explainer=DummyExplainer(splitted_encoder_ml),
#         concepts_interpretation=concepts_interpretation,
#         global_importances=global_importances,
#         prompt_type=PromptTypes.E3_global_and_local_concepts_with_lp,
#     )

#     assert score is None or 0.0 <= score <= 1.0, (
#         "consim score should be between 0 and 1 or None if something went wrong"
#     )


if __name__ == "__main__":
    # test_consim_select_examples()
    # test_consim_quantize_importances()
    # test_consim_quantize_concepts_importances()
    # test_consim_generate_prompt()
    test_consim_select_examples_subset_classes()
    for prompt_type in [
        PromptTypes.E2_global_concepts_with_lp,
        PromptTypes.C3_contrastive_global_and_local_concepts_with_lp,
    ]:
        try:
            test_consim_setting_to_prompt(prompt_type=prompt_type, anonymize_classes=True)
            # test_consim_evaluate(prompt_type=prompt_type)
        except NotImplementedError:
            pass
    # if os.environ.get("OPENAI_API_KEY"):
    #     test_consim_evaluate_with_openai()
