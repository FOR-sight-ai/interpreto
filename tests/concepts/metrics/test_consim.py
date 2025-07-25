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
    ConSim._get_predictions
    ConSim._extract_interesting_elements
    ConSim.select_examples
    ConSim.quantize_importances
    ConSim._filter_and_quantize_concepts_importances
    ConSim._setting_to_prompt
    ConSim._generate_prompt
    ConSim._extract_predictions_from_response
    ConSim._predictions_accuracy
    ConSim._compute_score
    ConSim.evaluate

In the unit tests listed above some configurations will be common:
- the `ModelWithSplitPoints` will be used around a Bert model,
- the `ConceptAutoEncoderExplainers` will be a `NeuronAsConcepts` explainer,
- the `LLMInterface` will be replaced by a place holder predicting the classes specified randomly,

Then the ConSim metric will be tested with different `ConceptAutoEncoderExplainers`.

Finally, an end to end test will include a call to the `OpenAILLM` if an API key is available.
"""

from __future__ import annotations

import os

import pytest
import torch

from interpreto import ModelWithSplitPoints
from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.concepts.metrics.consim import ConSim, PromptTypes
from interpreto.model_wrapping.llm_interface import LLMInterface, Role


class LLMInterfacePlaceholder(LLMInterface):
    def __init__(self):
        pass

    def generate(self, prompt: list[tuple[Role, str]]) -> str | None:
        system_prompt, user_prompt = prompt[0][1], prompt[1][1]

        # extract the classes from the system prompt
        classes_str = system_prompt.split("The classes are: [")[1].split("]")[0]
        classes = classes_str.split(", ")

        # extract the sample indices from the user prompt
        ep_samples_str = user_prompt.split("\n\nConcepts contributions for Sample_")[0]
        # Format:
        #     Sample_0: "this is the first sample"
        #     Sample_1: "this is the second sample"
        ep_sample_indices = [int(s.split(":")[0][6:]) for s in ep_samples_str.split("\n") if s]

        # generate the response, give a random class for each sample
        response = "\n".join([f"Sample_{i}: {classes[i % len(classes)]}" for i in ep_sample_indices])
        return response


def test_consim_init(splitted_encoder_ml: ModelWithSplitPoints, multi_split_model: ModelWithSplitPoints):
    """
    Test the `__init__` method of the ConSim metric.
    """
    llm = LLMInterfacePlaceholder()
    classes = [str(i) for i in range(int(splitted_encoder_ml._model.num_labels))]

    # when only one split point is available, it should be chosen automatically
    consim = ConSim(splitted_encoder_ml, llm, classes=classes)
    assert metric.split_point == splitted_encoder_ml.split_points[0]
    assert metric.user_llm is llm

    # invalid split point should raise an error
    with pytest.raises(ValueError):
        ConSim(splitted_encoder_ml, llm, classes, split_point="wrong.point")

    # when multiple split points exist, omitting split_point must fail
    with pytest.raises(ValueError):
        ConSim(multi_split_model, llm, classes)


def test_consim_get_predictions(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_get_predictions` method of the ConSim metric.
    """
    # Initialize the ConSim metric
    consim = ConSim(splitted_encoder_ml)

    inputs = ["This is a first sentence", "Another sentence"]

    # Compute predictions with nnsight
    with splitted_encoder_ml.trace(inputs):
        nnsight_preds = splitted_encoder_ml.output.save()

    # Verify nnsight predictions
    assert isinstance(nnsight_preds, torch.Tensor)
    assert nnsight_preds.shape == (len(inputs),)

    # Compute predictions with ConSim
    consim_preds = consim._get_predictions(inputs)

    # Verify ConSim predictions
    assert isinstance(consim_preds, torch.Tensor)
    assert consim_preds.shape == (len(inputs),)

    # Check that both predictions are equal
    assert torch.allclose(nnsight_preds, consim_preds, atol=1e-6)


def test_consim_extract_interesting_elements(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_extract_interesting_elements` method of the ConSim metric.
    """
    classes = ["0", "1"]
    consim = ConSim(splitted_encoder_ml, classes=classes)

    inputs = [f"sentence {i}" for i in range(6)]
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    predictions = torch.tensor([0, 0, 0, 1, 1, 1])

    # 2 correct and 2 incorrect elements should be returned
    samples, labels, predictions = consim._extract_interesting_elements(
        inputs, labels, predictions, nb_lp_samples=2, nb_ep_samples=2, seed=0
    )

    # ensure samples, labels, and predictions all have 4 elements, 2 correct and 2 incorrect
    assert len(samples) == 4
    assert labels.shape == (4,)
    assert predictions.shape == (4,)

    # ensure exactly half of the predictions match the labels
    assert torch.sum(labels == predictions) == 2

    # ensure each class represents half of the predictions and labels
    assert torch.sum(labels == 0) == 2
    assert torch.sum(predictions == 0) == 2

    # not enough correct predictions should raise a value error
    with pytest.raises(ValueError):
        consim._extract_interesting_elements(inputs[:2], labels[:2], predictions[:2], nb_lp_samples=2, nb_ep_samples=2)


def test_consim_select_examples(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `select_examples` method of the ConSim metric.
    """
    classes = ["0", "1"]
    consim = ConSim(splitted_encoder_ml, classes=classes)

    # prepare fake methods so we only test the logic of select_examples
    inputs = ["a", "b", "c", "d", "e", "f"]
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    predictions = torch.tensor([0, 0, 0, 1, 1, 1])

    def fake_get_preds(x, **kwargs):
        assert x == inputs
        return predictions

    consim._get_predictions = fake_get_preds  # type: ignore

    # 2 correct and 2 incorrect elements should be returned
    samples, labels, predictions = consim.select_examples(inputs, labels, nb_lp_samples=2, nb_ep_samples=2, seed=0)

    # ensure samples, labels, and predictions all have 4 elements, 2 correct and 2 incorrect
    assert len(samples) == 4
    assert labels.shape == (4,)
    assert predictions.shape == (4,)

    # ensure exactly half of the predictions match the labels
    assert torch.sum(labels == predictions) == 2

    # ensure each class represents half of the predictions and labels
    assert torch.sum(labels == 0) == 2
    assert torch.sum(predictions == 0) == 2


def test_consim_quantize_importances():
    """
    Test the `quantize_importances` method of the ConSim metric.
    """
    thr = 0.05
    assert ConSim.quantize_importances(0.4, thr) == "++"
    assert ConSim.quantize_importances(0.1, thr) == "+"
    assert ConSim.quantize_importances(-0.1, thr) == "-"
    assert ConSim.quantize_importances(-0.5, thr) == "--"

    with pytest.raises(ValueError):
        ConSim.quantize_importances(0.0, thr)


def test_consim_filter_and_quantize_concepts_importances():
    """
    Test the `_filter_and_quantize_concepts_importances` method of the ConSim metric.

    The function should filter concepts with values under the threshold in all case
    """
    # 4 concepts, 2 classes, and 2 samples
    concepts_interpretation = {
        "concept_0": "w0",
        "concept_1": "w1",
        "concept_2": "w2",
        "concept_3": "w3",
    }
    # concept_2 is always under the 0.05 threshold in global importance
    global_importances = {
        "A": {"concept_0": 0.5, "concept_1": 0.25, "concept_2": 0.01, "concept_3": -0.24},  # abs sum should be 1
        "B": {"concept_0": 0.1, "concept_1": 0.49, "concept_2": 0.01, "concept_3": -0.4},  # abs sum should be 1
    }
    # concept_1 and concept_3 are locally under the 0.05 threshold in local importance
    local_importances = torch.tensor(
        [
            [-0.55, 0.03, 0.3, 0.02],  # abs sum should be 1
            [0.3, -0.2, -0.39, 0.11],  # abs sum should be 1
        ]
    )

    # filter and quantize concepts importances
    interp, glob_imp, loc_imp = ConSim._filter_and_quantize_concepts_importances(
        concepts_interpretation,
        global_importances,
        local_importances,
        importance_threshold=0.05,
    )

    # global concepts under the threshold should be removed globally, in this case concept_2
    assert "concept_2" not in interp
    assert all("concept_2" not in gi for gi in glob_imp.values())
    assert all("concept_2" not in li for li in loc_imp)

    # ensure other concepts are not removed
    for c in ["concept_0", "concept_1", "concept_3"]:
        assert c in interp
        assert c in glob_imp["A"]
        assert c in glob_imp["B"]

    # local concepts under the threshold should be removed locally, in this case concept_1 and concept_3 in the first sample
    assert "concept_1" not in loc_imp[0]
    assert "concept_3" not in loc_imp[0]

    # ensure that values have been quantized and have the correct literal
    assert glob_imp["A"]["concept_0"] == "++"  # 0.5
    assert glob_imp["A"]["concept_1"] == "+"  # 0.25
    assert glob_imp["A"]["concept_3"] == "-"  # -0.24
    assert glob_imp["B"]["concept_0"] == "+"  # 0.1
    assert glob_imp["B"]["concept_1"] == "+"  # 0.49
    assert glob_imp["B"]["concept_3"] == "--"  # -0.4

    # ensure that values have been quantized and have the correct literal
    assert loc_imp[0]["concept_0"] == "--"  # -0.55
    assert loc_imp[0]["concept_2"] == "++"  # 0.3
    assert loc_imp[1]["concept_0"] == "++"  # 0.3
    assert loc_imp[1]["concept_1"] == "-"  # -0.2
    assert loc_imp[1]["concept_2"] == "--"  # -0.39
    assert loc_imp[1]["concept_3"] == "+"  # 0.11


@pytest.mark.parametrize(
    "prompt_type",
    [
        PromptTypes.L1_baseline_without_lp,
        PromptTypes.E1_global_concepts_without_lp,
        PromptTypes.L2_baseline_with_lp,
        PromptTypes.E2_global_concepts_with_lp,
        PromptTypes.E3_global_and_local_concepts_with_lp,
        PromptTypes.U1_upper_bound_concepts_at_ep,
    ],
)
@pytest.mark.parametrize("anonymize_classes", [False, True])
def test_consim_setting_to_prompt(prompt_type: PromptTypes, anonymize_classes: bool):
    """
    Test the `_setting_to_prompt` method of the ConSim metric.
    """
    # prepare fake method so we only test the logic of setting_to_prompt
    sentences = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 0, 1])
    classes = ["A", "B"]
    interp = {"concept_0": "word"}
    glob = {"A": {"concept_0": "++"}, "B": {"concept_0": "-"}}
    loc = [{"concept_0": "+"}, {"concept_0": "-"}, {"concept_0": "+"}, {"concept_0": "-"}]

    # convert the prompt type to settings
    prompt_settings = prompt_type.value

    # ==============================================================================================
    # Verify the settings based on the prompt type
    prompt_type_name = str(prompt_type)
    assert prompt_settings.ep_samples is True
    if "baseline" in prompt_type_name:
        assert prompt_settings.concepts_interpretation is False
        assert prompt_settings.concepts_global_importances is False
        assert prompt_settings.lp_concepts_local_contributions is False
        assert prompt_settings.ep_concepts_local_contributions is False
    if "without_lp" in prompt_type_name:
        assert prompt_settings.lp_samples is False
        assert prompt_settings.lp_labels is False
        assert prompt_settings.lp_concepts_local_contributions is False
    if "with_lp" in prompt_type_name:
        assert prompt_settings.lp_samples is True
        assert prompt_settings.lp_labels is True
    if "global" in prompt_type_name:
        assert prompt_settings.concepts_interpretation is True
        assert prompt_settings.concepts_global_importances is True
        assert prompt_settings.ep_concepts_local_contributions is False
    if "local" in prompt_type_name:
        assert prompt_settings.lp_concepts_local_contributions is True
        assert prompt_settings.ep_concepts_local_contributions is False
    else:
        assert prompt_settings.lp_concepts_local_contributions is False
    if "upper_bound" in prompt_type_name:
        assert prompt_settings.concepts_interpretation is True
        assert prompt_settings.concepts_global_importances is True
        assert prompt_settings.lp_samples is True
        assert prompt_settings.lp_labels is True
        assert prompt_settings.lp_concepts_local_contributions is True
        assert prompt_settings.ep_concepts_local_contributions is True

    # convert the settings to prompt
    system, user, literal = ConSim._setting_to_prompt(
        setting=prompt_settings,
        anonymize_classes=anonymize_classes,
        sentences=sentences,
        predictions=preds,
        classes=classes,
        concepts_interpretation=interp,
        global_importances=glob,
        local_importances=loc,
    )

    # ==============================================================================================
    # Verify the generated prompts elements

    # -------------
    # Initial Phase
    # task description
    assert "You are a classifier." in system
    assert "Sample_{i}: {predicted_class}" in system
    if anonymize_classes:
        assert "The classes are: [Class_0, Class_1]" in system
    else:
        assert "The classes are: [A, B]" in system

    # global concepts explanation
    #     concepts interpretation
    if prompt_settings.concepts_interpretation:
        assert "concept_0: word" in system
    #     global concepts importance
    if prompt_settings.concepts_global_importances:
        if anonymize_classes:
            assert "Class_0: {concept_0: ++}" in system
        else:
            assert "B: {concept_0: -}" in system

    # --------------
    # Learning Phase
    # examples
    if prompt_settings.lp_samples:
        # samples, only the 2 first samples for lp
        assert "Sample_0: s0\nSample_1: s1" in system
        assert "Sample_2" not in system and "Sample_3" not in system
        # labels
        assert "Sample_0: A\nSample_1: B" in system

    # local concepts explanation
    if prompt_settings.lp_concepts_local_contributions:
        assert "Sample_0: {'concept_0': '+'}" in system
        assert "Sample_1: {'concept_0': '-'}" in system

    # ----------------
    # Evaluation Phase
    # samples, only the 2 last samples for ep
    assert "Sample_2: s2\nSample_3: s3" in user
    assert "Sample_0" not in user and "Sample_1" not in user
    # concepts contributions
    if prompt_settings.ep_concepts_local_contributions:
        assert "Sample_2: {'concept_0': '+'}" in user
        assert "Sample_3: {'concept_0': '-'}" in user


def test_consim_generate_prompt():
    """
    Test the `_generate_prompt` method of the ConSim metric.
    """
    sentences = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 0, 1])
    classes = ["A", "B"]
    interp = {"concept_0": "word"}
    glob = {"A": {"concept_0": 0.6, "concept_1": 0.01}, "B": {"concept_0": -0.6, "concept_1": -0.02}}
    loc = torch.tensor([[0.4], [-0.4], [0.5], [-0.5]])

    prompts, literal = ConSim._generate_prompt(
        sentences=sentences,
        predictions=preds,
        classes=classes,
        concepts_interpretation=interp,
        global_importances=glob,
        local_importances=loc,
        prompt_type=PromptTypes.E3_global_and_local_concepts_with_lp,
    )

    # test prompt format
    assert prompts[0][0] is Role.SYSTEM
    system = prompts[0][1]
    assert isinstance(system, str)
    assert prompts[1][0] is Role.USER
    user = prompts[1][1]
    assert isinstance(user, str)

    # verify literal predictions
    assert isinstance(literal, list)
    assert len(literal) == len(sentences)
    assert all(isinstance(pred, str) for pred in literal)
    assert literal == ["A", "B", "A", "B"]

    # global importance should have been converted to:
    # glob = {"A": {"concept_0": 0.6}, "B": {"concept_0": -0.6}}
    # with the concept_1 being removed
    assert "concept_1" not in system
    assert "concept_0: word" in system  # E3 includes the concepts interpretation
    assert "A: {concept_0: ++}" in system  # E3 includes the global concepts importance
    assert "B: {concept_0: --}" in system  # E3 includes the global concepts importance
    assert "Sample_0: s0\nSample_1: s1" in system
    assert "Sample_2" not in system and "Sample_3" not in system
    assert "Sample_0: A\nSample_1: B" in system
    assert "Sample_0: {'concept_0': '+'}" in system  # E3 includes the local concepts contribution
    assert "Sample_1: {'concept_0': '-'}" in system  # E3 includes the local concepts contribution
    assert "Sample_2: s2\nSample_3: s3" in user
    assert "Sample_0" not in user and "Sample_1" not in user
    assert "concept" not in user  # E3 does not include the concepts contributions in the user prompt


def test_consim_extract_predictions_from_response():
    """
    Test the `_extract_predictions_from_response` method of the ConSim metric.
    """
    response = "Sample_0: A\nSample_1: B\n"
    preds = ConSim._extract_predictions_from_response(response, expected_length=2)
    assert preds == ["a", "b"]

    # wrong length should return None
    assert ConSim._extract_predictions_from_response("Sample_0: A", expected_length=2) is None

    # wrong format should return None
    assert ConSim._extract_predictions_from_response("[A, B]", expected_length=2) is None


def test_consim_predictions_accuracy():
    """
    Test the `_predictions_accuracy` method of the ConSim metric.
    """
    preds1 = ["a", "b", "a"]
    preds2 = ["a", "b", "c"]
    score = ConSim._predictions_accuracy(preds1, preds2)
    assert score == 2 / 3

    # empty predictions should return None
    assert ConSim._predictions_accuracy([], ["a", "b"]) is None
    assert ConSim._predictions_accuracy(["a", "b"], []) is None

    # different lengths should raise a ValueError
    with pytest.raises(ValueError):
        ConSim._predictions_accuracy(["a"], ["a", "b"])


def test_consim_compute_score(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_compute_score` method of the ConSim metric.
    """
    response = "Sample_0: A\nSample_1: B\nSample_2: A\nSample_3: B"
    score = ConSim._compute_score(response, ["A", "B", "B", "B"])
    assert score == 0.75

    # not matching lengths should return None
    assert ConSim._compute_score(response, ["A", "B"]) is None

    # bad formatted response should return None
    assert ConSim._compute_score("wrong", ["A", "B"]) is None


@pytest.mark.parametrize(
    "prompt_type",
    [
        PromptTypes.L1_baseline_without_lp,
        PromptTypes.E1_global_concepts_without_lp,
        PromptTypes.L2_baseline_with_lp,
        PromptTypes.E2_global_concepts_with_lp,
        PromptTypes.E3_global_and_local_concepts_with_lp,
        PromptTypes.U1_upper_bound_concepts_at_ep,
    ],
)
def test_consim_evaluate(splitted_encoder_ml: ModelWithSplitPoints, prompt_type: PromptTypes):
    """
    Test the `evaluate` method of the ConSim metric.

    Parameters
    ----------
    splitted_encoder_ml: ModelWithSplitPoints
        The model to explain. Is is a wrapper around a model and a tokenizer to easily get activations.
        Here a Bert model, but this is not used in this test apart from initializing the ConSim metric.
    llm_placeholder: LLMInterface
        The LLM interface that will serve as the meta-predictor.
        It randomly predicts the classes specified in the prompt.
    """
    llm = LLMInterfacePlaceholder()
    classes = ["A", "B", "C", "D"]
    consim = ConSim(splitted_encoder_ml, llm, classes=classes)

    samples = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
    preds = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    # creating a dummy explainer that will return a gradient of ones
    class DummyExplainer(ConceptAutoEncoderExplainer):
        fitted = True

        def __init__(self, model_with_split_points: ModelWithSplitPoints):  # type: ignore
            self.model_with_split_points = model_with_split_points

        def concept_output_gradient(self, samples, *args, **kwargs):
            """
            consim.evaluate calls this method
            """
            # Therefore we ensure that it is called only when necessary.
            assert prompt_type in [
                PromptTypes.E3_global_and_local_concepts_with_lp,
                PromptTypes.U1_upper_bound_concepts_at_ep,
            ]
            # Furthermore, we ensure it is called only with the necessary elements.
            if prompt_type == PromptTypes.E3_global_and_local_concepts_with_lp:
                # only lp samples local importances are computed
                assert samples == ["s0", "s1", "s2", "s3", "s4"]
            else:
                assert samples == ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]

            return torch.ones(len(samples), 1)

        def fit(self, *args, **kwargs):
            pass

    explainer = DummyExplainer(splitted_encoder_ml)

    # evaluate the ConSim metric
    score = consim.evaluate(
        interesting_samples=samples,
        predictions=preds,
        concept_explainer=explainer,
        concepts_interpretation={"concept_0": "word"},
        global_importances={"A": {"concept_0": 1.0}},
        prompt_type=prompt_type,
    )

    # None is allowed in the typing, but it should not happen in this case
    # because the llm placeholder always predicts in the expected format
    assert score is not None
    assert 0.0 <= score <= 1.0


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available.")
@pytest.mark.slow
def test_consim_evaluate_with_openai(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `evaluate` method of the ConSim metric with OpenAI API.
    """
    # lazy import to avoid importing openai
    from interpreto.model_wrapping.llm_interface import OpenAILLM

    open_ai_llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini-2024-07-18")

    # -------------------------------------------------
    # create a dummy dataset of prime-not-prime numbers
    # samples = ["s0", "s1", "s2", "s3", ...]
    # predictions = [0, 1, 0, 1, ...]
    def isprime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    nb_samples = 40
    samples = [f"s{i}" for i in range(nb_samples)]
    preds = torch.tensor([isprime(i) for i in range(nb_samples)])

    # shuffle the samples and predictions
    # this simulates the output of ConSim.select_examples
    torch.random.manual_seed(0)
    indices = torch.randperm(nb_samples)
    samples = [samples[i] for i in indices]
    preds = preds[indices]

    # -----------------------------------------------------------------
    # Initialize the ConSim metric with the open_ai_llm api as user_llm
    classes = ["not prime", "prime"]
    consim = ConSim(splitted_encoder_ml, user_llm=open_ai_llm, classes=classes)

    # construct a dummy explainer that will arbitrary local importances
    class DummyExplainer(ConceptAutoEncoderExplainer):
        fitted = True

        def __init__(self, model_with_split_points: ModelWithSplitPoints):  # type: ignore
            self.model_with_split_points = model_with_split_points

        def concept_output_gradient(self, inputs, *args, **kwargs):
            local_importances = torch.zeros(len(inputs), len(concepts_interpretation))
            for i, sentence in enumerate(inputs):
                index = int(sentence[1:])  # remove "s" prefix
                # generate concepts importances quite arbitrarily
                local_importances[i] = torch.tensor(
                    [index % 2, 1 - index % 2, (4 - index % 5) / 4, (2 - index % 3) / 2]
                )
            return local_importances

        def fit(self, *args, **kwargs):
            pass

    # ---------------------------------------------------------------------------------
    # make up concepts that could make sense with the prime-not-prime synthetic dataset
    concepts_interpretation = {
        "concept_0": "is odd",  # %2 == 1
        "concept_1": "lucky number",
        "concept_2": "%5 == 0",
        "concept_3": "%3 == 0",
    }

    global_importances = {
        "not prime": {"concept_0": -0.5, "concept_1": 0.0, "concept_2": 0.2, "concept_3": 0.3},
        "prime": {"concept_0": 0.8, "concept_1": 0.0, "concept_2": -0.2, "concept_3": -0.3},
    }

    # evaluate the ConSim metric
    score = consim.evaluate(
        interesting_samples=samples,
        predictions=preds,
        concept_explainer=DummyExplainer(splitted_encoder_ml),
        concepts_interpretation=concepts_interpretation,
        global_importances=global_importances,
        prompt_type=PromptTypes.E3_global_and_local_concepts_with_lp,
        anonymize_classes=True,
    )

    assert score is None or 0.0 <= score <= 1.0
