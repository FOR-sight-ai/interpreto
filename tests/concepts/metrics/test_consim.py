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
from interpreto.concepts.metrics.consim import ConSim, PromptTypes
from interpreto.model_wrapping.llm_interface import LLMInterface, Role


@pytest.fixture(scope="module")
class LLMInterfacePlaceholder(LLMInterface):
    def __init__(self):
        pass

    def generate(self, prompts: list[tuple[Role, str]]) -> str | None:
        system_prompt, user_prompt = prompts[0][1], prompts[1][1]

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


def test_consim_filter_and_quantize_concepts_importances(splitted_encoder_ml: ModelWithSplitPoints):
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
        "A": {"concept_0": 0.2, "concept_1": 0.2, "concept_2": 0.01, "concept_3": 0.1},
        "B": {"concept_0": 0.3, "concept_1": 0.49, "concept_2": 0.01, "concept_3": 0.2},
    }
    # concept_3 is always under the 0.05 threshold in local importance
    local_importances = torch.tensor(
        [
            [0.5, 0.03, 0.3, 0.01],
            [0.4, -0.2, -0.2, 0.0001],
        ]
    )

    # filter and quantize concepts importances
    interp, glob_imp, loc_imp = ConSim._filter_and_quantize_concepts_importances(
        concepts_interpretation,
        global_importances,
        local_importances,
        importance_threshold=0.05,
    )

    # TODO: update tests
    assert "concept_2" not in interp
    assert all("concept_2" not in gi for gi in glob_imp.values())
    for dic in loc_imp:
        assert "concept_2" not in dic
        for val in dic.values():
            assert val in {"++", "+", "-", "--"}

    # check that concept_3 is not in the interpretation
    assert "concept_3" not in interp
    assert all("concept_3" not in gi for gi in glob_imp.values())
    for dic in loc_imp:
        assert "concept_3" not in dic
        for val in dic.values():
            assert val in {"++", "+", "-", "--"}


def test_consim_setting_to_prompt(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_setting_to_prompt` method of the ConSim metric.
    """
    sentences = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 0, 1])
    classes = ["A", "B"]
    interp = {"concept_0": "word"}
    glob = {"A": {"concept_0": "++"}, "B": {"concept_0": "-"}}
    loc = [{"concept_0": "+"}, {"concept_0": "-"}, {"concept_0": "+"}, {"concept_0": "-"}]

    system, user, literal = ConSim._setting_to_prompt(
        PromptTypes.E3_global_and_local_concepts_with_lp.value,
        anonymize_classes=False,
        sentences=sentences,
        predictions=preds,
        classes=classes,
        concepts_interpretation=interp,
        global_importances=glob,
        local_importances=loc,
    )

    assert "You are a classifier" in system
    assert "Sample_0" in system and "Sample_2" in user
    assert literal == [classes[preds[i]] for i in range(len(sentences) // 2)]


def test_consim_generate_prompt(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_generate_prompt` method of the ConSim metric.
    """
    sentences = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 0, 1])
    classes = ["A", "B"]
    interp = {"concept_0": "word"}
    glob = {"A": {"concept_0": 0.6}, "B": {"concept_0": -0.6}}
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

    assert prompts[0][0] is Role.SYSTEM
    assert prompts[1][0] is Role.USER
    assert isinstance(literal, list)


def test_consim_extract_predictions_from_response(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_extract_predictions_from_response` method of the ConSim metric.
    """
    response = "Sample_0: A\nSample_1: B\n"
    preds = ConSim._extract_predictions_from_response(response, expected_length=2)
    assert preds == ["a", "b"]

    # wrong length should return None
    assert ConSim._extract_predictions_from_response("Sample_0: A", expected_length=2) is None


def test_consim_predictions_accuracy(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_predictions_accuracy` method of the ConSim metric.
    """
    preds1 = ["a", "b", "a"]
    preds2 = ["a", "b", "c"]
    score = ConSim._predictions_accuracy(preds1, preds2)
    assert score == 2 / 3

    assert ConSim._predictions_accuracy([], []) is None

    with pytest.raises(ValueError):
        ConSim._predictions_accuracy(["a"], ["a", "b"])


def test_consim_compute_score(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_compute_score` method of the ConSim metric.
    """
    llm = LLMInterfacePlaceholder()
    classes = ["A", "B"]
    consim = ConSim(splitted_encoder_ml, llm, classes=classes)

    response = "Sample_0: A\nSample_1: B"
    score = consim._compute_score(response, ["A", "B"])
    assert score == 1.0

    # bad formatted response should yield None
    assert metric._compute_score("wrong", ["A"]) is None


def test_consim_evaluate(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `evaluate` method of the ConSim metric.
    """
    llm = LLMInterfacePlaceholder()
    classes = ["A", "B"]
    consim = ConSim(splitted_encoder_ml, llm, classes=classes)

    samples = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 0, 1])

    class DummyExplainer:
        def concept_output_gradient(self, *args, **kwargs):
            return torch.ones(len(samples), 1)

    def fake_generate_prompt(**kwargs):
        return [(Role.SYSTEM, "sys"), (Role.USER, "user")], ["A", "B"]

    consim._generate_prompt = staticmethod(fake_generate_prompt)  # type: ignore
    llm.generate = lambda prompts: "Sample_0: A\nSample_1: B"  # type: ignore

    score = consim.evaluate(
        samples,
        preds,
        DummyExplainer(),
        {"concept_0": "word"},
        {"A": {"concept_0": 1.0}},
    )

    assert score == 1.0


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available.")
def test_consim_evaluate_with_openai(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `evaluate` method of the ConSim metric with OpenAI API.
    """
    from interpreto.model_wrapping.openai_interface import OpenAILLM

    llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-3.5-turbo")
    classes = ["A", "B"]
    consim = ConSim(splitted_encoder_ml, llm, classes=classes)
    samples = ["s0", "s1", "s2", "s3"]
    preds = torch.tensor([0, 1, 0, 1])

    class DummyExplainer:
        def concept_output_gradient(self, *args, **kwargs):
            return torch.ones(len(samples), 1)

    score = consim.evaluate(
        samples,
        preds,
        DummyExplainer(),
        {"concept_0": "word"},
        {"A": {"concept_0": 1.0}},
    )

    assert score is None or 0.0 <= score <= 1.0
