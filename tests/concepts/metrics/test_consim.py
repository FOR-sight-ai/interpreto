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


def test_consim_init(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `__init__` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_get_predictions(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_get_predictions` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_extract_interesting_elements(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_extract_interesting_elements` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_select_examples(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `select_examples` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_quantize_importances(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `quantize_importances` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_filter_and_quantize_concepts_importances(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_filter_and_quantize_concepts_importances` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_setting_to_prompt(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_setting_to_prompt` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_generate_prompt(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_generate_prompt` method of the ConSim metric.
    """
    # TODO: test with a LLMInterfacePlaceholder
    ...  # TODO: implement


def test_consim_extract_predictions_from_response(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_extract_predictions_from_response` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_predictions_accuracy(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_predictions_accuracy` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_compute_score(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `_compute_score` method of the ConSim metric.
    """
    ...  # TODO: implement


def test_consim_evaluate(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `evaluate` method of the ConSim metric.
    """
    ...  # TODO: implement


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key available.")
def test_consim_evaluate_with_openai(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the `evaluate` method of the ConSim metric with OpenAI API.
    """
    ...  # TODO: implement
