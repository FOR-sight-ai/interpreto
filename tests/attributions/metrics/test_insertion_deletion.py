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
"""This test module evaluates the Insertion and Deletion metrics."""

import pytest
import torch

from interpreto import Granularity, Occlusion
from interpreto.attributions.aggregations.insertion_deletion_aggregation import InsertionDeletionAggregator
from interpreto.attributions.metrics import Deletion, Insertion
from interpreto.attributions.perturbations import DeletionPerturbator, InsertionPerturbator

test_cases = [
    {  # Test case 1: Insertion metric with Token granularity and 10 perturbations
        "task": "classification",
        "metric_class": Insertion,
        "granularity": Granularity.TOKEN,
        "n_perturbations": 10,
        "expected_results": {
            "perturbator_type": InsertionPerturbator,
            "expected_auc": 0.505163,
            "scores_shape": [(11, 1), (11, 1), (11, 1)],
        },
    },
    {  # Test case 2: Deletion metric with Token granularity and 50 perturbations
        "task": "classification",
        "metric_class": Deletion,
        "granularity": Granularity.TOKEN,
        "n_perturbations": 50,
        "expected_results": {
            "perturbator_type": DeletionPerturbator,
            "expected_auc": 0.505124,
            "scores_shape": [(51, 1), (20, 1), (18, 1)],
        },
    },
    {  # Test case 3: Insertion metric with Word granularity and 5 perturbations
        "task": "classification",
        "metric_class": Insertion,
        "granularity": Granularity.WORD,
        "n_perturbations": 5,
        "expected_results": {
            "perturbator_type": InsertionPerturbator,
            "expected_auc": 0.505147,
            "scores_shape": [(6, 1), (4, 1), (3, 1)],
        },
    },
    {  # Test case 4: Deletion metric with Word granularity and 5 perturbations
        "task": "classification",
        "metric_class": Deletion,
        "granularity": Granularity.WORD,
        "n_perturbations": 5,
        "expected_results": {
            "perturbator_type": DeletionPerturbator,
            "expected_auc": 0.505140,
            "scores_shape": [(6, 1), (4, 1), (3, 1)],
        },
    },
]


def id_names(param):
    """Generate an id for each test case based on its parameter values."""
    return (
        param["metric_class"].__name__
        + "-task="
        + param["task"]
        + "-Granularity."
        + param["granularity"].upper()
        + "-n_perturbations="
        + str(param["n_perturbations"])
    )


@pytest.fixture(params=test_cases, ids=id_names)
def metric_test_case(request):
    """Fixture to provide test cases for insertion and deletion metrics."""
    if request.param["task"] == "classification":
        request.param["model"] = request.getfixturevalue("bert_model")
        request.param["tokenizer"] = request.getfixturevalue("bert_tokenizer")
    elif request.param["task"] == "generation":
        request.param["model"] = request.getfixturevalue("gpt2_model")
        request.param["tokenizer"] = request.getfixturevalue("gpt2_tokenizer")
    return request.param


def test_insertion_deletion_metric(sentences, metric_test_case):
    """Test the insertion and deletion metrics with various configurations.

    Multiple aspects are assessed:
    - Perturbator initialization and parameters
    - Perturbator functionality via get_mask
    - Aggregator type and functionality
    - AUC correctness and metric scores shape
    """

    model = metric_test_case["model"]
    tokenizer = metric_test_case["tokenizer"]
    metric_class = metric_test_case["metric_class"]
    granularity = metric_test_case["granularity"]
    n_perturbations = metric_test_case["n_perturbations"]
    expected_results = metric_test_case["expected_results"]

    torch.manual_seed(0)

    metric = metric_class(
        model=model,
        tokenizer=tokenizer,
        granularity=granularity,
        n_perturbations=n_perturbations,
    )

    # Assert that the "[REPLACE]" token has correctly been added to the tokenizer
    assert "[REPLACE]" in metric.tokenizer.get_vocab()

    # Assert that the perturbator stored its params correctly
    assert isinstance(metric.perturbator, expected_results["perturbator_type"])
    assert metric.perturbator.n_perturbations == n_perturbations
    assert metric.perturbator.granularity == granularity
    replace_id = metric.tokenizer.convert_tokens_to_ids("[REPLACE]")  # the token ID should match what we just added
    assert metric.perturbator.replace_token_id == replace_id

    # Assert the aggregator type
    assert isinstance(metric.aggregator, InsertionDeletionAggregator)

    # Assert that get_mask returns a tensor of the right shape
    seq_len = n_perturbations - 2  # to ensure we have as many perturbations as the sequence length
    num_targets = 3
    attributions = torch.arange(seq_len * num_targets).view(num_targets, seq_len).float()
    mask = metric.perturbator.get_mask(seq_len, attributions=attributions)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == ((seq_len + 1) * num_targets, seq_len)

    # Assert AUC correctness and metric scores shape using sentences from conftest
    explainer = Occlusion(model, tokenizer, granularity=granularity)
    attributions = explainer.explain(sentences)
    auc, metric_scores = metric.evaluate(attributions)

    assert auc == pytest.approx(expected_results["expected_auc"])
    assert [s.shape for s in metric_scores] == expected_results["scores_shape"]
