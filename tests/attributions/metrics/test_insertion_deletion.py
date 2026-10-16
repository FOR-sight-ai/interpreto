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

from interpreto import IntegratedGradients, KernelShap, Occlusion, SmoothGrad, TextGranularity
from interpreto.attributions.metrics import Deletion, Insertion

test_cases = [
    {  # Test case 1: Insertion metric with Token granularity and 10 perturbations
        "task": "classification",
        "metric_class": Insertion,
        "granularity": TextGranularity.TOKEN,
        "n_perturbations": 10,
    },
    {  # Test case 2: Deletion metric with Token granularity and 50 perturbations
        "task": "classification",
        "metric_class": Deletion,
        "granularity": TextGranularity.TOKEN,
        "n_perturbations": 50,
    },
    {  # Test case 3: Insertion metric with Word granularity and 5 perturbations
        "task": "classification",
        "metric_class": Insertion,
        "granularity": TextGranularity.WORD,
        "n_perturbations": 5,
    },
    {  # Test case 4: Deletion metric with Word granularity and 5 perturbations
        "task": "classification",
        "metric_class": Deletion,
        "granularity": TextGranularity.WORD,
        "n_perturbations": 5,
    },
]


def id_names(param):
    """Generate an id for each test case based on its parameter values."""
    return (
        param["metric_class"].__name__
        + "-task="
        + param["task"]
        + "-TextGranularity."
        + param["granularity"].value.upper()
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


def metric_on_method_model_pair(model, tokenizer, method_class, metric_class, sentences, targets=None):
    """Run metrics on a model and a method.

    This function is called by another

    Warnings are considered as errors.
    """
    # compute explanations
    explainer = method_class(model, tokenizer, granularity=TextGranularity.WORD)
    attributions = explainer.explain(sentences, targets)

    # compute metric
    metric_scores: list[list[torch.Tensor]]
    metric = metric_class(model, tokenizer)
    auc, metric_scores = metric.evaluate(attributions)

    # individual scores
    assert isinstance(metric_scores, list), f"Metric scores is not a list: {type(metric_scores)}"
    assert all(isinstance(s, list) for s in metric_scores), (
        f"Metric scores is not a list of list: {[type(s) for s in metric_scores]}"
    )
    assert all(isinstance(t, torch.Tensor) for s in metric_scores for t in s), (
        f"Metric scores is not a list of list of tensors: {[type(s) for s in metric_scores]}"
    )
    assert all(0 <= p.item() <= 1 for s in metric_scores for t in s for p in t), (
        f"All individual score should be between 0 and 1: {metric_scores}"
    )

    assert len(metric_scores) == len(sentences), (
        f"Metric scores len do not match sentences len: {len(metric_scores)} != {len(sentences)}"
    )
    if targets is None:  # not specifying targets (classification)
        assert all(len(s) == 1 for s in metric_scores), (
            f"Metric scores len do not match targets len: {len(metric_scores)} != {1}"
        )
    elif all(t == " word longword verylongword" for t in targets):  # using our specified targets (generation)
        assert all(len(s) == 3 for s in metric_scores), (
            f"Metric scores len for each sample do not match the number of targets, "
            f"expected 3 for all samples, got {[len(s) for s in metric_scores]}"
        )

    # AUC
    assert isinstance(auc, float), f"AUC is not a float: {auc}"
    assert 0 <= auc <= 1, f"AUC is not between 0 and 1: {auc}"


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("method_class", [Occlusion, IntegratedGradients])
@pytest.mark.parametrize("metric_class", [Insertion, Deletion])
def test_insertion_deletion_classification(bert_model, bert_tokenizer, method_class, metric_class, sentences):
    # Classification
    metric_on_method_model_pair(bert_model, bert_tokenizer, method_class, metric_class, sentences)


@pytest.mark.parametrize("method_class", [KernelShap, SmoothGrad])
@pytest.mark.parametrize("metric_class", [Insertion, Deletion])
def test_insertion_deletion_generation(gpt2_model, gpt2_tokenizer, method_class, metric_class, sentences):
    targets = [" word longword verylongword"] * len(sentences)
    # Generation
    metric_on_method_model_pair(gpt2_model, gpt2_tokenizer, method_class, metric_class, sentences, targets)


def test_qualitative_classification(bert_imdb_model, bert_imdb_tokenizer):
    torch.manual_seed(0)
    sentences = ["What a great movie! Too bad it is long."]
    expected_attribution = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]])
    random_attribution = torch.rand(1, 11)
    targets = torch.tensor([[1]])

    # Compute an explanation
    model_inputs = bert_imdb_tokenizer(sentences, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    explainer = Occlusion(bert_imdb_model, bert_imdb_tokenizer, granularity=TextGranularity.WORD)
    occlusion_attribution_outputs = explainer.explain(model_inputs, targets)

    assert occlusion_attribution_outputs[0].attributions.shape == (1, 11)

    # Construct attributions outputs for each attribution
    expected_attribution_outputs = occlusion_attribution_outputs.copy()
    expected_attribution_outputs[0].attributions = expected_attribution

    inverse_attribution_outputs = occlusion_attribution_outputs.copy()
    inverse_attribution_outputs[0].attributions = -expected_attribution

    random_attribution_outputs = occlusion_attribution_outputs.copy()
    random_attribution_outputs[0].attributions = random_attribution

    # Instantiate metrics
    deletion = Deletion(bert_imdb_model, bert_imdb_tokenizer)
    insertion = Insertion(bert_imdb_model, bert_imdb_tokenizer)

    # Compute deletion metrics
    expected_deletion, _ = deletion.evaluate(expected_attribution_outputs)
    inverse_deletion, _ = deletion.evaluate(inverse_attribution_outputs)
    random_deletion, _ = deletion.evaluate(random_attribution_outputs)
    occlusion_deletion, _ = deletion.evaluate(occlusion_attribution_outputs)

    # Compute insertion metrics
    expected_insertion, _ = insertion.evaluate(expected_attribution_outputs)
    inverse_insertion, _ = insertion.evaluate(inverse_attribution_outputs)
    random_insertion, _ = insertion.evaluate(random_attribution_outputs)
    occlusion_insertion, _ = insertion.evaluate(occlusion_attribution_outputs)

    # Ensure they all fall between 0 and 1
    assert 0 <= expected_deletion <= 1, f"Expected deletion metric is not between 0 and 1: {expected_deletion}"
    assert 0 <= expected_insertion <= 1, f"Expected insertion metric is not between 0 and 1: {expected_insertion}"
    assert 0 <= inverse_deletion <= 1, f"Inverse deletion metric is not between 0 and 1: {inverse_deletion}"
    assert 0 <= inverse_insertion <= 1, f"Inverse insertion metric is not between 0 and 1: {inverse_insertion}"
    assert 0 <= random_deletion <= 1, f"Random deletion metric is not between 0 and 1: {random_deletion}"
    assert 0 <= random_insertion <= 1, f"Random insertion metric is not between 0 and 1: {random_insertion}"
    assert 0 <= occlusion_deletion <= 1, f"Occlusion deletion metric is not between 0 and 1: {occlusion_deletion}"
    assert 0 <= occlusion_insertion <= 1, f"Occlusion insertion metric is not between 0 and 1: {occlusion_insertion}"

    # Ensure the order is coherent (deletion, lower is better) (insertion, higher is better)
    assert expected_deletion <= random_deletion <= inverse_deletion, (
        "Deletion metric order is not coherent (lower is better)"
        f"Expected deletion: {expected_deletion}, random deletion: {random_deletion}, inverse deletion: {inverse_deletion}"
    )
    assert occlusion_deletion <= inverse_deletion, (  # we cannot know the order between occlusion and expected
        "Deletion metric order is not coherent"
        f"Occlusion deletion: {occlusion_deletion}, inverse deletion: {inverse_deletion}"
    )
    assert expected_insertion >= random_insertion >= inverse_insertion, (
        "Insertion metric order is not coherent (higher is better)"
        f"Expected insertion: {expected_insertion}, random insertion: {random_insertion}, inverse insertion: {inverse_insertion}"
    )
    assert occlusion_insertion >= inverse_insertion, (  # we cannot know the order between occlusion and expected
        "Insertion metric order is not coherent"
        f"Occlusion insertion: {occlusion_insertion}, inverse insertion: {inverse_insertion}"
    )


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    # inputs
    sentences = [
        "test word longword verylongword",
        "Interpreto is the latin for 'to interpret'. But it also sounds like a spell from the Harry Potter books.",
        "Interpreto is magical",
        "Testing interpreto",
    ]

    targets = [" word longword verylongword"] * len(sentences)

    # models
    bert_model = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert")
    bert_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    bert_imdb_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    bert_imdb_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

    # tests
    metric_on_method_model_pair(bert_model, bert_tokenizer, IntegratedGradients, Insertion, sentences)

    metric_on_method_model_pair(gpt_model, gpt_tokenizer, Occlusion, Deletion, sentences, targets)

    test_qualitative_classification(bert_imdb_model, bert_imdb_tokenizer)
