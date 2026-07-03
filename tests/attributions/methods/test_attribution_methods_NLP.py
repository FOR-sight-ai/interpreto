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

import math
import os

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    # AutoModelForMultipleChoice,
    # AutoModelForQuestionAnswering,
    # AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
)

from interpreto.attributions import (
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
)
from interpreto.attributions.base import AttributionOutput
from interpreto.attributions.inference_wrappers.inference_wrapper import InferenceModes
from interpreto.commons.granularity import Granularity, GranularityAggregationStrategy
from interpreto.typing import IncompatibilityError

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

attribution_method_kwargs = {
    # -----------------------
    # Gradient based methods:
    GradientShap: {
        "baseline": 0.0,
        "n_perturbations": 2,
        "noise_std": 0.001,
    },
    Saliency: {},
    IntegratedGradients: {"n_perturbations": 3, "baseline": 0},
    SmoothGrad: {
        "n_perturbations": 3,
        "noise_std": 0.1,
    },
    VarGrad: {
        "inference_mode": InferenceModes.LOG_SOFTMAX,
        "input_x_gradient": False,
        "n_perturbations": 2,
        "noise_std": 0.05,
    },
    SquareGrad: {
        "n_perturbations": 2,
        "noise_std": 0.12,
    },
    # ---------------------------
    # Perturbation based methods:
    Occlusion: {"inference_mode": InferenceModes.SOFTMAX},
    KernelShap: {
        "n_perturbations": 3,
        "inference_mode": InferenceModes.LOG_SOFTMAX,
    },
    Lime: {"n_perturbations": 3},
    Sobol: {"n_token_perturbations": 3},
}


ALL_MODEL_LOADERS = {
    "hf-internal-testing/tiny-random-albert": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-bart": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-bert": AutoModelForSequenceClassification,
    # "hf-internal-testing/tiny-random-DebertaV2Model": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-distilbert": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-ElectraModel": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-roberta": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-t5": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-xlm-roberta": AutoModelForSequenceClassification,
    "hf-internal-testing/tiny-random-gpt2": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-gpt_neo": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-gptj": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-CodeGenForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-FalconModel": AutoModelForCausalLM,
    # "hf-internal-testing/tiny-random-Gemma3ForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-LlamaForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-MistralForCausalLM": AutoModelForCausalLM,
    "hf-internal-testing/tiny-random-Starcoder2ForCausalLM": AutoModelForCausalLM,
}

# A small subset to run on CI:
CI_MODEL_LOADERS = [
    "hf-internal-testing/tiny-random-bert",
    "hf-internal-testing/tiny-random-gpt2",
    "hf-internal-testing/tiny-random-roberta",
    "hf-internal-testing/tiny-random-t5",
]


def is_ci() -> bool:
    return os.getenv("GITHUB_ACTIONS", "").lower() == "true"


@pytest.mark.parametrize("model_name", CI_MODEL_LOADERS)
@pytest.mark.parametrize("attribution_explainer", attribution_method_kwargs.keys())
def test_attribution_methods_with_text_short(model_name, attribution_explainer):
    evaluate_attribution_methods_with_text(
        model_name, attribution_explainer, granularity=Granularity.TOKEN, aggregation_strategy=None
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [k for k in ALL_MODEL_LOADERS.keys() if k not in CI_MODEL_LOADERS])
@pytest.mark.parametrize("attribution_explainer", attribution_method_kwargs.keys())
def test_attribution_methods_with_text_long(model_name, attribution_explainer):
    evaluate_attribution_methods_with_text(
        model_name, attribution_explainer, granularity=Granularity.TOKEN, aggregation_strategy=None
    )


@pytest.mark.parametrize(
    "model_name", ["hf-internal-testing/tiny-random-bert", "hf-internal-testing/tiny-random-gpt2"]
)
@pytest.mark.parametrize("attribution_explainer", attribution_method_kwargs.keys())
@pytest.mark.parametrize(
    "granularity", [Granularity.ALL_TOKENS, Granularity.TOKEN, Granularity.WORD, Granularity.SENTENCE]
)
def test_attribution_methods_granularity(model_name, attribution_explainer, granularity):
    evaluate_attribution_methods_with_text(
        model_name=model_name,
        attribution_explainer=attribution_explainer,
        granularity=granularity,
        aggregation_strategy=None,
    )


@pytest.mark.parametrize(
    "model_name", ["hf-internal-testing/tiny-random-bert", "hf-internal-testing/tiny-random-gpt2"]
)
@pytest.mark.parametrize("attribution_explainer", attribution_method_kwargs.keys())
@pytest.mark.parametrize("granularity", [Granularity.WORD, Granularity.SENTENCE])
@pytest.mark.parametrize(
    "aggregation_strategy",
    [
        GranularityAggregationStrategy.MAX,
        GranularityAggregationStrategy.MIN,
        GranularityAggregationStrategy.SUM,
        GranularityAggregationStrategy.SIGNED_MAX,
    ],
)
def test_attribution_methods_granularity_aggregation_strategy(
    model_name, attribution_explainer, granularity, aggregation_strategy
):
    evaluate_attribution_methods_with_text(
        model_name=model_name,
        attribution_explainer=attribution_explainer,
        granularity=granularity,
        aggregation_strategy=aggregation_strategy,
    )


def evaluate_attribution_methods_with_text(model_name, attribution_explainer, granularity, aggregation_strategy):
    """Tests all combinations of models and loaders with an attribution method"""

    # Test are too memory heavy for the CI, hence we only run them on a subset of models:
    if is_ci() and model_name not in CI_MODEL_LOADERS:
        pytest.skip(f"Model {model_name} not available on CI")

    model_loader = ALL_MODEL_LOADERS[model_name]

    model = model_loader.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    assert model is not None, f"Model loading failed {model_name}"
    assert tokenizer is not None, f"Tokenizer failed to load for {model_name}"

    # To be changed according to the final form of the explainer:
    explainer_kwargs = attribution_method_kwargs.get(attribution_explainer, {})
    if aggregation_strategy is not None:
        explainer_kwargs["granularity_aggregation_strategy"] = aggregation_strategy
    explainer = attribution_explainer(
        model, tokenizer=tokenizer, batch_size=3, device=DEVICE, granularity=granularity, **explainer_kwargs
    )

    # we need to test both type of inputs: text, list_text, tokenized_text, tokenized_list_text:
    list_texts = [
        "I like this",
        "Oh it's cool",
        ["My dog is ", "this is very"],
        "Interpreto is",
        "This is two sentences. The goal is",
    ]
    list_tokenized_texts = [
        tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
        for text in list_texts
    ]

    if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
        list_targets = ["video", "and I like it.", ["nice", "good"], "a great library", "to test."]
        list_tokenized_targets = [
            tokenizer(target, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
            for target in list_targets
        ]
        list_texts_complete = list_texts + list_tokenized_texts + list_texts
        list_targets_complete = list_targets + list_targets + list_tokenized_targets
    else:
        list_targets = [None, 1, [0, 1], None, torch.tensor([[0, 1]])]
        list_texts_complete = list_texts + list_tokenized_texts
        list_targets_complete = list_targets + list_targets

    for input_text, target in zip(list_texts_complete, list_targets_complete, strict=False):
        if isinstance(input_text, BatchEncoding) and input_text["input_ids"].shape[0] > 1:
            # skip batch encoding with multiple rows
            continue

        try:
            attributions = explainer.explain(input_text, targets=target)
        except IncompatibilityError:
            continue

        # Checks:
        assert isinstance(attributions, list), "The output of the attribution explainer must be a list"

        if isinstance(input_text, str):
            assert len(attributions) == 1, (
                "The number of elements in the list must correspond to the number of inputs."
            )
        if isinstance(input_text, list):
            assert len(attributions) == len(input_text), (
                "The number of elements in the list must correspond to the number of inputs."
            )
        if isinstance(input_text, dict):
            assert len(attributions) == input_text["input_ids"].shape[0], (
                "The number of elements in the list must correspond to the number of inputs."
            )
        assert all(isinstance(attribution, AttributionOutput) for attribution in attributions), (
            "The elements of the list must be of type AttributionOutput."
        )
        assert all(
            len(attribution.elements) == (attribution.attributions).shape[-1] for attribution in attributions
        ), "In the AttributionOutput class, elements and attributions must have the same length."

        if model.__class__.__name__.endswith("ForCausalLM") or model.__class__.__name__.endswith("LMHeadModel"):
            for att_output in attributions:
                att = att_output.attributions
                t, l = att.shape
                assert l >= t, (
                    "The attributions must have the shape (t, l) where l is l_in + t. "
                    "Hence l must be greater than t. "
                    f"Got {att.shape} for {att_output.elements}."
                )

                # Example with l = 6 and t = 3
                # [[ x, x, x, NaN, NaN, NaN],
                #  [ x, x, x, x  , NaN, NaN],
                #  [ x, x, x, x  , x  , NaN]]
                # The number of NaNs corresponds to the sum of natural
                assert att.isnan().sum() == t * (t + 1) / 2, (
                    "In the case of generation attributions, only the upper triangular matrix should be filled with NaNs. "
                    f"Got {att.isnan().sum()} for shape {att.shape}. "
                    f"Expected {t * (t + 1) / 2}. "
                    f"The matrix is {att}."
                )
        else:
            for att_output in attributions:
                assert att_output.attributions.isnan().sum().item() == 0, (
                    "In the case of classification attributions, the attributions should not contain NaNs."
                    f"Got {att_output.attributions.isnan()} for {att_output.elements}."
                )


@pytest.mark.parametrize("method_class", [Lime, VarGrad])
def test_attribution_output_size(bert_model, bert_tokenizer, method_class, sentences):
    explainer = method_class(model=bert_model, tokenizer=bert_tokenizer, batch_size=3, device=DEVICE)

    attr_output = explainer.explain(sentences)

    for s, ao in zip(sentences, attr_output, strict=True):
        # (t, l)
        assert ao.attributions.shape == (1, len(ao.elements)), (
            "AttributionOutput: number of elements and attributions length mismatch"
        )
        assert math.prod(ao.attributions.shape) < 1000, "AttributionOutput: attributions tensor too large"

        for key, value in ao.model_inputs_to_explain.items():
            assert value.shape[0] == 1, (
                f"AttributionOutput: model_inputs_to_explain[{key}] should only contain one sample, no batching or perturbations"
            )
            assert math.prod(value.shape) < len(s) * 100, (
                f"AttributionOutput: model_inputs_to_explain[{key}] tensor too large"
                f"shape: {value.shape}, sentence length: {len(s)}"
            )

        assert "inputs_embeds" not in ao.model_inputs_to_explain.keys(), (
            "AttributionOutput: inputs_embeds should not be in model_inputs_to_explain"
        )


@pytest.mark.slow
@pytest.mark.parametrize("attribution_explainer", attribution_method_kwargs.keys())
def test_attribution_methods_memory_management_classification(attribution_explainer):
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bert",
        num_labels=2048,
        ignore_mismatched_sizes=True,
    )
    explainer_kwargs = attribution_method_kwargs.get(attribution_explainer, {}).copy()
    explainer = attribution_explainer(
        model,
        tokenizer=tokenizer,
        batch_size=16,
        device=DEVICE,
        granularity=Granularity.ALL_TOKENS,
        **explainer_kwargs,
    )

    samples = [f"token {i % 11} token {i % 7} token {i % 5}" for i in range(2048)]
    targets = [1] * len(samples)

    try:
        # Warm-up pass: if this fails, batch size/model sizes should be adjusted.
        explainer.explain(samples, targets=targets)
    except IncompatibilityError:
        pytest.skip(f"{attribution_explainer.__name__} is incompatible with this classification model.")

    try:
        explainer.explain(samples, targets=targets)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "can't allocate memory" in message:
            pytest.fail(f"OOM during classification stress test: {exc}")
        raise


@pytest.mark.slow
@pytest.mark.parametrize("attribution_explainer", attribution_method_kwargs.keys())
def test_attribution_methods_memory_management_generation(attribution_explainer):
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    explainer_kwargs = attribution_method_kwargs.get(attribution_explainer, {}).copy()
    explainer = attribution_explainer(
        model,
        tokenizer=tokenizer,
        batch_size=16,
        device=DEVICE,
        granularity=Granularity.ALL_TOKENS,
        **explainer_kwargs,
    )

    samples = [f"token {i % 11} token {i % 7} token {i % 5}" for i in range(2048)]
    targets = ["token token"] * len(samples)

    try:
        # Warm-up pass: if this fails, batch size/model sizes should be adjusted.
        explainer.explain(samples, targets=targets)
    except IncompatibilityError:
        pytest.skip(f"{attribution_explainer.__name__} is incompatible with this generation model.")

    try:
        explainer.explain(samples, targets=targets)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "can't allocate memory" in message:
            pytest.fail(f"OOM during generation stress test: {exc}")
        raise


# TODO: test that targets are correctly processed

# TODO: test batch size management with very different inputs, tensor mappings of shape: [(1, 10), (5, 10), (100, 10), (2, 10)...].
#       test that the output shapes are correct for each case.
#       There should be a counter wrapped around a model to verify that the number of calls to the model is correct.

if __name__ == "__main__":
    test_attribution_methods_with_text_short(
        model_name="hf-internal-testing/tiny-random-t5",
        attribution_explainer=IntegratedGradients,
    )
    test_attribution_methods_with_text_short(
        model_name="hf-internal-testing/tiny-random-gpt2",
        attribution_explainer=Lime,
    )
    test_attribution_methods_granularity(
        model_name="hf-internal-testing/tiny-random-bert",
        attribution_explainer=Occlusion,
        granularity=Granularity.WORD,
    )
    test_attribution_methods_granularity(
        model_name="hf-internal-testing/tiny-random-gpt2",
        attribution_explainer=VarGrad,
        granularity=Granularity.WORD,
    )
    test_attribution_methods_granularity(
        model_name="hf-internal-testing/tiny-random-bert",
        attribution_explainer=Saliency,
        granularity=Granularity.ALL_TOKENS,
    )
    test_attribution_methods_granularity_aggregation_strategy(
        model_name="hf-internal-testing/tiny-random-gpt2",
        attribution_explainer=GradientShap,
        granularity=Granularity.SENTENCE,
        aggregation_strategy=GranularityAggregationStrategy.SIGNED_MAX,
    )
    bert_model = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert")
    bert_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    sentences = [
        "Interpreto is the latin for 'to interpret'. But it also sounds like a spell from the Harry Potter books.",
        "Interpreto is magical",
        "Testing interpreto",
    ]
    test_attribution_output_size(bert_model, bert_tokenizer, Occlusion, sentences)
    test_attribution_output_size(bert_model, bert_tokenizer, VarGrad, sentences)
    test_attribution_methods_memory_management_classification(IntegratedGradients)
    test_attribution_methods_memory_management_generation(SmoothGrad)
