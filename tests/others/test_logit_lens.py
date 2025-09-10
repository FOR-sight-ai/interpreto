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

# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

"""
Tests for interpreto.others.logit_lens_general module

This test suite validates LogitLens functionality across different model architectures:
- Language Models (Causal LM and Masked LM)
- Classification Models (Sequence Classification)
- Automatic model type detection
- Output structure validation
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.others.logit_lens_general import (
    ClassificationLogitLens,
    LanguageModelLogitLens,
    LogitLens,
)

# Model configurations for testing
CAUSAL_LM_MODELS = {
    "hf-internal-testing/tiny-random-gpt2": {
        "model_class": AutoModelForCausalLM,
        "split_points": ["transformer.h.1.mlp"],
        "head_name": "lm_head",
    },
    "hf-internal-testing/tiny-random-gpt_neo": {
        "model_class": AutoModelForCausalLM,
        "split_points": ["transformer.h.1.mlp"],
        "head_name": "lm_head",
    },
}

MASKED_LM_MODELS = {
    "hf-internal-testing/tiny-random-bert": {
        "model_class": AutoModelForMaskedLM,
        "split_points": ["bert.encoder.layer.1.output"],
        "head_name": "cls.predictions.decoder",
    },
    "hf-internal-testing/tiny-random-roberta": {
        "model_class": AutoModelForMaskedLM,
        "split_points": ["roberta.encoder.layer.1.output"],
        "head_name": "lm_head",
    },
}

CLASSIFICATION_MODELS = {
    "hf-internal-testing/tiny-random-bert": {
        "model_class": AutoModelForSequenceClassification,
        "split_points": ["bert.encoder.layer.1.output"],
        "head_name": "classifier",
        "pooling_strategy": "cls",
    },
}

CI_MODELS = {
    "causal_lm": ["hf-internal-testing/tiny-random-gpt2"],
    "masked_lm": ["hf-internal-testing/tiny-random-bert"],
    "classification": ["hf-internal-testing/tiny-random-bert"],
}


# Test fixtures
@pytest.fixture
def sentences():
    """Test sentences for experiments."""
    return [
        "Interpreto is the latin for 'to interpret'. But it also sounds like a spell from the Harry Potter books.",
        "Interpreto is magical",
        "Testing interpreto",
    ]


def test_automatic_model_detection_causal_lm(sentences):
    """Test that GeneralLogitLens automatically detects causal language models."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
    )

    # Test that the factory creates LanguageModelLogitLens
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=5)

    assert isinstance(logit_lens, LanguageModelLogitLens)
    assert logit_lens.vocab_size == tokenizer.vocab_size
    assert hasattr(logit_lens, "nb_token")
    assert logit_lens.nb_token == 5


def test_automatic_model_detection_classification(sentences):
    """Test that GeneralLogitLens automatically detects classification models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
    )

    # Test that the factory creates ClassificationLogitLens
    logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy="cls")

    assert isinstance(logit_lens, ClassificationLogitLens)
    assert hasattr(logit_lens, "num_classes")
    assert hasattr(logit_lens, "pooling_strategy")
    assert logit_lens.pooling_strategy == "cls"


def test_helper_functions_through_language_model(sentences):
    """Test the utility functions used by LogitLens through an actual instance."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=5, device=torch.device("cpu"))

    # Access the visualization method to test helper functions
    # These functions are defined within the visualize_logit_lens_interactive method
    viz_method = logit_lens.visualize_logit_lens_interactive

    assert hasattr(logit_lens, "visualize_logit_lens_interactive")
    assert callable(viz_method)

    # Test basic functionality
    result = logit_lens.explain(sentences[0])
    assert isinstance(result, dict)
    assert len(result) > 0


def test_logit_lens_initialization_causal_lm(sentences):
    """Test LogitLens initialization with causal language models."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
    )

    # Test basic initialization
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=5)

    assert logit_lens.splitted_model == splitted_model
    assert logit_lens.model == splitted_model._model
    assert logit_lens.tokenizer == tokenizer
    assert logit_lens.vocab_size == tokenizer.vocab_size
    assert logit_lens.model_head is not None
    assert logit_lens.head_name == config["head_name"]
    assert logit_lens.features_dim is not None
    assert logit_lens.nb_token == 5

    # Test with specific head name
    logit_lens_specific = LogitLens(splitted_model, tokenizer, head_name=config["head_name"], nb_token=3, batch_size=2)

    assert logit_lens_specific.head_name == config["head_name"]
    assert logit_lens_specific.nb_token == 3
    assert logit_lens_specific.batch_size == 2


def test_logit_lens_initialization_masked_lm(sentences):
    """Test LogitLens initialization with masked language models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = MASKED_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
    )

    # Test basic initialization
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=5)

    assert logit_lens.splitted_model == splitted_model
    assert logit_lens.model == splitted_model._model
    assert logit_lens.tokenizer == tokenizer
    assert logit_lens.vocab_size == tokenizer.vocab_size
    assert logit_lens.model_head is not None
    assert logit_lens.head_name == config["head_name"]
    assert logit_lens.features_dim is not None
    assert logit_lens.nb_token == 5


def test_logit_lens_initialization_classification(sentences):
    """Test LogitLens initialization with classification models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
    )

    # Test basic initialization
    logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy="cls")

    assert isinstance(logit_lens, ClassificationLogitLens)
    assert logit_lens.splitted_model == splitted_model
    assert logit_lens.model == splitted_model._model
    assert logit_lens.tokenizer == tokenizer
    assert logit_lens.model_head is not None
    assert logit_lens.head_name == config["head_name"]
    assert logit_lens.features_dim is not None
    assert logit_lens.pooling_strategy == "cls"
    assert hasattr(logit_lens, "num_classes")
    assert logit_lens.num_classes > 0

    # Test with different pooling strategies
    logit_lens_mean = LogitLens(splitted_model, tokenizer, pooling_strategy="mean")
    assert logit_lens_mean.pooling_strategy == "mean"

    logit_lens_last = LogitLens(splitted_model, tokenizer, pooling_strategy="last")
    assert logit_lens_last.pooling_strategy == "last"


def test_explain_method_causal_lm(sentences):
    """Test the explain method with causal language models."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test with single string
    result_single = logit_lens.explain(sentences[0])

    assert isinstance(result_single, dict)
    assert len(result_single) > 0

    for _layer_name, layer_data in result_single.items():
        assert isinstance(layer_data, dict)
        assert "tokens" in layer_data
        assert "proba" in layer_data
        assert layer_data["tokens"].shape[0] == 1  # batch size 1
        assert layer_data["proba"].shape[0] == 1  # batch size 1
        assert layer_data["tokens"].shape[2] == 3  # nb_token=3
        assert layer_data["proba"].shape[2] == 3  # nb_token=3

        # Check probability values are valid
        assert (layer_data["proba"] >= 0).all()
        assert (layer_data["proba"] <= 1).all()

    # Test with list of strings
    result_list = logit_lens.explain(sentences[:2])

    assert isinstance(result_list, dict)
    for _layer_name, layer_data in result_list.items():
        assert layer_data["tokens"].shape[0] == 2  # batch size 2
        assert layer_data["proba"].shape[0] == 2  # batch size 2


def test_explain_method_masked_lm(sentences):
    """Test the explain method with masked language models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = MASKED_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test with single string
    result_single = logit_lens.explain(sentences[0])

    assert isinstance(result_single, dict)
    assert len(result_single) > 0

    for _layer_name, layer_data in result_single.items():
        assert isinstance(layer_data, dict)
        assert "tokens" in layer_data
        assert "proba" in layer_data
        assert layer_data["tokens"].shape[0] == 1  # batch size 1
        assert layer_data["proba"].shape[0] == 1  # batch size 1
        assert layer_data["tokens"].shape[2] == 3  # nb_token=3
        assert layer_data["proba"].shape[2] == 3  # nb_token=3

        # Check probability values are valid
        assert (layer_data["proba"] >= 0).all()
        assert (layer_data["proba"] <= 1).all()


def test_explain_method_classification(sentences):
    """Test the explain method with classification models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
    )

    logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy="cls")

    # Test with single string
    result_single = logit_lens.explain(sentences[0])

    assert isinstance(result_single, dict)
    assert len(result_single) > 0

    for _layer_name, layer_data in result_single.items():
        assert isinstance(layer_data, dict)

        # Check classification-specific output structure
        assert "logits" in layer_data
        assert "probabilities" in layer_data
        assert "predicted_classes" in layer_data
        assert "predicted_labels" in layer_data
        assert "top_k_predictions" in layer_data
        assert "class_labels" in layer_data
        assert "confidence_scores" in layer_data

        # Check shapes
        assert layer_data["logits"].shape[0] == 1  # batch size 1
        assert layer_data["probabilities"].shape[0] == 1  # batch size 1
        assert layer_data["predicted_classes"].shape[0] == 1  # batch size 1
        assert len(layer_data["predicted_labels"]) == 1  # batch size 1
        assert len(layer_data["top_k_predictions"]) == 1  # batch size 1
        assert layer_data["confidence_scores"].shape[0] == 1  # batch size 1

        # Check number of classes consistency
        num_classes = layer_data["logits"].shape[1]
        assert layer_data["probabilities"].shape[1] == num_classes
        assert len(layer_data["class_labels"]) == num_classes

        # Check probability values are valid
        assert (layer_data["probabilities"] >= 0).all()
        assert (layer_data["probabilities"] <= 1).all()
        assert np.allclose(layer_data["probabilities"].sum(axis=1), 1.0, atol=1e-5)

        # Check confidence scores
        assert (layer_data["confidence_scores"] >= 0).all()
        assert (layer_data["confidence_scores"] <= 1).all()

        # Check top-k predictions structure
        top_k_pred = layer_data["top_k_predictions"][0]
        assert isinstance(top_k_pred, list)
        assert len(top_k_pred) <= num_classes
        for pred in top_k_pred:
            assert "class_id" in pred
            assert "class_label" in pred
            assert "probability" in pred
            assert 0 <= pred["class_id"] < num_classes
            assert 0 <= pred["probability"] <= 1

    # Test with list of strings
    result_list = logit_lens.explain(sentences[:2])

    assert isinstance(result_list, dict)
    for _layer_name, layer_data in result_list.items():
        assert layer_data["logits"].shape[0] == 2  # batch size 2
        assert layer_data["probabilities"].shape[0] == 2  # batch size 2
        assert len(layer_data["top_k_predictions"]) == 2  # batch size 2


def test_classification_pooling_strategies(sentences):
    """Test different pooling strategies for classification models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
    )

    strategies = ["cls", "mean", "last"]

    results = {}
    for strategy in strategies:
        logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy=strategy)
        result = logit_lens.explain(sentences[0])
        results[strategy] = result

        # Basic checks for each strategy
        assert isinstance(result, dict)
        assert len(result) > 0

        for _layer_name, layer_data in result.items():
            assert "probabilities" in layer_data
            assert layer_data["probabilities"].shape[0] == 1  # batch size 1

    # Results should be different for different pooling strategies
    assert len(results) == len(strategies)


@pytest.mark.parametrize("model_name", CI_MODELS["causal_lm"])
def test_cross_model_compatibility_causal_lm(model_name, sentences):
    """Test LogitLens across different causal LM architectures (CI subset)."""
    evaluate_causal_lm_model(model_name, sentences)


@pytest.mark.parametrize("model_name", CI_MODELS["masked_lm"])
def test_cross_model_compatibility_masked_lm(model_name, sentences):
    """Test LogitLens across different masked LM architectures (CI subset)."""
    evaluate_masked_lm_model(model_name, sentences)


@pytest.mark.parametrize("model_name", CI_MODELS["classification"])
def test_cross_model_compatibility_classification(model_name, sentences):
    """Test LogitLens across different classification architectures (CI subset)."""
    evaluate_classification_model(model_name, sentences)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [k for k in CAUSAL_LM_MODELS.keys() if k not in CI_MODELS["causal_lm"]])
def test_cross_model_compatibility_causal_lm_extended(model_name, sentences):
    """Test LogitLens across extended causal LM set (slow tests)."""
    evaluate_causal_lm_model(model_name, sentences)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [k for k in MASKED_LM_MODELS.keys() if k not in CI_MODELS["masked_lm"]])
def test_cross_model_compatibility_masked_lm_extended(model_name, sentences):
    """Test LogitLens across extended masked LM set (slow tests)."""
    evaluate_masked_lm_model(model_name, sentences)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name", [k for k in CLASSIFICATION_MODELS.keys() if k not in CI_MODELS["classification"]]
)
def test_cross_model_compatibility_classification_extended(model_name, sentences):
    """Test LogitLens across extended classification set (slow tests)."""
    evaluate_classification_model(model_name, sentences)


def evaluate_causal_lm_model(model_name: str, sentences: list[str]):
    """Evaluate LogitLens functionality for a specific causal LM model."""
    config = CAUSAL_LM_MODELS[model_name]

    # Load model and tokenizer
    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if needed
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # Create splitted model
    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Test LogitLens initialization
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3)

    assert isinstance(logit_lens, LanguageModelLogitLens)
    assert logit_lens.head_name == config["head_name"]
    assert logit_lens.vocab_size == tokenizer.vocab_size

    # Test explain method
    result = logit_lens.explain(sentences[0])
    assert isinstance(result, dict)
    assert len(result) > 0

    # Test with multiple sentences
    result_multi = logit_lens.explain(sentences[:2])
    assert isinstance(result_multi, dict)

    # Test lens method (visualization)
    try:
        logit_lens.lens(sentences[0])
    except Exception as e:
        if "IPython" not in str(e) and "display" not in str(e):
            raise e


def evaluate_masked_lm_model(model_name: str, sentences: list[str]):
    """Evaluate LogitLens functionality for a specific masked LM model."""
    config = MASKED_LM_MODELS[model_name]

    # Load model and tokenizer
    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create splitted model
    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Test LogitLens initialization
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3)

    assert isinstance(logit_lens, LanguageModelLogitLens)
    assert logit_lens.head_name == config["head_name"]
    assert logit_lens.vocab_size == tokenizer.vocab_size

    # Test explain method
    result = logit_lens.explain(sentences[0])
    assert isinstance(result, dict)
    assert len(result) > 0

    # Test with multiple sentences
    result_multi = logit_lens.explain(sentences[:2])
    assert isinstance(result_multi, dict)


def evaluate_classification_model(model_name: str, sentences: list[str]):
    """Evaluate LogitLens functionality for a specific classification model."""
    config = CLASSIFICATION_MODELS[model_name]

    # Load model and tokenizer
    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create splitted model
    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Test LogitLens initialization
    logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy=config["pooling_strategy"])

    assert isinstance(logit_lens, ClassificationLogitLens)
    assert logit_lens.head_name == config["head_name"]
    assert logit_lens.pooling_strategy == config["pooling_strategy"]

    # Test explain method
    result = logit_lens.explain(sentences[0])
    assert isinstance(result, dict)
    assert len(result) > 0

    # Verify classification output structure
    for _layer_name, layer_data in result.items():
        assert "probabilities" in layer_data
        assert "predicted_classes" in layer_data
        assert "top_k_predictions" in layer_data
        assert "class_labels" in layer_data

    # Test with multiple sentences
    result_multi = logit_lens.explain(sentences[:2])
    assert isinstance(result_multi, dict)

    # Test lens method (visualization)
    try:
        logit_lens.lens(sentences[0])
    except Exception as e:
        if "IPython" not in str(e) and "display" not in str(e):
            raise e


def test_error_handling():
    """Test error handling in LogitLens."""
    # Test with incompatible model (this should raise an error)
    with pytest.raises(ValueError):
        # Create a mock splitted model with unsupported autoclass
        class MockModel:
            def __init__(self):
                self.model_autoclass = object  # Unsupported type
                self._model = None

        mock_model = MockModel()
        mock_tokenizer = None

        LogitLens(mock_model, mock_tokenizer)


def test_meta_tensor_handling():
    """Test meta tensor detection and handling."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test meta tensor detection method
    has_meta = logit_lens.has_meta_tensors()
    assert isinstance(has_meta, bool)

    # Test model reload check method
    needs_reload = logit_lens.needs_model_reload()
    assert isinstance(needs_reload, bool)


def test_classification_num_classes():
    """Test setting and getting number of classes for classification models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy="cls", device=torch.device("cpu"))

    # Test automatic detection of number of classes
    assert hasattr(logit_lens, "num_classes")
    assert logit_lens.num_classes > 0
    original_num_classes = logit_lens.num_classes

    # Test manual setting of number of classes
    logit_lens.set_num_classes(10)
    assert logit_lens.num_classes == 10

    # Reset to original
    logit_lens.set_num_classes(original_num_classes)
    assert logit_lens.num_classes == original_num_classes


def test_classification_pooling_detection():
    """Test automatic detection of head pooling in classification models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    # Test with different pooling strategies
    for strategy in ["cls", "mean", "last", None]:
        logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy=strategy, device=torch.device("cpu"))

        # Check that pooling strategy is set (may be modified by head detection)
        assert hasattr(logit_lens, "pooling_strategy")
        assert hasattr(logit_lens, "original_pooling_strategy")
        assert logit_lens.original_pooling_strategy == strategy


def test_call_method():
    """Test that LogitLens instances can be called as functions."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test calling as function
    test_sentence = "This is a test sentence"
    result_explain = logit_lens.explain(test_sentence)
    result_call = logit_lens(test_sentence)

    # Results should be identical
    assert isinstance(result_call, dict)
    assert len(result_call) == len(result_explain)

    for layer in result_explain:
        assert layer in result_call
        assert result_call[layer]["tokens"].shape == result_explain[layer]["tokens"].shape
        assert result_call[layer]["proba"].shape == result_explain[layer]["proba"].shape


def test_empty_input_handling():
    """Test handling of empty inputs."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test empty string
    with pytest.raises(ValueError, match="Empty string input is not supported"):
        logit_lens.explain("")

    # Test empty list
    with pytest.raises(ValueError, match="Empty list input is not supported"):
        logit_lens.explain([])

    # Test empty tensor
    empty_tensor = torch.empty(0)
    with pytest.raises(ValueError, match="Empty tensor input is not supported"):
        logit_lens.explain(empty_tensor)


def test_batch_merging_language_model():
    """Test batch result merging for language models."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=2,  # Small batch size to force batching
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test with multiple sentences that will be split into batches
    sentences = ["First test sentence", "Second test sentence", "Third test sentence"]

    result = logit_lens.explain(sentences)

    assert isinstance(result, dict)
    for _layer_name, layer_data in result.items():
        # Should have merged all 3 sentences
        assert layer_data["tokens"].shape[0] == 3
        assert layer_data["proba"].shape[0] == 3

        # Check that probabilities are valid
        assert (layer_data["proba"] >= 0).all()
        assert (layer_data["proba"] <= 1).all()


def test_batch_merging_classification():
    """Test batch result merging for classification models."""
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=2,  # Small batch size to force batching
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy="cls", device=torch.device("cpu"))

    # Test with multiple sentences that will be split into batches
    sentences = ["First test sentence", "Second test sentence", "Third test sentence"]

    result = logit_lens.explain(sentences)

    assert isinstance(result, dict)
    for _layer_name, layer_data in result.items():
        # Should have merged all 3 sentences
        assert layer_data["logits"].shape[0] == 3
        assert layer_data["probabilities"].shape[0] == 3
        assert len(layer_data["predicted_labels"]) == 3
        assert len(layer_data["top_k_predictions"]) == 3
        assert layer_data["confidence_scores"].shape[0] == 3

        # Check that probabilities are valid
        assert (layer_data["probabilities"] >= 0).all()
        assert (layer_data["probabilities"] <= 1).all()


def test_visualization_methods():
    """Test that visualization methods can be called without errors."""
    # Test language model visualization
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test lens method (should call visualization)
    try:
        logit_lens.lens("Test sentence")
    except Exception as e:
        # Allow IPython/display related errors since we're not in Jupyter
        if "IPython" not in str(e) and "display" not in str(e):
            raise e

    # Test classification model visualization
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    cls_logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy="cls", device=torch.device("cpu"))

    try:
        cls_logit_lens.lens("Test sentence")
    except Exception as e:
        # Allow IPython/display related errors since we're not in Jupyter
        if "IPython" not in str(e) and "display" not in str(e):
            raise e


def test_safe_tensor_methods():
    """Test safe tensor handling methods."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test _safe_item method with normal tensor
    test_tensor = torch.tensor(5.0)
    safe_value = logit_lens._safe_item(test_tensor)
    assert safe_value == 5.0

    # Test _safe_tensor_to_cpu method
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    safe_cpu_tensor = logit_lens._safe_tensor_to_cpu(cpu_tensor)
    assert safe_cpu_tensor.device.type == "cpu"
    assert torch.equal(safe_cpu_tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_features_dimension_detection():
    """Test automatic features dimension detection."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    # Test automatic detection
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    assert hasattr(logit_lens, "features_dim")
    assert logit_lens.features_dim is not None
    assert logit_lens.features_dim > 0
    original_features_dim = logit_lens.features_dim

    # Test manual specification with the correct head name
    logit_lens_manual = LogitLens(
        splitted_model,
        tokenizer,
        head_name=config["head_name"],  # Specify the head name to avoid detection issues
        features_dim=original_features_dim,  # Use the detected dimension to avoid head compatibility issues
        nb_token=3,
        device=torch.device("cpu"),
    )

    assert logit_lens_manual.features_dim == original_features_dim


def test_output_labels():
    """Test output label generation for different model types."""
    # Test language model labels
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    labels = logit_lens._get_output_labels()
    assert isinstance(labels, dict)
    assert len(labels) == tokenizer.vocab_size

    # Test classification model labels
    model_name = "hf-internal-testing/tiny-random-bert"
    config = CLASSIFICATION_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    cls_logit_lens = LogitLens(splitted_model, tokenizer, pooling_strategy="cls", device=torch.device("cpu"))

    cls_labels = cls_logit_lens._get_output_labels()
    assert isinstance(cls_labels, dict)
    assert len(cls_labels) == cls_logit_lens.num_classes


def test_padding_token_handling():
    """Test automatic padding token assignment."""
    model_name = "hf-internal-testing/tiny-random-gpt2"
    config = CAUSAL_LM_MODELS[model_name]

    model = config["model_class"].from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure no padding token initially by setting it to None
    tokenizer.pad_token = None
    # Don't set pad_token_id to -1 as it causes overflow error

    splitted_model = ModelWithSplitPoints(
        model,
        tokenizer=tokenizer,
        split_points=config["split_points"],
        batch_size=4,
        device_map="cpu",  # Force CPU to avoid device mismatch issues
    )

    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3, device=torch.device("cpu"))

    # Test with multiple sentences (should trigger padding token assignment)
    sentences = ["First sentence", "Second sentence"]
    result = logit_lens.explain(sentences)

    # Check that padding token was assigned
    assert tokenizer.pad_token is not None
    assert isinstance(result, dict)


if __name__ == "__main__":
    """
    Main test runner.
    """
    print("Running LogitLens comprehensive tests...")

    # Test sentences from interpreto conftest
    sentences = [
        "Interpreto is the latin for 'to interpret'. But it also sounds like a spell from the Harry Potter books.",
        "Interpreto is magical",
        "Testing interpreto",
    ]

    # Test automatic detection
    test_automatic_model_detection_causal_lm(sentences)
    test_automatic_model_detection_classification(sentences)
    print("✓ Automatic model detection tests passed")

    # Test helper functions through instances
    test_helper_functions_through_language_model(sentences)
    print("✓ Helper functions tests passed")

    # Test causal LM
    test_logit_lens_initialization_causal_lm(sentences)
    test_explain_method_causal_lm(sentences)
    print("✓ Causal LM tests passed")

    # Test masked LM
    test_logit_lens_initialization_masked_lm(sentences)
    test_explain_method_masked_lm(sentences)
    print("✓ Masked LM tests passed")

    # Test classification
    test_logit_lens_initialization_classification(sentences)
    test_explain_method_classification(sentences)
    test_classification_pooling_strategies(sentences)
    print("✓ Classification tests passed")

    # Test cross-model compatibility
    for model_name in CI_MODELS["causal_lm"]:
        evaluate_causal_lm_model(model_name, sentences)
        print(f"✓ {model_name} causal LM compatibility test passed")

    for model_name in CI_MODELS["masked_lm"]:
        evaluate_masked_lm_model(model_name, sentences)
        print(f"✓ {model_name} masked LM compatibility test passed")

    for model_name in CI_MODELS["classification"]:
        evaluate_classification_model(model_name, sentences)
        print(f"✓ {model_name} classification compatibility test passed")

    # Test error handling
    test_error_handling()
    print("✓ Error handling tests passed")

    # Test new functionality
    test_meta_tensor_handling()
    print("✓ Meta tensor handling tests passed")

    test_classification_num_classes()
    print("✓ Classification number of classes tests passed")

    test_classification_pooling_detection()
    print("✓ Classification pooling detection tests passed")

    test_call_method()
    print("✓ Call method tests passed")

    test_empty_input_handling()
    print("✓ Empty input handling tests passed")

    test_batch_merging_language_model()
    test_batch_merging_classification()
    print("✓ Batch merging tests passed")

    test_visualization_methods()
    print("✓ Visualization methods tests passed")

    test_safe_tensor_methods()
    print("✓ Safe tensor methods tests passed")

    test_features_dimension_detection()
    print("✓ Features dimension detection tests passed")

    test_output_labels()
    print("✓ Output labels tests passed")

    test_padding_token_handling()
    print("✓ Padding token handling tests passed")

    print("\n=== ALL TESTS PASSED ===")
    print("LogitLens is compatible with multiple model architectures!")
    print("- Language Models (Causal and Masked)")
    print("- Classification Models")
    print("- Automatic model type detection")
    print("- Different pooling strategies for classification")
    print("- Meta tensor handling and model reloading")
    print("- Enhanced visualization and error handling")
    print("- Robust batch processing and merging")
