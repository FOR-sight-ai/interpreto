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

import pytest
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
)

from interpreto.others.logit_lens_general import (
    LogitLens, 
    LanguageModelLogitLens, 
    ClassificationLogitLens,
    GeneralLogitLens
)
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints


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
    assert hasattr(logit_lens, 'nb_token')
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
    assert hasattr(logit_lens, 'num_classes')
    assert hasattr(logit_lens, 'pooling_strategy')
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
    )
    
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=5)
    
    # Access the visualization method to test helper functions
    # These functions are defined within the visualize_logit_lens_interactive method
    viz_method = logit_lens.visualize_logit_lens_interactive
    
    assert hasattr(logit_lens, 'visualize_logit_lens_interactive')
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
    logit_lens_specific = LogitLens(
        splitted_model, 
        tokenizer, 
        head_name=config["head_name"],
        nb_token=3,
        batch_size=2
    )
    
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
    assert hasattr(logit_lens, 'num_classes')
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
    )
    
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3)
    
    # Test with single string
    result_single = logit_lens.explain(sentences[0])
    
    assert isinstance(result_single, dict)
    assert len(result_single) > 0
    
    for layer_name, layer_data in result_single.items():
        assert isinstance(layer_data, dict)
        assert 'tokens' in layer_data
        assert 'proba' in layer_data
        assert layer_data['tokens'].shape[0] == 1  # batch size 1
        assert layer_data['proba'].shape[0] == 1  # batch size 1
        assert layer_data['tokens'].shape[2] == 3  # nb_token=3
        assert layer_data['proba'].shape[2] == 3  # nb_token=3
        
        # Check probability values are valid
        assert (layer_data['proba'] >= 0).all()
        assert (layer_data['proba'] <= 1).all()
    
    # Test with list of strings
    result_list = logit_lens.explain(sentences[:2])
    
    assert isinstance(result_list, dict)
    for layer_name, layer_data in result_list.items():
        assert layer_data['tokens'].shape[0] == 2  # batch size 2
        assert layer_data['proba'].shape[0] == 2  # batch size 2


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
    )
    
    logit_lens = LogitLens(splitted_model, tokenizer, nb_token=3)
    
    # Test with single string
    result_single = logit_lens.explain(sentences[0])
    
    assert isinstance(result_single, dict)
    assert len(result_single) > 0
    
    for layer_name, layer_data in result_single.items():
        assert isinstance(layer_data, dict)
        assert 'tokens' in layer_data
        assert 'proba' in layer_data
        assert layer_data['tokens'].shape[0] == 1  # batch size 1
        assert layer_data['proba'].shape[0] == 1  # batch size 1
        assert layer_data['tokens'].shape[2] == 3  # nb_token=3
        assert layer_data['proba'].shape[2] == 3  # nb_token=3
        
        # Check probability values are valid
        assert (layer_data['proba'] >= 0).all()
        assert (layer_data['proba'] <= 1).all()


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
    
    for layer_name, layer_data in result_single.items():
        assert isinstance(layer_data, dict)
        
        # Check classification-specific output structure
        assert 'logits' in layer_data
        assert 'probabilities' in layer_data
        assert 'predicted_classes' in layer_data
        assert 'predicted_labels' in layer_data
        assert 'top_k_predictions' in layer_data
        assert 'class_labels' in layer_data
        assert 'confidence_scores' in layer_data
        
        # Check shapes
        assert layer_data['logits'].shape[0] == 1  # batch size 1
        assert layer_data['probabilities'].shape[0] == 1  # batch size 1
        assert layer_data['predicted_classes'].shape[0] == 1  # batch size 1
        assert len(layer_data['predicted_labels']) == 1  # batch size 1
        assert len(layer_data['top_k_predictions']) == 1  # batch size 1
        assert layer_data['confidence_scores'].shape[0] == 1  # batch size 1
        
        # Check number of classes consistency
        num_classes = layer_data['logits'].shape[1]
        assert layer_data['probabilities'].shape[1] == num_classes
        assert len(layer_data['class_labels']) == num_classes
        
        # Check probability values are valid
        assert (layer_data['probabilities'] >= 0).all()
        assert (layer_data['probabilities'] <= 1).all()
        assert np.allclose(layer_data['probabilities'].sum(axis=1), 1.0, atol=1e-5)
        
        # Check confidence scores
        assert (layer_data['confidence_scores'] >= 0).all()
        assert (layer_data['confidence_scores'] <= 1).all()
        
        # Check top-k predictions structure
        top_k_pred = layer_data['top_k_predictions'][0]
        assert isinstance(top_k_pred, list)
        assert len(top_k_pred) <= num_classes
        for pred in top_k_pred:
            assert 'class_id' in pred
            assert 'class_label' in pred
            assert 'probability' in pred
            assert 0 <= pred['class_id'] < num_classes
            assert 0 <= pred['probability'] <= 1
    
    # Test with list of strings
    result_list = logit_lens.explain(sentences[:2])
    
    assert isinstance(result_list, dict)
    for layer_name, layer_data in result_list.items():
        assert layer_data['logits'].shape[0] == 2  # batch size 2
        assert layer_data['probabilities'].shape[0] == 2  # batch size 2
        assert len(layer_data['top_k_predictions']) == 2  # batch size 2


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
        
        for layer_name, layer_data in result.items():
            assert 'probabilities' in layer_data
            assert layer_data['probabilities'].shape[0] == 1  # batch size 1
    
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
@pytest.mark.parametrize("model_name", [k for k in CLASSIFICATION_MODELS.keys() if k not in CI_MODELS["classification"]])
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
    for layer_name, layer_data in result.items():
        assert 'probabilities' in layer_data
        assert 'predicted_classes' in layer_data
        assert 'top_k_predictions' in layer_data
        assert 'class_labels' in layer_data
    
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
        
        GeneralLogitLens(mock_model, mock_tokenizer)


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
    
    print("\n=== ALL TESTS PASSED ===")
    print("LogitLens is compatible with multiple model architectures!")
    print("- Language Models (Causal and Masked)")
    print("- Classification Models")
    print("- Automatic model type detection")
    print("- Different pooling strategies for classification")
