"""Session fixtures reserved for NNsight-backed concept tests.

NNsight instruments models in place. Override the root text model fixtures with
independent instances so instrumentation cannot leak into attribution tests,
which deep-copy embedding layers onto CPU. Tokenizers are independent too,
because splitters may change their padding configuration.
"""

import torch
from pytest import fixture
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

from interpreto.concepts.splitters.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import LatentActivations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@fixture(scope="session")
def bert_model():
    return AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert").to(DEVICE)


@fixture(scope="session")
def bert_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")


@fixture(scope="session")
def gpt2_model():
    return AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(DEVICE)


@fixture(scope="session")
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")


@fixture(scope="session")
def multi_splitter() -> ModelWithSplitPoints:
    return ModelWithSplitPoints(
        "hf-internal-testing/tiny-random-bert",
        split_point="bert.encoder.layer.1",
        automodel=AutoModelForMaskedLM,  # type: ignore
        batch_size=4,
    )


@fixture(scope="session")
def splitted_encoder_ml(bert_model, bert_tokenizer) -> ModelWithSplitPoints:
    """Reuse the concept-only BERT instance, isolated from attribution models."""
    return ModelWithSplitPoints(
        bert_model,
        split_point="bert.encoder.layer.1.output",
        tokenizer=bert_tokenizer,
        batch_size=4,
        device_map=DEVICE,
    )


@fixture(scope="session")
def activations(splitted_encoder_ml: ModelWithSplitPoints, sentences: list[str]) -> LatentActivations:
    latent_activations, _ = splitted_encoder_ml.get_activations(
        sentences, activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN
    )
    return latent_activations
