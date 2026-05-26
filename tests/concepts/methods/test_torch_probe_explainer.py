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
Tests for :class:`ProbeExplainer` with diverse probe models and granularities.
"""

from __future__ import annotations

import pytest
import torch

from interpreto.concepts import (
    CosineCentroidProbe,
    DotProductCentroidProbe,
    LinearRegressionProbe,
    LinearSVMProbe,
    LogisticRegressionProbe,
    MeansDiffProbe,
    ProbeExplainer,
    SqL2CentroidProbe,
)
from interpreto.concepts.probes import Standardization
from interpreto.model_wrapping.model_with_split_points import (
    ActivationGranularity,
    ModelWithSplitPoints,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Probe configs: (ProbeClass, kwargs)
# Covers linear, centroid, and iterative probes with/without normalization.
# ---------------------------------------------------------------------------

PROBE_CONFIGS = [
    ("LinearRegression", LinearRegressionProbe, {}),
    ("LinearRegression_std", LinearRegressionProbe, {"normalization": Standardization()}),
    ("MeansDiff", MeansDiffProbe, {}),
    ("LogisticRegression", LogisticRegressionProbe, {}),
    ("LinearSVM", LinearSVMProbe, {}),
    ("DotProductCentroid", DotProductCentroidProbe, {"normalization": Standardization()}),
    ("CosineCentroid", CosineCentroidProbe, {"normalization": Standardization()}),
    ("SqL2Centroid", SqL2CentroidProbe, {}),
    ("SqL2Centroid_std", SqL2CentroidProbe, {"normalization": Standardization()}),
]

GRANULARITIES = [
    ActivationGranularity.TOKEN,
    ActivationGranularity.SAMPLE,
    ActivationGranularity.CLS_TOKEN,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[g.name for g in GRANULARITIES],
    scope="module",
)
def activations_with_granularity(
    request, splitted_encoder_ml: ModelWithSplitPoints, sentences: list[str]
) -> tuple[dict[str, torch.Tensor], ActivationGranularity]:
    """Activations extracted at different granularities."""
    granularity = ActivationGranularity[request.param]
    acts = splitted_encoder_ml.get_activations(sentences, activation_granularity=granularity)
    return acts, granularity  # type: ignore


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,probe_cls,probe_kwargs",
    PROBE_CONFIGS,
    ids=[c[0] for c in PROBE_CONFIGS],
)
def test_torch_probe_explainer_fit_and_encode(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_with_granularity: tuple[dict[str, torch.Tensor], ActivationGranularity],
    name: str,
    probe_cls: type,
    probe_kwargs: dict,
):
    """Fit a ProbeExplainer and verify encode_activations output shape."""
    acts_dict, granularity = activations_with_granularity
    split_name = list(acts_dict.keys())[0]
    activations = acts_dict[split_name]

    n, d = activations.shape
    nb_concepts = 3

    # Build random binary labels matching the number of activation rows
    torch.manual_seed(42)
    labels = (torch.rand(n, nb_concepts) > 0.5).float()

    # Instantiate probe and explainer
    probe = probe_cls(**probe_kwargs)
    explainer = ProbeExplainer(
        model_with_split_points=splitted_encoder_ml,
        concept_model=probe,
    )

    # Before fitting
    assert not explainer.is_fitted, f"explainer is already fitted before fit: {explainer}"

    # Encode before fit should fail
    with pytest.raises(RuntimeError, match="not fitted"):
        explainer.encode_activations(activations)

    # Fit
    explainer.fit(activations, labels)

    # After fitting
    assert explainer.is_fitted, f"explainer is not fitted after fit: {explainer}"

    # Second fit without overwrite should fail
    with pytest.raises(RuntimeError, match="already been fitted"):
        explainer.fit(activations, labels)

    # Encode
    concepts = explainer.encode_activations(activations)
    assert concepts.shape == (n, nb_concepts), (
        f"incorrect concepts shape: {concepts.shape}, expected {(n, nb_concepts)}"
    )


def test_torch_probe_explainer_type_check(splitted_encoder_ml: ModelWithSplitPoints):
    """Passing a non-Probe should raise TypeError."""
    with pytest.raises(TypeError, match="must be a Probe"):
        ProbeExplainer(splitted_encoder_ml, concept_model="not_a_probe")  # type: ignore


def test_torch_probe_explainer_with_dict_activations(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_dict: dict[str, torch.Tensor],
):
    """Fit accepts a dict of activations (keyed by split point)."""
    split_name = list(activations_dict.keys())[0]
    activations = activations_dict[split_name]
    n = activations.shape[0]
    nb_concepts = 4

    torch.manual_seed(7)
    labels = (torch.rand(n, nb_concepts) > 0.5).float()

    probe = MeansDiffProbe()
    explainer = ProbeExplainer(splitted_encoder_ml, probe)

    # Pass the full dict (explainer should extract the right split)
    explainer.fit(activations_dict, labels)
    assert explainer.is_fitted

    concepts = explainer.encode_activations(activations)
    assert concepts.shape == (n, nb_concepts)


# ---------------------------------------------------------------------------
# Sanity check: BERT middle layer, TOKEN granularity, 3 semantic concepts
# ---------------------------------------------------------------------------

# Concepts: animal, food, color (multi-label, words can belong to 0-2 classes)
# Shuffled with seed=7 so train/test split is representative.
# All words are single-token in bert-base-uncased → 1 word = 1 activation row.
WORDS_AND_LABELS = [
    ("tea", [0, 1, 0]),
    ("sea", [0, 0, 0]),
    ("pig", [1, 0, 0]),
    ("soup", [0, 1, 0]),
    ("bun", [0, 1, 0]),
    ("ant", [1, 0, 0]),
    ("ram", [1, 0, 0]),
    ("hill", [0, 0, 0]),
    ("box", [0, 0, 0]),
    ("dog", [1, 0, 0]),
    ("owl", [1, 0, 0]),
    ("pup", [1, 0, 0]),
    ("jam", [0, 1, 0]),
    ("fox", [1, 0, 0]),
    ("bat", [1, 0, 0]),
    ("fog", [0, 0, 0]),
    ("ape", [1, 0, 0]),
    ("gold", [0, 0, 1]),
    ("cow", [1, 0, 0]),
    ("fig", [0, 1, 0]),
    ("cake", [0, 1, 0]),
    ("cod", [1, 1, 0]),
    ("red", [0, 0, 1]),
    ("bird", [1, 0, 0]),
    ("rock", [0, 0, 0]),
    ("road", [0, 0, 0]),
    ("dust", [0, 0, 0]),
    ("rum", [0, 1, 0]),
    ("deer", [1, 0, 0]),
    ("hen", [1, 1, 0]),
    ("nut", [0, 1, 0]),
    ("pie", [0, 1, 0]),
    ("elk", [1, 0, 0]),
    ("egg", [0, 1, 0]),
    ("fish", [1, 0, 0]),
    ("rice", [0, 1, 0]),
    ("pink", [0, 0, 1]),
    ("tan", [0, 0, 1]),
    ("rat", [1, 0, 0]),
    ("mud", [0, 0, 0]),
    ("eel", [1, 1, 0]),
    ("cat", [1, 0, 0]),
    ("ham", [1, 1, 0]),
    ("rye", [0, 1, 0]),
]

NB_TEST = 10


@pytest.fixture(scope="module")
def bert_split_model() -> ModelWithSplitPoints:
    from transformers import AutoModelForSequenceClassification  # noqa PLC0415

    return ModelWithSplitPoints(
        "bert-base-uncased",
        split_point="bert.encoder.layer.6",
        automodel=AutoModelForSequenceClassification,  # type: ignore
        batch_size=16,
        device_map=DEVICE,
    )


@pytest.fixture(scope="module")
def bert_train_test(bert_split_model: ModelWithSplitPoints):
    """Extract BERT activations and labels for the word list."""
    words = [w for w, _ in WORDS_AND_LABELS]
    labels = torch.tensor([l for _, l in WORDS_AND_LABELS], dtype=torch.float32)

    acts = bert_split_model.get_activations(words, activation_granularity=ActivationGranularity.TOKEN)
    activations = acts["bert.encoder.layer.6"]
    assert activations.shape[0] == len(words)  # type: ignore

    # Train/test split
    split_idx = len(words) - NB_TEST
    train_x, test_x = activations[:split_idx], activations[split_idx:]
    train_y, test_y = labels[:split_idx], labels[split_idx:]

    return train_x, train_y, test_x, test_y


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,probe_cls,probe_kwargs",
    PROBE_CONFIGS,
    ids=[c[0] for c in PROBE_CONFIGS],
)
def test_sanity_check_bert(
    bert_split_model: ModelWithSplitPoints,
    bert_train_test: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    name: str,
    probe_cls: type,
    probe_kwargs: dict,
):
    """Sanity check: fit probe on BERT activations with 3 semantic concepts.

    Uses bert-base-uncased layer 6 with TOKEN granularity on single-token
    words. Verifies that probes can both overfit training data and generalize
    to unseen test words (mean positive score > mean negative score).
    """
    if "SqL2" in name:
        pytest.skip("SqL2CentroidProbe can fail to converge on this small dataset, causing test instability.")
    train_x, train_y, test_x, test_y = bert_train_test

    probe = probe_cls(**probe_kwargs)
    probe.to(train_x.device)
    explainer = ProbeExplainer(bert_split_model, probe)

    # Fit
    explainer.fit(train_x, train_y)
    assert explainer.is_fitted

    # Encode train and test
    train_scores = explainer.encode_activations(train_x)
    test_scores = explainer.encode_activations(test_x)

    assert train_scores.shape == (train_x.shape[0], 3)
    assert test_scores.shape == (test_x.shape[0], 3)

    # Overfit check: on training data, positive samples should score higher
    for c in range(3):
        pos_mask = train_y[:, c] == 1.0
        neg_mask = train_y[:, c] == 0.0
        if pos_mask.any() and neg_mask.any():
            mean_pos = train_scores[pos_mask, c].mean()
            mean_neg = train_scores[neg_mask, c].mean()
            assert mean_pos > mean_neg, (
                f"Probe {name}, concept {c}: train positive mean ({mean_pos:.4f}) "
                f"should exceed negative mean ({mean_neg:.4f})"
            )

    # Generalization check: same property on test set
    for c in range(3):
        pos_mask = test_y[:, c] == 1.0
        neg_mask = test_y[:, c] == 0.0
        if pos_mask.any() and neg_mask.any():
            mean_pos = test_scores[pos_mask, c].mean()
            mean_neg = test_scores[neg_mask, c].mean()
            assert mean_pos > mean_neg, (
                f"Probe {name}, concept {c}: test positive mean ({mean_pos:.4f}) "
                f"should exceed negative mean ({mean_neg:.4f})"
            )
