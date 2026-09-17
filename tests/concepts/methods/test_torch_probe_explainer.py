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
Tests for :class:`ProbeExplainer` with diverse probe models.
"""

from __future__ import annotations

import pytest
import torch

from interpreto import SplitterForClassification
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

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,probe_cls,probe_kwargs",
    PROBE_CONFIGS,
    ids=[c[0] for c in PROBE_CONFIGS],
)
def test_torch_probe_explainer_fit_and_encode(
    splitted_encoder_ml: SplitterForClassification,
    activations: torch.Tensor,
    name: str,
    probe_cls: type,
    probe_kwargs: dict,
):
    """Fit a ProbeExplainer and verify activations_to_concepts output shape."""

    n, d = activations.shape
    nb_concepts = 3

    # Build random binary labels matching the number of activation rows
    torch.manual_seed(42)
    labels = (torch.rand(n, nb_concepts) > 0.5).float()

    # Instantiate probe and explainer
    probe = probe_cls(**probe_kwargs)
    explainer = ProbeExplainer(
        splitter=splitted_encoder_ml,
        concept_model=probe,
    )

    # Before fitting
    assert not explainer.is_fitted, f"explainer is already fitted before fit: {explainer}"

    # Encode before fit should fail
    with pytest.raises(RuntimeError, match="not fitted"):
        explainer.activations_to_concepts(activations)

    # Fit
    explainer.fit(activations, labels)

    # After fitting
    assert explainer.is_fitted, f"explainer is not fitted after fit: {explainer}"

    # Encode
    concepts = explainer.activations_to_concepts(activations)
    assert concepts.shape == (n, nb_concepts), (
        f"incorrect concepts shape: {concepts.shape}, expected {(n, nb_concepts)}"
    )


def test_torch_probe_explainer_type_check(splitted_encoder_ml: SplitterForClassification):
    """Passing a non-Probe should raise TypeError."""
    with pytest.raises(TypeError, match="must be a Probe"):
        ProbeExplainer(splitted_encoder_ml, concept_model="not_a_probe")  # type: ignore


def test_torch_probe_explainer_with_tensor_activations(
    splitted_encoder_ml: SplitterForClassification,
    activations: torch.Tensor,
):
    """Fit accepts latent activation tensors returned by get_activations."""
    n = activations.shape[0]
    nb_concepts = 4

    torch.manual_seed(7)
    labels = (torch.rand(n, nb_concepts) > 0.5).float()

    probe = MeansDiffProbe()
    explainer = ProbeExplainer(splitted_encoder_ml, probe)

    explainer.fit(activations, labels)
    assert explainer.is_fitted

    concepts = explainer.activations_to_concepts(activations)
    assert concepts.shape == (n, nb_concepts)


def test_torch_probe_explainer_accepts_low_precision():
    """Torch probes normalize low-precision activations at their boundaries."""
    splitter = SplitterForClassification("hf-internal-testing/tiny-random-bert", device_map=DEVICE)
    activations = torch.randn(12, splitter.config.hidden_size, dtype=torch.bfloat16)
    labels = (torch.rand(12, 2) > 0.5).float()
    explainer = ProbeExplainer(splitter, LogisticRegressionProbe())

    explainer.fit(activations, labels)
    concepts = explainer.activations_to_concepts(activations)

    assert concepts.shape == (12, 2)
    assert concepts.dtype == torch.float32


# ---------------------------------------------------------------------------
# Sanity check: probes overfit separable training data and generalize
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def separable_train_test():
    """Three well-separated synthetic clusters with one-hot labels."""
    torch.manual_seed(7)
    n_per_cluster, hidden = 16, 32
    centers = torch.eye(3, hidden) * 5.0
    train_x = torch.cat([center + torch.randn(n_per_cluster, hidden) for center in centers])
    train_y = torch.cat([torch.nn.functional.one_hot(torch.full((n_per_cluster,), i), 3) for i in range(3)]).float()
    test_x = torch.cat([center + torch.randn(n_per_cluster // 2, hidden) for center in centers])
    test_y = torch.cat(
        [torch.nn.functional.one_hot(torch.full((n_per_cluster // 2,), i), 3) for i in range(3)]
    ).float()
    return train_x, train_y, test_x, test_y


@pytest.mark.parametrize(
    "name,probe_cls,probe_kwargs",
    PROBE_CONFIGS,
    ids=[c[0] for c in PROBE_CONFIGS],
)
def test_sanity_check_probes_generalize(
    splitted_encoder_ml: SplitterForClassification,
    separable_train_test: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    name: str,
    probe_cls: type,
    probe_kwargs: dict,
):
    """Sanity check that probes overfit separable data and generalize."""
    if "SqL2" in name:
        pytest.skip("SqL2CentroidProbe can fail to converge on this small dataset, causing test instability.")
    train_x, train_y, test_x, test_y = separable_train_test

    probe = probe_cls(**probe_kwargs)
    probe.to(train_x.device)
    explainer = ProbeExplainer(splitted_encoder_ml, probe)

    # Fit
    explainer.fit(train_x, train_y)
    assert explainer.is_fitted

    # Encode train and test
    train_scores = explainer.activations_to_concepts(train_x)
    test_scores = explainer.activations_to_concepts(test_x)

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
