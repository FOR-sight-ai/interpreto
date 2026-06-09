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
Tests for :class:`SklearnProbeExplainer` with diverse sklearn classifiers.

Covers RidgeClassifier, SVC, and LinearDiscriminantAnalysis — all of which
expose ``decision_function`` as required by :class:`SklearnProbe`.
"""

from __future__ import annotations

import pytest
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from interpreto.concepts.probes.sklearn import SklearnProbe, SklearnProbeExplainer
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints

# ---------------------------------------------------------------------------
# Sklearn classifier configs: (name, sklearn_class, sklearn_kwargs)
# All classifiers must expose decision_function.
# ---------------------------------------------------------------------------

SKLEARN_CONFIGS = [
    ("RidgeClassifier", RidgeClassifier, {}),
    ("SVC", SVC, {"kernel": "linear"}),
    ("LDA", LinearDiscriminantAnalysis, {}),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,sklearn_class,sklearn_kwargs",
    SKLEARN_CONFIGS,
    ids=[c[0] for c in SKLEARN_CONFIGS],
)
def test_sklearn_probe_explainer_fit_and_encode(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_dict: dict[str, torch.Tensor],
    name: str,
    sklearn_class: type,
    sklearn_kwargs: dict,
):
    """Fit a SklearnProbeExplainer and verify encode_activations output shape."""
    split_name = list(activations_dict.keys())[0]
    activations = activations_dict[split_name]
    n = activations.shape[0]

    # Binary labels for single concept
    torch.manual_seed(42)
    labels = (torch.rand(n) > 0.5).float()

    explainer = SklearnProbeExplainer(
        model_with_split_points=splitted_encoder_ml,
        sklearn_class=sklearn_class,
        sklearn_kwargs=sklearn_kwargs,
    )

    # Before fitting
    assert not explainer.is_fitted

    # Fit
    explainer.fit(activations, labels)

    # After fitting
    assert explainer.is_fitted

    # Encode — SklearnProbe always has nb_concepts=1
    concepts = explainer.encode_activations(activations)
    assert concepts.shape == (n, 1), f"Expected shape ({n}, 1), got {concepts.shape}"


@pytest.mark.parametrize(
    "name,sklearn_class,sklearn_kwargs",
    SKLEARN_CONFIGS,
    ids=[c[0] for c in SKLEARN_CONFIGS],
)
def test_sklearn_probe_explainer_encode_before_fit(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_dict: dict[str, torch.Tensor],
    name: str,
    sklearn_class: type,
    sklearn_kwargs: dict,
):
    """Encoding before fitting should raise RuntimeError."""
    split_name = list(activations_dict.keys())[0]
    activations = activations_dict[split_name]

    explainer = SklearnProbeExplainer(
        model_with_split_points=splitted_encoder_ml,
        sklearn_class=sklearn_class,
        sklearn_kwargs=sklearn_kwargs,
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        explainer.encode_activations(activations)


def test_sklearn_probe_explainer_with_dict_activations(
    splitted_encoder_ml: ModelWithSplitPoints,
    activations_dict: dict[str, torch.Tensor],
):
    """Fit accepts a dict of activations (keyed by split point)."""
    split_name = list(activations_dict.keys())[0]
    activations = activations_dict[split_name]
    n = activations.shape[0]

    torch.manual_seed(7)
    labels = (torch.rand(n) > 0.5).float()

    explainer = SklearnProbeExplainer(
        model_with_split_points=splitted_encoder_ml,
        sklearn_class=SVC,
        sklearn_kwargs={"kernel": "linear"},
    )

    # Pass the full dict — explainer should extract the right split
    explainer.fit(activations_dict, labels)
    assert explainer.is_fitted

    concepts = explainer.encode_activations(activations)
    assert concepts.shape == (n, 1)


@pytest.mark.parametrize(
    "name,sklearn_class,sklearn_kwargs",
    SKLEARN_CONFIGS,
    ids=[c[0] for c in SKLEARN_CONFIGS],
)
def test_sklearn_probe_explainer_separation(
    name: str,
    sklearn_class: type,
    sklearn_kwargs: dict,
):
    """On linearly separable data, positive scores should exceed negative scores."""
    torch.manual_seed(0)
    n, d = 100, 16

    # Create two well-separated clusters
    pos_data = torch.randn(n // 2, d) + 2.0
    neg_data = torch.randn(n // 2, d) - 2.0
    X = torch.cat([pos_data, neg_data], dim=0)
    y = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])

    # Shuffle
    perm = torch.randperm(n)
    X, y = X[perm], y[perm]

    # Fit directly on the SklearnProbe (no need for ModelWithSplitPoints)
    probe = SklearnProbe(sklearn_class, sklearn_kwargs)
    probe.fit(X, y)

    scores = probe.encode(X)  # (n, 1)
    assert scores.shape == (n, 1)

    pos_mask = y == 1.0
    neg_mask = y == 0.0
    mean_pos = scores[pos_mask, 0].mean()
    mean_neg = scores[neg_mask, 0].mean()

    assert mean_pos > mean_neg, f"{name}: positive mean ({mean_pos:.4f}) should exceed negative mean ({mean_neg:.4f})"
