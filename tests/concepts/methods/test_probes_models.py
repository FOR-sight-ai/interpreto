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

from functools import partial

import pytest
import torch

# Bias calibrators
from interpreto.concepts.probes.bias_calibrators import (
    bce_bias,
    fpr_bias,
    lda_shared_var_bias,
    midpoint_bias,
    prevalence_bias,
)

# Centroid probes
from interpreto.concepts.probes.centroid import (
    CosineCentroidProbe,
    DiagonalMahalanobisCentroidProbe,
    DotProductCentroidProbe,
    SqL2CentroidProbe,
    SVDDCentroidProbe,
)

# Linear probes
from interpreto.concepts.probes.linear import (
    LinearRegressionProbe,
    LinearSVMProbe,
    LogisticRegressionProbe,
    MeansDiffProbe,
)

# Normalizations
from interpreto.concepts.probes.normalizations import (
    Standardization,
    Whitening,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic_dataset(c: int, d: int, nc: int, seed: int):
    """
    Synthetic multi-class / multi-label dataset (one-vs-rest targets).

    - c classes, d-dimensional data
    - nc samples per class for train, nc per class for test
    - centroid per class in {-1, 1}^d
    - samples ~ N(centroid, I)
    - labels: (n_samples, c) with 1 for the class, 0 for others
    """
    g = torch.Generator().manual_seed(seed)

    centroids = (torch.randint(0, 2, (c, d), generator=g) * 2 - 1).float()  # (c, d)

    n_train = c * nc
    n_test = c * nc

    X_train = torch.empty(n_train, d)
    X_test = torch.empty(n_test, d)
    y_train = torch.zeros(n_train, c)
    y_test = torch.zeros(n_test, c)
    train_labels = torch.empty(n_train, dtype=torch.long)
    test_labels = torch.empty(n_test, dtype=torch.long)

    idx_train = 0
    idx_test = 0
    for class_id in range(c):
        mu = centroids[class_id]  # (d,)
        X_train[idx_train : idx_train + nc] = mu + torch.randn(nc, d, generator=g)
        y_train[idx_train : idx_train + nc, class_id] = 1.0
        train_labels[idx_train : idx_train + nc] = class_id

        X_test[idx_test : idx_test + nc] = mu + torch.randn(nc, d, generator=g)
        y_test[idx_test : idx_test + nc, class_id] = 1.0
        test_labels[idx_test : idx_test + nc] = class_id

        idx_train += nc
        idx_test += nc

    # Shuffle train/test independently
    perm = torch.randperm(n_train, generator=g)
    X_train, y_train, train_labels = X_train[perm], y_train[perm], train_labels[perm]

    perm = torch.randperm(n_test, generator=g)
    X_test, y_test, test_labels = X_test[perm], y_test[perm], test_labels[perm]

    return X_train, y_train, train_labels, X_test, y_test, test_labels, centroids


# =============================================================================
# PROBE CONFIGURATIONS
# =============================================================================
# Each entry: (ProbeClass, {params})
#
# Pertinent hyperparameters per probe:
#
# LinearRegressionProbe:
#   - l2: float (ridge regularization, 0.0 = OLS)
#   - bias_calibrator: BiasCalibrator | None
#   - normalization: NormalizationBase | None
#
# MeansDiffProbe:
#   - bias_calibrator: BiasCalibrator | None
#   - normalization: NormalizationBase | None
#
# LogisticRegressionProbe / LinearSVMProbe:
#   - lr: float (learning rate)
#   - max_iter: int
#   - l2: float (weight decay)
#   - init_from_means_diff: bool
#   - init_bias_calibrator: BiasCalibrator | None
#   - normalization: NormalizationBase | None
#
# DotProductCentroidProbe / CosineCentroidProbe / SqL2CentroidProbe:
#   - normalization: NormalizationBase | None
#   - bias_calibrator: BiasCalibrator | None
#
# DiagonalMahalanobisCentroidProbe:
#   - normalization: NormalizationBase | None (default: Standardization)
#   - bias_calibrator: BiasCalibrator | None
#   - shrinkage: float in [0, 1] (0=classwise var, 1=global var)
#
# SVDDCentroidProbe:
#   - lr: float
#   - max_iter: int
#   - C: float (hinge loss weight)
#   - l2: float (center regularization)
#   - normalization: NormalizationBase | None
# =============================================================================

PROBE_CONFIGS = {
    # -------------------------------------------------------------------------
    # Linear Regression Probe
    # -------------------------------------------------------------------------
    "LinearRegression": (LinearRegressionProbe, {}),
    "LinearRegression_l2": (LinearRegressionProbe, {"l2": 1e-2}),
    "LinearRegression_standardization": (
        LinearRegressionProbe,
        {"normalization": Standardization()},
    ),
    "LinearRegression_l2_standardization": (
        LinearRegressionProbe,
        {"l2": 1e-2, "normalization": Standardization()},
    ),
    "LinearRegression_midpoint_bias": (
        LinearRegressionProbe,
        {"bias_calibrator": midpoint_bias},
    ),
    # -------------------------------------------------------------------------
    # MeansDiff Probe
    # -------------------------------------------------------------------------
    "MeansDiff": (MeansDiffProbe, {}),
    "MeansDiff_midpoint": (MeansDiffProbe, {"bias_calibrator": midpoint_bias}),
    "MeansDiff_prevalence": (MeansDiffProbe, {"bias_calibrator": prevalence_bias}),
    "MeansDiff_bce": (MeansDiffProbe, {"bias_calibrator": bce_bias}),
    "MeansDiff_lda": (MeansDiffProbe, {"bias_calibrator": lda_shared_var_bias}),
    "MeansDiff_fpr": (MeansDiffProbe, {"bias_calibrator": partial(fpr_bias, target_fpr=0.05)}),
    "MeansDiff_standardization": (MeansDiffProbe, {"normalization": Standardization()}),
    "MeansDiff_standardization_midpoint": (
        MeansDiffProbe,
        {"normalization": Standardization(), "bias_calibrator": midpoint_bias},
    ),
    # -------------------------------------------------------------------------
    # Logistic Regression Probe
    # -------------------------------------------------------------------------
    "LogisticRegression": (LogisticRegressionProbe, {}),
    "LogisticRegression_no_init": (
        LogisticRegressionProbe,
        {"init_from_means_diff": False},
    ),
    "LogisticRegression_l2": (LogisticRegressionProbe, {"l2": 1e-2}),
    "LogisticRegression_standardization": (
        LogisticRegressionProbe,
        {"normalization": Standardization()},
    ),
    "LogisticRegression_l2_standardization": (
        LogisticRegressionProbe,
        {"l2": 1e-2, "normalization": Standardization()},
    ),
    # -------------------------------------------------------------------------
    # Linear SVM Probe
    # -------------------------------------------------------------------------
    "LinearSVM": (LinearSVMProbe, {}),
    "LinearSVM_no_init": (LinearSVMProbe, {"init_from_means_diff": False}),
    "LinearSVM_l2": (LinearSVMProbe, {"l2": 1e-2}),
    "LinearSVM_standardization": (LinearSVMProbe, {"normalization": Standardization()}),
    "LinearSVM_l2_standardization": (
        LinearSVMProbe,
        {"l2": 1e-2, "normalization": Standardization()},
    ),
    # -------------------------------------------------------------------------
    # Dot Product Centroid Probe
    # -------------------------------------------------------------------------
    "DotProductCentroid": (DotProductCentroidProbe, {}),
    "DotProductCentroid_standardization": (
        DotProductCentroidProbe,
        {"normalization": Standardization()},
    ),
    "DotProductCentroid_midpoint": (
        DotProductCentroidProbe,
        {"bias_calibrator": midpoint_bias},
    ),
    "DotProductCentroid_standardization_midpoint": (
        DotProductCentroidProbe,
        {"normalization": Standardization(), "bias_calibrator": midpoint_bias},
    ),
    # -------------------------------------------------------------------------
    # Cosine Centroid Probe
    # -------------------------------------------------------------------------
    "CosineCentroid": (CosineCentroidProbe, {}),
    "CosineCentroid_standardization": (
        CosineCentroidProbe,
        {"normalization": Standardization()},
    ),
    "CosineCentroid_midpoint": (
        CosineCentroidProbe,
        {"bias_calibrator": midpoint_bias},
    ),
    # -------------------------------------------------------------------------
    # Squared L2 Centroid Probe
    # -------------------------------------------------------------------------
    "SqL2Centroid": (SqL2CentroidProbe, {}),
    "SqL2Centroid_standardization": (
        SqL2CentroidProbe,
        {"normalization": Standardization()},
    ),
    "SqL2Centroid_whitening": (
        SqL2CentroidProbe,
        {"normalization": Whitening()},
    ),
    "SqL2Centroid_midpoint": (
        SqL2CentroidProbe,
        {"bias_calibrator": midpoint_bias},
    ),
    "SqL2Centroid_standardization_lda": (
        SqL2CentroidProbe,
        {"normalization": Standardization(), "bias_calibrator": lda_shared_var_bias},
    ),
    # -------------------------------------------------------------------------
    # Diagonal Mahalanobis Centroid Probe
    # -------------------------------------------------------------------------
    "DiagMahalanobis_global": (
        DiagonalMahalanobisCentroidProbe,
        {"shrinkage": 1.0},  # default normalization is Standardization
    ),
    "DiagMahalanobis_classwise": (
        DiagonalMahalanobisCentroidProbe,
        {"shrinkage": 0.0},
    ),
    "DiagMahalanobis_mixed": (
        DiagonalMahalanobisCentroidProbe,
        {"shrinkage": 0.5},
    ),
    "DiagMahalanobis_global_midpoint": (
        DiagonalMahalanobisCentroidProbe,
        {"shrinkage": 1.0, "bias_calibrator": midpoint_bias},
    ),
    "DiagMahalanobis_whitening": (
        DiagonalMahalanobisCentroidProbe,
        {"normalization": Whitening(), "shrinkage": 1.0},
    ),
    # -------------------------------------------------------------------------
    # SVDD Centroid Probe
    # -------------------------------------------------------------------------
    "SVDD": (SVDDCentroidProbe, {}),
    "SVDD_standardization": (
        SVDDCentroidProbe,
        {"normalization": Standardization()},
    ),
    "SVDD_l2": (SVDDCentroidProbe, {"l2": 1e-4}),
    "SVDD_standardization_l2": (
        SVDDCentroidProbe,
        {"normalization": Standardization(), "l2": 1e-4},
    ),
    "SVDD_high_C": (SVDDCentroidProbe, {"C": 10.0}),
}

_probe_items = list(PROBE_CONFIGS.items())


@pytest.mark.parametrize(
    "probe_name,probe_spec",
    _probe_items,
    ids=[name for name, _ in _probe_items],
)
@pytest.mark.parametrize("nb_classes", [1, 10])
@pytest.mark.parametrize("features_dimensions", [50])
def test_multilabel_probes_on_synthetic_data(probe_name, probe_spec, nb_classes, features_dimensions):
    torch.manual_seed(0)
    # Make dataset
    X_train, y_train, train_labels, X_test, y_test, test_labels, centroids = make_synthetic_dataset(
        nb_classes, features_dimensions, nb_classes * features_dimensions, 0
    )
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    centroids = centroids.to(DEVICE)

    # Train
    probe_cls, params = probe_spec
    probe = probe_cls(**params)
    probe.to(DEVICE)
    probe.fit(X_train, y_train)

    assert probe.fitted is True, "After fitting, probe should be fitted"

    # Encode train and test
    train_scores = probe.encode(X_train)  # (n_train, c)
    test_scores = probe.encode(X_test)  # (n_test, c)

    assert train_scores.shape == (X_train.shape[0], nb_classes), "Wrong encoded shape"
    assert test_scores.shape == (X_test.shape[0], nb_classes), "Wrong encoded shape"

    if nb_classes == 1:
        # Single concept: check that positives score higher than negatives on average
        pos_mask = y_test[:, 0] == 1.0
        neg_mask = ~pos_mask
        if pos_mask.any() and neg_mask.any():
            mean_pos = test_scores[pos_mask, 0].mean()
            mean_neg = test_scores[neg_mask, 0].mean()
            assert mean_pos > mean_neg, (
                f"Probe {probe_name}: positive mean score ({mean_pos:.3f}) "
                f"should exceed negative mean ({mean_neg:.3f})"
            )
    else:
        # Multiclass prediction by argmax over outputs
        train_pred = train_scores.argmax(dim=1)
        test_pred = test_scores.argmax(dim=1)

        train_acc = (train_pred == train_labels).float().mean().item()
        test_acc = (test_pred == test_labels).float().mean().item()

        assert train_acc > 0.9, f"Probe {probe_name} training accuracy too low: {train_acc:.3f}"
        assert test_acc > 0.9, f"Probe {probe_name} test accuracy too low: {test_acc:.3f}"


@pytest.mark.parametrize(
    "probe_name,probe_spec",
    _probe_items,
    ids=[name for name, _ in _probe_items],
)
def test_probe_encode_determinism_and_save_load(probe_name, probe_spec, tmp_path):
    """Verify that encode is deterministic and survives state_dict round-trip."""
    c, d, nc = 5, 20, 100
    X_train, y_train, _, X_test, _, _, _ = make_synthetic_dataset(c, d, nc, seed=42)

    probe_cls, params = probe_spec
    probe = probe_cls(**params)
    probe.fit(X_train, y_train)

    # Two consecutive encodes must be identical
    scores_1 = probe.encode(X_test)
    scores_2 = probe.encode(X_test)
    assert torch.equal(scores_1, scores_2), f"Probe {probe_name}: encode is not deterministic across calls"

    # state_dict round-trip: save, create fresh probe, load weights
    path = tmp_path / "probe_state.pt"
    torch.save(probe.state_dict(), path)

    loaded_probe = probe_cls(**params)
    loaded_probe.load_state_dict(torch.load(path, weights_only=True))

    scores_loaded = loaded_probe.encode(X_test)
    assert torch.equal(scores_1, scores_loaded), (
        f"Probe {probe_name}: encode results differ after state_dict round-trip"
    )


if __name__ == "__main__":
    name = "MeansDiff_midpoint"
    cls, params = PROBE_CONFIGS[name]
    test_multilabel_probes_on_synthetic_data(name, (cls, params), 10, 50)
