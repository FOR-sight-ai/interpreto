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

import pytest
import torch

from interpreto.concepts.probe import (
    CentroidCosineProbe,
    CentroidDotProbe,
    CentroidMahalanobisClasswiseVarProbe,
    CentroidMahalanobisCommonVarProbe,
    CentroidSqL2Probe,
    EllipsoidalBoundaryProbe,
    GaussianLikelihoodProbe,
    LinearRegressionProbe,
    LinearSVMProbe,
    LogisticRegressionProbe,
    SVDDProbe,
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


PROBE_CONFIGS = {
    # Linear probes with variants
    "LinearRegression": (LinearRegressionProbe, {}),
    "LinearRegression_l2=1e-2": (LinearRegressionProbe, {"l2": 1e-2}),
    "LogisticRegression": (LogisticRegressionProbe, {}),
    "LogisticRegression_l2=1e-2_meansdiff": (LogisticRegressionProbe, {"l2": 1e-2, "means_diff_init": True}),
    "LinearSVM": (LinearSVMProbe, {}),
    "LinearSVM_l2=1e-2_meansdiff": (LinearSVMProbe, {"l2": 1e-2, "means_diff_init": True}),
    # MeansDiff with bias variants
    # "MeansDiff_zero": (MeansDiffProbe, {"bias": "zero"}),
    # "MeansDiff_midpoint": (MeansDiffProbe, {"bias": "midpoint"}),
    # "MeansDiff_prevalence": (MeansDiffProbe, {"bias": "prevalence"}),
    # "MeansDiff_bce": (MeansDiffProbe, {"bias": "bce"}),
    # Centroid-based probes with normalization variants
    "CentroidDot_none": (CentroidDotProbe, {"normalization": "none"}),
    "CentroidDot_zscore": (CentroidDotProbe, {"normalization": "zscore"}),
    "CentroidCosine_none": (CentroidCosineProbe, {"normalization": "none"}),
    "CentroidCosine_zscore": (CentroidCosineProbe, {"normalization": "zscore"}),
    "CentroidSqL2_none": (CentroidSqL2Probe, {"normalization": "none"}),
    "CentroidSqL2_zscore": (CentroidSqL2Probe, {"normalization": "zscore"}),
    "MahalanobisCommon_zscore": (CentroidMahalanobisCommonVarProbe, {"normalization": "zscore"}),
    "MahalanobisClasswise_zscore": (CentroidMahalanobisClasswiseVarProbe, {"normalization": "zscore"}),
    # Ellipsoidal and SVDD with normalization/l2 variants
    "Ellipsoidal_none": (EllipsoidalBoundaryProbe, {"normalization": "none"}),
    "Ellipsoidal_zscore_shrink=0.1": (EllipsoidalBoundaryProbe, {"normalization": "zscore", "var_shrink": 0.1}),
    "SVDD_none": (SVDDProbe, {"normalization": "none"}),
    "SVDD_zscore_l2=1e-4": (SVDDProbe, {"normalization": "zscore", "l2": 1e-4}),
    # Gaussian likelihood (QDA-style)
    "GaussianLikelihood_standardization": (GaussianLikelihoodProbe, {"normalization": "standardization"}),
    "GaussianLikelihood_whitening": (GaussianLikelihoodProbe, {"normalization": "whitening"}),
    "GaussianLikelihood_lowrank_whitening_r3": (
        GaussianLikelihoodProbe,
        {"normalization": "lowrank_whitening", "lowrank_rank": 3},
    ),
    "GaussianLikelihood_lowrank_whitening_r5": (
        GaussianLikelihoodProbe,
        {"normalization": "lowrank_whitening", "lowrank_rank": 5},
    ),
}

_probe_items = list(PROBE_CONFIGS.items())


@pytest.mark.parametrize(
    "probe_name,probe_spec",
    _probe_items,
    ids=[name for name, _ in _probe_items],
)
@pytest.mark.parametrize("nb_classes", [10])  # , 20, 100
@pytest.mark.parametrize("features_dimensions", [50])  # , 200, 1000])
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

    # Multiclass prediction by argmax over outputs
    train_pred = train_scores.argmax(dim=1)
    test_pred = test_scores.argmax(dim=1)

    train_acc = (train_pred == train_labels).float().mean().item()
    test_acc = (test_pred == test_labels).float().mean().item()

    assert train_acc > 0.9, "Probe training accuracy should be high on easy dataset"
    assert test_acc > 0.9, "Probe test accuracy should be high on easy dataset"

    # # Cosine similarity between centroids and learned weights
    # # centroids: (c, d), weights: (d, c) -> transpose to (c, d)
    # weights = probe.weight  # (d, c)
    # weight_rows = weights.t()  # (c, d)

    # cos_sim = F.cosine_similarity(weight_rows, centroids, dim=1)  # (c,)
    # # Cosine similarity should be high
    # assert torch.all(cos_sim > 0.7), (
    #     "Probe concept vector should be aligned with centroids"
    # )


if __name__ == "__main__":
    name = "LinearSVM_l2=1e-2_meansdiff"
    cls, params = PROBE_CONFIGS[name]
    test_multilabel_probes_on_synthetic_data(name, (cls, params), 10, 50)
