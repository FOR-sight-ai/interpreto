import pytest
import torch
from interpreto.concepts.probe import (
    LinearRegressionProbe,
    LinearSVMProbe,
    LogisticRegressionProbe,
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


@pytest.mark.parametrize(
    "probe_cls", [LinearRegressionProbe, LogisticRegressionProbe, LinearSVMProbe]
)
@pytest.mark.parametrize("nb_classes", [10, 20, 100])
@pytest.mark.parametrize("features_dimensions", [50, 200, 1000])
def test_multilabel_probes_on_synthetic_data(
    probe_cls, nb_classes, features_dimensions
):
    torch.manual_seed(0)
    # Make dataset
    X_train, y_train, train_labels, X_test, y_test, test_labels, centroids = (
        make_synthetic_dataset(
            nb_classes, features_dimensions, nb_classes * features_dimensions, 0
        )
    )
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    centroids = centroids.to(DEVICE)

    # Train
    probe = probe_cls()
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
    test_multilabel_probes_on_synthetic_data(LinearSVMProbe, 10, 50)
