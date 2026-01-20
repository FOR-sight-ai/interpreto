"""
Utility functions for the optimization module of Overcomplete.
"""

import torch


def batched_matrix_nnls(D, X, max_iter=50, tol=1e-5, Z_init=None):
    """
    Batched non-negative least squares (nnls) via projected gradient descent.

    solves for Z in:
        min_Z ||Z @ D - X||^2  subject to Z >= 0
    for each sample in the batch.
    Z is initialized to zero if not provided and is of shape (n, k).
    D is of shape (n, k, d) and X is of shape (n, d).
    n is the batch size, k is the number of concepts, and d is the dimensionality of the data.

    Parameters
    ----------
    D : torch.Tensor
        dictionary of shape (n, k, d)
    X : torch.Tensor
        targets of shape (n, d)
    max_iter : int
        number of pgd steps
    tol : float
        convergence threshold on gradient norm
    Z_init : torch.Tensor or None
        optional initial value for Z, shape (n, k)

    Returns
    -------
    Z : torch.Tensor
        codes of shape (n, k), with Z >= 0
    """
    n, k, d = D.shape

    # initialize z
    if Z_init is not None:
        assert Z_init.shape == (n, k), f"z_init must have shape ({n}, {k})"
        Z = Z_init.to(D.device).clone()
    else:
        Z = torch.zeros(n, k, device=D.device)

    # precompute q = d d^t and p = d x^t
    Q = torch.bmm(D, D.transpose(2, 1))
    P = torch.bmm(D, X.unsqueeze(2)).squeeze(2)

    # compute per-batch learning rate from q norm
    # as ||Z @ D - X|| is L-lipschitz
    Q_norms = Q.square().sum(dim=(1, 2)).sqrt() + 1e-8
    lr = 1.0 / Q_norms

    for _ in range(max_iter):
        # gradient step
        grad = torch.bmm(Q, Z.unsqueeze(2)).squeeze(2) - P
        Z = Z - lr.unsqueeze(1) * grad

        # project onto non-negative orthant
        Z = torch.clamp(Z, min=0)

        # early stop if gradient norm is below threshold
        if torch.norm(grad, p='fro') < tol:
            break

    return Z


def matrix_nnls(A, B, max_iter=50, tol=1e-3):
    """
    Non-negative least squares problem solver (NNLS):
        min_x ||AX - B||^2 such that x >= 0.
    Using projected gradient descent scheme.

    Parameters
    ----------
    A : torch.Tensor
        Matrix of shape (m, n)
    B : torch.Tensor
        Matrix of shape (m, d)
    max_iter : int, optional
        Maximum number of iterations, by default 300

    Returns
    -------
    X : torch.Tensor
        Solution to the NNLS problem
    """
    m, n = A.shape
    _, d = B.shape

    X = torch.rand(n, d, device=A.device)

    Q = A.T @ A
    P = A.T @ B

    # from operator norm inequality, as ||Ax - b|| is L-lipschitz (easy to prove)
    # with L = ||A.T @ A||_2
    lr = 1.0 / (torch.sqrt(Q.square().sum()) + 1e-8)

    for _ in range(max_iter):
        # grad on X
        grad = Q @ X - P
        X = X - lr * grad
        # project onto non-negative orthant
        X = torch.clamp(X, min=0)
        # naive stopping criterion
        if torch.norm(grad, p='fro') < tol:
            break
    return X


def stopping_criterion(Z_next, Z_old, tol):
    """
    Define the stopping criterion for the nmf optimization algorithms.

    Parameters
    ----------
    Z_next : torch.Tensor
        Updated codes tensor.
    Z_old : torch.Tensor
        Previous codes tensor.
    tol : float
        Tolerance value.

    Returns
    -------
    bool
        True if the stopping criterion is met.
    """
    if tol <= 0:
        return False

    diff = torch.norm(Z_next - Z_old) / torch.norm(Z_next)
    return diff <= tol


def _assert_shapes(A, Z, D):
    """
    Ensure that the input tensors have the correct shapes for this module.

    Parameters
    ----------
    A : torch.Tensor
        Activation tensor, should be (batch_size, n_features).
    Z : torch.Tensor
        Codes tensor, should (batch_size, nb_concepts).
    D : torch.Tensor
        Dictionary tensor, should be (nb_concepts, n_features).
    """
    assert A.ndim == Z.ndim == D.ndim == 2, "All tensors must be 2D tensors"
    assert A.shape[1] == D.shape[1], "A and D must have the same number of features"
    assert A.shape[0] == Z.shape[0], "A and Z must have the same number of samples"
    assert Z.shape[1] == D.shape[0], "Z and D must have the same number of concepts"


def pos_part(A):
    """
    Compute the positive part of a tensor.

    A+ = max(0, A)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output tensor with negative values set to zero.
    """
    return torch.relu(A)


def neg_part(A):
    """
    Compute the negative part of a tensor.

    A- = (|A| - A) / 2.0

    Parameters
    ----------
    A : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output tensor with positive values set to zero.
    """
    return torch.relu(-A)
