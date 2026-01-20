"""
Semi-NMF module for PyTorch.

For sake of simplicity, we will use the following notation:
- A: pattern of activations of a neural net, tensor of shape (batch_size, n_features)
- Z: codes in the concepts (overcomplete) basis, tensor of shape (batch_size, nb_concepts)
- D: dictionary of concepts, tensor of shape (nb_concepts, n_features)
"""

from tqdm import tqdm
import torch
from torch.linalg import pinv

from .base import BaseOptimDictionaryLearning
from .utils import stopping_criterion, _assert_shapes, pos_part, neg_part


def _one_step_snmf_multiplicative_update(A, Z, D, update_Z=True, update_D=True):
    """
    One step of the Semi-NMF update rules.
    The Semi-NMF algorithm updates Z and D alternately:
    1. Update Z by solving
       Z = Z * ((A @ D.T)^+ + (Z @ (D @ D.T)^-)) / ((A @ D.T)^- + (Z @ (D @ D.T)^+))^-1.
    2. Update D by solving
       D = ((A.T @ Z) @ (Z.T @ Z))^T.

    Parameters
    ----------
    A : torch.Tensor
        Activation tensor, should be (batch_size, n_features).
    Z : torch.Tensor
        Codes tensor, should be (batch_size, nb_concepts).
    D : torch.Tensor
        Dictionary tensor, should be (nb_concepts, n_features).
    update_Z : bool, optional
        Whether to update Z, by default True.
    update_D : bool, optional
        Whether to update D, by default True.

    Returns
    -------
    Z : torch.Tensor
        Updated codes tensor.
    D : torch.Tensor
        Updated dictionary tensor.
    """
    if update_Z:
        # @tfel: one could also use nnls here
        # Z = matrix_nnls(D.T, A.T).T
        # instead we use the update rule from the original paper
        ATD = A @ D.T
        DDT = D @ D.T
        numerator = pos_part(ATD) + (Z @ neg_part(DDT))
        denominator = neg_part(ATD) + (Z @ pos_part(DDT))
        Z = Z * torch.sqrt((numerator / (denominator+1e-8)) + 1e-8)

    if update_D:
        ZtZ_inv = torch.linalg.pinv((Z.T @ Z) + torch.eye(Z.shape[1], device=Z.device) * 1e-8)
        D = ((A.T @ Z) @ ZtZ_inv).T

    return Z, D


def snmf_multiplicative_update(A, Z, D, update_Z=True, update_D=True, max_iter=500, tol=1e-5,
                               verbose=False, **kwargs):
    """
    Semi-NMF optimizer.

    Alternately updates Z and D using the Semi-NMF update rules.
    See: "Convex and Semi-Nonnegative Matrix Factorizations"
    Chris Ding, Tao Li, and Michael I. Jordan (2008).

    Parameters
    ----------
    A : torch.Tensor
        Activation tensor, should be (batch_size, n_features).
    Z : torch.Tensor
        Codes tensor, should (batch_size, nb_concepts).
    D : torch.Tensor
        Dictionary tensor, should be (nb_concepts, n_features).
    update_Z : bool, optional
        Whether to update Z, by default True.
    update_D : bool, optional
        Whether to update D, by default True.
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    tol : float, optional
        Tolerance value for the stopping criterion, by default 1e-5.
    verbose : bool, optional
        Whether to print the loss at each iteration, by default False.

    Returns
    -------
    Z : torch.Tensor
        Updated codes tensor.
    D : torch.Tensor
        Updated dictionary tensor.
    """
    _assert_shapes(A, Z, D)

    for _ in tqdm(range(max_iter), disable=not verbose):
        Z_old = Z.clone()
        Z, D = _one_step_snmf_multiplicative_update(A, Z, D, update_Z, update_D)

        if update_Z and tol > 0 and stopping_criterion(Z, Z_old, tol):
            break

    return Z, D


def snmf_projected_gradient_descent(A, Z, D, lr=5e-2, update_Z=True, update_D=True, max_iter=500, tol=1e-5,
                                    l1_penalty=0.0, verbose=False, **kwargs):
    """
    Projected gradient descent optimizer for NMF.

    See: Chih-Jen Lin, "Projected Gradient Methods for Nonnegative Matrix Factorization".
    Neural Computation (2007).

    Parameters
    ----------
    A : torch.Tensor
        Activation tensor, should be (batch_size, n_features).
    Z : torch.Tensor
        Codes tensor, should (batch_size, nb_concepts).
    D : torch.Tensor
        Dictionary tensor, should be (nb_concepts, n_features).
    lr : float, optional
        Learning rate, by default 1e-3.
    update_Z : bool, optional
        Whether to update Z, by default True.
    update_D : bool, optional
        Whether to update D, by default True.
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    tol : float, optional
        Tolerance value for the stopping criterion, by default 1e-5.
    l1_penalty : float, optional
        L1 penalty for the sparsity constraint, by default 0.0.
    verbose : bool, optional
        Whether to print the loss at each iteration, by default False.

    Returns
    -------
    Z : torch.Tensor
        Updated codes tensor.
    D : torch.Tensor
        Updated dictionary tensor.
    """
    _assert_shapes(A, Z, D)

    if update_Z:
        Z = torch.nn.Parameter(Z)
    if update_D:
        D = torch.nn.Parameter(D)

    to_optimize = []
    if update_Z:
        to_optimize.append(Z)
    if update_D:
        to_optimize.append(D)

    optimizer = torch.optim.Adam(to_optimize, lr=lr, weight_decay=1e-5)

    for iter_i in tqdm(range(max_iter), disable=not verbose):
        optimizer.zero_grad()
        # @tfel: see if we could pass a custom loss function
        loss = torch.mean(torch.square(A - (Z @ D))) + l1_penalty * torch.mean(torch.abs(Z))

        if update_Z:
            Z_old = Z.data.clone()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if update_Z:
                Z.clamp_(min=0)

        if update_Z:
            should_stop = stopping_criterion(Z, Z_old, tol)
            if should_stop:
                break

    # return pure torch tensor (not a parameter)
    Z_return = Z.detach() if update_Z else Z
    D_return = D.detach() if update_D else D
    return Z_return, D_return


class SemiNMF(BaseOptimDictionaryLearning):
    """
    Torch Semi-NMF-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||A - ZD||_F^2, s.t. D >= 0.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    solver : str, optional
        Solver to use, by default 'mu'. Possible values are 'mu' (multiplicative update)
        and 'pgd' (projected gradient descent), 'pgd' allows for sparsity penalty.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    tol : float, optional
        Tolerance value for the stopping criterion, by default 1e-4.
    l1_penalty : float, optional
        L1 penalty for the sparsity constraint, by default 0.0. Only used with 'pgd' solver.
    verbose : bool, optional
        Whether to print the status of the optimization, by default False.
    """

    _SOLVERS = {
        'mu': snmf_multiplicative_update,
        'pgd': snmf_projected_gradient_descent,
    }

    def __init__(self, nb_concepts, solver='mu', device='cpu',
                 tol=1e-4, l1_penalty=0.0, verbose=False, **kwargs):
        assert solver in self._SOLVERS, f"Solver '{solver}' not found in registry."

        super().__init__(nb_concepts, device)
        self.tol = tol
        self.solver_fn = self._SOLVERS[solver]
        self.l1_penalty = l1_penalty
        self.register_buffer('D', None)
        self.verbose = verbose

    def encode(self, A, max_iter=300, tol=None):
        """
        Encode the input tensor (the activations) using Semi-NMF.

        Parameters
        ----------
        A : torch.Tensor or Iterable
            Input tensor of shape (batch_size, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 300.
        tol : float, optional
            Tolerance value for the stopping criterion, by default the value set in the constructor.

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()
        if tol is None:
            tol = self.tol

        A = A.to(self.device)
        Z = self.init_random_z(A)

        Z, _ = self.solver_fn(A, Z, self.D, update_Z=True, update_D=False, max_iter=max_iter, tol=tol,
                              l1_penalty=self.l1_penalty)

        return Z

    def decode(self, Z):
        """
        Decode the input tensor (the codes) using Semi-NMF.

        Parameters
        ----------
        Z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_concepts).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        Z = Z.to(self.device)
        A_hat = Z @ self.D

        return A_hat

    def fit(self, A, max_iter=500):
        """
        Fit the Semi-NMF model to the input data.

        Parameters
        ----------
        A : torch.Tensor or Iterable
            Input tensor of shape (batch_size, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 500.
        """
        A = A.to(self.device)

        # @tfel: we could warm start here, or/and use nnsvdvar instead
        # of (non-negative) random
        Z = self.init_random_z(A)
        D = self.init_random_d(A, Z)

        Z, D = self.solver_fn(A, Z, D, max_iter=max_iter, tol=self.tol,
                              l1_penalty=self.l1_penalty, verbose=self.verbose)

        self.D = D
        self._set_fitted()

        return Z, self.D

    def get_dictionary(self):
        """
        Return the learned dictionary components from Semi-NMF.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return self.D

    def init_random_d(self, A, Z):
        """
        Initialize the dictionary D using matrix inversion.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).
        Z : torch.Tensor
            Codes tensor of shape (batch_size, nb_concepts).

        Returns
        -------
        D : torch.Tensor
            Initialized dictionary tensor.
        """
        ZtZ = (Z.T @ Z)
        ZtZ_inv = torch.linalg.pinv(ZtZ)
        D = ((A.T @ Z) @ ZtZ_inv).T
        return D

    def init_random_z(self, A):
        """
        Initialize the codes Z using random values (can be negative).

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        Z : torch.Tensor
            Initialized codes tensor.
        """
        # for semi-nmf, mean of A could be negative
        mu = torch.sqrt(torch.mean(torch.abs(A) / self.nb_concepts))

        Z = torch.randn(A.shape[0], self.nb_concepts, device=self.device) * mu
        Z = torch.abs(Z)

        return Z
