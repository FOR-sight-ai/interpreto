"""
NMF module for PyTorch.

For sake of simplicity, we will use the following notation:
- A: pattern of activations of a neural net, tensor of shape (batch_size, n_features)
- Z: codes in the concepts (overcomplete) basis, tensor of shape (batch_size, nb_concepts)
- D: dictionary of concepts, tensor of shape (nb_concepts, n_features)
"""

from tqdm import tqdm
import torch
from sklearn.decomposition._nmf import _initialize_nmf
from scipy.optimize import nnls as scipy_nnls


from .base import BaseOptimDictionaryLearning
from .utils import matrix_nnls, stopping_criterion, _assert_shapes


def _one_step_hals(A, Z, D, update_Z=True, update_D=True):
    """
    One step of the Hierarchical Alternating Least Squares (HALS) update rules for NMF.

    The original HALS algorithm is defined as follows:
    1. Update Z_i <- max(0, (R_i D_i) / (||D_i||_2^2))
    2. Update D_i <- max(0, (R^T_i Z_i) / (||Z_i||_2^2))
    with R_i = A - sum_{j!=i} Z_j D_j

    However, it is more costly to re-compute R_i at each iteration since it depends on D and Z.
    Instead, we note that this update can also be written:

    1. Z_i <- max(0, (C_i - sum_{j!=i} Z_j B_ji) / Bii)
    with C = A D^T and B = D D^T (don't depend on Z when updating Z!)

    2. D_i <- max(0, (M_i - sum_{j!=i} N_ji D_j) / Nii)
    with M = Z^T A and N = Z^T Z (don't depend on D when updating D!)

    See the excellent Gillis book "Non-negative Matrix Factorization".

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
        C = A @ D.T
        B = D @ D.T
        for k in range(Z.shape[1]):
            numerator = C[:, k] - Z @ B[:, k] + Z[:, k] * B[k, k]
            denominator = B[k, k] + 1e-10
            Z[:, k] = torch.relu(numerator / denominator)

    if update_D:
        M = Z.T @ A
        N = Z.T @ Z
        for k in range(D.shape[0]):
            numerator = M[k, :] - N[k, :] @ D + D[k, :] * N[k, k]
            denominator = N[k, k] + 1e-10
            D[k, :] = torch.relu(numerator / denominator)

    return Z, D


def _one_step_multiplicative_rule(A, Z, D, update_Z=True, update_D=True):
    """
    One step of the multiplicative update rules for NMF.

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
    torch.Tensor
        Updated codes tensor.
    torch.Tensor
        Updated dictionary tensor.
    """
    if update_Z:
        Z = Z * ((A @ D.T) / (Z @ D @ D.T + 1e-10))

    if update_D:
        D = D * ((Z.T @ A) / (Z.T @ Z @ D + 1e-10))

    return Z, D


def _one_step_nnls_scipy(A, Z, D, update_Z=True, update_D=True):
    """
    Slow but stable implementation of the NMF update rules using SciPy.

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

    Returns
    -------
    torch.Tensor
        Updated codes tensor.
    torch.Tensor
        Updated dictionary tensor.
    """
    if update_Z:

        Z_update = []
        D_np = D.cpu().numpy().T
        for i in range(A.shape[0]):
            Ai = A[i].cpu().numpy()
            Zi, _ = scipy_nnls(D_np, Ai)
            Z_update.append(Zi)
        Z_update = torch.tensor(Z_update, device=A.device)
    else:
        Z_update = Z

    if update_D:

        D_update = []
        Z_np = Z.cpu().numpy().T
        for i in range(A.shape[1]):
            Ai = A[:, i].cpu().numpy()
            Di, _ = scipy_nnls(Z_np, Ai)
            D_update.append(Di)
        D_update = torch.tensor(D_update, device=A.device).T
    else:
        D_update = D

    return Z_update, D_update


def _one_step_nnls(A, Z, D, update_Z=True, update_D=True):
    """
    One step of the non-negative least squares update rules for NMF using custom
    Pytorch nnls implementation.

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
    Z = matrix_nnls(D.T, A.T).T if update_Z else Z
    D = matrix_nnls(Z, A) if update_D else D

    return Z, D


def multiplicative_update(A, Z, D, update_Z=True, update_D=True, max_iter=500, tol=1e-5, verbose=False):
    """
    Multiplicative update rules optimizer for NMF.

    See:  Lee & Seung "Learning the parts of objects by non-negative matrix factorization".
    Nature (1999).

    The use of multiplicative rule has been introduced in
    Daube-Witherspoon et al. (86) and then proposed in the seminal work of
    Lee and Seung (1999) to solve the NMF algorithm.

    Z <- Z * (A @ D)   / (Z @ D @ D^T).
    D <- D * (Z^T @ A) / (Z^T @ Z @ D).

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
        Maximum number of iterations, by default 1000.
    tol : float, optional
        Tolerance value, by default 1e-5.
    verbose : bool, optional
        Whether to display a progress bar, by default False.

    Returns
    -------
    torch.Tensor
        Updated codes tensor.
    torch.Tensor
        Updated dictionary tensor.
    """
    _assert_shapes(A, Z, D)

    for _ in tqdm(range(max_iter), disable=not verbose):
        Z_old = Z.clone()
        Z, D = _one_step_multiplicative_rule(A, Z, D, update_Z, update_D)

        if update_Z:
            should_stop = stopping_criterion(Z, Z_old, tol)
            if should_stop:
                break

    return Z, D


def alternating_nnls(A, Z, D, update_Z=True, update_D=True, max_iter=500, tol=1e-5, verbose=False):
    """
    Alternating non-negative least squares (ANLS) optimizer for NMF.

    See: Kim, Park "Sparse non-negative matrix factorizations via alternating
         non-negativity-constrained least squares for microarray data analysis".
    Bioinformatics (2007).

    The current custom nnls implementation is based on projected gradient descent.

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
        Maximum number of iterations, by default 1000.
    tol : float, optional
        Tolerance value, by default 1e-5.
    verbose : bool, optional
        Whether to display a progress bar, by default False.

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
        Z, D = _one_step_nnls(A, Z, D, update_Z, update_D)

        if update_Z:
            should_stop = stopping_criterion(Z, Z_old, tol)
            if should_stop:
                break

    return Z, D


def projected_gradient_descent(A, Z, D, lr=5e-2, update_Z=True, update_D=True,
                               max_iter=500, tol=1e-5, verbose=False):
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
        Maximum number of iterations, by default 1000.
    tol : float, optional
        Tolerance value, by default 1e-5.
    verbose : bool, optional
        Whether to display a progress bar, by default False.

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
        loss = torch.mean(torch.square(A - (Z @ D)))

        if update_Z:
            Z_old = Z.data.clone()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if update_Z:
                Z.clamp_(min=0)
            if update_D:
                D.clamp_(min=0)

        if update_Z:
            should_stop = stopping_criterion(Z, Z_old, tol)
            if should_stop:
                break

    # return pure torch tensor (not a parameter)
    Z_return = Z.detach() if update_Z else Z
    D_return = D.detach() if update_D else D
    return Z_return, D_return


def hierarchical_als(A, Z, D, update_Z=True, update_D=True, max_iter=500, tol=1e-5, verbose=False):
    """
    Hierarchical Alternating Least Squares optimizer for NMF.

    See: Cichocki, Zdunek, Amari "Hierarchical ALS Algorithms for Nonnegative Matrix and 3D
         Tensor Factorization".
    Lecture Notes in Computer Science (2007).

    For an explanation and efficient implementation, see Gillis book
    "Non-negative Matrix Factorization" (Data Science Book serie).

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
        Maximum number of iterations, by default 1000.
    tol : float, optional
        Tolerance value, by default 1e-5.
    verbose : bool, optional
        Whether to display a progress bar, by default False.

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
        Z, D = _one_step_hals(A, Z, D, update_Z, update_D)

        if update_Z:
            should_stop = stopping_criterion(Z, Z_old, tol)
            if should_stop:
                break

    return Z, D


class NMF(BaseOptimDictionaryLearning):
    """
    Torch NMF-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||A - ZD||_F^2, s.t. Z >= 0, D >= 0.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    solver : str, optional
        Optimization algorithm to use, by default 'hals'. Can be one of:
        - 'hals': Hierarchical Alternating Least Squares
        - 'mu': Multiplicative update rules
        - 'pgd': Projected Gradient Descent
        - 'anls': Alternating Non-negative Least Squares
    tol : float, optional
        Tolerance value for the stopping criterion, by default 1e-4.
    verbose : bool, optional
        Whether to display a progress bar, by default False.
    """

    _SOLVERS = {
        'hals': hierarchical_als,
        'mu': multiplicative_update,
        'pgd': projected_gradient_descent,
        'anls': alternating_nnls
    }

    def __init__(self, nb_concepts, device='cpu', solver='hals', tol=1e-4, verbose=False, **kwargs):
        super().__init__(nb_concepts, device)

        assert solver in self._SOLVERS, f'Unknown solver {solver}'

        self.solver = solver
        self.solver_fn = self._SOLVERS[solver]
        self.tol = tol
        self.verbose = verbose

        self.register_buffer('D', None)

    def encode(self, A, max_iter=300, tol=None):
        """
        Encode the input tensor (the activations) using NMF.

        Parameters
        ----------
        A : torch.Tensor or Iterable
            Input tensor of shape (batch_size, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 100.
        tol : float, optional
            Tolerance value for the stopping criterion, by default the value set in the constructor.

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()
        assert (A >= 0).all(), 'Input tensor must be non-negative'

        A = A.to(self.device)

        if tol is None:
            tol = self.tol

        Z = self.init_random_z(A)
        Z, _ = self.solver_fn(A, Z, self.D, update_Z=True, update_D=False, max_iter=max_iter, tol=tol,
                              verbose=self.verbose)

        return Z

    def decode(self, Z):
        """
        Decode the input tensor (the codes) using NMF.

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
        Fit the NMF model to the input data.

        Parameters
        ----------
        A : torch.Tensor or Iterable
            Input tensor of shape (batch_size, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 500.
        """
        assert (A >= 0).all(), 'Input tensor must be non-negative'

        A = A.to(self.device)

        if self.nb_concepts <= min(A.shape[0], A.shape[1]):
            Z, D = self.init_nndsvda(A)
        else:
            Z = self.init_random_z(A)
            D = self.init_random_d(A)

        Z, D = self.solver_fn(A, Z, D, max_iter=max_iter, tol=self.tol)

        self.D = D
        self._set_fitted()

        return Z, self.D

    def get_dictionary(self):
        """
        Return the learned dictionary components from NMF.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return self.D

    def init_random_d(self, A):
        """
        Initialize the dictionary D using non negative random values.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        D : torch.Tensor
            Initialized dictionary tensor.
        """
        mu = torch.sqrt(torch.mean(A) / self.nb_concepts)

        D = torch.randn(self.nb_concepts, A.shape[1], device=self.device) * mu
        D = torch.abs(D)

        return D

    def init_random_z(self, A):
        """
        Initialize the codes Z using non negative random values.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        Z : torch.Tensor
            Initialized codes tensor.
        """
        mu = torch.sqrt(torch.mean(A) / self.nb_concepts)

        Z = torch.randn(A.shape[0], self.nb_concepts, device=self.device) * mu
        Z = torch.abs(Z)

        return Z

    def init_nndsvda(self, A):
        """
        Initialize the dictionary and the code using Sklearn NNDsvdA algorithm.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        Z : torch.Tensor
            Initialized codes tensor.
        D : torch.Tensor
            Initialized dictionary tensor.
        """
        Z, D = _initialize_nmf(A.detach().cpu().numpy(), self.nb_concepts, init='nndsvda')

        Z = torch.tensor(Z, device=self.device)
        D = torch.tensor(D, device=self.device)

        return Z, D
