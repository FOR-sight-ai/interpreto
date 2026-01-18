"""
Archetypal Analysis (AA) module.

We use the following notation:
- A: data matrix, tensor of shape (n_samples, n_features)
- Z: codes matrix, tensor of shape (n_samples, nb_concepts)
- W: coefficient matrix, tensor of shape (nb_concepts, n_samples)
- D: dictionary matrix, computed as D = W @ A

The objective is:
    min_{Z,W} ||A - Z D||_F^2
    subject to Z in Δ^nb_concepts and D in conv(A)

Say it otherwise, Z row stochastic and W row stochastic.
Currently supports projected gradient descent (PGD) solver.

For a complete and more faithful implementation, see the great SPAM toolbox:
https://thoth.inrialpes.fr/people/mairal/spams/
"""

import torch
from tqdm import tqdm

from .base import BaseOptimDictionaryLearning
from .utils import stopping_criterion


def project_simplex(W, temperature=0.0, dim=1):
    """
    Project matrix W onto the simplex using softmax.

    Parameters
    ----------
    W : torch.Tensor
        Input tensor.
    temperature : float, optional
        Temperature parameter for scaling before softmax, by default 0.0.
    dim : int, optional
        Dimension along which to apply softmax, by default 1.

    Returns
    -------
    torch.Tensor
        Row- or column-stochastic matrix.
    """
    return torch.softmax(W / torch.exp(torch.tensor(temperature)), dim=dim)


def aa_pgd_solver(A, Z, W, lr=1e-2, update_Z=True, update_W=True,
                  max_iter=500, tol=1e-5, verbose=False):
    """
    Alternating Projected Gradient Descent (PGD) solver for Archetypal Analysis.

    Parameters
    ----------
    A : torch.Tensor
        Input data matrix (n_samples, n_features).
    Z : torch.Tensor
        Initial codes matrix (n_samples, nb_concepts).
    W : torch.Tensor
        Initial coefficient matrix (nb_concepts, n_samples).
    lr : float
        Learning rate.
    update_Z : bool
        Whether to update Z.
    update_W : bool
        Whether to update W.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to display a progress bar.

    Returns
    -------
    Z : torch.Tensor
        Final codes matrix.
    W : torch.Tensor
        Final coefficient matrix.
    """
    if update_Z:
        Z = torch.nn.Parameter(Z)
    if update_W:
        W = torch.nn.Parameter(W)

    params = [p for p in [Z, W] if isinstance(p, torch.nn.Parameter)]
    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in tqdm(range(max_iter), disable=not verbose):
        optimizer.zero_grad()
        D = W @ A
        loss = torch.mean((A - Z @ D).pow(2))

        if update_Z:
            Z_old = Z.data.clone()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if update_Z:
                Z.copy_(project_simplex(Z))
            if update_W:
                W.copy_(project_simplex(W))

        if update_Z and tol > 0 and stopping_criterion(Z, Z_old, tol):
            break

    return Z.detach(), W.detach()


class ArchetypalAnalysis(BaseOptimDictionaryLearning):
    """
    PyTorch Archetypal Analysis Dictionary Learning model.

    Objective:
        min_{Z,W} ||A - Z D||_F^2  with  D = W A,
        rows(Z) simplex, rows(W) simplex.

    Parameters
    ----------
    nb_concepts : int
        Number of archetypes (concepts).
    device : str, optional
        Computation device.
    tol : float, optional
        Convergence tolerance.
    solver : str, optional
        Solver to use ('pgd').
    verbose : bool, optional
        Verbosity flag.
    """
    _SOLVERS = {
        'pgd': aa_pgd_solver,
    }

    def __init__(self, nb_concepts, device='cpu', tol=1e-4, solver='pgd', verbose=False):
        super().__init__(nb_concepts, device)
        assert solver in self._SOLVERS, f"Unknown solver '{solver}'"
        self.tol = tol
        self.verbose = verbose
        self.solver = solver
        self.solver_fn = self._SOLVERS[solver]

    def encode(self, A, max_iter=300, tol=None):
        """
        Encode the input data matrix into codes Z using fixed dictionary D.

        Parameters
        ----------
        A : torch.Tensor
            Input data matrix (n_samples, n_features).
        max_iter : int, optional
            Maximum number of solver iterations.
        tol : float, optional
            Convergence tolerance.

        Returns
        -------
        torch.Tensor
            Codes matrix Z (n_samples, nb_concepts).
        """
        self._assert_fitted()
        A = A.to(self.device)
        tol = tol or self.tol
        Z = self.init_random_z(A)
        Z, _ = self.solver_fn(A, Z, self.W, update_Z=True, update_W=False,
                              max_iter=max_iter, tol=tol, verbose=self.verbose)
        return Z

    def decode(self, Z):
        """
        Decode the codes matrix Z into reconstructed data using dictionary D.

        Parameters
        ----------
        Z : torch.Tensor
            Codes matrix (n_samples, nb_concepts).

        Returns
        -------
        torch.Tensor
            Reconstructed data matrix (n_samples, n_features).
        """
        self._assert_fitted()
        return Z.to(self.device) @ self.D

    def fit(self, A, max_iter=500):
        """
        Fit the AA model by jointly optimizing Z and W.

        Parameters
        ----------
        A : torch.Tensor
            Input data matrix (n_samples, n_features).
        max_iter : int, optional
            Maximum number of solver iterations.

        Returns
        -------
        Z : torch.Tensor
            Final codes matrix (n_samples, nb_concepts).
        D : torch.Tensor
            Learned dictionary matrix (nb_concepts, n_features).
        """
        A = A.to(self.device)
        Z = self.init_random_z(A)
        W = self.init_random_w(A)
        Z, W = self.solver_fn(A, Z, W, update_Z=True, update_W=True,
                              max_iter=max_iter, tol=self.tol, verbose=self.verbose)
        self.register_buffer('W', W)
        self.register_buffer('D', W @ A)
        self._set_fitted()
        return Z, self.D

    def get_dictionary(self):
        """
        Return the learned dictionary D = W @ A.

        Returns
        -------
        torch.Tensor
            Dictionary matrix (nb_concepts, n_features).
        """
        self._assert_fitted()
        return self.D

    def init_random_z(self, A):
        """
        Initialize the codes matrix Z with random values projected onto the simplex.

        Parameters
        ----------
        A : torch.Tensor
            Input data matrix (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Initialized codes matrix (n_samples, nb_concepts).
        """
        mu = torch.sqrt(torch.mean(torch.abs(A)) / self.nb_concepts)
        Z = torch.randn(A.shape[0], self.nb_concepts, device=self.device) * mu
        return project_simplex(Z)

    def init_random_w(self, A):
        """
        Initialize the coefficient matrix W with random values projected onto the simplex.

        Parameters
        ----------
        A : torch.Tensor
            Input data matrix (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Initialized coefficient matrix (nb_concepts, n_samples).
        """
        mu = torch.sqrt(torch.mean(torch.abs(A)) / self.nb_concepts)
        W = torch.randn(self.nb_concepts, A.shape[0], device=self.device) * mu
        return project_simplex(W)
