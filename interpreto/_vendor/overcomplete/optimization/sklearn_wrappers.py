"""
Sklearn-based wrappers for dictionary learning models.
"""

import numpy as np
import torch
from sklearn.decomposition import (PCA, NMF, FastICA, TruncatedSVD, SparsePCA, DictionaryLearning)
from sklearn.cluster import KMeans

from .base import BaseOptimDictionaryLearning


class SkPCA(BaseOptimDictionaryLearning):
    """
    PCA-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2 s.t D is orthogonal and Z are centered.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def __init__(self, nb_concepts, device='cpu', **kwargs):
        super().__init__(nb_concepts, device)
        self.model = PCA(n_components=nb_concepts, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using PCA.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        # @tfel: see to adapt for online algorithms
        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using PCA.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = self.model.inverse_transform(z)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the PCA model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)
        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from PCA.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(self.model.components_, dtype=torch.float32, device=self.device)


class SkICA(BaseOptimDictionaryLearning):
    """
    ICA-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2. s.t Z are statistically independent.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def __init__(self, nb_concepts, device='cpu', **kwargs):
        super().__init__(nb_concepts, device)
        self.model = FastICA(n_components=nb_concepts, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using ICA.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using ICA.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = np.dot(z, self.model.mixing_.T)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the ICA model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from ICA.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(self.model.components_, dtype=torch.float32, device=self.device)


class SkNMF(BaseOptimDictionaryLearning):
    """
    NMF-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2. s.t Z >=0, D >= 0.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def __init__(self, nb_concepts, device='cpu', **kwargs):
        super().__init__(nb_concepts, device)
        self.model = NMF(n_components=nb_concepts, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using NMF.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)
        assert (x >= 0).all(), 'Input tensor must be non-negative'

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using NMF.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = np.dot(z, self.model.components_)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the NMF model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)
        assert (x >= 0).all(), 'Input tensor must be non-negative'

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from NMF.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(self.model.components_, dtype=torch.float32, device=self.device)


class SkKMeans(BaseOptimDictionaryLearning):
    """
    KMeans-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2. s.t Z ∈ {0, 1}^{n_samples x nb_concepts}.

    Parameters
    ----------
    nb_concepts : int
        Number of clusters to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def __init__(self, nb_concepts, device='cpu', **kwargs):
        super().__init__(nb_concepts, device)
        self.model = KMeans(n_clusters=nb_concepts, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using KMeans.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using KMeans.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        cluster_centers = self.model.cluster_centers_
        x_hat = np.dot(z, cluster_centers)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the KMeans model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from KMeans.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(self.model.cluster_centers_, dtype=torch.float32, device=self.device)


class SkDictionaryLearning(BaseOptimDictionaryLearning):
    """
    Dictionary Learning model using sklearn.

    Solve the following optimization problem:
    min ||X - ZD||_F^2 + λ * ||Z||_1.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    sparsity : float, optional
        Sparsity parameter,  λ in the equation, by default 1.0
    fit_algorithm : str, optional
        Algorithm for sklearn for fitting the model, by default "cd"
    transform_algorithm : str, optional
        Algorithm for sklearn for transforming the data, by default "lasso_cd"
    tolerance : float, optional
        Tolerance for sklearn optimization algorithm, by default 1e-3
    """

    def __init__(
            self, nb_concepts, device='cpu', sparsity=1.0, fit_algorithm="cd", transform_algorithm="lasso_cd",
            tolerance=1e-3, **kwargs):
        super().__init__(nb_concepts, device)
        self.model = DictionaryLearning(n_components=nb_concepts,
                                        alpha=sparsity,
                                        positive_code=True,
                                        fit_algorithm=fit_algorithm,
                                        transform_algorithm=transform_algorithm,
                                        tol=tolerance, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using DictionaryLearning.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using DictionaryLearning.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = z @ self.model.components_
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the DictionaryLearning model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from DictionaryLearning.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(self.model.components_, dtype=torch.float32, device=self.device)


class SkSparsePCA(BaseOptimDictionaryLearning):
    """
    SparsePCA-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2 + λ * ||Z||_1 s.t D is orthogonal and X is centered.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    sparsity : float, optional
        Sparsity parameter,  λ in the equation, by default 1.0
    fit_method : str, optional
        Algorithm for sklearn for fitting the model, by default "cd"
    tolerance : float, optional
        Tolerance for sklearn optimization algorithm, by default 1e-3
    """

    def __init__(self, nb_concepts, device='cpu', sparsity=1.0, fit_method="cd", tolerance=1e-3, **kwargs):
        super().__init__(nb_concepts, device)
        self.model = SparsePCA(n_components=nb_concepts,
                               alpha=sparsity,
                               method=fit_method,
                               tol=tolerance,
                               **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using SparsePCA.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using SparsePCA.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = self.model.inverse_transform(z)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the SparsePCA model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from SparsePCA.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(self.model.components_, dtype=torch.float32, device=self.device)


class SkSVD(BaseOptimDictionaryLearning):
    """
    SVD-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2 s.t D is orthogonal.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def __init__(self, nb_concepts, device='cpu', **kwargs):
        super().__init__(nb_concepts, device)
        self.model = TruncatedSVD(n_components=nb_concepts, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using SVD.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using SVD.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = self.model.inverse_transform(z)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the SVD model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from TruncatedSVD.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(self.model.components_, dtype=torch.float32, device=self.device)
