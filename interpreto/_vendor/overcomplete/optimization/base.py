"""
Module dedicated to optimization-based dictionary learning models.

The general formulation of dictionary learning consist in finding codes (Z) and
dictionary (D) that allow to reconstruct the original input (X) by minimizing the
following objective function:
min ||X - ZD||_F^2 s.t Ω_1(Z) and Ω_2(D) where Ω are constraint.
"""

import numpy as np
from sklearn.decomposition import PCA, NMF, FastICA, TruncatedSVD, SparsePCA, DictionaryLearning
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

from ..base import BaseDictionaryLearning
from ..data import to_npf32, unwrap_dataloader


class BaseOptimDictionaryLearning(BaseDictionaryLearning):
    """
    Abstract base class for optimization-based Dictionary Learning models.

    Parameters
    ----------
    nb_concepts : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def sanitize_np_input(self, x):
        """
        Ensure the input tensor is a numpy array of shape (batch_size, dims).
        Convert from pytorch tensor or DataLoader if necessary.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Sanitized input tensor.
        """
        if isinstance(x, DataLoader):
            x = unwrap_dataloader(x)
        x = to_npf32(x)
        assert x.ndim == 2, 'Input tensor must have 2 dimensions'
        return x

    def sanitize_np_codes(self, z):
        """
        Ensure the codes tensor (Z) is a numpy array of shape (batch_size, nb_concepts).
        Convert from pytorch tensor or DataLoader if necessary.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_concepts).

        Returns
        -------
        torch.Tensor
            Sanitized codes tensor.
        """
        if isinstance(z, DataLoader):
            z = unwrap_dataloader(z)
        z = to_npf32(z)
        assert z.ndim == 2 and z.shape[1] == self.nb_concepts, \
            'Input tensor must have 2 dimensions and nb_concepts columns'
        return z
