"""
Base classes for (overcomplete) dictionary learning methods.
"""

from abc import ABC, abstractmethod

from torch import nn


class BaseDictionaryLearning(ABC, nn.Module):
    """
    Abstract base class for Dictionary Learning models.

    Parameters
    ----------
    nb_concepts : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > in_dimensions.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    encode(x):
        Encode the input tensor (the activations).
    decode(z):
        Decode the input tensor (the codes).
    fit(x):
        Fit the model to the input data.
    get_dictionary():
        Return the learned dictionary components.
    """

    def __init__(self, nb_concepts, device='cpu'):
        super().__init__()
        self.nb_concepts = nb_concepts
        self.device = device
        self.fitted = False
        self._fitted_info = {}

    @abstractmethod
    def encode(self, x):
        """
        Encode the input tensor (the activations).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features.
        """
        pass

    @abstractmethod
    def decode(self, z):
        """
        Decode the input tensor (the codes).

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor of shape (batch_size, nb_concepts).

        Returns
        -------
        torch.Tensor
            Decoded output.
        """
        pass

    @abstractmethod
    def fit(self, x):
        """
        Fit the model to the input data (the activations).

        Parameters
        ----------
        x : torch.Tensor, Iterable
            Input tensor of shape (batch_size, ...).
        """
        pass

    @abstractmethod
    def get_dictionary(self):
        """
        Return the learned dictionary components.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        pass

    def _assert_fitted(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before calling this method.")

    def _set_fitted(self, **kwargs):
        """
        Set the model as fitted.

        Parameters
        ----------
        kwargs : dict
            Additional information to store.
        """
        self.fitted = True
        self._fitted_info = kwargs
