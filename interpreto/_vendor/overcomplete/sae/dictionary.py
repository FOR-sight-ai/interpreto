"""
Module dedicated to everything around the Dictionary Layer of SAE.
"""

import torch
from torch import nn


def normalize_l2(x):
    """
    Project each concept in the dictionary "on" the l2 ball.

    Parameters
    ----------
    d : torch.Tensor
        Dictionary tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor along the concept dimension, such that
        each concept has l2 norm of 1.
    """
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)


def normalize_max_l2(x):
    """
    Project each concept in the dictionary "in" the l2 ball.

    Parameters
    ----------
    d : torch.Tensor
        Dictionary tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor along the concept dimension, such that
        the l2 norm of the concepts is 1 or less.
    """
    return x / torch.amax(torch.norm(x, p=2, dim=1, keepdim=True), dim=0, keepdim=True)


def normalize_l1(x):
    """
    Project each concept in the dictionary "on" the l1 ball.

    Parameters
    ----------
    d : torch.Tensor
        Dictionary tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor along the concept dimension, such that
        the l1 norm of the concepts is 1.
    """
    return x / (torch.norm(x, p=1, dim=1, keepdim=True) + 1e-8)


def normalize_max_l1(x):
    """
    Project each concept in the dictionary "in" the l1 ball.

    Parameters
    ----------
    d : torch.Tensor
        Dictionary tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor along the concept dimension, such that
        the l1 norm of the concepts is 1 or less.
    """
    return x / torch.amax(torch.norm(x, p=1, dim=1, keepdim=True), dim=0, keepdim=True)


def normalize_identity(x):
    """
    Identity "normalization".

    Parameters
    ----------
    d : torch.Tensor
        Dictionary tensor.

    Returns
    -------
    torch.Tensor
        The input tensor.
    """
    return x


class DictionaryLayer(nn.Module):
    """
    A neural network layer representing a dictionary for reconstructing input data.

    Parameters
    ----------
    in_dimensions : int
        Dimensionality of the input data (e.g number of channels).
    nb_concepts : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > in_dimensions.
    normalization : str or callable, optional
        Whether to normalize the dictionary, by default 'l2' normalization is applied.
        Current options are 'l2', 'max_l2', 'l1', 'max_l1', 'identity'.
        If a custom normalization is needed, a callable can be passed.
    initializer : torch.Tensor, optional
        Initial dictionary tensor, by default None.
    use_multiplier : bool, optional
        Whether to train a positive scalar to multiply the dictionary after normalization,
        (e.g. to control the radius of the l2 ball when l2 norm is applied), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
    """

    NORMALIZATIONS = {
        "l2": normalize_l2,
        "max_l2": normalize_max_l2,
        "l1": normalize_l1,
        "max_l1": normalize_max_l1,
        "identity": normalize_identity,
    }

    def __init__(
        self, in_dimensions, nb_concepts,
        normalization="l2", initializer=None, use_multiplier=False,
        device="cpu"
    ):
        super().__init__()
        self.in_dimensions = in_dimensions
        self.nb_concepts = nb_concepts
        self.device = device

        # prepare normalization function
        if isinstance(normalization, str):
            self.normalization = self.NORMALIZATIONS[normalization]
        elif callable(normalization):
            self.normalization = normalization
        else:
            raise ValueError("Invalid normalization function")

        # init weights
        self._weights = nn.Parameter(torch.empty(nb_concepts, in_dimensions, device=device))
        if initializer is None:
            nn.init.xavier_uniform_(self._weights)
        else:
            assert initializer.shape == self._weights.shape, "Invalid initializer, must have the same shape as the dictionary."
            self._weights.data = torch.tensor(initializer, device=device)

        # init multiplier
        if use_multiplier:
            self.multiplier = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        else:
            # by default, the multiplier will be exp(0) = 1 and not trainable
            self.register_buffer("multiplier", torch.tensor(0.0, device=device))

        self._fused_dictionary = None

    def forward(self, z):
        """
        Reconstruct input data from latent representation.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor of shape (batch_size, dimensions).
        """
        dictionary = self.get_dictionary()
        x_hat = torch.matmul(z, dictionary)
        return x_hat

    def get_dictionary(self):
        """
        Get the dictionary.

        Returns
        -------
        torch.Tensor
            The dictionary tensor of shape (nb_components, dimensions).
        """
        if self.training:
            # we are in training mode, apply normalization
            with torch.no_grad():
                self._weights.data = self.normalization(self._weights)
            return self._weights * torch.exp(self.multiplier)
        else:
            # we are in eval mode, return the fused dictionary
            assert self._fused_dictionary is not None, "Dictionary is not initialized."
            return self._fused_dictionary

    def train(self, mode=True):
        """
        Hook called when switching between training and evaluation mode.
        We use it to fuse W, C, Relax and multiplier into a single dictionary.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to training mode or not, by default True.
        """
        if not mode:
            # we are in .eval() mode, fuse the dictionary
            self._fused_dictionary = self.get_dictionary()

        return super().train(mode)
