"""
Module for Quantized Sparse SAE (Q-SAE).
"""

import torch
from torch import nn

from .base import SAE


class QSAE(SAE):
    """
    Quantized SAE.

    @tfel: The Quantized Sparse Autoencoder is a sparse autoencoder that will learn
    level of codes shared across dimension, and project the pre_code to the closest
    quantization point in a set of q points. The quantization points are learned
    during training and are initialize around 0:
        Q0 = [-1, ..., 1], with #|Q| = q.

    @tfel: this is an unpublished work and purely experimental.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data, do not include batch dimensions.
        It is usually 1d (dim), 2d (seq length, dim) or 3d (dim, height, width).
    nb_concepts : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > in_dimensions.
    q : int, optional
        Number of top quantized steps to keep in the latent representation.
    hard : bool, optional
        Whether to use hard quantization (True) or soft quantization (False),
        by default False. Hard quantization is slower and more memory intensive.
    encoder_module : nn.Module or string, optional
        Custom encoder module, by default None.
        If None, a simple Linear + BatchNorm default encoder is used.
        If string, the name of the registered encoder module.
    dictionary_params : dict, optional
        Parameters that will be passed to the dictionary layer.
        See DictionaryLayer for more details.
    device : str, optional
        Device to run the model on, by default 'cpu'.

    Methods
    -------
    get_dictionary():
        Return the learned dictionary.
    forward(x):
        Perform a forward pass through the autoencoder.
    encode(x):
        Encode input data to latent representation.
    decode(z):
        Decode latent representation to reconstruct input data.
    """

    def __init__(self, input_shape, nb_concepts, q=4, hard=False,
                 encoder_module=None, dictionary_params=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))
        assert q > 1, "You need at least 2 quantization levels."

        super().__init__(input_shape, nb_concepts, encoder_module,
                         dictionary_params, device)

        # initialize linearly around [-1, 1]
        Q = torch.linspace(0.0, 1.0, q, device=device).float()
        self.Q = nn.Parameter(Q, requires_grad=True)

        self.hard = hard

    def encode(self, x):
        """
        Encode input data to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        pre_codes : torch.Tensor
            Pre-codes tensor of shape (batch_size, n_components), before the relu and quantization
            operation.
        codes : torch.Tensor
            Codes, latent representation tensor (z) of shape (batch_size, n_components
        """
        pre_codes, codes = self.encoder(x)

        # compute the distance from pre_codes to each quantization state
        dist = (pre_codes[:, :, None] - self.Q[None, None, :]).square()

        if self.hard:
            # take the closest id, costly
            closest_idx = dist.argmin(dim=-1)

            # @tfel: could be more efficient
            one_hot = torch.nn.functional.one_hot(closest_idx, num_classes=self.Q.size(-1)).float()
            quantized_codes = torch.sum(one_hot * self.Q[None, None, :], -1)
        else:
            # soft quantization
            # compute the softmax of the negative distance
            closest_idx = torch.nn.functional.softmax(-dist, dim=-1)
            quantized_codes = torch.sum(closest_idx * self.Q[None, None, :], -1)

        # straight-through estimator
        quantized_codes = codes + quantized_codes - codes.detach()

        quantized_codes = torch.relu(quantized_codes)

        return pre_codes, quantized_codes
