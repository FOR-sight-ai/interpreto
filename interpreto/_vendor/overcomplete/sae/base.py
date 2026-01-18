"""
Base module for Sparse Autoencoder (SAE) model for dictionary learning.
"""

import torch
from torch import nn
from dataclasses import dataclass

from ..base import BaseDictionaryLearning
from .dictionary import DictionaryLayer
from .factory import EncoderFactory
from .modules import TieableEncoder


class SAE(BaseDictionaryLearning):
    """
    Sparse Autoencoder (SAE) model for dictionary learning.

    The SAE for Overcomplete models follows a common structure, consisting of:
    - (i) An encoder that returns code embeddings (concepts) for each token. This
          embedding is always 2D (N, n_components). The encoder can handle 2D
          (seq_len, dim) or 3D (dim, height, width) input data. At the end of
          encoding, a rearrangement/flattening is applied so that each token has a
          unique description.
    - (ii) A dictionary layer, which is a matrix multiplication between the code
           values produced by the encoder and a dictionary. To reconstruct the
           activation, the dimension 'dim' will be chosen. In the 1D case, it is
           the unique dimension; in the 2D case (seq_len, dim), it is the second
           dimension; and in the 3D case (dim, height, width), it is the first
           dimension.
    After the dictionary layer, we end up with a vector of dimension 2 (N, dim),
    where dim is the 'channel' dimension.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data, do not include batch dimensions.
        It is usually 1d (dim), 2d (seq length, dim) or 3d (dim, height, width).
    nb_concepts : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > in_dimensions.
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
        Perform a forward pass through the autoencoder and return a triplet of tensors composed of
        of pre_codes, codes and reconstructed input tensor.
    encode(x):
        Encode input data to latent representation.
    decode(z):
        Decode latent representation to reconstruct input data.
    """

    def __init__(self, input_shape, nb_concepts, encoder_module=None,
                 dictionary_params=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))

        super().__init__(nb_concepts=nb_concepts, device=device)

        # initialize the encoder,
        # - either string and we use the fatctory
        # - directly a module
        # - or None and we use the default encoder (linear)
        if isinstance(encoder_module, str):
            assert encoder_module in EncoderFactory.list_modules(), f"Encoder '{encoder_module}' not found in registry."
            self.encoder = EncoderFactory.create_module(encoder_module, input_shape, nb_concepts, device=device)
        elif encoder_module is not None:
            self.encoder = encoder_module
        else:
            assert isinstance(input_shape, int), "Default encoder assumes input_shape is an int."
            self.encoder = EncoderFactory.create_module("linear", input_shape, nb_concepts, device=device)

        # initialize the dictionary, but first find the channel dimension
        # tfel: do we really need this parameter if an encoder module is passed?
        if isinstance(input_shape, int):
            in_dim = input_shape
        elif len(input_shape) == 2:
            # attention model case with shape (T C)
            in_dim = input_shape[1]
        elif len(input_shape) == 3:
            # convolutional model case with shape (C H W)
            in_dim = input_shape[0]
        else:
            raise ValueError("Input shape must be 1D, 2D or 3D.")

        self.dictionary = DictionaryLayer(in_dim, nb_concepts, device=device, **(dictionary_params or {}))

    def get_dictionary(self):
        """
        Return the learned dictionary.

        Returns
        -------
        torch.Tensor
            Learned dictionary tensor of shape (nb_components, input_size).
        """
        return self.dictionary.get_dictionary()

    def forward(self, x):
        """
        Perform a forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        SAEOuput
            Return the pre_codes (z_pre), codes (z) and reconstructed input tensor (x_hat).
        """
        pre_codes, codes = self.encode(x)

        x_reconstructed = self.decode(codes)

        return pre_codes, codes, x_reconstructed

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
            Pre-codes tensor of shape (batch_size, nb_components), before the activation function.
        codes : torch.Tensor
            Codes, latent representation tensor (z) of shape (batch_size, nb_components).
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent representation to reconstruct input data.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor of shape (batch_size, input_size).
        """
        return self.dictionary(z)

    def fit(self, x):
        """
        Method not implemented for SAE. See train_sae function for training the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        """
        raise NotImplementedError('SAE does not support fit method. You have to train the model \
                                  using a custom training loop.')

    def tied(self, bias=False):
        """
        Tie encoder weights to dictionary (use D^T as encoder).

        Parameters
        ----------
        bias : bool, optional
            Whether to include bias in encoder, by default False.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self.encoder = TieableEncoder(
            in_dimensions=self.dictionary.in_dimensions,
            nb_concepts=self.nb_concepts,
            bias=bias,
            tied_to=self.dictionary,
            device=self.device
        )
        return self

    def untied(self, bias=False, copy_from_dictionary=True):
        """
        Create a new encoder with weight from the current dictionary (or random init).

        Parameters
        ----------
        bias : bool, optional
            Whether to include bias in encoder, by default False.
        copy_from_dictionary : bool, optional
            If True, initialize encoder with current dictionary weights, by default True.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        weight_init = self.get_dictionary().clone().detach() if copy_from_dictionary else None
        self.encoder = TieableEncoder(
            in_dimensions=self.dictionary.in_dimensions,
            nb_concepts=self.nb_concepts,
            bias=bias,
            tied_to=None,
            weight_init=weight_init,
            device=self.device
        )
        return self
