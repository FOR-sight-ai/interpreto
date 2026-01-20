"""
Module for Top-k Sparse SAE (TopKSAE).
"""

import torch
from torch import nn

from ..sae.base import SAE


class TopKSAE(SAE):
    """
    Top-k Sparse SAE.

    The Top-k Sparse Autoencoder is a sparse autoencoder that enforces sparsity
    by keeping only the k largest activations in the latent representation.

             z = {zi if i in top_k_indices(z),
                  0  otherwise}.
             and x_hat = zD.

    For more information, see:
        - "Scaling and evaluating sparse autoencoders"
            by Gao et al. (2024), https://arxiv.org/abs/2406.04093v1

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data, do not include batch dimensions.
        It is usually 1d (dim), 2d (seq length, dim) or 3d (dim, height, width).
    nb_concepts : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > in_dimensions.
    top_k : int, optional
        Number of top activations to keep in the latent representation,
        by default n_components // 10 (sparsity of 90%).
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

    def __init__(self, input_shape, nb_concepts, top_k=None,
                 encoder_module=None, dictionary_params=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))
        if isinstance(top_k, int):
            assert top_k > 0

        super().__init__(input_shape, nb_concepts, encoder_module,
                         dictionary_params, device)

        self.top_k = top_k if top_k is not None else max(nb_concepts // 10, 1)

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
            Pre-codes tensor of shape (batch_size, nb_components) before the relu and top-k operation.
        z : torch.Tensor
            Codes, latent representation tensor (z) of shape (batch_size, nb_components).
        """
        pre_codes, codes = self.encoder(x)

        z_topk = torch.topk(codes, self.top_k, dim=-1)
        z_topk = torch.zeros_like(codes).scatter(-1, z_topk.indices, z_topk.values)

        return pre_codes, z_topk
