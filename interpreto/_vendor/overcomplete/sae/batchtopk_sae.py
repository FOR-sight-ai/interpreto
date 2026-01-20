"""
Module for Batch Top-k Sparse Autoencoder (BatchTopKSAE).
"""

import torch
from torch import nn

from ..sae import SAE


class BatchTopKSAE(SAE):
    """
    Batch Top-k Sparse SAE.

    This autoencoder retains only the top-k global activations across the entire
    batch of latent codes. The kth highest activation (when flattening all codes)
    is used as the threshold. During training, a running threshold is updated with
    a specified momentum. In evaluation mode, the running threshold is used.

    For more information, see:
        "Batch Top-k Sparse Autoencoders",
        by Bussmann et al. (2024) https://arxiv.org/pdf/2412.06410

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of latent dimensions (components) of the autoencoder.
    top_k : int
        The number of activations to keep (the kth highest activation is used as threshold).
    threshold_momentum : float, optional
        Momentum for the running threshold update (default is 0.9).
    encoder_module : nn.Module or str, optional
        Custom encoder module (or its registered name). If None, a default encoder is used.
    dictionary_params : dict, optional
        Parameters that will be passed to the dictionary layer.
        See DictionaryLayer for more details.
    device : str, optional
        Device on which to run the model (default is 'cpu').
    """

    def __init__(self, input_shape, nb_concepts, top_k=None, threshold_momentum=0.9,
                 encoder_module=None, dictionary_params=None, device='cpu'):
        super().__init__(input_shape, nb_concepts, encoder_module,
                         dictionary_params, device)
        self.top_k = top_k if top_k is not None else max(nb_concepts // 10, 1)
        self.threshold_momentum = threshold_momentum
        self.running_threshold = None

    def encode(self, x):
        """
        Encode input data and apply global top-k thresholding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        pre_codes : torch.Tensor
            The raw outputs from the encoder.
        z : torch.Tensor
            The sparse latent representation after thresholding.
        """
        pre_codes, codes = self.encoder(x)

        if self.training:
            # @tfel: for simplicity we avoid checking for the weird case where top_k > numel(codes)
            flattened = codes.view(-1)
            topk_vals, _ = torch.topk(flattened, self.top_k)
            current_threshold = topk_vals[-1]

            if self.running_threshold is None:
                # init running threshold (first time)
                self.running_threshold = current_threshold.detach()
            else:
                self.running_threshold = (
                    self.threshold_momentum * self.running_threshold +
                    (1 - self.threshold_momentum) * current_threshold.detach()
                )
            threshold = current_threshold
        else:
            # we are in eval, use the running threshold if available.
            assert self.running_threshold is not None, "Running threshold is not initialized."
            threshold = self.running_threshold

        mask = (codes >= threshold).float().detach()
        z = codes * mask

        return pre_codes, z
