"""
Module for Matching Pursuit Sparse Autoencoder (MpSAE).
"""

import torch
from torch import nn

from ..sae import SAE


class MpSAE(SAE):
    """
    Matching Pursuit Sparse Autoencoder (MpSAE).

    This autoencoder uses a greedy Matching Pursuit strategy to obtain sparse
    codes. Specifically, at each iteration the dictionary atom most correlated
    with the current residual is chosen and its contribution is subtracted
    from the residual. This process is repeated k times. An optional dropout
    can be applied on the dictionary elements at each iteration.
    Warning: for this SAE, the encoding is returning (1) the residual and (2) the
    codes -- as the pre_codes are just the input.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of latent dimensions (components) of the autoencoder.
    k : int, optional
        The number of matching pursuit iterations to perform (must be > 0).
    dropout : float, optional
        Probability of dropping a dictionary element at each iteration
        (range: 0.0 - 1.0). If None, no dropout is applied.
    encoder_module : nn.Module or str, optional
        Custom encoder module (or its registered name). If None, a default encoder is used.
    dictionary_params : dict, optional
        Parameters that will be passed to the dictionary layer.
        See DictionaryLayer for more details.
    device : str, optional
        Device on which to run the model (default is 'cpu').
    """

    def __init__(self, input_shape, nb_concepts, k=1, dropout=None,
                 encoder_module="identity", dictionary_params=None, device='cpu'):
        # input shape must be int or length-1 tuple
        assert isinstance(input_shape, int) or len(input_shape) == 1, \
            "MpSAE Doesn't handle 3d or 4d data format."
        if isinstance(k, int):
            assert k > 0, "k must be a positive integer."

        super().__init__(input_shape, nb_concepts,
                         encoder_module, dictionary_params, device)
        self.k = k
        self.dropout = dropout

    def encode(self, x):
        """
        Encode input data with a greedy Matching Pursuit approach.

        The dictionary (W) is optionally subjected to dropout. Then, at each
        of k iterations:
            1) Compute the correlation (dot product) between the residual and each
               dictionary atom.
            2) Identify the maximum correlation and corresponding dictionary atom index.
            3) Update the codes by adding that contribution.
            4) Subtract the chosen atom (scaled by the correlation) from the residual.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        residual : torch.Tensor
            The residual after k Matching Pursuit iterations.
        codes : torch.Tensor
            The final sparse codes obtained after k Matching Pursuit iterations.
        """
        W = self.get_dictionary()

        if self.dropout is not None:
            # dropout directly on the weights (dictionary) atoms
            drop_w = torch.bernoulli((1.0 - self.dropout) *
                                     torch.ones(self.nb_concepts, 1, device=self.device))
            W = W * drop_w

        codes = torch.zeros(x.shape[0], self.nb_concepts, device=self.device)
        residual = x.clone()

        # greedy selection of dictionary atoms
        for _ in range(self.k):
            z = residual @ W.T  # pre_codes as projection of the current residual
            val, idx = torch.max(z, dim=1)

            # add top concept to the current codes
            to_add = torch.nn.functional.one_hot(idx, num_classes=self.nb_concepts).float()
            to_add = to_add * val.unsqueeze(1)

            # accumulate contribution and update residual
            codes = codes + to_add
            residual = residual - to_add @ W

        return residual, codes

    def train(self, mode=True):
        """
        Hook called when switching between training and evaluation mode.
        We use it to ensure no dropout is applied during evaluation.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to training mode or not, by default True.
        """
        if not mode:
            self.dropout = None

        return super().train(mode)
