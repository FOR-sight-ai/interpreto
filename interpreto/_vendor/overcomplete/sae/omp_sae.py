"""
Module for Orthogonal Matching Pursuit Sparse Autoencoder (OMPSAE).
"""

import torch
from ..sae import SAE
from ..optimization import batched_matrix_nnls


class OMPSAE(SAE):
    """
    Orthogonal Matching Pursuit Sparse Autoencoder (OMPSAE).

    This autoencoder uses an Orthogonal Matching Pursuit strategy to find sparse
    codes. At each iteration, the atom most correlated with the current residual
    is selected, and the full NNLS problem is solved on the selected atoms to update
    the codes. Optionally, dictionary elements can be randomly dropped at each iteration.
    Warning: for this SAE, the encoding is returning (1) the residual and (2) the
    codes -- as the pre_codes are just the input.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data (excluding batch dimension).
    nb_concepts : int
        Number of latent components (atoms) in the dictionary.
    k : int, optional
        Default number of pursuit iterations (must be > 0).
    dropout : float, optional
        Dropout rate applied to dictionary atoms (range [0.0, 1.0]).
    encoder_module : str or nn.Module, optional
        Encoder module or name of registered encoder.
    dictionary_params : dict, optional
        Parameters passed to the dictionary layer.
    device : str, optional
        Device to run the model on (default is 'cpu').
    max_iter : int, optional
        Default number of NNLS iterations (default: 10).
    """

    def __init__(self, input_shape, nb_concepts, k=1, dropout=None,
                 encoder_module="identity", dictionary_params=None, device='cpu',
                 max_iter=10):

        assert isinstance(input_shape, int) or len(input_shape) == 1, \
            "OMPSAE doesn't support 3D or 4D input format."
        assert isinstance(k, int) and k > 0, "k must be a positive integer."
        if dropout is not None:
            assert 0.0 <= dropout <= 1.0, "Dropout must be in range [0, 1]."
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be a positive integer."

        super().__init__(input_shape, nb_concepts, encoder_module, dictionary_params, device)
        self.k = k
        self.dropout = dropout
        self.max_iter = max_iter

    def encode(self, x, k=None, max_iter=None):
        """
        Encode input using Orthogonal Matching Pursuit.

        At each iteration of the pursuit, (1) select the atom most correlated with the residual,
        (2) solve NNLS over all selected atoms, and (3) update codes and residual.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        k : int, optional
            Override the number of pursuit iterations.
        max_iter : int, optional
            Override the number of NNLS iterations.

        Returns
        -------
        residual : torch.Tensor
            Final residual after k iterations.
        codes : torch.Tensor
            Sparse codes of shape (batch_size, nb_concepts).
        """
        k = k if k is not None else self.k
        max_iter = max_iter if max_iter is not None else self.max_iter

        assert isinstance(k, int) and k > 0, "k must be a positive integer."
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be a positive integer."

        W = self.get_dictionary()

        if self.dropout is not None:
            drop_mask = torch.bernoulli(
                (1.0 - self.dropout) * torch.ones(W.shape[0], device=self.device)
            )
            W = W * drop_mask.unsqueeze(1)

        batch_size = x.shape[0]
        codes = torch.zeros(batch_size, self.nb_concepts, device=self.device)
        residual = x.clone()
        selected_atoms = None

        with torch.no_grad():
            for _ in range(k):
                z = residual @ W.T

                if selected_atoms is not None:
                    z.scatter_(dim=1, index=selected_atoms, value=-torch.inf)

                _, idx = torch.topk(z, k=1, dim=1)
                selected_atoms = idx if selected_atoms is None else torch.cat([selected_atoms, idx], dim=1)

                W_sel = W[selected_atoms]
                Z_init = torch.gather(codes, 1, selected_atoms)

                codes_sel = batched_matrix_nnls(
                    W_sel,
                    x,
                    max_iter=max_iter,
                    tol=1e-5,
                    Z_init=Z_init
                )

                codes.scatter_(dim=1, index=selected_atoms, src=codes_sel)
                residual = x - codes @ W

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
