"""
Collections of torch modules for the encoding of SAE.
"""

import torch
from torch import nn
from einops import rearrange


class IdentityEncoder(nn.Module):
    """
    Identity Encoder for SAE.

    This encoder simply returns the input as both the pre-activation and encoded representation.
    It is useful for some testing purposes or SAEs where no transformation is needed.

    Parameters
    ----------
    input_shape : int
        The input size of the encoder. Must be a single integer.
    """

    def __init__(self, input_shape, n_components=None):
        super().__init__()
        assert isinstance(input_shape, int), "Input shape must be a single integer."
        self.input_size = input_shape

    def forward(self, x):
        """
        Forward pass through the Identity Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        pre_z : torch.Tensor
            Pre-activation tensor (same as input).
        z : torch.Tensor
            Encoded tensor (same as input).
        """
        assert x.shape[1] == self.input_size, "Input tensor shape mismatch."

        return x, x


class MLPEncoder(nn.Module):
    """
    Linear + Normalisation + Residual (optional) Encoder module for the SAE.

    Parameters
    ----------
    input_shape : int
        The input size of the encoder. Only accept a single dimension.
    n_components : int
        The number of components (or concepts) in the latent representation.
    hidden_dim : int, optional
        The hidden dimension of the encoder, by default use the input_size.
    nb_blocks : int, optional
        The number of blocks in the encoder, by default 1.
        Each block consists of a Linear layer, normalisation and an activation.
    hidden_activation : nn.Module, optional
        The activation function to use in the hidden layers, by default nn.ReLU.
    output_activation : nn.Module, optional
        The activation function to use in the output layer, by default nn.ReLU.
    norm_layer : nn.Module, optional
        The normalization layer to use, by default nn.BatchNorm1d.
    residual : bool, optional
        Whether to use residual connections, by default False.
    device : torch.device, optional
        The device to use, by default 'cpu'.
    """

    def __init__(self, input_shape, n_components, hidden_dim=None, nb_blocks=1,
                 hidden_activation=nn.ReLU, output_activation=nn.ReLU, norm_layer=nn.BatchNorm1d,
                 residual=False, device='cpu'):
        # we authorize nb_blocks = 0 that give the simplest linear + norm block
        assert nb_blocks >= 0, "The number of blocks must be greater than 0."
        assert isinstance(input_shape, int), "The input size must be a single integer."

        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_shape

        self.input_size = input_shape
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.nb_blocks = nb_blocks
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.norm_layer = norm_layer
        self.residual = residual

        last_dim = input_shape
        self.mlp_blocks = nn.ModuleList()

        # construct the n-1 hidden blocks
        for _ in range(nb_blocks-1):
            block = nn.Sequential(
                nn.Linear(last_dim, hidden_dim),
                norm_layer(hidden_dim),
                hidden_activation(),
            ).to(device)
            last_dim = hidden_dim
            self.mlp_blocks.append(block)

        # last block is just a linear layer to the components/concepts space
        self.final_block = nn.Sequential(
            nn.Linear(last_dim, n_components),
            norm_layer(n_components),
        ).to(device)
        self.final_activation = output_activation()

    def forward(self, x):
        """
        Forward pass through the encoder with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        pre_z : torch.Tensor
            Pre-activation (pre-codes) tensor of shape (batch_size, n_components).
        z : torch.Tensor
            Encoded tensor (codes) of shape (batch_size, n_components).
        """
        assert len(x.shape) == 2, "Input tensor must have 2 dimensions (batch, input_size)."

        for layer_i, layer in enumerate(self.mlp_blocks):
            residual = x
            x = layer(x)
            # if hidden_dim is not None, we may not be able to do the
            # first residual connection (as we reduce the dim, but only once)
            if self.residual and (layer_i != 0 or self.hidden_dim == self.input_size):
                x = x + residual

        pre_z = self.final_block(x)
        z = self.final_activation(pre_z)

        return pre_z, z


class AttentionBlock(nn.Module):
    """
    Vision Transformer (ViT) style Attention Block with LayerNorm and residual connection.

    Parameters
    ----------
    dims : tuple (int, int)
        The dimensions of the input tensor, nb_token and nb_dim.
    num_heads : int
        The number of attention heads.
    mlp_ratio : float, optional
        The ratio of the hidden dimension to the input dimension in the MLP, by default 4.0.
    drop : float, optional
        Dropout rate, by default 0.0.
    attn_drop : float, optional
        Attention dropout rate, by default 0.0.
    drop_path : float, optional
        Drop path rate, by default 0.0.
    act_layer : nn.Module, optional
        Activation layer, by default nn.GELU.
    device : torch.device, optional
        The device to use, by default 'cpu'.
    """

    def __init__(self, dims, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 act_layer=nn.GELU, device='cpu'):
        assert len(dims) == 2, "The input dimensions must be a tuple of (nb_token, nb_dim)."

        super().__init__()
        self.seq_len, self.embed_dim = dims
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dims).to(device)
        self.attn = nn.MultiheadAttention(self.embed_dim,
                                          self.num_heads, dropout=attn_drop).to(device)
        self.norm2 = nn.LayerNorm(dims).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(int(self.embed_dim * mlp_ratio), self.embed_dim),
            nn.Dropout(drop),
        ).to(device)

    def forward(self, x):
        """
        Forward pass through the attention block with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size).

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, seq_length, input_size).
        """
        assert len(x.shape) == 3, "Input tensor must have 3 dimensions (batch, seq_length, input_size)."

        x1 = self.norm1(x)
        x = x + self.attn(x1, x1, x1)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionEncoder(nn.Module):
    """
    Attention Encoder module for the SAE with flattening and linear transformation at the end.
    Uses ViT style attention blocks by default.

    Parameters
    ----------
    input_shape : tuple (int, int)
        The input size of the encoder input composed of shape (seq len, dim).
    n_components : int
        The number of components (or concepts) in the latent representation.
    hidden_dim : int, optional
        The hidden dimension of the encoder, by default use the input_size.
    nb_blocks : int, optional
        The number of attention blocks in the encoder, by default 1.
    output_activation : nn.Module, optional
        The activation function to use in the output layer, by default nn.ReLU.
    norm_layer : nn.Module, optional
        The normalization layer to use for the final block, by default nn.LayerNorm.
    residual : bool, optional
        Whether to use residual connections, by default True.
    attention_heads : int, optional
        The number of attention heads, by default 4.
    mlp_ratio : float, optional
        The ratio of the hidden dimension to the input dimension in the MLP, by default 4.0.
    device : torch.device, optional
        The device to use, by default 'cpu'.
    """

    def __init__(self, input_shape, n_components, hidden_dim=None, nb_blocks=1,
                 output_activation=nn.ReLU, norm_layer=nn.LayerNorm,
                 residual=True, attention_heads=4, mlp_ratio=4.0, device='cpu'):
        assert nb_blocks > 0, "The number of blocks must be greater than 0."
        assert len(input_shape) == 2, "The input shape must be a tuple of (nb_token, nb_dim)."

        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_shape

        self.input_size = input_shape
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.nb_blocks = nb_blocks
        self.output_activation = output_activation
        self.norm_layer = norm_layer
        self.residual = residual
        self.attention_heads = attention_heads
        self.mlp_ratio = mlp_ratio

        self.attention_blocks = nn.ModuleList()

        # construct the n attention blocks using ViT-style attention
        for _ in range(nb_blocks):
            block = AttentionBlock(
                dims=input_shape,
                num_heads=attention_heads,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
            ).to(device)
            self.attention_blocks.append(block)

        # flatten and apply final linear transformation, no more mixing between tokens
        # each token will have a unique concept representation allowed to be reconstructed
        self.final_block = nn.Sequential(
            nn.Linear(input_shape[-1], n_components),
            norm_layer(n_components),
        ).to(device)

        self.final_activation = output_activation()

    def forward(self, x):
        """
        Forward pass through the encoder with attention blocks and residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size).

        Returns
        -------
        pre_z : torch.Tensor
            Pre-activation (pre-codes) tensor of shape (batch_size * seq_length, n_components).
        z : torch.Tensor
            Encoded tensor (codes) of shape (batch_size * seq_length, n_components).
        """
        assert len(x.shape) == 3, "Input tensor must have 3 dimensions (batch, seq_length, input_size)."

        for block in self.attention_blocks:
            x = block(x)

        # flatten, no more mixing between token of the same input
        # from (N, t, d) to (N * t, d)
        x = rearrange(x, 'n t d -> (n t) d')

        # final block to get concepts/features embeds for each token
        pre_z = self.final_block(x)
        z = self.final_activation(pre_z)

        return pre_z, z


class ResNetBlock(nn.Module):
    """
    ResNet Block with normalization and activation.

    Parameters
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    stride : int, optional
        The stride of the convolution, by default 1.
    activation : nn.Module, optional
        The activation function to use, by default nn.ReLU.
    device : torch.device, optional
        The device to use, by default 'cpu'.
    """

    def __init__(self, in_channels, out_channels, stride=1, activation=nn.GELU, device='cpu'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False).to(device)
        self.norm1 = nn.BatchNorm2d(out_channels).to(device)
        self.act = activation().to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False).to(device)
        self.norm2 = nn.BatchNorm2d(out_channels).to(device)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ).to(device)
        else:
            self.downsample = nn.Identity().to(device)

    def forward(self, x):
        """
        Forward pass through the ResNet block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        assert len(x.shape) == 4, "Input tensor must have 4 dimensions (batch, channels, height, width)."

        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)

        return out


class ResNetEncoder(nn.Module):
    """
    ResNetV2-like Encoder module for the SAE.

    Parameters
    ----------
    input_shape : tuple (int, int, int)
        The input size of the encoder input composed of shape (channels, height, width).
    n_components : int
        The number of components (or concepts) in the latent representation.
    hidden_dim : int, optional
        The hidden dimension of the encoder, by default 64.
    nb_blocks : int, optional
        The number of ResNet blocks in the encoder, by default 3.
    output_activation : nn.Module, optional
        The activation function to use in the output layer, by default nn.ReLU.
    device : torch.device, optional
        The device to use, by default 'cpu'.
    """

    def __init__(
            self, input_shape, n_components, hidden_dim=None, nb_blocks=1,
            output_activation=nn.ReLU, device='cpu'):
        assert nb_blocks > 0, "The number of blocks must be greater than 0."
        assert len(input_shape) == 3, "The input shape must contain 3 dimensions (channels, height, width)."

        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_shape[0]

        self.input_channels = input_shape[0]
        self.spatial_dims = input_shape[1:]
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.nb_blocks = nb_blocks

        self.resnet_blocks = nn.ModuleList()
        last_dim = self.input_channels
        for _ in range(nb_blocks):
            self.resnet_blocks.append(ResNetBlock(last_dim, hidden_dim).to(device))
            last_dim = hidden_dim
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks).to(device)

        # flattening / reshape is done in the forward pass
        # no more mixing between tokens, each token will have a unique concept
        # representation allowed to be reconstructed
        self.final_block = nn.Sequential(
            nn.Linear(last_dim, n_components),
            nn.BatchNorm1d(n_components),
        ).to(device)
        self.final_activation = output_activation()

    def forward(self, x):
        """
        Forward pass through the ResNetV2-like encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        pre_z : torch.Tensor
            Pre-activation (pre-codes) tensor of shape (batch_size, n_components).
        z :
            Encoded tensor (codes) of shape (batch_size, n_components).
        """
        assert len(x.shape) == 4, "Input tensor must have 4 dimensions (batch, channels, height, width)."

        x = self.resnet_blocks(x)

        # flatten, no more mixing between 'tokens'/ 'spatial dims' of the same input
        x = rearrange(x, 'n c h w -> (n h w) c')

        # then get the final codes / concepts
        pre_z = self.final_block(x)
        z = self.final_activation(pre_z)

        return pre_z, z


class TieableEncoder(nn.Module):
    """
    Linear encoder that can be tied to a dictionary or use independent weights.

    Parameters
    ----------
    in_dimensions : int
        Input dimensionality.
    nb_concepts : int
        Number of latent dimensions.
    bias : bool, optional
        Whether to include bias, by default True.
    tied_to : DictionaryLayer, optional
        If provided, uses D^T (tied weights). If None, uses independent weights, by default None.
    weight_init : torch.Tensor, optional
        Initial weights for untied mode, by default None (uses Xavier initialization).
    device : str, optional
        Device for parameters, by default 'cpu'.
    """

    def __init__(self, in_dimensions, nb_concepts, bias=False,
                 tied_to=None, weight_init=None, device='cpu'):
        super().__init__()
        self.tied_to = tied_to

        if tied_to is None:
            # untied: create own weights
            self.weight = nn.Parameter(torch.empty(nb_concepts, in_dimensions, device=device))
            if weight_init is not None:
                self.weight.data.copy_(weight_init)
            else:
                nn.init.xavier_uniform_(self.weight)
        else:
            # tied weights: we use the dictionary transpose as encoder
            # no weights needed
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(nb_concepts, device=device))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        Encode input.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, in_dimensions).

        Returns
        -------
        z_pre : torch.Tensor
            Pre-activation codes.
        z : torch.Tensor
            Activated codes (ReLU applied).
        """
        if self.tied_to is not None:
            z_pre = x @ self.tied_to.get_dictionary().T
        else:
            # untied mode: use own weights
            z_pre = x @ self.weight.T

        if self.bias is not None:
            z_pre = z_pre + self.bias

        z = torch.relu(z_pre)
        return z_pre, z
