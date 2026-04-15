"""
Collection of module creation functions for SAE encoder
and a factory class to create modules using string identifier.

example usage:

model = EncoderFactory.create_module("mlp_ln_1") # 1 block mlp with layer norm
model = EncoderFactory.create_module("mlp_bn_3") # 3 blocks mlp with batch norm

model = EncoderFactory.create_module("resnet_1b") # 1 block resnet
model = EncoderFactory.create_module("attention_3b") # 3 blocks attention

you can also pass additional arguments to the module creation function:

model = EncoderFactory.create_module("mlp_ln_1", hidden_dim=128)
model = EncoderFactory.create_module("attention_1b", attention_heads=2)
"""
# ruff: noqa: D103

from torch import nn

from .modules import IdentityEncoder, MLPEncoder, AttentionEncoder, ResNetEncoder


class EncoderFactory:
    """
    Factory class to create modules using registered module creation functions.
    """
    _module_registry = {}

    @staticmethod
    def register_module(name):
        """
        Decorator to register a module creation function.

        Parameters
        ----------
        name : str
            The name to register the module creation function under.
        """
        def decorator(func):
            EncoderFactory._module_registry[name] = func
            return func
        return decorator

    @staticmethod
    def create_module(name, *args, **kwargs):
        """
        Creates a module based on the registered name.

        Parameters
        ----------
        name : str
            The name of the registered module creation function.
        *args : tuple
            Positional arguments to pass to the module creation function.
        **kwargs : dict
            Keyword arguments to pass to the module creation function.

        Returns
        -------
        nn.Module
            The initialized module.
        """
        if name not in EncoderFactory._module_registry:
            raise ValueError(f"Module '{name}' not found in registry.")
        return EncoderFactory._module_registry[name](*args, **kwargs)

    @staticmethod
    def list_modules():
        """
        Lists all registered modules.

        Returns
        -------
        list
            A list of names of all registered modules.
        """
        return list(EncoderFactory._module_registry.keys())


@EncoderFactory.register_module("identity")
def identity(input_shape, n_components=None, **kwargs):
    """
    Creates an Identity Encoder.

    Parameters
    ----------
    input_shape : int
        The input size of the encoder.
    """
    return IdentityEncoder(input_shape, n_components)


@EncoderFactory.register_module("linear")
def linear(input_shape, n_components, **kwargs):
    return MLPEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=0,
        norm_layer=nn.Identity,
        ** kwargs
    )


@EncoderFactory.register_module("mlp_ln_1")
def mlp_ln_1(input_shape, n_components, **kwargs):
    return MLPEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=1,
        norm_layer=nn.LayerNorm,
        **kwargs
    )


@EncoderFactory.register_module("mlp_ln_3")
def mlp_ln_3(input_shape, n_components, **kwargs):
    return MLPEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=3,
        norm_layer=nn.LayerNorm,
        **kwargs
    )


@EncoderFactory.register_module("mlp_bn_1")
def mlp_bn_1(input_shape, n_components, **kwargs):
    return MLPEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=1,
        norm_layer=nn.BatchNorm1d,
        **kwargs
    )


@EncoderFactory.register_module("mlp_bn_3")
def mlp_bn_3(input_shape, n_components, **kwargs):
    return MLPEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=3,
        norm_layer=nn.BatchNorm1d,
        **kwargs
    )


@EncoderFactory.register_module("resnet_1b")
def resnet_1b(input_shape, n_components, **kwargs):
    return ResNetEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=1,
        **kwargs
    )


@EncoderFactory.register_module("resnet_3b")
def resnet_3b(input_shape, n_components, **kwargs):
    return ResNetEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=3,
        **kwargs
    )


@EncoderFactory.register_module("attention_1b")
def attention_1b(input_shape, n_components, **kwargs):
    return AttentionEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=1,
        **kwargs
    )


@EncoderFactory.register_module("attention_3b")
def attention_3b(input_shape, n_components, **kwargs):
    return AttentionEncoder(
        input_shape=input_shape,
        n_components=n_components,
        nb_blocks=3,
        **kwargs
    )
