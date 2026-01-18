"""
Module containing kernel used principally in the JumpReLU activation function.

The kernels implemented in this module are symmetric.
For more information, see https://en.wikipedia.org/wiki/Kernel_(statistics).
"""

import torch


def rectangle_kernel(x, bandwith):
    """
    Rectangle kernel.

    y = 1 if |x| <= bandwith / 2, 0 otherwise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwith of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwith must be positive."
    half_eps = bandwith / 2
    return ((x >= -half_eps) & (x <= half_eps)).float() / bandwith


def gaussian_kernel(x, bandwith):
    """
    Gaussian kernel.

    y = e^(-0.5 * (x / (bandwith / 2))^2) / (bandwith * (2 * pi)^0.5)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwidth of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwidth must be positive."
    coeff = 1 / (bandwith * (2.0 * torch.pi)**0.5)
    x_scaled = x / (bandwith / 2.0)
    return torch.exp(-0.5 * x_scaled ** 2) * coeff


def triangular_kernel(x, bandwith):
    """
    Triangular kernel.

    y = (1 - |x| / (bandwith / 2)) if |x| <= bandwith / 2, 0 otherwise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwidth of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwidth must be positive."
    half_eps = bandwith / 2
    x_abs = x.abs()
    return ((1 - (x_abs / half_eps)) * (x_abs <= half_eps).float()) / half_eps


def cosine_kernel(x, bandwith):
    """
    Cosine kernel.

    y = (1 + cos(pi * x / bandwith)) / 2 if |x| <= bandwith, 0 otherwise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwidth of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwidth must be positive."
    coeff = torch.pi / bandwith
    x_abs = x.abs()
    return ((1 + torch.cos(x * coeff)) / 2) * (x_abs <= bandwith).float()


def epanechnikov_kernel(x, bandwith):
    """
    Epanechnikov kernel.

    y = (3 / 4) * (1 - (x / (bandwith / 2))^2) if |x| <= bandwith / 2, 0 otherwise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwidth of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwidth must be positive."
    half_eps = bandwith / 2
    x_scaled = x / half_eps
    return (3 / 4) * (1 - x_scaled ** 2) * (x_scaled.abs() <= 1).float() / half_eps


def quartic_kernel(x, bandwith):
    """
    Quartic (biweight) kernel.

    y = (15 / 16) * (1 - (x / (bandwith / 2))^2)^2 if |x| <= bandwith / 2, 0 otherwise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwidth of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwidth must be positive."
    half_eps = bandwith / 2
    x_scaled = x / half_eps
    return (15 / 16) * (1 - x_scaled ** 2) ** 2 * (x_scaled.abs() <= 1).float() / half_eps


def silverman_kernel(x, bandwith):
    """
    Silverman kernel.

    y = exp(-|x| / (sqrt(2) * bandwith / 2)) * sin(|x| / (sqrt(2) * bandwith / 2) + pi / 4) / (bandwith / 2)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwidth of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwidth must be positive."
    half_eps = bandwith / 2
    x_scaled = x.abs() / half_eps
    return (torch.exp(-x_scaled / (2**0.5)) * torch.sin(x_scaled / (2**0.5) + torch.pi / 4)).float() / half_eps


def cauchy_kernel(x, bandwith):
    """
    Cauchy kernel.

    y = 1 / (1 + (x / (bandwith / 2))^2) / (pi * (bandwith / 2))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    bandwith : float
        Bandwidth of the kernel.

    Returns
    -------
    torch.Tensor
        Kernel values.
    """
    assert bandwith > 0, "Bandwidth must be positive."
    half_eps = bandwith / 2
    x_scaled = x / half_eps
    return (1 / (1 + x_scaled ** 2)) / (torch.pi * half_eps)
