"""
Module for data loading and conversion utilities.
"""

import numpy as np
import torch



def to_npf32(tensor):
    """
    Check if tensor is torch, ensure it is on CPU and convert to NumPy.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Input tensor.
    """
    # return as is if already npf32
    if isinstance(tensor, np.ndarray) and tensor.dtype == np.float32:
        return tensor
    # torch case
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    # pil case (and other)
    return np.array(tensor).astype(np.float32)


def unwrap_dataloader(dataloader):
    """
    Unwrap a DataLoader into a single tensor.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object.
    """
    return torch.cat([batch[0] if isinstance(batch, (tuple, list))
                      else batch for batch in dataloader], dim=0)
