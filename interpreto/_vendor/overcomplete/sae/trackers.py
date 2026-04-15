"""
Module dedicated to trackers for SAEs training.
"""

import torch


class DeadCodeTracker:
    """Track dead codes in an online manner."""

    def __init__(self, nb_concepts, device):
        """
        Initialize dead code tracker.

        Parameters
        ----------
        nb_concepts : int
            Number of features to track
        device : str or torch.device
            Device to store the tracking tensor on
        """
        self.alive_features = torch.zeros(nb_concepts, dtype=torch.bool, device=device)

    def update(self, z):
        """
        Update alive features based on new activations.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor of shape (batch_size, n_features)
        """
        self.alive_features |= (z > 0).any(dim=0)

    def get_dead_ratio(self):
        """
        Get the ratio of dead features.

        Returns
        -------
        float
            Ratio of dead features (between 0 and 1)
        """
        return 1 - self.alive_features.float().mean().item()
