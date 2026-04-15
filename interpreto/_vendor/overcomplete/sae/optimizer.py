"""
Module dedicated to everything around the Schedulers of SAE.
"""


import numpy as np
import torch


class CosineScheduler:
    """
    Cosine learning rate scheduler with optional warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to adjust the learning rate.
    base_value : float
        The initial learning rate value after warmup.
    final_value : float
        The final learning rate value.
    total_iters : int
        The total number of iterations over which to schedule the learning rate.
    warmup_iters : int, optional
        The number of iterations for the warmup phase, by default 0.
    start_warmup_value : float, optional
        The starting value of the learning rate during warmup, by default 0.

    Methods
    -------
    __getitem__(it):
        Get the learning rate at a specific iteration.
    step():
        Update the learning rate for the current iteration.
    """

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        super().__init__()
        self.optimizer = optimizer
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = torch.tensor(np.concatenate((warmup_schedule, schedule))).float()

        self.iter = 0

        assert len(self.schedule) == self.total_iters, "The length of the schedule must equal total_iters."

    def __getitem__(self, it):
        """
        Get the learning rate at a specific iteration.

        Parameters
        ----------
        it : int
            The iteration number.

        Returns
        -------
        float
            The learning rate at the specified iteration.
        """
        if it >= self.total_iters:
            return self.final_value
        return self.schedule[it]

    def step(self):
        """
        Update the learning rate for the current iteration.
        """
        self.iter += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self[self.iter]
