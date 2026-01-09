# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Perturbation for the insertion and deletion metric.
"""

from __future__ import annotations

import math
from abc import abstractmethod

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from transformers import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer

from interpreto.commons.granularity import Granularity
from interpreto.typing import SingleAttribution


class InsertionDeletionPerturbator:
    """
    Base class for insertion and deletion perturbations.

    The perturbations are based on the importance of the attributions: the most important elements are perturbed first
    (inserted for Insertion or removed for Deletion). The perturbations are done iteratively, where at each step the
    next perturbation is applied to the next most important elements in the input sequence.

    Unlike standard perturbations, the perturbations are different for each target, as the perturbations are based on
    the importance of the attributions. For `t` targets and `p` perturbations per target, the `get_mask` method returns
    a mask of shape `(p * t, mask_dim)`, where `mask_dim` is the length of the input sequence. The first `p` rows
    correspond to the first target, the next `p` rows to the second target, and so on.

    Attributes:
        tokenizer (PreTrainedTokenizer | None): Hugging Face tokenizer associated with the model.
        granularity (Granularity): Level at which deletion should be applied. Set after initialization.
        n_perturbations (int): Number of perturbations to generate.
        max_percentage_perturbed (float): Maximum percentage of tokens in the sequence to be perturbed.
        replace_token_id (int): Token used to replace deleted elements.
        device (torch.device): Device on which the perturbator will be run.
    """

    __slots__ = (
        "tokenizer",
        "n_perturbations",
        "replace_token_id",
        "granularity",
        "max_percentage_perturbed",
        "device",
    )

    granularity: Granularity

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None = None,
        n_perturbations: int = 100,
        max_percentage_perturbed: float = 1.0,
        replace_token_id: int = 0,
        device: torch.device | None = None,
    ) -> None:
        if n_perturbations < 1:
            raise ValueError("The number of perturbations must be at least 1.")

        self.tokenizer = tokenizer

        # number of perturbations made by the "perturb" method
        self.n_perturbations = n_perturbations

        # token id used to replace the masked tokens
        self.replace_token_id = replace_token_id

        self.max_percentage_perturbed = max_percentage_perturbed

        self.device = device or torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device

    @abstractmethod
    def _baseline_mask(self, dims) -> Float[torch.Tensor, "p l"]:
        """Returns the baseline mask for the perturbator, i.e. a mask with all zeros for
        deletion and a mask with all ones for insertion.

        Returns:
            torch.Tensor: Tensor of shape ``(n_perturbations, mask_dim)``
        """
        raise NotImplementedError()

    @jaxtyped(typechecker=beartype)
    def get_mask(  # type: ignore[override]
        self, mask_dim: int, attributions: SingleAttribution
    ) -> Float[torch.Tensor, "total_perturbations {mask_dim}"]:
        """Return a mask performing single-token insertions/deletions based on the attributions.

        Args:
            mask_dim (int): Length of the input sequence.
            attributions (SingleAttribution): the attribution tensor for a single target, with shape ``(mask_dim,)`` or
                ``(1, mask_dim)``.

        Returns:
            torch.Tensor: Tensor of shape ``(n_perturbations, mask_dim)``
        """

        if mask_dim < 2:
            raise ValueError(
                "The mask dimension (i.e. the sequence length) must be greater than 1, in order to have meaningful "
                "perturbations."
            )

        if attributions.ndim == 1:
            attributions = attributions.unsqueeze(0)

        if attributions.shape[0] != 1:
            raise ValueError("Only single target attributions are supported. Received shape: {attributions.shape}")

        if attributions.shape[1] != mask_dim:
            raise ValueError(
                f"The attributions length ({attributions.shape[1]}) must match the mask dimension ({mask_dim})."
            )

        # Get indices of most important tokens
        most_important_idx = torch.argsort(attributions, descending=True)

        # Compute the real number of perturbations "p": take the minimum between n_perturbations and the number of
        # elements to perturb (+1 for the baseline)
        num_elements_to_perturb = math.ceil(mask_dim * self.max_percentage_perturbed)
        if num_elements_to_perturb < 1:
            raise RuntimeError(
                "The number of perturbed elements is too low, leading to no elements being perturbed. "
                "Please increase the value of max_percentage_perturbed."
            )
        p = min(self.n_perturbations, num_elements_to_perturb) + 1

        # Build the mask
        mask: Float[torch.Tensor, p, mask_dim] = self._baseline_mask((p, mask_dim))
        steps = torch.linspace(0, num_elements_to_perturb, p)
        fill_value = int(1 - mask[0, 0])  # Value to fill the mask with (1 for deletion, 0 for insertion)
        for i, step in enumerate(steps):
            mask[i, most_important_idx[0, : int(step)]] = fill_value

        return mask

    @jaxtyped(typechecker=beartype)
    @staticmethod
    def apply_mask(
        inputs: Int[torch.Tensor, "l 1"],
        mask: Float[torch.Tensor, "p g"],
        mask_value: torch.Tensor,
    ) -> Float[torch.Tensor, "p l"]:
        """
        Basic mask application method.

        If last dimension `d` is 1 (in case of tokens and not embeddings), this last dimension will be squeezed out
        and the returned tensor will have shape (num_sequences, n_perturbations, mask_dim).

        Args:
            inputs (torch.Tensor): inputs to mask
            mask (torch.Tensor): mask matrix to apply
            mask_value (torch.Tensor): tensor used as a mask (mask token, zero tensor, etc.)

        Returns:
            torch.Tensor: masked inputs
        """
        base: Float[torch.Tensor, "p l d"] = inputs.unsqueeze(-3) * (1 - mask).unsqueeze(
            -1
        )  # torch.einsum("ld,pl->pld", inputs, 1 - mask)
        masked: Float[torch.Tensor, "p l d"] = mask_value * mask.unsqueeze(
            -1
        )  # torch.einsum("pl,d->pld", mask, mask_value)
        return (base + masked).squeeze(-1)

    def perturb(
        self,
        model_inputs: BatchEncoding,
        granularity_indices: list[list[list[int]]],
        attributions: SingleAttribution,
    ) -> BatchEncoding:
        """
        Method called to perturb the inputs of the model

        Args:
            model_inputs (BatchEncoding): mapping given by the tokenizer

        Returns:
            model_inputs with perturbations applied
        """
        model_inputs = model_inputs.copy()  # type: ignore
        input_ids: torch.Tensor = model_inputs["input_ids"]  # type: ignore

        if input_ids.shape[0] != 1:
            raise ValueError(
                "Inputs are treated one by on one in the perturbator."
                + "But received a batch of inputs."
                + f"input_ids shape: {input_ids.shape} - expected shape: (1, l)"
            )

        # compute association matrix between the granularity level and ALL_TOKENS
        association_matrix: Int[torch.Tensor, "g l"] = (
            self.granularity.get_association_matrix(model_inputs, self.tokenizer, indices_list=granularity_indices)[0]
            .float()
            .to(self.device)
        )

        # compute granularity-wise perturbation mask based on the length of the sequence (granularity-wise)
        gran_mask: Float[torch.Tensor, "p g"] = self.get_mask(association_matrix.shape[0], attributions).to(
            self.device
        )

        # compute real perturbation mask
        real_mask: Float[torch.Tensor, "p l"] = torch.einsum("pg,gl->pl", gran_mask, association_matrix)

        model_inputs["input_ids"] = (
            self.apply_mask(
                inputs=input_ids.T.to(self.device),
                mask=real_mask,
                mask_value=torch.Tensor([self.replace_token_id]).to(self.device),
            )
            .squeeze(-1)
            .to(torch.int)
        )

        # Repeat other keys in encoding for each perturbation
        for k, v in model_inputs.items():
            v: torch.Tensor
            if k != "input_ids":
                repeats = [1] * (v.dim())
                repeats[0] = real_mask.shape[0]
                model_inputs[k] = v.repeat(*repeats)
        return model_inputs


class DeletionPerturbator(InsertionDeletionPerturbator):
    """Perturbator for deletion metric."""

    def _baseline_mask(self, dims) -> Float[torch.Tensor, "p l"]:
        """Return the baseline mask for the perturbator, i.e. a mask full of zeros, meaning that initially nothing
        is perturbed."""
        return torch.zeros(dims)


class InsertionPerturbator(InsertionDeletionPerturbator):
    """Perturbator for insertion metric."""

    def _baseline_mask(self, dims) -> Float[torch.Tensor, "p l"]:
        """Return the baseline mask for the perturbator, i.e. a mask full of ones, meaning that initially everything is
        perturbed."""
        return torch.ones(dims)
