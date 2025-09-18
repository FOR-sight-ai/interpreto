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
Base classes for perturbations used in attribution methods
"""
# TODO : remake all the docstrings of this file to fit with new method signatures

from __future__ import annotations

from abc import abstractmethod

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from transformers.tokenization_utils import PreTrainedTokenizer

from interpreto.commons.granularity import Granularity
from interpreto.typing import TensorMapping


class Perturbator:
    """
    Base class for perturbators
    If this class is instantiated, it behaves as a no-op perturbator
    Perturbator may be subclassed to define custom perturbations, we recommend to use either IdsPerturbator or EmbeddingsPerturbator as base classes
    """

    __slots__ = ("_device",)

    @property
    def device(self) -> torch.device:
        """
        Get the device of the perturbator
        """
        return self._device if hasattr(self, "_device") else torch.device("cpu")

    @device.setter
    def device(self, device: torch.device):
        """
        Set the device of the perturbator
        """
        self._device = device

    def to(self, device: torch.device):
        """
        Set the device of the perturbator
        """
        self._device = device

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        """
        Method called when we ask the perturbator to perturb a mapping of tensors, generally the output of a tokenizer
        The mapping should be similar to mappings returned by the tokenizer.
        It should at least have "input_ids" and "attention_mask".
        Optionally, the "offsets_mapping" might be required for the `SENTENCE` granularity.
        Give directly the output of the tokenizer without modifying it would be the best and most common way to use this method

        Args:
            model_inputs (TensorMapping): output of the tokenizers
        """
        # add perturbation dimension
        if model_inputs["input_ids"].ndim <= 1:
            model_inputs["input_ids"] = model_inputs["input_ids"].unsqueeze(0)
        # model_inputs["input_ids"] = model_inputs["input_ids"].unsqueeze(0)
        # if "inputs_embeds" in model_inputs:
        #    model_inputs["inputs_embeds"] = model_inputs["inputs_embeds"].unsqueeze(0)
        # TODO : eventually add perturbation dimension to other keys in the mapping ?

        return model_inputs, torch.zeros_like(model_inputs["input_ids"], dtype=torch.float)

    def __call__(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        return self.perturb(model_inputs)


class EmbeddingsPerturbator(Perturbator):
    """
    Specific abstract class for perturbators working on input embeddings
    All perturbators working on input embeddings only should inherit from this class
    """

    __slots__ = ("inputs_embedder",)

    def __init__(self, inputs_embedder: torch.nn.Module | None = None):
        """Create a perturbator.

        Args:
            inputs_embedder: Optional module used to embed input IDs when only
                ``input_ids`` are provided.
        """
        # Embedders is optional
        self.inputs_embedder = inputs_embedder

    @property
    def device(self) -> torch.device:
        """
        Get the device of the inputs embedder
        """
        if self.inputs_embedder is not None:
            return self.inputs_embedder.weight.device  # type: ignore
        return self._device

    @device.setter
    def device(self, device: torch.device):
        """
        Set the device of the inputs embedder
        """
        if self.inputs_embedder is not None:
            self.inputs_embedder.to(device)
        self._device = device

    def to(self, device: torch.device):
        """
        Set the device of the inputs embedder
        """
        self.device = device

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        embeddings = self._embed(model_inputs)
        return self.perturb_embeds(embeddings)

    @abstractmethod
    def perturb_embeds(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        """
        Perturb the input of the model, given as embeddings

        Args:
            model_inputs (MutableMapping): Mapping given by the tokenizer, should contain "inputs_embeds", otherwise, the given inputs_embedder will be used to compute them from "input_ids"
        Returns:
            TensorMapping: Perturbed mapping
            torch.Tensor | None: Perturbation mask, if applicable
        """

    # TODO : this function is replicated in the inference wrapper, eventually merge them
    def _embed(self, model_inputs: TensorMapping) -> TensorMapping:
        """
        Embed the inputs using the inputs_embedder

        Args:
            model_inputs (TensorMapping): input mapping containing either "input_ids" or "inputs_embeds".

        Raises:
            ValueError: If neither "input_ids" nor "inputs_embeds" are present in the input mapping.

        Returns:
            TensorMapping: The input mapping with "inputs_embeds" added.
        """
        # If input embeds are already present, return the unmodified model inputs
        if "inputs_embeds" in model_inputs:
            return model_inputs
        # If no inputs embedder is provided, raise an error
        if self.inputs_embedder is None:
            raise ValueError("Cannot call _embed method from a Perturbator without an inputs embedder")
        # If input ids are present, get the embeddings and add them to the model inputs
        if "input_ids" in model_inputs:
            base_shape = model_inputs["input_ids"].shape
            flatten_embeds = self.inputs_embedder(model_inputs["input_ids"].flatten(0, -2).to(self.device))
            model_inputs["inputs_embeds"] = flatten_embeds.view(*base_shape, flatten_embeds.shape[-1])
            return model_inputs
        # If neither input ids nor input embeds are present, raise an error
        raise ValueError("model_inputs should contain either 'input_ids' or 'inputs_embeds'")


class IdsPerturbator(Perturbator):
    """
    Base class for perturbations consisting in applying masks on token (or groups of tokens)
    All perturbators working on input IDs by applying a mask should inherit from this class
    """

    __slots__ = ("tokenizer", "n_perturbations", "replace_token_id", "granularity")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | None,
        replace_token_id: int,
        n_perturbations: int = 1,
        granularity: Granularity = Granularity.TOKEN,
    ):
        self.tokenizer = tokenizer

        # number of perturbations made by the "perturb" method
        self.n_perturbations = n_perturbations

        # token id used to replace the masked tokens
        self.replace_token_id = replace_token_id

        # granularity level of the perturbation (token masking, word masking...)
        # in most commons cases, this should be set to Granularity.TOKEN
        self.granularity = granularity

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

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def get_mask(self, mask_dim: int) -> Float[torch.Tensor, "{self.n_perturbations} {mask_dim}"]:
        """
        Method returning a perturbation mask for a given set of inputs
        This method should be implemented in subclasses

        The created mask should be of size (n_perturbations, mask_dim)
        where mask_dim is the length of the sequence according to the granularity level (number of tokens, number of words, number of sentences...)

        Args:
            mask_dim (int): length of the sequence according to the granularity level

        Returns:
            torch.Tensor: mask to apply on the inputs, of shape (n_perturbations, mask_dim)
        """
        raise NotImplementedError()

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        """
        Method called to perturb the inputs of the model

        Args:
            model_inputs (MutableMapping): mapping given by the tokenizer

        Returns:
            tuple: model_inputs with perturbations and the specific granularity mask
        """
        model_inputs = model_inputs.copy()  # type: ignore

        if model_inputs["input_ids"].shape[0] != 1:
            raise ValueError(
                "Inputs are treated one by on one in the perturbator."
                + "But received a batch of inputs."
                + f"input_ids shape: {model_inputs['input_ids'].shape} - expected shape: (1, l)"
            )

        # compute association matrix between the granularity level and ALL_TOKENS
        association_matrix: Int[torch.Tensor, "g l"] = (
            Granularity.get_association_matrix(model_inputs, self.granularity, self.tokenizer)[0]
            .float()
            .to(self.device)
        )

        # compute granularity-wise perturbation mask based on the length of the sequence (granularity-wise)
        gran_mask: Float[torch.Tensor, "p g"] = self.get_mask(association_matrix.shape[0]).to(self.device)

        # compute real perturbation mask
        real_mask: Float[torch.Tensor, "p l"] = torch.einsum("pg,gl->pl", gran_mask, association_matrix)

        model_inputs["input_ids"] = (
            self.apply_mask(
                inputs=model_inputs["input_ids"].T.to(self.device),
                mask=real_mask,
                mask_value=torch.Tensor([self.replace_token_id]).to(self.device),
            )
            .squeeze(-1)
            .to(torch.int)
        )

        # Repeat other keys in encoding for each perturbation
        for k in model_inputs.keys():
            if k != "input_ids":
                repeats = [1] * (model_inputs[k].dim())
                repeats[0] = model_inputs["input_ids"].shape[0]
                model_inputs[k] = model_inputs[k].repeat(*repeats)
        return model_inputs, gran_mask
