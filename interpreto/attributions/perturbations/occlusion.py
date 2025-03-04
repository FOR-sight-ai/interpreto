"""
Simple occlusion perturbations for tokenized inputs
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import singledispatchmethod

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.perturbations.base import TokenPerturbation

# TODO : Add a mixin in perturbationts.base to avoid code duplication between multiples token-wise or word-wise perturbations
# TODO : tests pour les méthodes de word occlusion

class TokenOcclusionPerturbator(TokenPerturbation):
    """
    Perturbator removing tokens from the input
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, inputs_embeddings: torch.nn.Module, mask_value: str | None = None
    ):
        # TODO : Currently only deals with Huggingface PreTrainedTokenizer (or equivalents), should be more general
        self.tokenizer = tokenizer
        self.inputs_embeddings = inputs_embeddings
        self.mask_value = mask_value or tokenizer.mask_token

    @singledispatchmethod
    def perturb(self, inputs) -> tuple[torch.Tensor] | tuple[list[torch.Tensor]]:
        """
        Perturb a sentence or a collection of sentences by applying token occlusion

        Args:
            inputs (str|Iterable[str]): sentence to perturb

        Returns:
            tuple[torch.Tensor]|tuple[list[torch.Tensor]]: embeddings of perturbed sentences and associated masks
        """
        raise NotImplementedError(f"Type {type(inputs)} not supported")

    @perturb.register(str)
    def _(self, inputs: str) -> tuple[torch.Tensor]:
        # Get tokens from str input
        tokens = self.tokenizer.tokenize(inputs)
        n_perturbations = len(tokens)
        # Create variations by masking each token
        variations = [
            self.tokenizer.convert_tokens_to_ids(tokens[:i] + [self.mask_value] + tokens[i + 1 :])
            for i in range(n_perturbations)
        ]
        # Get words embeddings for each variation
        embeddings = torch.stack(
            [self.inputs_embeddings(torch.tensor(variation)) for variation in variations]
        ).unsqueeze(0)
        # Return embeddings and identity matrix as mask
        return embeddings, torch.eye(n_perturbations)

    @perturb.register(Iterable)
    def _(self, inputs: Iterable) -> tuple[list[torch.Tensor]]:
        # Perturb a batch of inputs (or nested batchs of inputs)
        return [self.perturb(item) for item in inputs]


class WordOcclusionPerturbator(TokenPerturbation):
    """
    Perturbator removing words from the input
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, inputs_embeddings: torch.nn.Module, mask_value: str | None = None
    ):
        # TODO : Currently only deals with Huggingface PreTrainedTokenizer (or equivalents), should be more general
        self.tokenizer = tokenizer
        self.inputs_embeddings = inputs_embeddings
        self.mask_value = mask_value or tokenizer.mask_token

    @singledispatchmethod
    def perturb(self, inputs) -> tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Perturb a sentence or a collection of sentences by applying word occlusion

        Args:
            inputs (str|Iterable[str]): sentence to perturb

        Returns:
            tuple[torch.Tensor]|tuple[list[torch.Tensor]]: embeddings of perturbed sentences and associated masks
        """
        raise NotImplementedError(f"Type {type(inputs)} not supported")

    @perturb.register(Iterable)
    def _(self, inputs: Iterable) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # TODO : put this in a mixin to avoid code duplication
        # Perturb a batch of inputs (or nested batchs of inputs)
        return [self.perturb(item) for item in inputs]

    @perturb.register(str)
    def _(self, inputs: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Get tokens from str input
        words = inputs.split()
        n_perturbations = len(words)
        # Create variations by masking each word
        variations = []
        for index, word in enumerate(words):
            first_part = self.tokenizer.tokenize(" ".join(words[:index]))
            second_part = self.tokenizer.tokenize(" " + " ".join(words[index + 1 :]))

            tokens = first_part + [self.tokenizer.mask_token for _ in self.tokenizer.tokenize(word)] + second_part
            # add truncation ?
            # tokens = tokens[: max_nb_tokens]
            variations.append(self.tokenizer.convert_tokens_to_ids(tokens))

        max_length = max([len(tokens) for tokens in variations])
        # TODO : put the padding in a separate reusable utility function
        variations = [v + [self.tokenizer.pad_token_id] * (max_length - len(v)) for v in variations]

        # Get words embeddings for each variation
        embeddings = [self.inputs_embeddings(torch.tensor(variation)) for variation in variations]

        # Return embeddings and identity matrix as mask
        return torch.stack(embeddings).unsqueeze(0), torch.eye(n_perturbations)


