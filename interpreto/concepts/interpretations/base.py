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
Base class for concept interpretation methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from jaxtyping import Float

from interpreto import Granularity, ModelWithSplitPoints
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations


class BaseConceptInterpretationMethod(ABC):
    """Code: [:octicons-mark-github-24: `concepts/interpretations/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/interpretations/base.py)

    Abstract class defining an interface for concept interpretation.
    Its goal is to make the dimensions of the concept space interpretable by humans.

    Attributes:
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModelProtocol,
        activation_granularity: ActivationGranularity,
        split_point: str | None = None,
        concept_encoding_batch_size: int = 1024,
        device: torch.device | str | None = "cpu",
    ):
        if not hasattr(concept_model, "encode"):
            raise TypeError(
                f"Concept model should be able to encode activations into concepts. Got: {type(concept_model)}."
            )

        if split_point is None:
            if len(model_with_split_points.split_points) > 1:
                raise ValueError(
                    "If the model has more than one split point, a split point for fitting the concept model should "
                    f"be specified. Got split point: '{split_point}' with model split points: "
                    f"{', '.join(model_with_split_points.split_points)}."
                )
            split_point = model_with_split_points.split_points[0]

        if split_point not in model_with_split_points.split_points:
            raise ValueError(
                f"Split point '{split_point}' not found in model split points: "
                f"{', '.join(model_with_split_points.split_points)}."
            )

        self.model_with_split_points: ModelWithSplitPoints = model_with_split_points
        self.split_point: str = split_point
        self.concept_model: ConceptModelProtocol = concept_model
        self.activation_granularity: ActivationGranularity = activation_granularity
        self.concept_encoding_batch_size: int = concept_encoding_batch_size
        self.device: torch.device | str | None = device

    @abstractmethod
    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        """
        Interpret the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and an object allowing to interpret them.
        It can be a label, a description, examples, etc.

        Args:
            concepts_indices (int | list[int]): The indices of the concepts to interpret.
            inputs (ModelInput | None): The inputs to use for the interpretation.
            latent_activations (LatentActivations | None): The latent activations to use for the interpretation.
            concepts_activations (ConceptsActivations | None): The concepts activations to use for the interpretation.

        Returns:
            Mapping[int, Any]: The interpretation of each of the specified concepts.
        """
        raise NotImplementedError

    def concepts_activations_from_source(
        self,
        *,
        inputs: list[str] | None = None,
        latent_activations: Float[torch.Tensor, "nl d"] | None = None,
        concepts_activations: Float[torch.Tensor, "nl cpt"] | None = None,
    ) -> Float[torch.Tensor, "nl cpt"]:
        """
        Computes the concepts activations from the given samples.
        Samples can be provided as raw text (`inputs`), latent activations (`latent_activations`),
        or directly concept activations (`concepts_activations`).

        Args:
            inputs (list[str] | None): The indices of the concepts to interpret.
            latent_activations (Float[torch.Tensor, "nl d"] | None): The latent activations
            concepts_activations (Float[torch.Tensor, "nl cpt"] | None): The concepts activations

        Returns:
            Float[torch.Tensor, "nl cpt"] :
        """

        if concepts_activations is not None:
            return concepts_activations

        if latent_activations is not None:
            # TODO: remove the if case when all overcomplete models are nn modules
            if hasattr(self.concept_model, "to") and self.device is not None:
                self.concept_model.to(self.device)  # type: ignore

            # batch over latent activations for concept encoding
            concepts_activations_list = []
            for batch_idx in range(0, latent_activations.shape[0], self.concept_encoding_batch_size):
                # extract and encode a batch of latent activations
                batch_latent_activations = latent_activations[batch_idx : batch_idx + self.concept_encoding_batch_size]

                if hasattr(batch_latent_activations, "to"):
                    batch_latent_activations = batch_latent_activations.to(self.device)  # type: ignore
                batch_concepts_activations = self.concept_model.encode(batch_latent_activations)

                # SAEs outputs from overcomplete are tuples, we need to get the codes
                if isinstance(batch_concepts_activations, tuple):
                    batch_concepts_activations = batch_concepts_activations[1]  # temporary fix, issue #65

                concepts_activations_list.append(batch_concepts_activations.cpu())  # type: ignore
            concepts_activations = torch.cat(concepts_activations_list, dim=0)

            if hasattr(self.concept_model, "to"):
                self.concept_model.to("cpu")  # type: ignore
            return concepts_activations

        if inputs is not None:
            activations_dict: dict[str, LatentActivations] = self.model_with_split_points.get_activations(
                inputs,
                activation_granularity=self.activation_granularity,
            )
            latent_activations = self.model_with_split_points.get_split_activations(
                activations_dict, split_point=self.split_point
            )
            return self.concepts_activations_from_source(latent_activations=latent_activations, inputs=inputs)

        raise ValueError(
            "No source provided. Please provide either `inputs`, `latent_activations`, or `concepts_activations`."
        )

    def concepts_activations_from_vocab(
        self,
    ) -> tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
        """
        Computes the concepts activations for each token of the vocabulary

        Args:
            model_with_split_points (ModelWithSplitPoints):
            split_point (str):
            concept_model (ConceptModelProtocol):

        Returns:
            tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
                - The list of tokens in the vocabulary
                - The concept activations for each token
        """
        # extract and sort the vocabulary
        vocab_dict: dict[str, int] = self.model_with_split_points.tokenizer.get_vocab()
        input_ids: list[int]
        inputs, input_ids = zip(*vocab_dict.items(), strict=True)  # type: ignore

        # compute the vocabulary's latent activations
        input_tensor: Float[torch.Tensor, "v 1"] = torch.tensor(input_ids).unsqueeze(1)
        activations_dict: dict[str, LatentActivations] = self.model_with_split_points.get_activations(  # type: ignore
            input_tensor, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS
        )
        latent_activations = self.model_with_split_points.get_split_activations(
            activations_dict, split_point=self.split_point
        )
        concepts_activations = self.concept_model.encode(latent_activations)  # type: ignore
        if isinstance(concepts_activations, tuple):
            concepts_activations = concepts_activations[1]  # temporary fix, issue #65
        return inputs, concepts_activations  # type: ignore

    def get_granular_inputs(
        self,
        inputs: list[str],  # (n)
    ) -> tuple[list[str], list[int]]:  # (ng,)
        """Split texts from the inputs based on the target granularity
        (for instance into tokens, words, sentences, ...)

        Args:
            inputs (list[str]): n text samples

        Returns:
            tuple[list[str], list[int]]:
                - list[str]: The granular texts from the inputs, flatened
                - list[int]: The sample id for each granular text, to keep track of which sample the text belongs to.
        """
        if self.activation_granularity is ActivationGranularity.SAMPLE:
            # no activation_granularity is needed
            return inputs, list(range(len(inputs)))

        # Get granular texts from the inputs
        tokens = self.model_with_split_points.tokenizer(
            inputs, return_tensors="pt", padding=True, return_offsets_mapping=True
        )
        granular_texts: list[list[str]] = Granularity.get_decomposition(
            tokens,
            granularity=self.activation_granularity.value,  # type: ignore
            tokenizer=self.model_with_split_points.tokenizer,
            return_text=True,
        )  # type: ignore

        granular_flattened_texts = [text for sample_texts in granular_texts for text in sample_texts]
        granular_flattened_sample_id = [i for i, sample_texts in enumerate(granular_texts) for _ in sample_texts]
        return granular_flattened_texts, granular_flattened_sample_id


def verify_concepts_indices(
    concepts_activations: ConceptsActivations,
    concepts_indices: int | list[int],
) -> list[int]:
    # take subset of concepts as specified by the user
    if isinstance(concepts_indices, int):
        concepts_indices = [concepts_indices]

    if not isinstance(concepts_indices, list) or not all(isinstance(c, int) for c in concepts_indices):  # type: ignore
        raise ValueError(f"`concepts_indices` should be 'all', an int, or a list of int. Received {concepts_indices}.")

    if max(concepts_indices) >= concepts_activations.shape[1] or min(concepts_indices) < 0:
        raise ValueError(
            f"At least one concept index out of bounds. `max(concepts_indices)`: {max(concepts_indices)} >= {concepts_activations.shape[1]}."
        )

    return concepts_indices


def verify_granular_inputs(
    granular_inputs: list[str],
    sure_concepts_activations: ConceptsActivations,
    latent_activations: LatentActivations | None = None,
    concepts_activations: ConceptsActivations | None = None,
):
    if len(granular_inputs) != len(sure_concepts_activations):
        if latent_activations is not None and len(granular_inputs) != len(latent_activations):
            raise ValueError(
                f"The lengths of the granulated inputs do not match the number of provided latent activations {len(granular_inputs)} != {len(latent_activations)}"
                "If you provide latent activations, make sure they have the same granularity as the inputs."
            )
        if concepts_activations is not None and len(granular_inputs) != len(concepts_activations):
            raise ValueError(
                f"The lengths of the granulated inputs do not match the number of provided concepts activations {len(granular_inputs)} != {len(concepts_activations)}"
                "If you provide concepts activations, make sure they have the same granularity as the inputs."
            )
        raise ValueError(
            f"The lengths of the granulated inputs do not match the number of concepts activations {len(granular_inputs)} != {len(sure_concepts_activations)}"
        )
