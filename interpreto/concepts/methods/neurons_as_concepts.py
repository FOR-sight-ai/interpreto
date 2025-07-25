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
Concept Bottleneck Explainer based on Overcomplete concept-encoder-decoder framework.
"""

from __future__ import annotations

import torch
from overcomplete.base import BaseDictionaryLearning

from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import ConceptsActivations, LatentActivations


class IdentityConceptModel(BaseDictionaryLearning):
    """
    Identity concept model that does not change the input activations.

    Attributes:
        nb_concepts (int): The number of concepts, which is the same as the input size.
        fitted (bool): Whether the concept model has been fitted.
    """

    def __init__(self, input_size: int):
        super().__init__(nb_concepts=input_size)
        self.fitted = True

    def fit(self, *args, **kwargs):
        """
        Overwrite to raise error as the concept model does not need to be fitted.
        """
        raise NotImplementedError("Identity concept model does not need to be fitted.")

    def encode(self, activations: LatentActivations) -> ConceptsActivations:  # type: ignore
        """
        Forwards the latent activations as concepts.

        Args:
            activations (LatentActivations): The activations to return as concepts.

        Returns:
            ConceptsActivations: The activations as concepts.
        """
        return activations

    def decode(self, concepts: ConceptsActivations) -> LatentActivations:  # type: ignore
        """
        Forwards the concepts as latent activations.

        Args:
            concepts (ConceptsActivations): The concepts to return as activations.

        Returns:
            LatentActivations: The concepts as activations.
        """
        return concepts

    def get_dictionary(self) -> torch.Tensor:  # type: ignore
        """
        Returns the identity matrix as the dictionary.

        Returns:
            torch.Tensor: The identity matrix as the dictionary.
        """
        return torch.eye(self.nb_concepts)

    def to(self, device: torch.device | str) -> None:  # type: ignore
        """
        Move the concept model to the given device.
        """
        self.device = device

    def cpu(self) -> None:  # type: ignore
        """
        Move the concept model to the CPU.
        """
        self.device = "cpu"


class NeuronsAsConcepts(ConceptAutoEncoderExplainer):
    """Code: [:octicons-mark-github-24: `concepts/methods/neurons_as_concepts.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/neurons_as_concepts.py)
    Concept Bottleneck Explainer where the latent space is considered as the concept space.

    # TODO: Add doc with papers we can redo with it.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str): The split point used to train the `concept_model`.
        concept_model (IdentityConceptModel): An identity concept model for harmonization.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        split_point: str | None = None,
    ):
        """
        Initializes the concept explainer with a given splitted model.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
        """
        # extract the input size from the model activations
        self.model_with_split_points = model_with_split_points
        self.split_point: str = split_point  # type: ignore
        input_size = self.model_with_split_points.get_latent_shape()[self.split_point][-1]

        # initialize
        super().__init__(
            model_with_split_points=model_with_split_points,
            concept_model=IdentityConceptModel(input_size),
            split_point=self.split_point,
        )
        self.has_differentiable_concept_encoder = True
        self.has_differentiable_concept_decoder = True

    def fit(self, *args, **kwargs):
        """
        Overwrite to raise error as the concept model does not need to be fitted.
        """
        raise NotImplementedError("NeuronsAsConcepts does not need to be fitted.")
