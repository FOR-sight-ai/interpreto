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
Bases Classes for Concept-based Explainers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from textwrap import dedent
from typing import Any, Generic, TypeVar

import torch
from overcomplete.base import BaseDictionaryLearning

from interpreto.attributions.base import AttributionExplainer
from interpreto.commons.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import ConceptsActivations, LatentActivations, ModelInput

ConceptModel = TypeVar("ConceptModel")
MethodOutput = TypeVar("MethodOutput")


# Decorator that checks if the concept model is fitted before calling the method
def check_fitted(func: Callable[..., MethodOutput]) -> Callable[..., MethodOutput]:
    @wraps(func)
    def wrapper(self: AbstractConceptExplainer, *args, **kwargs) -> MethodOutput:
        if not self.is_fitted or self.split_point is None:
            raise RuntimeError("Concept encoder is not fitted yet. Use the .fit() method to fit the explainer.")
        return func(self, *args, **kwargs)

    return wrapper


class AbstractConceptExplainer(ABC, Generic[ConceptModel]):
    """Code: [:octicons-mark-github-24: `concepts/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/base.py)

    Abstract class defining an interface for concept explanation.
    Child classes should implement the `fit` and `encode_activations` methods, and only assume the presence of an
        encoding step using the `concept_model` to convert activations to latent concepts.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str): The split point used to train the `concept_model`.
        concept_model (Any): The model used to extract concepts from the activations of
            `model_with_split_points`. The only assumption for classes inheriting from this class is that
            the `concept_model` can encode activations into concepts with `encode_activations`.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModel,
        split_point: str | None = None,
    ):
        """Initializes the concept explainer with a given splitted model.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            concept_model (Any): The model used to extract concepts from
                the activations of `model_with_split_points`.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
        """
        if not isinstance(model_with_split_points, ModelWithSplitPoints):
            raise TypeError(
                f"The given model should be a ModelWithSplitPoints, but {type(model_with_split_points)} was given."
            )
        self.model_with_split_points: ModelWithSplitPoints = model_with_split_points
        self.concept_model: ConceptModel = concept_model
        self.is_fitted: bool = False
        self.has_differentiable_concept_encoder = False

    def __repr__(self):
        return dedent(f"""\
            {self.__class__.__name__}(
                split_point={self.split_point},
                concept_model={type(self.concept_model).__name__},
                is_fitted={self.is_fitted},
                has_differentiable_concept_encoder={self.has_differentiable_concept_encoder},
            )""")

    @abstractmethod
    def fit(self, activations: LatentActivations | dict[str, LatentActivations], *args, **kwargs) -> Any:
        """Fits `concept_model` on the given activations.

        Args:
            activations (dict[str, torch.Tensor]): A dictionary with model paths as keys and the corresponding
                tensors as values.

        Returns:
            `None`, `concept_model` is fitted in-place, `is_fitted` is set to `True` and `split_point` is set.
        """
        pass

    @abstractmethod
    def encode_activations(self, activations: LatentActivations) -> ConceptsActivations:
        """Abstract method defining how activations are converted into concepts by the concept encoder.

        Args:
            activations (torch.Tensor): The activations to encode.

        Returns:
            A `torch.Tensor` of encoded activations produced by the fitted concept encoder.
        """
        pass

    def verify_activations(self, activations: dict[str, LatentActivations]) -> None:
        """
        Verify that the given activations are valid for the `model_with_split_points` and `self.split_point`.
        Cases in which the activations are not valid include:

        * Activations are not a valid dictionary.
        * Specified split point does not exist in the activations.

        Args:
            activations (dict[str, torch.Tensor]): A dictionary with model paths as keys and the corresponding
                tensors as values.
        """
        if not isinstance(activations, dict) or not all(isinstance(act, torch.Tensor) for act in activations.values()):
            raise TypeError(
                "Invalid activations for the concept explainer. "
                "Activations should be a dictionary of model paths and torch.Tensor activations. "
                f"Got: '{type(activations)}'"
            )
        activations_split_points = list(activations.keys())
        if self.split_point not in activations_split_points:
            raise ValueError(
                f"Fitted split point '{self.split_point}' not found in activations.\n"
                f"Available split_points: {', '.join(activations_split_points)}."
            )

    @property
    def split_point(self) -> str:
        return self._split_point

    @split_point.setter
    def split_point(self, split_point: str | None) -> None:
        if split_point is None and len(self.model_with_split_points.split_points) > 1:
            raise ValueError(
                "If the model has more than one split point, a split point for fitting the concept model should "
                f"be specified. Got split point: '{split_point}' with model split points: "
                f"{', '.join(self.model_with_split_points.split_points)}."
            )
        if split_point is None:
            self._split_point: str = self.model_with_split_points.split_points[0]
        if split_point is not None:
            if split_point not in self.model_with_split_points.split_points:
                raise ValueError(
                    f"Split point '{split_point}' not found in model split points: "
                    f"{', '.join(self.model_with_split_points.split_points)}."
                )
            self._split_point: str = split_point

    def prepare_fit(
        self,
        activations: LatentActivations | dict[str, LatentActivations],
        overwrite: bool,
    ) -> LatentActivations:
        if self.is_fitted and not overwrite:
            raise RuntimeError(
                "Concept explainer has already been fitted. Refitting will overwrite the current model."
                "If this is intended, use `overwrite=True` in fit(...)."
            )
        if isinstance(activations, dict):
            self.verify_activations(activations)
            split_activations = activations[self.split_point]
        else:
            split_activations = activations
        assert len(split_activations.shape) == 2, (
            f"Input activations should be a 2D tensor of shape (batch_size, n_features) but got {split_activations.shape}."
        )
        return split_activations

    @check_fitted
    def top_k_inputs_for_concept(self, inputs: ModelInput, concept_id: int, k: int = 5) -> list[tuple[str, float]]:
        """
        Retrieves the top-k most important input elements (tokens/words/clauses/phrases) related to a given concept.

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences
                or a dataset.
            concept_id (int): Index identifying the position of the concept of interest (score in the
                `ConceptActivations` tensor) for which relevant elements should be retrieved.
            k: The number of input elements to retrieve. Defaults to 5.

        Returns:
            An ordered list of k tuples, each containing containing one of the top-k relevant input elements
                for the specified concept and its importance scores.
        """
        activations = self.model_with_split_points.get_activations(inputs)
        return self.top_k_inputs_for_concept_from_activations(inputs, activations[self.split_point], concept_id, k)

    @check_fitted
    def top_k_inputs_for_concept_from_activations(
        self,
        inputs: ModelInput,
        activations: LatentActivations,
        concept_id: int,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieves the top-k most important input elements (tokens/words/clauses/phrases) related to a given concept.
        This version of the method uses pre-computed activations

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences
                or a dataset.
            activations: A torch.Tensor of activations to use for obtaining concept activations.
            concept_id (int): Index identifying the position of the concept of interest (score in the
                `ConceptActivations` tensor) for which relevant elements should be retrieved.
            k: The number of input elements to retrieve. Defaults to 5.

        Returns:
            An ordered list of k tuples, each containing containing one of the top-k relevant input elements
                for the specified concept and its importance scores.
        """
        concept_activations: ConceptsActivations = self.encode_activations(activations)
        # TODO: Implement the method alongside these lines, depending on specific shapes of concept_activations and inputs
        # - Extract the concept activations for the given concept_id
        # - Obtain a string version of inputs (decode tokens)
        # - Attribute input tokens to the concept activations
        # - Select top-k values from attributions
        # - Build the list of tuples and return it
        print(inputs, activations, concept_activations, concept_id, k)
        raise NotImplementedError()

    @check_fitted
    def input_concept_attribution(
        self,
        inputs: ModelInput,
        concept: int,
        attribution_method: type[AttributionExplainer],
        **attribution_kwargs,
    ) -> list[float]:
        """Attributes model inputs for a selected concept.

        Args:
            inputs (ModelInput): The input data, which can be a string, a list of tokens/words/clauses/sentences
                or a dataset.
            concept (int): Index identifying the position of the concept of interest (score in the
                `ConceptActivations` tensor) for which relevant input elements should be retrieved.
            attribution_method: The attribution method to obtain importance scores for input elements.

        Returns:
            A list of attribution scores for each input.
        """
        raise NotImplementedError("Input-to-concept attribution method is not implemented yet.")


class ConceptBottleneckExplainer(AbstractConceptExplainer[BaseDictionaryLearning], ABC):
    """Code: [:octicons-mark-github-24: `concepts/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/base.py)

    A concept bottleneck explainer wraps a `concept_model` that should be able to encode activations into concepts
    and decode concepts into activations.

    We use the term "concept bottleneck" loosely, as the latent space can be overcomplete compared to activation
        space, as in the case of sparse autoencoders.

    We assume that the concept model follows the structure of an [`overcomplete.BaseDictionaryLearning`](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)
    model, which defines the `encode` and `decode` methods for encoding and decoding activations into concepts.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str): The split point used to train the `concept_model`.
        concept_model ([BaseDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)): The model used to extract concepts from the
            activations of  `model_with_split_points`. The only assumption for classes inheriting from this class is
            that the `concept_model` can encode activations into concepts with `encode_activations`.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: BaseDictionaryLearning,
        split_point: str | None,
    ):
        """Initializes the concept explainer with a given splitted model.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            concept_model ([BaseDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)): The model used to extract concepts from
                the activations of `model_with_split_points`.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
        """
        super().__init__(model_with_split_points, concept_model, split_point)
        self.has_differentiable_concept_decoder = False

    def __repr__(self):
        return dedent(f"""\
            {self.__class__.__name__}(
                split_point={self.split_point},
                concept_model={type(self.concept_model).__name__},
                is_fitted={self.is_fitted},
                has_differentiable_concept_encoder={self.has_differentiable_concept_encoder},
                has_differentiable_concept_decoder={self.has_differentiable_concept_decoder},
            )""")

    @check_fitted
    def encode_activations(self, activations: LatentActivations) -> torch.Tensor:  # ConceptActivations
        """Encode the given activations using the `concept_model` encoder.

        Args:
            activations (torch.Tensor): The activations to encode.

        Returns:
            The encoded concept activations.
        """
        self.verify_activations({self.split_point: activations})
        return self.concept_model.encode(activations)  # type: ignore

    @check_fitted
    def decode_concepts(self, concepts: ConceptsActivations) -> torch.Tensor:  # LatentActivations
        """Decode the given concepts using the `concept_model` decoder.

        Args:
            concepts (ConceptsActivations): The concepts to decode.

        Returns:
            The decoded model activations.
        """
        return self.concept_model.decode(concepts)  # type: ignore

    @check_fitted
    def get_dictionary(self) -> torch.Tensor:
        """Get the dictionary learned by the fitted `concept_model`.

        Returns:
            torch.Tensor: A `torch.Tensor` containing the learned dictionary.
        """
        return self.concept_model.get_dictionary()  # type: ignore

    @check_fitted
    def concept_output_attribution(
        self,
        inputs: ModelInput,
        concepts: ConceptsActivations,
        target: int,
        attribution_method: type[AttributionExplainer],
        **attribution_kwargs,
    ) -> list[float]:
        """Computes the attribution of each concept for the logit of a target output element.

        Args:
            inputs (ModelInput): An input datapoint for the model.
            concepts (torch.Tensor): Concept activation tensor.
            target (int): The target class for which the concept output attribution should be computed.
            attribution_method: The attribution method to obtain importance scores for input elements.

        Returns:
            A list of attribution scores for each concept.
        """
        raise NotImplementedError("Concept-to-output attribution method is not implemented yet.")
