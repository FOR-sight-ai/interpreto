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
from collections.abc import Callable, Mapping
from functools import wraps
from textwrap import dedent
from typing import Any, Generic, Literal, TypeVar

import torch
from jaxtyping import Float
from overcomplete.base import BaseDictionaryLearning
from transformers.tokenization_utils_base import BatchEncoding

from interpreto.attributions.base import AttributionExplainer
from interpreto.concepts.interpretations.base import BaseConceptInterpretationMethod
from interpreto.model_wrapping.model_with_split_points import (
    ActivationGranularity,
    GranularityAggregationStrategy,
    ModelWithSplitPoints,
)
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations, ModelInputs

ConceptModel = TypeVar("ConceptModel", bound=ConceptModelProtocol)
BDL = TypeVar("BDL", bound=BaseDictionaryLearning)
MethodOutput = TypeVar("MethodOutput")


# Decorator that checks if the concept model is fitted before calling the method
def check_fitted(func: Callable[..., MethodOutput]) -> Callable[..., MethodOutput]:
    @wraps(func)
    def wrapper(self: ConceptEncoderExplainer, *args, **kwargs) -> MethodOutput:
        if not self.is_fitted or self.split_point is None:
            raise RuntimeError("Concept encoder is not fitted yet. Use the .fit() method to fit the explainer.")
        return func(self, *args, **kwargs)

    return wrapper


class ConceptEncoderExplainer(ABC, Generic[ConceptModel]):
    """Code: [:octicons-mark-github-24: `concepts/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/base.py)

    Abstract class defining an interface for concept explanation.
    Child classes should implement the `fit` and `encode_activations` methods, and only assume the presence of an
        encoding step using the `concept_model` to convert activations to latent concepts.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str): The split point used to train the `concept_model`.
        concept_model (ConceptModelProtocol): The model used to extract concepts from the activations of
            `model_with_split_points`. The only assumption for classes inheriting from this class is that
            the `concept_model` can encode activations into concepts with `encode_activations`.
            The `ConceptModelProtocol` is defined in `interpreto.typing`. It is basically a `torch.nn.Module` with an `encode` method.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
    """

    has_differentiable_concept_encoder = False

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModelProtocol,
        split_point: str | None = None,
    ):
        """Initializes the concept explainer with a given splitted model.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            concept_model (ConceptModelProtocol): The model used to extract concepts from
                the activations of `model_with_split_points`.
                The `ConceptModelProtocol` is defined in `interpreto.typing`. It is basically a `torch.nn.Module` with an `encode` method.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
        """
        if not isinstance(model_with_split_points, ModelWithSplitPoints):
            raise TypeError(
                f"The given model should be a ModelWithSplitPoints, but {type(model_with_split_points)} was given."
            )
        self.model_with_split_points: ModelWithSplitPoints = model_with_split_points
        self._concept_model = concept_model
        self.split_point = split_point  # Verified by `split_point.setter`
        self.__is_fitted: bool = False

    @property
    def concept_model(self) -> ConceptModelProtocol:
        """
        Returns:
            The concept model used to extract concepts from the activations of `model_with_split_points`.
            The `ConceptModelProtocol` is defined in `interpreto.typing`. It is basically a `torch.nn.Module` with an `encode` method.
        """
        # Declare the concept model as read-only property for inheritance typing flexibility
        return self._concept_model

    @property
    def is_fitted(self) -> bool:
        return self.__is_fitted

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
            activations (torch.Tensor | dict[str, torch.Tensor]): A dictionary with model paths as keys and the corresponding
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

    def _sanitize_activations(
        self,
        activations: LatentActivations | dict[str, LatentActivations],
    ) -> LatentActivations:
        if isinstance(activations, dict):
            split_activations: LatentActivations = self.model_with_split_points.get_split_activations(activations)  # type: ignore
        else:
            split_activations = activations
        assert len(split_activations.shape) == 2, (
            f"Input activations should be a 2D tensor of shape (batch_size, n_features) but got {split_activations.shape}. "
            + "If you use `ModelWithSplitPoints.get_activations()`, "
            + "make sure to set `activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS` to get a 2D activation tensor."
        )
        return split_activations

    def _prepare_fit(
        self,
        activations: LatentActivations | dict[str, LatentActivations],
        overwrite: bool,
    ) -> LatentActivations:
        if self.is_fitted and not overwrite:
            raise RuntimeError(
                "Concept explainer has already been fitted. Refitting will overwrite the current model."
                "If this is intended, use `overwrite=True` in fit(...)."
            )
        return self._sanitize_activations(activations)

    @check_fitted
    def interpret(
        self,
        interpretation_method: type[BaseConceptInterpretationMethod],
        concepts_indices: int | list[int] | Literal["all"],
        inputs: list[str] | None = None,
        latent_activations: dict[str, LatentActivations] | LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
        **kwargs,
    ) -> Mapping[int, Any]:
        """
        Interpret the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and an object allowing to interpret them.
        It can be a label, a description, examples, etc.

        Args:
            interpretation_method: The interpretation method to use to interpret the concepts.
            concepts_indices (int | list[int] | Literal["all"]): The indices of the concepts to interpret.
                If "all", all concepts are interpreted.
            inputs (list[str] | None): The inputs to use for the interpretation.
                Necessary if the source is not `VOCABULARY`, as examples are extracted from the inputs.
            latent_activations (LatentActivations | dict[str, LatentActivations] | None): The latent activations to use for the interpretation.
                Necessary if the source is `LATENT_ACTIVATIONS`.
                Otherwise, it is computed from the inputs or ignored if the source is `CONCEPT_ACTIVATIONS`.
            concepts_activations (ConceptsActivations | None): The concepts activations to use for the interpretation.
                Necessary if the source is not `CONCEPT_ACTIVATIONS`. Otherwise, it is computed from the latent activations.
            **kwargs: Additional keyword arguments to pass to the interpretation method.

        Returns:
            Mapping[int, Any]: A mapping between the concepts indices and the interpretation of the concepts.
        """
        if concepts_indices == "all":
            concepts_indices = list(range(self.concept_model.nb_concepts))

        # verify
        if latent_activations is not None:
            split_latent_activations = self._sanitize_activations(latent_activations)
        else:
            split_latent_activations = None

        # initialize the interpretation method
        method = interpretation_method(
            model_with_split_points=self.model_with_split_points,
            split_point=self.split_point,
            concept_model=self.concept_model,
            **kwargs,
        )

        # compute the interpretation from inputs and activations
        return method.interpret(
            concepts_indices=concepts_indices,
            inputs=inputs,
            latent_activations=split_latent_activations,
            concepts_activations=concepts_activations,
        )

    @check_fitted
    def input_concept_attribution(
        self,
        inputs: ModelInputs,
        concept: int,
        attribution_method: type[AttributionExplainer],
        **attribution_kwargs,
    ) -> list[float]:
        """Attributes model inputs for a selected concept.

        Args:
            inputs (ModelInputs): The input data, which can be a string, a list of tokens/words/clauses/sentences
                or a dataset.
            concept (int): Index identifying the position of the concept of interest (score in the
                `ConceptsActivations` tensor) for which relevant input elements should be retrieved.
            attribution_method: The attribution method to obtain importance scores for input elements.

        Returns:
            A list of attribution scores for each input.
        """
        raise NotImplementedError("Input-to-concept attribution method is not implemented yet.")


class ConceptAutoEncoderExplainer(ConceptEncoderExplainer[BaseDictionaryLearning], Generic[BDL]):
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

    has_differentiable_concept_decoder = False

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: BaseDictionaryLearning,
        split_point: str | None = None,
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
        self.concept_model: BaseDictionaryLearning
        super().__init__(model_with_split_points, concept_model, split_point)

    @property
    def is_fitted(self) -> bool:
        return self.concept_model.fitted

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
    def encode_activations(self, activations: LatentActivations) -> torch.Tensor:  # ConceptsActivations
        """Encode the given activations using the `concept_model` encoder.

        Args:
            activations (LatentActivations): The activations to encode.

        Returns:
            The encoded concept activations.
        """
        if hasattr(self.concept_model, "device") and self.concept_model.device != activations.device:
            activations = activations.to(self.concept_model.device, non_blocking=True)
            self.concept_model.to(activations.device)
        return self.concept_model.encode(activations)  # type: ignore

    @check_fitted
    def decode_concepts(self, concepts: ConceptsActivations) -> torch.Tensor:  # LatentActivations
        """Decode the given concepts using the `concept_model` decoder.

        Args:
            concepts (ConceptsActivations): The concepts to decode.

        Returns:
            The decoded model activations.
        """
        if hasattr(self.concept_model, "device") and self.concept_model.device != concepts.device:
            concepts = concepts.to(self.concept_model.device, non_blocking=True)
            self.concept_model.to(concepts.device)
        return self.concept_model.decode(concepts)  # type: ignore

    @check_fitted
    def get_dictionary(self) -> torch.Tensor:  # TODO: add this to tests
        """Get the dictionary learned by the fitted `concept_model`.

        Returns:
            torch.Tensor: A `torch.Tensor` containing the learned dictionary.
        """
        return self.concept_model.get_dictionary()  # type: ignore

    @check_fitted
    def concept_output_attribution(
        self,
        inputs: ModelInputs,
        concepts: ConceptsActivations,
        target: int,
        attribution_method: type[AttributionExplainer],
        **attribution_kwargs,
    ) -> list[float]:
        """Computes the attribution of each concept for the logit of a target output element.

        Args:
            inputs (ModelInputs): An input data-point for the model.
            concepts (torch.Tensor): Concept activation tensor.
            target (int): The target class for which the concept output attribution should be computed.
            attribution_method: The attribution method to obtain importance scores for input elements.

        Returns:
            A list of attribution scores for each concept.
        """
        raise NotImplementedError("Concept-to-output attribution method is not implemented yet.")

    @check_fitted
    def concept_output_gradient(
        self,
        inputs: torch.Tensor | list[str] | BatchEncoding,
        targets: list[int] | None = None,
        split_point: str | None = None,
        activation_granularity: ActivationGranularity = ActivationGranularity.TOKEN,
        aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        concepts_x_gradients: bool = False,
        tqdm_bar: bool = True,
    ) -> list[Float[torch.Tensor, "t g c"]]:
        """ """
        self.concept_model.to(self.model_with_split_points.device)
        gradients = self.model_with_split_points.get_concepts_output_gradients(
            inputs=inputs,
            targets=targets,
            encode_activations=self.encode_activations,
            decode_activations=self.decode_concepts,
            split_point=split_point,
            activation_granularity=activation_granularity,
            aggregation_strategy=aggregation_strategy,
            concepts_x_gradients=concepts_x_gradients,
            tqdm_bar=tqdm_bar,
        )
        return gradients
