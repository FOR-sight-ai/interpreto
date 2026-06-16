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
from types import SimpleNamespace
from typing import Any, Generic, TypeVar

import torch
from jaxtyping import Float
from transformers.tokenization_utils_base import BatchEncoding

from interpreto._vendor.overcomplete.base import BaseDictionaryLearning
from interpreto.concepts.splitters.base_splitter import BaseSplitter
from interpreto.concepts.splitters.model_with_split_points import (
    ActivationGranularity,
    GranularityAggregationStrategy,
)
from interpreto.concepts.splitters.splitter_for_classification import SplitterForClassification
from interpreto.typing import (
    ConceptModelProtocol,
    ConceptsActivations,
    IncompatibilityError,
    LatentActivations,
)

ConceptModel = TypeVar("ConceptModel", bound=ConceptModelProtocol)
BDL = TypeVar("BDL", bound=BaseDictionaryLearning)
MethodOutput = TypeVar("MethodOutput")


# Decorator that checks if the concept model is fitted before calling the method
def check_fitted(func: Callable[..., MethodOutput]) -> Callable[..., MethodOutput]:
    @wraps(func)
    def wrapper(self: ConceptEncoderExplainer, *args, **kwargs) -> MethodOutput:
        if not self.is_fitted:
            raise RuntimeError("Concept encoder is not fitted yet. Use the .fit() method to fit the explainer.")
        return func(self, *args, **kwargs)

    return wrapper


class ModelForInputsToConcepts:
    """Bridge model that maps raw inputs to concept activations.

    Composes a ``SplitterForClassification`` (inputs → latent activations)
    with a concept model encoder (latent activations → concept activations).

    The goal is to return this in concept_explainer.get_inputs_to_concepts_model() method.
    Which will then be used for in attribution methods.

    The resulting object quacks enough like a ``PreTrainedModel`` to be usable
    inside ``InputsToConceptsInferenceWrapper``: it exposes ``.eval()``,
    ``.config.pad_token_id``, and ``__call__`` returns an object with a
    ``.logits`` attribute.
    """

    def __init__(
        self,
        concept_explainer: ConceptEncoderExplainer,
    ):
        self.concept_explainer = concept_explainer
        splitter = concept_explainer.splitter  # type: ignore

        if not isinstance(splitter, SplitterForClassification):
            raise IncompatibilityError(
                f"The split model must be a SplitterForClassification model. Got {splitter.__class__.__name__}."
            )

        self.to(self.concept_explainer.splitter.device)  # type: ignore

        self.nb_concepts = concept_explainer.concept_model.nb_concepts

        # Expose a minimal config so InferenceWrapper.__init__ and setup_token_ids can work
        self.config = SimpleNamespace(
            pad_token_id=splitter.tokenizer.pad_token_id,
            vocab_size=getattr(splitter._model.config, "vocab_size", None),
        )

    def eval(self):
        """No-op: the underlying models are already in eval mode via nnsight."""
        return self

    def resize_token_embeddings(self, new_num_tokens: int):
        """No-op: the concept model does not have token embeddings."""
        self.concept_explainer.splitter._model.resize_token_embeddings(new_num_tokens)

    def __call__(self, **kwargs):
        """Run inputs → activations → concepts and return a BaseModelOutput-like object.

        Returns:
            SimpleNamespace with a ``.logits`` attribute containing concept activations.
        """
        concepts = self.concept_explainer.inputs_to_concepts(**kwargs)
        return SimpleNamespace(logits=concepts)

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: The device on which the model is loaded.
        """
        if self.concept_explainer.splitter.device != self.concept_explainer.device:
            self.concept_explainer.to(self.concept_explainer.splitter.device)  # type: ignore
        return self.concept_explainer.splitter.device  # type: ignore

    @device.setter
    def device(self, device: torch.device):
        """
        Sets the device on which the model is loaded.

        Args:
            device (torch.device): wanted device (e.g., "cpu" or "cuda").
        """
        self.to(device)

    def to(self, device: torch.device):
        """
        Move the model to the specified device.

        Args:
            device (torch.device): The device to which the model should be moved.
        """
        self.concept_explainer.splitter.to(device)  # type: ignore
        self.concept_explainer.to(device)  # type: ignore


class ConceptEncoderExplainer(ABC, Generic[ConceptModel]):
    """Code: [:octicons-mark-github-24: `concepts/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/base.py)

    Abstract class defining an interface for concept explanation.
    Child classes should implement the `fit` and `activations_to_concepts` methods, and only assume the presence of an
        encoding step using the `concept_model` to convert activations to latent concepts.

    Attributes:
        splitter (BaseSplitter): The model to apply the explanation on.
            The split point is determined by the model's `split_point` attribute.
        concept_model (ConceptModelProtocol): The model used to extract concepts from the activations of
            `splitter`. The only assumption for classes inheriting from this class is that
            the `concept_model` can encode activations into concepts with `activations_to_concepts`.
            The `ConceptModelProtocol` is defined in `interpreto.typing`. It is basically a `torch.nn.Module` with an `encode` method.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `activations_to_concepts` operation is differentiable.
    """

    has_differentiable_concept_encoder = False

    def __init__(
        self,
        splitter: BaseSplitter,
        concept_model: ConceptModelProtocol,
    ):
        """Initializes the concept explainer with a given splitted model.

        Args:
            splitter (BaseSplitter): The model to apply the explanation on.
                Its `split_point` attribute determines where activations are extracted.
            concept_model (ConceptModelProtocol): The model used to extract concepts from
                the activations of `splitter`.
                The `ConceptModelProtocol` is defined in `interpreto.typing`. It is basically a `torch.nn.Module` with an `encode` method.
        """
        if not isinstance(splitter, BaseSplitter):
            raise TypeError(f"The given model should be a BaseSplitter (or subclass), but {type(splitter)} was given.")
        self.splitter: BaseSplitter = splitter
        self._concept_model = concept_model
        self.__is_fitted: bool = False

    @property
    def concept_model(self) -> ConceptModelProtocol:
        """
        Returns:
            The concept model used to extract concepts from the activations of `splitter`.
            The `ConceptModelProtocol` is defined in `interpreto.typing`. It is basically a `torch.nn.Module` with an `encode` method.
        """
        # Declare the concept model as read-only property for inheritance typing flexibility
        return self._concept_model

    @property
    def is_fitted(self) -> bool:
        return self.__is_fitted

    @property
    def device(self) -> torch.device:
        """Return the device of the concept model, independent of the splitter device."""
        concept_model = self.concept_model
        if isinstance(concept_model, torch.nn.Module):
            try:
                return next(concept_model.parameters()).device
            except StopIteration:
                try:
                    return next(concept_model.buffers()).device
                except StopIteration:
                    pass
        return torch.device(getattr(concept_model, "device", "cpu"))

    def to(self, device: torch.device | str) -> None:
        """Move only the concept model to ``device``; the splitter remains user-managed."""
        device = torch.device(device)
        if hasattr(self.concept_model, "to"):
            self.concept_model.to(device)  # type: ignore[call-arg]
        if hasattr(self.concept_model, "device"):
            self.concept_model.device = device  # type: ignore[attr-defined]

    @device.setter
    def device(self, device: torch.device) -> None:
        """Set the device on which the concept model is stored."""
        self.to(device)

    @abstractmethod
    def fit(self, activations: LatentActivations, *args, **kwargs) -> Any:
        """Fits `concept_model` on the given activations.

        Args:
            activations (torch.Tensor): The latent activations used to fit the concept model.

        Returns:
            `None`, `concept_model` is fitted in-place, `is_fitted` is set to `True` and `split_point` is set.
        """
        pass

    @abstractmethod
    def activations_to_concepts(self, activations: LatentActivations) -> ConceptsActivations:
        """Abstract method defining how activations are converted into concepts by the concept encoder.

        Args:
            activations (torch.Tensor): The activations to encode.

        Returns:
            A `torch.Tensor` of encoded activations produced by the fitted concept encoder.
        """
        pass

    def inputs_to_concepts(self, **kwargs) -> ConceptsActivations:
        """Abstract method defining how inputs are converted into concepts by the concept encoder.

        Args:
            kwargs (Any): The inputs to encode.

        Returns:
            A `torch.Tensor` of encoded activations produced by the fitted concept encoder.
        """
        activations: Float[torch.Tensor, "n d"] = self.splitter.inputs_to_activations(kwargs)
        return self.activations_to_concepts(activations)

    def get_inputs_to_concepts_model(self) -> ModelForInputsToConcepts:
        """Returns a model that maps raw inputs to concept activations.

        The model can be passed to an attribution method,
        to obtain inputs to concepts attributions.
        Which are ways to interpret the concepts.

        Returns:
            ModelForInputsToConcepts: A model that maps raw inputs to concept activations.
        """
        return ModelForInputsToConcepts(self)


class ConceptAutoEncoderExplainer(ConceptEncoderExplainer[BaseDictionaryLearning], Generic[BDL]):
    """Code: [:octicons-mark-github-24: `concepts/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/base.py)

    A concept bottleneck explainer wraps a `concept_model` that should be able to encode activations into concepts
    and decode concepts into activations.

    We use the term "concept bottleneck" loosely, as the latent space can be overcomplete compared to activation
        space, as in the case of sparse autoencoders.

    We assume that the concept model follows the structure of an [`overcomplete.BaseDictionaryLearning`](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)
    model, which defines the `encode` and `decode` methods for encoding and decoding activations into concepts.

    Attributes:
        splitter (ModelWithSplitPoints): The model to apply the explanation on.
            The split point is determined by the model's `split_point` attribute.
        concept_model ([BaseDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)): The model used to extract concepts from the
            activations of  `splitter`. The only assumption for classes inheriting from this class is
            that the `concept_model` can encode activations into concepts with `activations_to_concepts`.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `activations_to_concepts` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `concepts_to_activations` operation is differentiable.
    """

    has_differentiable_concept_decoder = False

    def __init__(
        self,
        splitter: BaseSplitter,
        concept_model: BaseDictionaryLearning,
    ):
        """Initializes the concept explainer with a given splitted model.

        Args:
            splitter (BaseSplitter): The model to apply the explanation on.
                Its `split_point` attribute determines where activations are extracted.
            concept_model ([BaseDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/24568ba5736cbefca4b78a12246d92a1be04a1f4/overcomplete/base.py#L10)): The model used to extract concepts from
                the activations of `splitter`.
        """
        self.concept_model: BaseDictionaryLearning
        super().__init__(splitter, concept_model)  # type: ignore

    @property
    def is_fitted(self) -> bool:
        return self.concept_model.fitted

    @check_fitted
    def activations_to_concepts(self, activations: LatentActivations) -> torch.Tensor:  # ConceptsActivations
        """Encode the given activations using the `concept_model` encoder.

        Args:
            activations (LatentActivations): The activations to encode.

        Returns:
            The encoded concept activations.
        """
        if self.device != activations.device:
            activations = activations.to(self.device, non_blocking=True)
        return self.concept_model.encode(activations)  # type: ignore

    @check_fitted
    def concepts_to_activations(self, concepts: ConceptsActivations) -> torch.Tensor:  # LatentActivations
        """Decode the given concepts using the `concept_model` decoder.

        Args:
            concepts (ConceptsActivations): The concepts to decode.

        Returns:
            The decoded model activations.
        """
        if self.device != concepts.device:
            concepts = concepts.to(self.device, non_blocking=True)
        return self.concept_model.decode(concepts)  # type: ignore

    @check_fitted
    def get_dictionary(self) -> torch.Tensor:  # TODO: add this to tests
        """Get the dictionary learned by the fitted `concept_model`.

        Returns:
            torch.Tensor: A `torch.Tensor` containing the learned dictionary.
        """
        return self.concept_model.get_dictionary()  # type: ignore

    def __normalize_gradients(self, gradients: Float[torch.Tensor, "t g c"]) -> Float[torch.Tensor, "t g c"]:
        """
        Normalize the gradients as described in parameter `normalization` of `concept_output_gradient`.
        But for a single sample.

        Args:
            gradients (Float[torch.Tensor, "t g c"]):
                The gradients to normalize.

        Returns:
            The normalized gradients.
        """
        # normalize the gradients
        target_importance_sum: Float[torch.Tensor, "t 1 1"] = gradients.abs().sum(dim=-1).sum(dim=-1).view(-1, 1, 1)
        normalized_gradients: Float[torch.Tensor, "t g c"] = gradients / target_importance_sum

        return normalized_gradients

    @check_fitted
    def concept_output_gradient(
        self,
        inputs: torch.Tensor | list[str] | BatchEncoding,
        targets: list[int] | None = None,
        activation_granularity: ActivationGranularity = ActivationGranularity.TOKEN,
        aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        concepts_x_gradients: bool = True,
        normalization: bool = True,
        tqdm_bar: bool = False,
        batch_size: int | None = None,
    ) -> list[Float[torch.Tensor, "t g c"]]:
        """
        Compute the gradients of the predictions with respect to the concepts.

        To clarify what this function does, lets detail some notations.
        Suppose the initial model was splitted such that $f = g \\circ h$.
        Hence the concept model was fitted on $A = h(X)$ with $X$ a dataset of samples.
        The resulting concept model encoders and decoders are noted $t$ and $t^{-1}$.
        $t$ can be seen as projections from the latent space to the concept space.
        Hence, the function going from the inputs to the concepts is $f_{ic} = t \\circ h$
        and the function going from the concepts to the outputs is $f_{co} = g \\circ t^-1$.

        Given a set of samples $X$, and the functions $(h, t, t^{-1}, g)$
        This function first compute $C = t(A) = t \\circ h(X)$, then returns $\\nabla{f_{co}}(C)$.

        In practice all computations are done by `ModelWithSplitPoints._get_concept_output_gradients`,
        which relies on NNsight. The current method only forwards the $t$ and $t^{-1}$,
        respectively `self.activations_to_concepts` and `self.concepts_to_activations` methods.

        Args:
            inputs (list[str] | torch.Tensor | BatchEncoding):
                The input data, either a list of samples, the tokenized input or a batch of samples.

            targets (list[int] | None):
                Specify which outputs of the model should be used to compute the gradients.
                Note that $f_{co}$ often has several outputs, by default gradients are computed for each output.
                The `t` dimension of the returned tensor is equal to the number of selected targets.
                (For classification, those are the classes logits and for generation, those are the most probable tokens probabilities).

            activation_granularity (ActivationGranularity):
                The granularity of the activations to use for the attribution.
                It is highly recommended to to use the same granularity as the one used in the `fit` method.
                Possibles values are:

                - ``ModelWithSplitPoints.activation_granularities.CLS_TOKEN``:
                    only the first token (e.g. ``[CLS]``) activation is returned ``(batch, d_model)``.

                - ``ModelWithSplitPoints.activation_granularities.ALL_TOKENS``:
                    every token activation is treated as a separate element ``(batch x seq_len, d_model)``.

                - ``ModelWithSplitPoints.activation_granularities.TOKEN``: remove special tokens.

                - ``ModelWithSplitPoints.activation_granularities.WORD``:
                    aggregate by words following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.WORD`.

                - ``ModelWithSplitPoints.activation_granularities.SENTENCE``:
                    aggregate by sentences following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.SENTENCE`.

            aggregation_strategy:
                Strategy to aggregate token activations into larger inputs granularities.
                Applied for `WORD` and `SENTENCE` activation strategies.
                Token activations of shape  n * (l, d) are aggregated on the sequence length dimension.
                The concatenated into (ng, d) tensors.
                Existing strategies are:

                - ``ModelWithSplitPoints.aggregation_strategies.SUM``:
                    Tokens activations are summed along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MEAN``:
                    Tokens activations are averaged along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MAX``:
                    The maximum of the token activations along the sequence length dimension is selected.

                - ``ModelWithSplitPoints.aggregation_strategies.SIGNED_MAX``:
                    The maximum of the absolute value of the activations multiplied by its initial sign.
                    signed_max([[-1, 0, 1, 2], [-3, 1, -2, 0]]) = [-3, 1, -2, 2]

            concepts_x_gradients (bool):
                If the resulting gradients should be multiplied by the concepts activations.
                True by default (similarly to attributions), because of mathematical properties.
                Therefore the out put is $C * \\nabla{f_{co}}(C)$.

            normalization (bool):
                Whether to normalize the gradients.
                Gradients will be normalized on the concept (c) and sequence length (g) dimensions.
                Such that for a given sample-target-granular pair,
                the sum of the absolute values of the gradients is equal to 1.
                (The granular elements depend on the :arg:`activation_granularity`).

            tqdm_bar (bool):
                Whether to display a progress bar.

            batch_size (int | None):
                Batch size for the model.
                It might be different from the one used in `ModelWithSplitPoints.get_activations`
                because gradients have a much larger impact on the memory.

        Returns:
            list[Float[torch.Tensor, "t g c"]]:
                The gradients of the model output with respect to the concept activations.
                List length: correspond to the number of inputs.
                    Tensor shape: (t, g, c) with t the target dimension, g the number of granularity elements in one input, and c the number of
                    concepts.
        """
        if not self.has_differentiable_concept_decoder:
            raise ValueError(
                "The concept decoder of this explainer is not differentiable. This is required to compute concept-to-output gradients. "
                f"Current explainer class: {self.__class__.__name__}."
            )

        # put everything on device
        self.to(self.splitter.device)  # type: ignore

        # forward all computations to
        gradients = self.splitter._get_concept_output_gradients(
            inputs=inputs,
            targets=targets,
            activations_to_concepts=self.activations_to_concepts,
            concepts_to_activations=self.concepts_to_activations,
            activation_granularity=activation_granularity,
            aggregation_strategy=aggregation_strategy,
            concepts_x_gradients=concepts_x_gradients,
            tqdm_bar=tqdm_bar,
            batch_size=batch_size,
        )

        # normalize the gradients if required
        if normalization:
            gradients = [self.__normalize_gradients(g) for g in gradients]
        return gradients
