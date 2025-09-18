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
"""Concept encoder/decoder models wrapping scikit-learn components."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from overcomplete.optimization import BaseOptimDictionaryLearning
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from torch import nn

from interpreto import ModelWithSplitPoints
from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.concepts.methods.overcomplete import DictionaryLearningExplainer

__all__ = [
    "ICAConcepts",
    "KMeansConcepts",
    "PCAConcepts",
    "SVDConcepts",
]


class SkLearnWrapper(BaseOptimDictionaryLearning):
    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ):
        """
        Abstract concept model built around scikit-learn.
        The inheriting classes convert sklearn models into torch modules.

        Args:
            nb_concepts (int): Number of concepts to extract.
            input_size (int): Size of the input to the model. Often referred to as d in our code.
            random_state (int): Random state for the model.
            device (torch.device | str): Device to use for the model.

        """
        super().__init__(nb_concepts, device=device)  # type:ignore[override]
        self.input_size = input_size
        self.random_state = random_state

    @abstractmethod
    def fit(  # type:ignore[override]
        self, x: Float[torch.Tensor, "n {self.input_size}"], return_sklearn_model: bool = False
    ) -> TransformerMixin | None:
        pass

    @abstractmethod
    def encode(self, x: Float[torch.Tensor, "n {self.input_size}"]) -> Float[torch.Tensor, "n {self.nb_concepts}"]:  # type:ignore[override]
        pass

    @abstractmethod
    def decode(self, z: Float[torch.Tensor, "n {self.nb_concepts}"]) -> Float[torch.Tensor, "n {self.input_size}"]:  # type:ignore[override]
        pass

    @abstractmethod
    def get_dictionary(self) -> nn.Parameter:  # type:ignore[override]
        pass


class ICAWrapper(SkLearnWrapper):
    """Independent Component Analysis concept model."""

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ):
        super().__init__(nb_concepts=nb_concepts, input_size=input_size, random_state=random_state, device=device)

        self.mean = nn.Parameter(torch.zeros(input_size))
        self.components = nn.Parameter(torch.zeros(input_size, nb_concepts))
        self.mixing = nn.Parameter(torch.zeros(nb_concepts, input_size))
        self.to(device)

    @jaxtyped(typechecker=beartype)
    def fit(self, x: Float[torch.Tensor, "n {self.input_size}"], return_sklearn_model: bool = False) -> FastICA | None:
        ica = FastICA(n_components=self.nb_concepts, random_state=self.random_state, max_iter=500)
        ica.fit(x.detach().cpu().numpy())

        self.mean.data = torch.as_tensor(ica.mean_, dtype=torch.float32, device=self.mean.device)
        self.components.data = torch.as_tensor(ica.components_.T, dtype=torch.float32, device=self.components.device)
        self.mixing.data = torch.as_tensor(ica.mixing_.T, dtype=torch.float32, device=self.mixing.device)  # type: ignore
        self.fitted = True

        if return_sklearn_model:
            return ica

        del ica

    @jaxtyped(typechecker=beartype)
    def encode(self, x: Float[torch.Tensor, "n {self.input_size}"]) -> Float[torch.Tensor, "n {self.nb_concepts}"]:
        self._assert_fitted()
        x = x.to(self.mean.device)
        return (x - self.mean) @ self.components

    @jaxtyped(typechecker=beartype)
    def decode(self, z: Float[torch.Tensor, "n {self.nb_concepts}"]) -> Float[torch.Tensor, "n {self.input_size}"]:
        self._assert_fitted()
        z = z.to(self.mixing.device)
        return (z @ self.mixing) + self.mean

    def get_dictionary(self):
        self._assert_fitted()
        return self.mixing


class PCAWrapper(SkLearnWrapper):
    """Principal Component Analysis concept model."""

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(nb_concepts=nb_concepts, input_size=input_size, random_state=random_state, device=device)
        self.input_size = input_size
        self.random_state = random_state

        self.mean = nn.Parameter(torch.zeros(input_size))
        self.components = nn.Parameter(torch.zeros(nb_concepts, input_size))
        self.to(device)

    @jaxtyped(typechecker=beartype)
    def fit(
        self, x: Float[torch.Tensor, "n {self.input_size}"], return_sklearn_model: bool = False, **kwargs
    ) -> PCA | None:
        pca = PCA(n_components=self.nb_concepts, random_state=self.random_state, **kwargs)
        pca.fit(x.detach().cpu().numpy())
        self.mean.data = torch.as_tensor(pca.mean_, dtype=torch.float32, device=self.mean.device)
        self.components.data = torch.as_tensor(pca.components_, dtype=torch.float32, device=self.components.device)
        self.fitted = True

        if return_sklearn_model:
            return pca

        del pca

    @jaxtyped(typechecker=beartype)
    def encode(self, x: Float[torch.Tensor, "n {self.input_size}"]) -> Float[torch.Tensor, "n {self.nb_concepts}"]:
        self._assert_fitted()
        x = x.to(self.mean.device)
        return (x - self.mean) @ self.components.T

    @jaxtyped(typechecker=beartype)
    def decode(self, z: Float[torch.Tensor, "n {self.nb_concepts}"]) -> Float[torch.Tensor, "n {self.input_size}"]:
        self._assert_fitted()
        z = z.to(self.components.device)
        return (z @ self.components) + self.mean

    def get_dictionary(self):
        self._assert_fitted()
        return self.components


class SVDWrapper(SkLearnWrapper):
    """
    Singular Value Decomposition concept model.

    Solve A = U @ S @ V^T

    Then consider the concept x as U @ S
    and the concept encoding as V^T

    Finally, as $V^(-1) = v^T$, for any new A called A', we have
    $U' @ S = A' @ V$
    """

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(nb_concepts=nb_concepts, input_size=input_size, random_state=random_state, device=device)
        self.input_size = input_size
        self.random_state = random_state

        self.components = nn.Parameter(torch.zeros(nb_concepts, input_size))
        self.to(device)

    @jaxtyped(typechecker=beartype)
    def fit(
        self, x: Float[torch.Tensor, "n {self.input_size}"], return_sklearn_model: bool = False, **kwargs
    ) -> TruncatedSVD | None:
        svd = TruncatedSVD(n_components=self.nb_concepts, random_state=self.random_state, **kwargs)
        svd.fit(x.detach().cpu().numpy())

        self.components.data = torch.as_tensor(svd.components_, dtype=torch.float32, device=self.components.device)
        self.fitted = True

        if return_sklearn_model:
            return svd

        del svd

    @jaxtyped(typechecker=beartype)
    def encode(self, x: Float[torch.Tensor, "n {self.input_size}"]) -> Float[torch.Tensor, "n {self.nb_concepts}"]:
        self._assert_fitted()
        x = x.to(self.components.device)
        return x @ self.components.T

    @jaxtyped(typechecker=beartype)
    def decode(self, z: Float[torch.Tensor, "n {self.nb_concepts}"]) -> Float[torch.Tensor, "n {self.input_size}"]:
        self._assert_fitted()
        z = z.to(self.components.device)
        return z @ self.components

    def get_dictionary(self):
        self._assert_fitted()
        return self.components


class KMeansWrapper(SkLearnWrapper):
    """K-means concept model."""

    def __init__(
        self,
        nb_concepts: int,
        input_size: int,
        *,
        random_state: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(nb_concepts=nb_concepts, input_size=input_size, random_state=random_state, device=device)
        self.input_size = input_size
        self.random_state = random_state

        self.components = nn.Parameter(torch.zeros(nb_concepts, input_size))
        self.to(device)

    @jaxtyped(typechecker=beartype)
    def fit(self, x: Float[torch.Tensor, "n d"], return_sklearn_model: bool = False, **kwargs) -> KMeans | None:
        kmeans = KMeans(n_clusters=self.nb_concepts, random_state=self.random_state, **kwargs)
        kmeans.fit(x.detach().cpu().numpy())
        self.components.data = torch.as_tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=self.components.device
        )
        self.fitted = True

        if return_sklearn_model:
            return kmeans

        del kmeans

    @jaxtyped(typechecker=beartype)
    def encode(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "n {self.nb_concepts}"]:
        self._assert_fitted()
        x = x.to(self.components.device)

        # Compute distances to cluster centers
        return torch.cdist(x, self.components, p=2)

    @jaxtyped(typechecker=beartype)
    def decode(self, z: Float[torch.Tensor, "n {self.nb_concepts}"]) -> Float[torch.Tensor, "n d"]:
        self._assert_fitted()
        z = z.to(self.components.device)
        return z @ self.components

    def get_dictionary(self):
        self._assert_fitted()
        return self.components


_SkLearnWrapper_co = TypeVar("_SkLearnWrapper_co", bound=SkLearnWrapper, covariant=True)


class SkLearnWrapperExplainer(DictionaryLearningExplainer[SkLearnWrapper], Generic[_SkLearnWrapper_co]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/sklearn_wrappers.py)

    Implementation of a concept explainer using wrappers around sklearn decompositions as `concept_model`.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str | None): The split point used to train the `concept_model`. Default: `None`, set only when
            the concept explainer is fitted.
        concept_model (SkLearnWrapper): A wrapper around a sklearn decomposition for concept extraction.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.
    """

    has_differentiable_concept_encoder = True
    has_differentiable_concept_decoder = True

    @property
    @abstractmethod
    def concept_model_class(self) -> type[SkLearnWrapper]:
        pass

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        *,
        nb_concepts: int,
        split_point: str | None = None,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete BaseOptimDictionaryLearning framework.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            nb_concepts (int): Size of the SAE concept space.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
            device (torch.device | str): Device to use for the `concept_module`.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_module`.
                See the sklearn documentation of the provided `concept_model_class` for more details.
        """
        self.model_with_split_points = model_with_split_points
        self.split_point: str = split_point  # type:ignore[assignment]
        shapes = model_with_split_points.get_latent_shape()

        concept_model = self.concept_model_class(
            input_size=shapes[self.split_point][-1],
            nb_concepts=nb_concepts,
            device=device,
            **kwargs,
        )
        ConceptAutoEncoderExplainer.__init__(
            self, model_with_split_points, concept_model=concept_model, split_point=split_point
        )


class PCAConcepts(SkLearnWrapperExplainer[PCAWrapper]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/sklearn_wrappers.py)

    `ConceptAutoEncoderExplainer` with the PCA from Pearson (1901)[^3] as concept model.

    [^3]:
        K. Pearson, [On lines and planes of closest fit to systems of points in space](https://doi.org/10.1080/14786440109462720).
        Philosophical Magazine, 2(11), 1901, pp. 559-572.
    """

    @property
    def concept_model_class(self) -> type[PCAWrapper]:
        return PCAWrapper


class ICAConcepts(SkLearnWrapperExplainer[ICAWrapper]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/sklearn_wrappers.py)

    `ConceptAutoEncoderExplainer` with the ICA from Hyvarinen and Oja (2000)[^4] as concept model.

    [^4]:
        A. Hyvarinen and E. Oja, [Independent Component Analysis: Algorithms and Applications](https://www.sciencedirect.com/science/article/pii/S0893608000000265),
        Neural Networks, 13(4-5), 2000, pp. 411-430.
    """

    @property
    def concept_model_class(self) -> type[ICAWrapper]:
        return ICAWrapper


class KMeansConcepts(SkLearnWrapperExplainer[KMeansWrapper]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/sklearn_wrappers.py)

    `ConceptAutoEncoderExplainer` with the K-Means as concept model.
    """

    @property
    def concept_model_class(self) -> type[KMeansWrapper]:
        return KMeansWrapper


class SVDConcepts(SkLearnWrapperExplainer[SVDWrapper]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/sklearn_wrappers.py)

    `ConceptAutoEncoderExplainer` with SVD as concept model.
    """

    @property
    def concept_model_class(self) -> type[SVDWrapper]:
        return SVDWrapper
