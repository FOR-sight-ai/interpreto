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

from __future__ import annotations

from enum import Enum

from interpreto.commons.distances import DistanceFunctionProtocol, DistanceFunctions
from interpreto.concepts.base import ConceptAutoEncoderExplainer
from interpreto.typing import ConceptsActivations, LatentActivations


class ReconstructionSpaces(Enum):
    """
    Enumeration of possible reconstruction spaces.
    Latent activations go through the concept autoencoder to obtain reconstructed latent activations.
    Then it is possible to compute the distance between the original and reconstructed latent activations.
    First directly in the latent space, second in the logits space.

    Attributes:
        LATENT_ACTIVATIONS (str): Reconstruction space in the latent space.
        LOGITS (str): Reconstruction space in the logits space.
    """

    LATENT_ACTIVATIONS = "latent_activations"
    LOGITS = "logits"


class ReconstructionError:
    """Code [:octicons-mark-github-24: `concepts/metrics/reconstruction_metrics.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/reconstruction_metrics.py)

    Evaluates wether the information reconstructed by the concept autoencoder corresponds to the original latent
    activations. It corresponds to a faithfulness metric.
    The space where the distance thus error is computed and the distance function used can be specified.

    Attributes:
        concept_explainer (ConceptAutoEncoderExplainer): The explainer used to compute concepts.
        reconstruction_space (ReconstructionSpaces): The space in which the reconstruction error is computed.
        distance_function (DistanceFunctionProtocol): The distance function used to compute the reconstruction error.
    """

    def __init__(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
        reconstruction_space: ReconstructionSpaces,
        distance_function: DistanceFunctionProtocol,
    ):
        self.concept_explainer = concept_explainer
        self.reconstruction_space = reconstruction_space
        self.distance_function = distance_function

    def compute(self, latent_activations: LatentActivations | dict[str, LatentActivations]) -> float:
        """Compute the reconstruction error.

        Args:
            latent_activations (LatentActivations | dict[str, LatentActivations]): The latent activations to use for the computation.

        Returns:
            float: The reconstruction error.
        """
        split_latent_activations: LatentActivations = self.concept_explainer._sanitize_activations(latent_activations)

        concepts_activations: ConceptsActivations = self.concept_explainer.encode_activations(split_latent_activations)

        reconstructed_latent_activations: LatentActivations = self.concept_explainer.decode_concepts(
            concepts_activations
        )

        if self.reconstruction_space is ReconstructionSpaces.LATENT_ACTIVATIONS:
            return self.distance_function(split_latent_activations, reconstructed_latent_activations).item()

        raise NotImplementedError("Only LATENT_ACTIVATIONS reconstruction space is supported.")


class MSE(ReconstructionError):
    r"""Code [:octicons-mark-github-24: `concepts/metrics/reconstruction_metrics.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/reconstruction_metrics.py)

    Evaluates wether the information reconstructed by the concept autoencoder corresponds to the original latent
    activations. It is a faithfulness metric.
    It is computed in the latent activations space through the Euclidean distance.
    It is also known as the reconstruction error.

    With $A$ latent activations obtained through $A = h(X)$,
    $t$ and $t^{-1}$ the concept encoder and decoders, the MSE is defined as:
    $$ \sum_{a}^{A} ||t^{-1}(t(a)) - a||_2 $$

    TODO: make formula work

    Attributes:
        concept_explainer (ConceptAutoEncoderExplainer): The explainer used to compute concepts.
    """

    def __init__(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
    ):
        super().__init__(
            concept_explainer=concept_explainer,
            reconstruction_space=ReconstructionSpaces.LATENT_ACTIVATIONS,
            distance_function=DistanceFunctions.EUCLIDEAN,
        )


class FID(ReconstructionError):
    r"""Code [:octicons-mark-github-24: `concepts/metrics/reconstruction_metrics.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/reconstruction_metrics.py)

    Evaluates wether the information reconstructed by the concept autoencoder corresponds to the original latent activations.
    It corresponds to a faithfulness metric, it measures if the reconstructed distribution matches the original distribution.
    It is computed in the latent activations space through the Wasserstein 1D distance.

    This metric was introduced by Fel et al. (2023)[^1]

    With $A$ latent activations obtained through $A = h(X)$,
    $t$ and $t^{-1}$ the concept encoder and decoders, and
    $\mathcal{W}_1$ the 1-Wassertein distance, the FID is defined as:

    $$ \mathcal{W}_1(A, t^{-1}(t(A))) $$

    TODO: make formula work

    [^1]:
        Fel, T., Boutin, V., Béthune, L., Cadène, R., Moayeri, M., Andéol, L., Chavidal, M., & Serre, T.
        [A holistic approach to unifying automatic concept extraction and concept importance estimation.](https://arxiv.org/abs/2306.07304)
        Advances in Neural Information Processing Systems. 2023.

    Attributes:
        concept_explainer (ConceptAutoEncoderExplainer): The explainer used to compute concepts.
    """

    def __init__(
        self,
        concept_explainer: ConceptAutoEncoderExplainer,
    ):
        super().__init__(
            concept_explainer=concept_explainer,
            reconstruction_space=ReconstructionSpaces.LATENT_ACTIVATIONS,
            distance_function=DistanceFunctions.WASSERSTEIN_1D,
        )


# TODO: implement when the concept to output forward works
# class Completeness(ReconstructionError):
#     pass
