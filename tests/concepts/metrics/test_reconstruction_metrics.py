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

import torch

from interpreto.commons.distances import DistanceFunctions
from interpreto.concepts import NeuronsAsConcepts, PCAConcepts
from interpreto.concepts.metrics import (
    FID,
    MSE,
    ReconstructionError,
    ReconstructionSpaces,
)
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_reconstruction_error(splitted_encoder_ml: ModelWithSplitPoints, activations: torch.Tensor):
    """
    Test the reconstruction error metrics
    """

    neurons_concept_explainer = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml)
    pca_concept_explainer = PCAConcepts(model_with_split_points=splitted_encoder_ml, nb_concepts=5)
    pca_concept_explainer.fit(activations)

    for metric_class in [MSE, FID]:
        try:
            # The neurons concept model if the identity function, the the reconstruction error should be 0
            metric = metric_class(neurons_concept_explainer)
            assert metric.compute(activations) == 0.0

            # The reconstruction error should be higher than 0 for non-identity functions
            metric = metric_class(pca_concept_explainer)
            assert metric.compute(activations) > 0.0
        except Exception as e:
            raise AssertionError(f"Error with {metric_class.__name__}") from e


def test_latent_activations_reconstruction_error(splitted_encoder_ml: ModelWithSplitPoints, activations: torch.Tensor):
    """
    Test the latent activations reconstruction error metrics
    """

    pca_concept_explainer = PCAConcepts(model_with_split_points=splitted_encoder_ml, nb_concepts=5)
    pca_concept_explainer.fit(activations)

    base_metric = ReconstructionError(
        pca_concept_explainer, ReconstructionSpaces.LATENT_ACTIVATIONS, DistanceFunctions.EUCLIDEAN
    )
    base_score = base_metric.compute(activations)

    metric = MSE(pca_concept_explainer)
    score = metric.compute(activations)

    # Test the MSE metric corresponds to the reconstruction error with the latent activations and the euclidean distance
    assert score == base_score


def test_fid(splitted_encoder_ml: ModelWithSplitPoints, activations: torch.Tensor):
    """
    Test the fid metrics
    """
    pca_concept_explainer = PCAConcepts(model_with_split_points=splitted_encoder_ml, nb_concepts=5)
    pca_concept_explainer.fit(activations)

    base_metric = ReconstructionError(
        pca_concept_explainer, ReconstructionSpaces.LATENT_ACTIVATIONS, DistanceFunctions.WASSERSTEIN_1D
    )
    base_score = base_metric.compute(activations)

    metric = FID(pca_concept_explainer)
    score = metric.compute(activations)

    # Test the FID metric corresponds to the reconstruction error with the latent activations and the Wasserstein 1D distance
    assert score == base_score
