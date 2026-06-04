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

import pytest
import torch

from interpreto.concepts import NeuronsAsConcepts
from interpreto.concepts.metrics import ConceptMatchingAlgorithm, Stability
from interpreto.concepts.splitters.model_with_split_points import ModelWithSplitPoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_stability_values():
    """
    Test the stability metric on corner cases
    """
    eps = 1e-4
    shape = (50, 50)
    range_up = torch.arange(0, 50 * 50, dtype=torch.float32).reshape(shape)
    range_down_T = torch.arange(50 * 50, 0, -1, dtype=torch.float32).reshape(shape).T

    for algo in [
        ConceptMatchingAlgorithm.COSINE_HUNGARIAN,
        ConceptMatchingAlgorithm.COSINE_MAXIMUM,
    ]:
        try:
            # Stability of two identical matrices should be really close to 1
            stability_identical = Stability(range_up, range_up, matching_algorithm=algo).compute()
            assert abs(stability_identical - 1.0) < eps

            # Stability should be between 0 and 1
            stability_different = Stability(range_up, range_down_T, matching_algorithm=algo).compute()
            assert 0 < stability_different < 1.0

            # Stability of orthogonal matrices should be really close to 0
            eye = torch.eye(50)
            stability_orthogonal = Stability(eye[:25], eye[25:], matching_algorithm=algo).compute()
            assert stability_orthogonal < eps

        except Exception as e:
            raise AssertionError(f"Error with {algo}") from e


def test_dictionary_metrics_with_dict_and_ce(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test the dictionary metric give similar results via dictionaries and concept explainers
    """
    split = "bert.encoder.layer.1.output"
    splitted_encoder_ml.split_point = split

    rand1 = torch.rand(32, 32)
    concept_explainer1 = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml)
    concept_explainer1.get_dictionary = lambda: rand1

    rand2 = torch.rand(32, 32)
    concept_explainer2 = NeuronsAsConcepts(model_with_split_points=splitted_encoder_ml)
    concept_explainer2.get_dictionary = lambda: rand2

    for metric in [Stability]:
        try:
            # Creating metrics with either matrices or explainers
            score_dd = metric(rand1, rand2).compute()
            score_cece = metric(concept_explainer1, concept_explainer2).compute()
            score_ced = metric(concept_explainer1, rand2).compute()
            score_dce = metric(rand1, concept_explainer2).compute()
            score_all = metric(concept_explainer1, concept_explainer2, rand1, rand2).compute()
            # Calculating the distance between two matrices must be equivalent to calculating the distance between two explainers created from these matrices.
            # Similarly when matrices and explainers are mixed.
            assert score_cece == score_dd
            assert score_ced == score_dd
            assert score_dce == score_dd
            assert score_dd < score_all < 1  # some matrices are the same in the list
        except Exception as e:
            raise AssertionError(f"Error with {metric}") from e


def test_stability_error_raising():
    """
    Test the stability metric error raising
    """
    # no dictionary provided
    with pytest.raises(ValueError):
        Stability()

    # wrong type
    with pytest.raises(ValueError):
        Stability(1, 1)  # type: ignore

    # not a 2D tensor
    with pytest.raises(ValueError):
        Stability(torch.rand(32, 32, 1))
    with pytest.raises(ValueError):
        Stability(torch.rand(32, 32), torch.rand(32, 32, 1))

    # not matching shapes
    with pytest.raises(ValueError):
        Stability(torch.rand(32, 32), torch.rand(32, 5))
