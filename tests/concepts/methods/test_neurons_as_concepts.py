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
Tests for `NeuronsAsConcepts` concept explainer
"""

from __future__ import annotations

import pytest
import torch

from interpreto.concepts import NeuronsAsConcepts
from interpreto.concepts.splitters.model_with_split_points import ModelWithSplitPoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_neurons_as_concepts(splitted_encoder_ml: ModelWithSplitPoints, activations: torch.Tensor):
    """
    Test that the concept encoding and decoding of the `NeuronsAsConcepts` is the identity
    """
    d = activations.shape[1]

    concept_explainer = NeuronsAsConcepts(splitter=splitted_encoder_ml)

    assert concept_explainer.is_fitted is True  # splitted_model has a single split so it is fitted
    assert hasattr(concept_explainer, "has_differentiable_concept_encoder")
    assert hasattr(concept_explainer, "has_differentiable_concept_decoder")
    assert concept_explainer.concept_model.nb_concepts == d

    concepts = concept_explainer.activations_to_concepts(activations)
    reconstructed_activations = concept_explainer.concepts_to_activations(concepts)

    assert torch.allclose(concepts, activations)  # type: ignore
    assert torch.allclose(reconstructed_activations, activations)  # type: ignore

    with pytest.raises(NotImplementedError):
        concept_explainer.fit(activations)
