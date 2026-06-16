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

from interpreto.concepts import NeuronsAsConcepts
from interpreto.concepts.metrics import Sparsity, SparsityRatio
from interpreto.concepts.splitters.model_with_split_points import ModelWithSplitPoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_sparsity(splitted_encoder_ml: ModelWithSplitPoints):
    """
    Test that the sparsity metric measures expected sparsity
    """
    torch.manual_seed(0)
    eps = 1e-5
    n = 50
    d = 32
    sparsity_ratio = 0.1  # should make an integer through 1 / sparsity_ratio
    split = "bert.encoder.layer.1.output"
    splitted_encoder_ml.split_point = split

    concept_explainer = NeuronsAsConcepts(splitter=splitted_encoder_ml)
    activations = torch.arange(n * d, device=DEVICE).reshape(n, d)
    sparse_activations = activations * (activations % (1 / sparsity_ratio) == 0).float()

    # Ensure constructed activations are sparse
    assert torch.norm(sparse_activations, p=0, dim=1).mean() - sparsity_ratio * d < eps

    # Compare computed sparsity with specified/constructed sparsity, it should be close
    sparsity_metric = Sparsity(concept_explainer)
    metric = sparsity_metric.compute(sparse_activations)
    assert metric - sparsity_ratio * d < eps

    # Compare computed sparsity with specified/constructed sparsity, it should be close
    sparsity_ratio_metric = SparsityRatio(concept_explainer)
    metric = sparsity_ratio_metric.compute(sparse_activations)
    assert metric - sparsity_ratio < eps
