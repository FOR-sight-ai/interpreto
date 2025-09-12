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
Test that all aggregator classes return correct shapes for both (p, t, l) and (p, l) inputs.
"""

import pytest
import torch

from interpreto.attributions.aggregations import (
    LinearRegressionAggregator,
    MaskwiseMeanAggregator,
    MeanAggregator,
    OcclusionAggregator,
    SobolAggregator,
    SquaredMeanAggregator,
    SumAggregator,
    TrapezoidalMeanAggregator,
    VarianceAggregator,
)

AGGREGATOR_CLASSES = [
    MeanAggregator,
    SquaredMeanAggregator,
    SumAggregator,
    VarianceAggregator,
    TrapezoidalMeanAggregator,
]


AGGREGATOR_CLASSES_WITH_MASK = [
    MaskwiseMeanAggregator,
    OcclusionAggregator,
    LinearRegressionAggregator,
]


@pytest.mark.parametrize("aggregator_cls", AGGREGATOR_CLASSES)
def test_aggregator_shapes(aggregator_cls):
    """
    Test that aggregators without a mask return the correct shape
    when given input of shape (p, t, l).
    """
    p, t, l = 4, 3, 5
    x = torch.randn(p, t, l)
    expected_shape = (t, l)

    agg = aggregator_cls()
    result = agg(x, None)

    assert result.shape == expected_shape, (
        f"{aggregator_cls.__name__} with shape {x.shape} returned {result.shape}, expected {expected_shape}"
    )


@pytest.mark.parametrize("aggregator_cls_with_mask", AGGREGATOR_CLASSES_WITH_MASK)
def test_aggregator_shapes_with_mask(aggregator_cls_with_mask):
    """
    Test that aggregators with a mask return the correct shape
    when given input of shape (p, t) and a mask of shape (p, l).
    """
    p, t, l = 4, 3, 5
    x = torch.randn(p, t)  # shape (p, t)
    expected_shape = (t, l)

    mask = torch.randint(0, 2, (p, l)).float()  # shape (p, l)
    agg = aggregator_cls_with_mask()
    result = agg(x, mask)

    print("Testing", aggregator_cls_with_mask.__name__, "with input shape:", x.shape, " Result shape:", result.shape)
    assert result.shape == expected_shape, (
        f"{aggregator_cls_with_mask.__name__} with input shape {x.shape} and mask shape {mask.shape} returned {result.shape}, expected {expected_shape}"
    )


def test_aggregator_shapes_sobol():
    """
    Test SobolAggregator with specific shapes.
    """
    p, t, l = 14, 3, 5  # p = (l + 2) * k, where k = 2
    x = torch.randn(p, t)
    mask = torch.randint(0, 2, (p, l)).float()  # shape (p, l)

    expected_shape = (t, l)

    agg = SobolAggregator(n_token_perturbations=2)
    result = agg(x, mask)

    assert result.shape == expected_shape, (
        f"SobolAggregator with input shape {x.shape} and mask shape {mask.shape} returned {result.shape}, expected {expected_shape}"
    )
