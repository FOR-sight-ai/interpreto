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
from torch.nn.functional import interpolate

from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity

# --------
# Fixtures


@pytest.fixture(scope="module")
def null_matrix():
    """Creates a 10*10 null tensor"""
    return torch.zeros(1, 10, 10)


@pytest.fixture(scope="module")
def simple_matrix():
    """A simple matrix"""
    return torch.tensor([[1, 0], [0, 1]])


@pytest.fixture(scope="module")
def matrix():
    """slightly more difficult matrix"""
    return torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]]])


@pytest.fixture(scope="module")
def null_output_size():
    return (0, 0)


@pytest.fixture(scope="module")
def normal_output_size():
    return (5, 7)


@pytest.fixture(scope="module")
def huge_output_size():
    return (50, 13)


@pytest.fixture(scope="module")
def normal_patch_size():
    return 16


@pytest.fixture(scope="module")
def small_patch_size():
    return 8


@pytest.fixture(scope="module")
def wrong_patch_size():
    return 17


@pytest.mark.parametrize(
    "input",
    [
        torch.zeros(1, 10, 10),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 11.0], [7.0, 8.0, 9.0, 12.0], [7.0, 8.0, 9.0, 12.0]]]),
    ],
)
@pytest.mark.parametrize("output", [(5, 7), (50, 13), None])
@pytest.mark.parametrize(
    "strategy",
    [
        GranularityResizeStrategy.NEAREST,
        GranularityResizeStrategy.BILINEAR,
        GranularityResizeStrategy.BICUBIC,
        GranularityResizeStrategy.AREA,
    ],
)
@pytest.mark.parametrize("patch_size", [1, 2])
def test_resize_strategy_output_size(input, output, strategy, patch_size):
    if output is not None:
        assert strategy.resize(input, output, patch_size).shape == (1, *output)
    else:
        c, h, w = input.shape
        h_out = h // patch_size
        w_out = w // patch_size
        assert strategy.resize(input, output, patch_size).shape == (c, h_out, w_out)


def test_resize_fail(null_matrix, wrong_patch_size):
    strategy = GranularityResizeStrategy.NEAREST
    with pytest.raises(AssertionError):
        strategy.resize(input=null_matrix, output_size=None, patch_size=wrong_patch_size)


def test_resize_nearest(matrix):
    strategy = GranularityResizeStrategy.NEAREST
    bool_tensor = strategy.resize(matrix, output_size=(2, 2)) == torch.tensor([[1, 0], [0, 3]])
    assert bool_tensor[0, 0, 0] and bool_tensor[0, 0, 1] and bool_tensor[0, 1, 0] and bool_tensor[0, 1, 1]


def test_resize_bilinear(matrix):
    strategy = GranularityResizeStrategy.BILINEAR
    bool_tensor = strategy.resize(matrix, output_size=(2, 2)) - torch.tensor([[30 / 49, 15 / 49], [15 / 49, 65 / 49]])
    bool_tensor = bool_tensor.abs() <= 10 ** (-5)
    assert bool_tensor[0, 0, 0] and bool_tensor[0, 0, 1] and bool_tensor[0, 1, 0] and bool_tensor[0, 1, 1]


def test_resize_bicubic(matrix):
    strategy = GranularityResizeStrategy.BICUBIC
    expected = interpolate(
        matrix.unsqueeze(0), size=(2, 2), mode="bicubic", align_corners=False, antialias=True
    ).squeeze(0)
    bool_tensor = strategy.resize(matrix, output_size=(2, 2)) - expected
    bool_tensor = bool_tensor.abs() <= 10 ** (-5)
    assert bool_tensor[0, 0, 0] and bool_tensor[0, 0, 1] and bool_tensor[0, 1, 0] and bool_tensor[0, 1, 1]


def test_resize_area(matrix):
    strategy = GranularityResizeStrategy.AREA
    print(strategy.resize(matrix, output_size=(2, 2)))
    bool_tensor = strategy.resize(matrix, output_size=(2, 2)) - torch.tensor([[0.75, 0.0], [0.0, 1.75]])
    print(bool_tensor)
    bool_tensor = bool_tensor.abs() <= 10 ** (-5)
    print(bool_tensor)
    assert bool_tensor[0, 0, 0] and bool_tensor[0, 0, 1] and bool_tensor[0, 1, 0] and bool_tensor[0, 1, 1]


# NOTE: this used to be a test for the now useless granularity_resize function. We keep it for now in case it proves useful later
# @pytest.mark.parametrize(
#     "input",
#     [
#         torch.zeros(1, 10, 10),
#         torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
#         torch.tensor([[[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 11.0], [7.0, 8.0, 9.0, 12.0], [7.0, 8.0, 9.0, 12.0]]]),
#     ],
# )
# @pytest.mark.parametrize("contributions", [(5, 7), (50, 13), None])
# @pytest.mark.parametrize(
#     "strategy",
#     [
#         GranularityResizeStrategy.NEAREST,
#         GranularityResizeStrategy.BILINEAR,
#         GranularityResizeStrategy.BICUBIC,
#         GranularityResizeStrategy.AREA,
#     ],
# )
# @pytest.mark.parametrize("patch_size", [1, 2])
# @pytest.mark.parametrize("granularity", [ImageGranularity.PIXEL, ImageGranularity.PATCH])
# def test_granularity_resize(input, output, strategy, patch_size, granularity):
#     if output != None:
#         assert strategy.resize(input, output, patch_size).shape == (1, *output)
#     else:
#         c, h, w = input.shape
#         h_out = h // patch_size
#         w_out = w // patch_size
#         assert strategy.resize(input, output, patch_size).shape == (c, h_out, w_out)


PATCH_SIZE = 16


@pytest.mark.parametrize("granularity", [ImageGranularity.PIXEL, ImageGranularity.PATCH])
@pytest.mark.parametrize("h_in, w_in", [(224, 224), (240, 208)])
@pytest.mark.parametrize("t", [1, 3])
@pytest.mark.parametrize(
    "strategy",
    [
        GranularityResizeStrategy.NEAREST,
        GranularityResizeStrategy.BILINEAR,
        GranularityResizeStrategy.BICUBIC,
        GranularityResizeStrategy.AREA,
    ],
)
def test_resize_to_image_output_size(granularity, h_in, w_in, t, strategy):
    """
    `resize_to_image` must always land on the pixel grid `(t, h_in, w_in)`, whatever the
    granularity it starts from. Only the shape is checked — the numerical correctness of the
    interpolation itself belongs to `test_resize_*` above.

    The two granularities reach that shape by different routes: PIXEL starts at `g = h*w` and
    is a pure reshape, PATCH starts at the much smaller `g = (h//16)*(w//16)` and genuinely
    interpolates. 240x208 is there to catch anything that silently assumes a square image.
    """
    if granularity is ImageGranularity.PIXEL:
        g = h_in * w_in
    else:
        g = (h_in // PATCH_SIZE) * (w_in // PATCH_SIZE)

    contribution = torch.rand(t, g)
    inputs = {"pixel_values": torch.zeros(1, 3, h_in, w_in)}

    resized = granularity.resize_to_image(
        contribution=contribution,
        resize_strategy=strategy,
        inputs=inputs,
        patch_size=PATCH_SIZE,
    )

    assert resized.shape == (t, h_in, w_in)
