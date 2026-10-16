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

from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity


@pytest.mark.parametrize(
    "input, output, strategy",
    [
        (torch.zeros(1, 10, 10), (5, 7), GranularityResizeStrategy.NEAREST),
        (torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), (50, 13), GranularityResizeStrategy.BILINEAR),
        (
            torch.tensor(
                [[[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 11.0], [7.0, 8.0, 9.0, 12.0], [7.0, 8.0, 9.0, 12.0]]]
            ),
            None,
            GranularityResizeStrategy.BICUBIC,
        ),
        (torch.zeros(1, 10, 10), (50, 13), GranularityResizeStrategy.AREA),
    ],
)
def test_resize_strategy_output_size(input, output, strategy):
    if output is not None:
        assert strategy.resize(input, output).shape == (1, *output), (
            "resize() with an explicit output_size must return shape (1, *output). The leading "
            "1 comes from this test's single-channel input fixtures, not output_size"
        )


@pytest.fixture(scope="module")
def matrix():
    """matrix with easily computable resize values."""
    return torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]]])


@pytest.mark.parametrize(
    "strategy, expected",
    [
        (GranularityResizeStrategy.NEAREST, [[1, 0], [0, 3]]),
        (GranularityResizeStrategy.BILINEAR, [[30 / 49, 15 / 49], [15 / 49, 65 / 49]]),
        (GranularityResizeStrategy.BICUBIC, [[0.981445, -0.278320], [-0.278320, 2.075195]]),  # from chatgpt
        (GranularityResizeStrategy.AREA, [[0.75, 0.0], [0.0, 1.75]]),
    ],
)
def test_resize(strategy, expected, matrix):
    resized = strategy.resize(matrix, (2, 2))
    assert torch.allclose(resized, expected, atol=1e-5), (
        f"{strategy} resize of the 4x4 diagonal matrix to 2x2 must match the analytically expected values within 1e-5"
    )


PATCH_SIZE = 16


@pytest.mark.parametrize(
    "granularity, h_in, w_in, t, strategy",
    [
        (ImageGranularity.PIXEL, 240, 208, 3, GranularityResizeStrategy.NEAREST),
        (ImageGranularity.PATCH, 224, 224, 1, GranularityResizeStrategy.NEAREST),
        (ImageGranularity.PATCH, 240, 208, 3, GranularityResizeStrategy.BILINEAR),
        (ImageGranularity.PATCH, 224, 224, 3, GranularityResizeStrategy.BICUBIC),
        (ImageGranularity.PATCH, 240, 208, 1, GranularityResizeStrategy.AREA),
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

    assert resized.shape == (t, h_in, w_in), (
        "resize_to_image must land on the full pixel grid (t, h_in, w_in) regardless of "
        "starting granularity (PIXEL vs PATCH) or resize strategy"
    )
