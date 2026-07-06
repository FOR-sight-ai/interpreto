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
Test file to test the visualization functions
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from interpreto.visualizations.image_attributions import _clip_percentile, _color_limits, _prepare_heatmap, _to_displayable


def test_clip_percentile_constant_is_unchanged():
    # A constant array is a fixed point: lo == hi == 10, clip is a no-op.
    arr = np.full(20, 10.0)
    out = _clip_percentile(arr, percentile=np.random.uniform(0, 50))
    assert np.array_equal(out, arr) 


def test_clip_percentile_pulls_tails_to_bounds():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = _clip_percentile(arr, percentile=0.2)
    expected = np.array([1.008, 2.0, 3.0, 4.0, 4.992])
    assert np.allclose(out, expected)

def _random_tensor():
    # channel-first (C, H, W), uniform [0, 1000) -> exercises the transpose branch
    return torch.rand(3, 8, 8) * 1000.0

def _random_array():
    return np.random.uniform(0, 1000, size=(3, 8, 8)).astype(np.float32)

def _random_pil():
    # PIL is uint8 [0, 255], already channel-last (H, W, 3) -> no transpose
    data = np.random.uniform(0, 255, size=(8, 8, 3)).astype(np.uint8)
    return Image.fromarray(data)


@pytest.mark.parametrize("factory", [_random_tensor, _random_array, _random_pil])
def test_to_displayable_range_and_channels(factory):
    out = _to_displayable(factory())
    assert issubclass(out.dtype.type,np.floating )
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.ndim == 3 and out.shape[-1] in (1, 3)


def test_prepare_heatmap():
    # _prepare_heatmap only reads `.attributions_image`, so a SimpleNamespace stand-in is enough.
    # (1, 2, 2): t=1 target, 2x2 spatial. Mixed signs + a value >1 on purpose.
    output = SimpleNamespace(attributions_image=torch.tensor([[[10.0, 20.0], [-3.0, 0.0]]]))

    # absolute_value=True -> no negative values
    abs_out = _prepare_heatmap(output, target_idx=0, clip_percentile=None, absolute_value=True)
    assert (abs_out >= 0).all()

    # absolute_value=False -> floating dtype, and scores kept in real scale (NOT rescaled to [0,1])
    raw_out = _prepare_heatmap(output, target_idx=0, clip_percentile=None, absolute_value=False)
    assert issubclass(raw_out.dtype.type, np.floating)
    assert raw_out.max() > 1.0 or raw_out.min() < 0.0


@pytest.mark.parametrize(
    "values, expected",
    [
        ([-3.0, 1.0, 20.0], (-20.0, 20.0, True)),    # straddles zero -> centered at +/- m, m=max(|lo|,|hi|)=20
        ([1.0, 2.0, 20.0], (1.0, 20.0, False)),      # all positive -> plain (min, max), not centered
        ([-5.0, -2.0, -1.0], (-5.0, -1.0, False)),   # all negative -> plain (min, max), not centered
    ],
)
def test_color_limits(values, expected):
    out = _color_limits(np.array(values, dtype=np.float32))
    assert out == expected