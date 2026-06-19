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

import re
import sys

import pytest
import torch


# local import – the module is supplied alongside this test file
from interpreto import ImageGranularity
from interpreto.commons.granularity import GranularityResizeStrategy

# --------
# Fixtures


@pytest.fixture(scope="module")
def null_matrix():
    """Creates a 10*10 null tensor"""
    return torch.zeros(10,10)


@pytest.fixture(scope="module")
def simple_matrix():
    """A simple matrix"""
    return torch.tensor([[1,0],[0,1]])


@pytest.fixture(scope="module")
def matrix():
    """slightly more difficult matrix"""
    return torch.tensor([[1,2,3],[4,5,6],[7,8,9]])


@pytest.fixture(scope="module")
def null_output_size():
    return (0,0)

@pytest.fixture(scope="module")
def normal_output_size():
    return (5,7)

@pytest.fixture(scope="module")
def huge_output_size():
    return (50,13)

@pytest.fixture(scope="module")
def normal_patch_size():
    return 16

@pytest.fixture(scope="module")
def small_patch_size():
    return 8

@pytest.fixture(scope="module")
def wrong_patch_size():
    return 17


@pytest.mark.parametrize("input", [torch.zeros(1,10,10), torch.tensor([[[1.,0.],[0.,1.]]]), torch.tensor([[[1.,2.,3.,10.],[4.,5.,6.,11.],[7.,8.,9.,12.],[7.,8.,9.,12.]]])])
@pytest.mark.parametrize("output", [(5,7), (50, 13), None])
@pytest.mark.parametrize("strategy", [GranularityResizeStrategy.NEAREST, GranularityResizeStrategy.BILINEAR, GranularityResizeStrategy.BICUBIC, GranularityResizeStrategy.AREA])
@pytest.mark.parametrize("patch_size", [1,2])
def test_verify_output_size(input, output, strategy, patch_size):
    if output != None: 
        assert strategy.resize(input, output, patch_size).shape == (1,*output) 
    else:
        c, h, w = input.shape 
        h_out = h // patch_size
        w_out = w // patch_size 
        assert strategy.resize(input, output, patch_size).shape == (c ,h_out, w_out)
