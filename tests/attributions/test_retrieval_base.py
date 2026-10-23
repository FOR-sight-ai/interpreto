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

"""Raw-image flattening for retrieval explainers, without loading a model."""

import numpy as np
import pytest
import torch
from PIL import Image

from interpreto.attributions.base import RetrievalAttributionExplainer


def _images():
    pil = Image.new("RGB", (4, 6))
    array = np.zeros((6, 4, 3), dtype=np.uint8)
    tensor = torch.zeros(3, 6, 4)
    return pil, array, tensor


def test_flatten_raw_images_single_and_nested():
    """One raw image per sample, in traversal order, mirroring process_model_inputs."""
    pil, array, tensor = _images()
    assert RetrievalAttributionExplainer._flatten_raw_images(pil) == [pil]
    nested = [pil, (array, [tensor])]
    flat = RetrievalAttributionExplainer._flatten_raw_images(nested)
    assert flat == [pil, array, tensor]
    assert len(flat) == 3


def test_flatten_raw_images_generator():
    """One-shot iterables flatten fully, so explain can traverse inputs twice."""

    def gen():
        pil, array, _ = _images()
        yield pil
        yield array

    flat = RetrievalAttributionExplainer._flatten_raw_images(gen())
    assert len(flat) == 2
    assert isinstance(flat[0], Image.Image)


def test_flatten_raw_images_rejects_unsupported():
    with pytest.raises(ValueError, match="not supported for method process_model_inputs"):
        RetrievalAttributionExplainer._flatten_raw_images(42)
