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

"""Logit Lens implementation."""

from ._lens_base import BaseLens


class LogitLens(BaseLens):
    """Project every residual-stream state through the model prediction head.

    The residual states are collected in one model trace and projected together
    through :class:`~interpreto.concepts.splitters.AllLayersSplitter`.

    Args:
        splitter (AllLayersSplitter): Model wrapper used to collect and project all layer states.
        top_k (int): Maximum number of token or class scores returned per prediction.

    Examples:
        >>> from interpreto import AllLayersSplitter, LogitLens
        >>> splitter = AllLayersSplitter("hf-internal-testing/tiny-random-gpt2")
        >>> lens = LogitLens(splitter, top_k=3)
        >>> results = lens("Interpreto is useful.")
        >>> list(results) == splitter.activation_names
        True
    """
