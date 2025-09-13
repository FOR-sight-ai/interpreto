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
"""Tests for interpreto.concepts.methods.sklearn_wrappers."""

from __future__ import annotations

import pytest
import torch

from interpreto.concepts.methods.sklearn_wrappers import (
    ICAWrapper,
    KMeansWrapper,
    PCAWrapper,
    SVDWrapper,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "wrapper_cls",
    [ICAWrapper, PCAWrapper, KMeansWrapper, SVDWrapper],
)
def test_sklearn_wrapper_encode_decode(wrapper_cls):
    """Ensure wrappers behave like their sklearn counterparts."""
    torch.manual_seed(0)
    n, d, c = 12, 6, 3

    data = torch.randn(n, d)

    sk_wrapper = wrapper_cls(nb_concepts=c, input_size=d, random_state=0, device="cpu")
    sk_model = sk_wrapper.fit(data, return_sklearn_model=True)

    sk_wrapper.to(DEVICE)

    # Compare encodings
    encoded_wrapper = sk_wrapper.encode(data)
    assert encoded_wrapper is not None, f"{wrapper_cls.__name__}.encode returned None"
    encoded_sklearn = torch.as_tensor(sk_model.transform(data.cpu().numpy()), dtype=torch.float32)

    assert isinstance(encoded_wrapper, torch.Tensor), (
        f"{wrapper_cls.__name__}.encode returned type {type(encoded_wrapper)} instead of torch.Tensor"
    )
    assert encoded_wrapper.shape == encoded_sklearn.shape == torch.Size([n, c]), (
        f"Encoded shape mismatch for {wrapper_cls.__name__}:\n"
        f"wrapper={tuple(encoded_wrapper.shape)}, sklearn={tuple(encoded_sklearn.shape)}, expected={(n, c)}"
    )
    assert torch.allclose(encoded_wrapper.cpu(), encoded_sklearn, atol=1e-5, rtol=1e-5), (
        f"Encoded values differ from sklearn baseline for {wrapper_cls.__name__} beyond tolerance"
    )

    # Compare reconstructions
    if not isinstance(sk_wrapper, KMeansWrapper):
        decoded_wrapper = sk_wrapper.decode(encoded_wrapper)
        assert decoded_wrapper is not None, f"{wrapper_cls.__name__}.decode returned None"
        decoded_sklearn = torch.as_tensor(sk_model.inverse_transform(encoded_sklearn.numpy()), dtype=torch.float32)

        assert isinstance(decoded_wrapper, torch.Tensor), (
            f"{wrapper_cls.__name__}.decode returned type {type(decoded_wrapper)} instead of torch.Tensor"
        )
        assert decoded_wrapper.shape == decoded_sklearn.shape == torch.Size([n, d]), (
            f"Decoded shape mismatch for {wrapper_cls.__name__}:\n"
            f"wrapper={tuple(decoded_wrapper.shape)}, sklearn={tuple(decoded_sklearn.shape)}, expected={(n, d)}"
        )
        assert torch.allclose(decoded_wrapper.cpu(), decoded_sklearn, atol=1e-5, rtol=1e-5), (
            f"Decoded values differ for {wrapper_cls.__name__} beyond tolerance"
        )


if __name__ == "__main__":
    test_sklearn_wrapper_encode_decode(ICAWrapper)
    test_sklearn_wrapper_encode_decode(PCAWrapper)
    test_sklearn_wrapper_encode_decode(KMeansWrapper)
    test_sklearn_wrapper_encode_decode(SVDWrapper)
