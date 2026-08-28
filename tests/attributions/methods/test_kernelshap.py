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

import pytest
import torch

from interpreto.attributions import KernelShap
from interpreto.attributions.aggregations.linear_regression_aggregation import (
    Kernels,
    LinearRegressionAggregator,
)
from interpreto.attributions.base import AttributionOutput
from interpreto.commons.granularity import TextGranularity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "granularity, n_perturbations",
    [
        (TextGranularity.TOKEN, 5),
        (TextGranularity.WORD, 100),
    ],
)
def test_kernel_shap_init_and_mask(bert_model, bert_tokenizer, granularity, n_perturbations):
    torch.manual_seed(0)
    batch_size = 2

    # 2) init explainer
    explainer = KernelShap(
        model=bert_model,
        processor=bert_tokenizer,
        batch_size=batch_size,
        granularity=granularity,
        n_perturbations=n_perturbations,
        device=DEVICE,
    )

    # 4) perturbator is ShapTokenPerturbator with correct params
    assert isinstance(explainer.perturbator, ShapTokenPerturbator)
    pert = explainer.perturbator
    assert pert.n_perturbations == n_perturbations
    assert pert.granularity == granularity
    # replace_token_id matches tokenizer
    rid = bert_tokenizer.mask_token_id
    if rid is None:
        assert "[REPLACE]" in bert_tokenizer.get_vocab()
        rid = bert_tokenizer.convert_tokens_to_ids("[REPLACE]")
    expected_id = rid if isinstance(rid, int) else rid[0]
    assert pert.replace_token_id == expected_id

    # 5) aggregator is LinearRegressionAggregator with ONES kernel
    assert isinstance(explainer.aggregator, LinearRegressionAggregator)
    assert explainer.aggregator.similarity_kernel == Kernels.ONES

    # 6) get_mask returns float32 tensor of shape (n_perturbations, seq_len)
    seq_len = 8
    mask = pert.get_mask(seq_len)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (n_perturbations, seq_len)
    assert mask.dtype == torch.float32

    # 7) explain(...) returns a list of AttributionOutput, one per input
    inputs = ["Interpreto is magic", "I hope the test passes"]
    attributions = explainer.explain(inputs)

    assert isinstance(attributions, list)
    assert len(attributions) == len(inputs)
    for out in attributions:
        assert isinstance(out, AttributionOutput)
        assert isinstance(out.attributions, torch.Tensor)
        assert isinstance(out.elements, list)
        assert all(isinstance(el, str) for el in out.elements)
