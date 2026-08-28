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

from interpreto.attributions import Sobol
from interpreto.attributions.aggregations.sobol_aggregation import SobolAggregator, SobolIndicesOrders
from interpreto.attributions.base import AttributionOutput
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers, SobolPerturbator
from interpreto.commons.granularity import TextGranularity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "granularity, order, sampler, n_token_perturbations",
    [
        (TextGranularity.TOKEN, SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.SOBOL, 2),
        (TextGranularity.TOKEN, SobolIndicesOrders.TOTAL_ORDER, SequenceSamplers.HALTON, 5),
        (TextGranularity.TOKEN, SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.LatinHypercube, 50),
        (TextGranularity.WORD, SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.SOBOL, 50),
        (TextGranularity.WORD, SobolIndicesOrders.TOTAL_ORDER, SequenceSamplers.HALTON, 5),
        (TextGranularity.WORD, SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.LatinHypercube, 10),
    ],
)
def test_sobol_attribution_init_and_mask(
    bert_model, bert_tokenizer, granularity, order, sampler, n_token_perturbations
):
    batch_size = 2

    explainer = Sobol(
        model=bert_model,
        processor=bert_tokenizer,
        batch_size=batch_size,
        device=DEVICE,
        granularity=granularity,
        n_token_perturbations=n_token_perturbations,
        sobol_indices_order=order,
        sampler=sampler,
    )

    # 2) check the perturbator stored our enums correctly
    assert isinstance(explainer.perturbator, SobolPerturbator)
    assert isinstance(explainer.aggregator, SobolAggregator)
    assert explainer.perturbator.sampler_class == sampler.value
    assert explainer.aggregator.sobol_indices_order == order.value
    expected_replace_id = bert_tokenizer.mask_token_id
    if expected_replace_id is None:
        assert "[REPLACE]" in bert_tokenizer.get_vocab()
        expected_replace_id = bert_tokenizer.convert_tokens_to_ids("[REPLACE]")
    assert explainer.perturbator.replace_token_id == expected_replace_id

    # 3) get_mask returns a float32 tensor of shape (n_perturbations, seq_len)   # TODO: put this in a common perturbator test
    seq_len = 8
    mask = explainer.perturbator.get_mask(seq_len)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == ((seq_len + 2) * n_token_perturbations, seq_len)
    assert mask.dtype == torch.float32

    # 4) check explanations TODO add this to the main test
    input_text = ["Interpreto is magic", "I hope the test passes"]
    attributions = explainer.explain(input_text)

    assert isinstance(attributions, list)
    assert len(attributions) == len(input_text)
    for attribution in attributions:
        assert isinstance(attribution, AttributionOutput)
        assert isinstance(attribution.attributions, torch.Tensor)
        assert isinstance(attribution.elements, list)
        assert all(isinstance(element, str) for element in attribution.elements)
