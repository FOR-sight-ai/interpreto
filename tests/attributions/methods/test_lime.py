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

from interpreto.attributions import Lime
from interpreto.attributions.aggregations.linear_regression_aggregation import (
    DistancesFromMask,
    Kernels,
    LinearRegressionAggregator,
    default_kernel_width_fn,
)
from interpreto.attributions.base import AttributionOutput
from interpreto.commons.granularity import TextGranularity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "granularity, n_perturbations, perturb_probability, distance_function, kernel_width",
    [
        (TextGranularity.TOKEN, 10, 0.5, DistancesFromMask.HAMMING, None),
        (TextGranularity.TOKEN, 20, 0.8, DistancesFromMask.EUCLIDEAN, 5),
        (TextGranularity.TOKEN, 10, 0.5, DistancesFromMask.COSINE, 0.5),
        (TextGranularity.WORD, 15, 0.3, DistancesFromMask.HAMMING, None),
        (TextGranularity.WORD, 25, 0.7, DistancesFromMask.EUCLIDEAN, None),
        (TextGranularity.WORD, 15, 0.3, DistancesFromMask.COSINE, 0.5),
    ],
)
def test_lime_attribution_init_and_mask(
    bert_model, bert_tokenizer, granularity, n_perturbations, perturb_probability, distance_function, kernel_width
):
    torch.manual_seed(0)
    batch_size = 2

    explainer = Lime(
        model=bert_model,
        processor=bert_tokenizer,
        batch_size=batch_size,
        granularity=granularity,
        n_perturbations=n_perturbations,
        perturb_probability=perturb_probability,
        distance_function=distance_function,
        kernel_width=kernel_width,
        device=DEVICE,
    )

    # 2) check the perturbator stored our params correctly
    assert isinstance(explainer.perturbator, RandomMaskedTokenPerturbator)
    perturbator = explainer.perturbator
    assert perturbator.n_perturbations == n_perturbations
    assert pytest.approx(perturbator.perturb_probability, rel=1e-6) == perturb_probability
    assert perturbator.granularity == granularity
    replace_id = bert_tokenizer.mask_token_id
    if replace_id is None:
        assert "[REPLACE]" in bert_tokenizer.get_vocab()
        replace_id = bert_tokenizer.convert_tokens_to_ids("[REPLACE]")
    assert perturbator.replace_token_id == (replace_id if isinstance(replace_id, int) else replace_id[0])

    # 3) check the aggregator
    assert isinstance(explainer.aggregator, LinearRegressionAggregator)
    aggregator = explainer.aggregator
    assert aggregator.distance_function == distance_function
    assert aggregator.similarity_kernel == Kernels.EXPONENTIAL
    if kernel_width is None:
        assert aggregator.kernel_width == default_kernel_width_fn
    else:
        assert aggregator.kernel_width == kernel_width

    # 4) get_mask returns a float32 tensor of shape (n_perturbations, seq_len)  # TODO: put this in a common perturbator test
    seq_len = 8
    mask = perturbator.get_mask(seq_len)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (n_perturbations, seq_len)
    assert mask.dtype == torch.float32

    # 5) check explanations TODO add this to the main test
    input_text = ["Interpreto is magic", "I hope the test passes"]
    attributions = explainer.explain(input_text)

    assert isinstance(attributions, list)
    assert len(attributions) == len(input_text)
    for attribution in attributions:
        assert isinstance(attribution, AttributionOutput)
        assert isinstance(attribution.attributions, torch.Tensor)
        assert isinstance(attribution.elements, list)
        assert all(isinstance(el, str) for el in attribution.elements)
