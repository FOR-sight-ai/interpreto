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

"""Tests for the decorators catching the arguments renamed by the vision update."""

from __future__ import annotations

import pytest

from interpreto import (
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
)
from interpreto.commons.bad_arguments_decorator import (
    AggregationStrategyError,
    PerturbationsError,
    TokenizerError,
)

ALL_METHODS = [
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
]


@pytest.mark.parametrize("explainer_class", ALL_METHODS)
def test_new_arguments_pass_through(explainer_class, bert_model, bert_tokenizer):
    """The decorators are transparent when the new argument names are used."""
    explainer = explainer_class(bert_model, processor=bert_tokenizer, batch_size=2)
    assert explainer.tokenizer is bert_tokenizer, (
        f"Expected the explainer's tokenizer to be {bert_tokenizer}, got {explainer.tokenizer}"
    )
    assert explainer.inference_wrapper.batch_size == 2, (
        f"Expected batch_size to be 2, got {explainer.inference_wrapper.batch_size}"
    )


@pytest.mark.parametrize("explainer_class", ALL_METHODS)
@pytest.mark.parametrize(
    "argument,error",
    [
        ("tokenizer", TokenizerError),
        ("granularity_aggregation_strategy", AggregationStrategyError),
    ],
)
def test_old_general_arguments_raise(explainer_class, argument, error, bert_model, bert_tokenizer):
    """Every method catches the two generally renamed arguments, each with its own error."""
    with pytest.raises(error) as raised:
        explainer_class(bert_model, **{argument: bert_tokenizer})
    assert argument in str(raised.value), (
        "The error message should indicate which argument caused the raise and explain how it should be changed"
    )
    assert "tokenizer -> processor" in str(raised.value), (
        "The error message should indicate how to change the tokenizer argument"
    )
    assert "granularity_aggregation_strategy -> combination_strategy" in str(raised.value), (
        "The error message should indicate how to change the granularity_aggregation_stategy argument"
    )


def test_old_sobol_argument_raises(bert_model, bert_tokenizer):
    """`n_token_perturbations` is only renamed for Sobol, the only method taking it."""
    with pytest.raises(PerturbationsError) as raised:
        Sobol(bert_model, processor=bert_tokenizer, n_token_perturbations=8)
    assert "n_token_perturbations -> n_input_perturbations" in str(raised.value), (
        "The error message should indicate how to change the n_token_perturbations argument"
    )
