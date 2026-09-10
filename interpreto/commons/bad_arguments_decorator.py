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
Decorators catching arguments renamed by the vision update, and guiding the user to the new names.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

RENAMING_MESSAGE = (
    "Interpreto has been updated and now also support vision models ! "
    "We have renamed some arguments notably : tokenizer -> processor, "
    "granularity_aggregation_strategy -> combination_strategy and "
    "n_token_perturbations -> n_input_perturbations. "
    "An error was raised because you used {argument}"
)


class TokenizerError(TypeError):
    """Raised when `tokenizer` is used instead of `processor`."""

    def __init__(self):
        super().__init__(RENAMING_MESSAGE.format(argument="tokenizer"))


class AggregationStrategyError(TypeError):
    """Raised when `granularity_aggregation_strategy` is used instead of `combination_strategy`."""

    def __init__(self):
        super().__init__(RENAMING_MESSAGE.format(argument="granularity_aggregation_strategy"))


class PerturbationsError(TypeError):
    """Raised when `n_token_perturbations` is used instead of `n_input_perturbations`."""

    def __init__(self):
        super().__init__(RENAMING_MESSAGE.format(argument="n_token_perturbations"))


def general_bad_argument(func: Callable) -> Callable:
    """
    Raise `TokenizerError` or `AggregationStrategyError` if the wrapped function is called with
    `tokenizer` or `granularity_aggregation_strategy` as a keyword argument.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "tokenizer" in kwargs:
            raise TokenizerError()
        if "granularity_aggregation_strategy" in kwargs:
            raise AggregationStrategyError()
        return func(*args, **kwargs)

    return wrapper


def sobol_bad_argument(func: Callable) -> Callable:
    """
    Raise `PerturbationsError` if the wrapped function is called with `n_token_perturbations` as a
    keyword argument.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "n_token_perturbations" in kwargs:
            raise PerturbationsError()
        return func(*args, **kwargs)

    return wrapper
