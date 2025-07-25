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
Integrated Gradients method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations import MeanAggregator
from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.attributions.perturbations import LinearInterpolationPerturbator
from interpreto.commons.granularity import Granularity, GranularityAggregationStrategy
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class IntegratedGradients(MultitaskExplainerMixin, AttributionExplainer):
    """
    Integrated Gradients (IG) is a gradient-based interpretability method that attributes
    importance scores to input features (e.g., tokens) by integrating the model’s gradients
    along a path from a baseline input to the actual input.

    The method is designed to address some of the limitations of standard gradients, such as
    saturation and noise, by averaging gradients over interpolated inputs rather than relying
    on a single local gradient.

    **Reference:**
    Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.*
    [Paper](http://proceedings.mlr.press/v70/sundararajan17a.html)

    Examples:
        >>> from interpreto import IntegratedGradients
        >>> method = IntegratedGradients(model=model, tokenizer=tokenizer,
        >>>                              batch_size=4, n_interpolations=50)
        >>> explanations = method.explain(model_inputs=text)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        granularity: Granularity = Granularity.WORD,
        granularity_aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        device: torch.device | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        n_interpolations: int = 10,
        baseline: torch.Tensor | float | None = None,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            batch_size (int): batch size for the attribution method
            granularity (Granularity, optional): The level of granularity for the explanation.
                Options are: `ALL_TOKENS`, `TOKEN`, `WORD`, or `SENTENCE`.
                Defaults to Granularity.WORD.
                To obtain it, `from interpreto import Granularity` then `Granularity.WORD`.
            granularity_aggregation_strategy (GranularityAggregationStrategy): how to aggregate token-level attributions into granularity scores.
                Options are: MEAN, MAX, MIN, SUM, and SIGNED_MAX.
                Ignored for `granularity` set to `ALL_TOKENS` or `TOKEN`.
            device (torch.device): device on which the attribution method will be run
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
            n_interpolations (int): the number of interpolations to generate
            baseline (torch.Tensor | float | None): the baseline to use for the interpolations
        """
        perturbator = LinearInterpolationPerturbator(
            inputs_embedder=model.get_input_embeddings(), baseline=baseline, n_perturbations=n_interpolations
        )
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            perturbator=perturbator,
            aggregator=MeanAggregator(),  # TODO: check if we need a trapezoidal mean
            granularity=granularity,
            granularity_aggregation_strategy=granularity_aggregation_strategy,
            inference_mode=inference_mode,
            use_gradient=True,
        )
