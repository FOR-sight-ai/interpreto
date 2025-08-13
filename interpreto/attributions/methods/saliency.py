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
Saliency method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.commons.granularity import Granularity, GranularityAggregationStrategy
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class Saliency(MultitaskExplainerMixin, AttributionExplainer):
    """
    Saliency maps are a simple and widely used gradient-based method for interpreting
    neural network predictions. The idea is to compute the gradient of the model's output
    with respect to its input embeddings to estimate which input tokens most influence the output.

    Procedure:

    - Pass the input through the model to obtain an output (e.g., class logit, token probability).
    - Compute the gradient of the output with respect to the input embeddings.
    - For each token, reduce the gradient vector (e.g., via norm with the embedding) to obtain a scalar importance score.

    **Reference:**
    Simonyan et al. (2013). *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.*
    [Paper](https://arxiv.org/abs/1312.6034)

    Examples:
        >>> from interpreto import Saliency
        >>> method = Saliency(model, tokenizer, batch_size=4)
        >>> explanations = method.explain(text)
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
        input_x_gradient: bool = True,
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
            input_x_gradient (bool, optional): If True, multiplies the input embeddings with
                their gradients before aggregation. Defaults to ``True``.
        """

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            perturbator=None,
            aggregator=None,
            granularity=granularity,
            granularity_aggregation_strategy=granularity_aggregation_strategy,
            inference_mode=inference_mode,
            use_gradient=True,
            input_x_gradient=input_x_gradient,
        )
