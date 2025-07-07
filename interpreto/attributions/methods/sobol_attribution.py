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
Sobol attribution method
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations.sobol_aggregation import SobolAggregator, SobolIndicesOrders
from interpreto.attributions.base import AttributionExplainer, InferenceModes, MultitaskExplainerMixin
from interpreto.attributions.perturbations.sobol_perturbation import (
    SequenceSamplers,
    SobolTokenPerturbator,
)
from interpreto.commons.granularity import Granularity


class Sobol(MultitaskExplainerMixin, AttributionExplainer):
    """
    Sobol is a variance-based sensitivity analysis method used to quantify the contribution
    of each input component to the output variance of the model.

    It estimates both the first-order (main) and total (interaction) effects of features using
    Monte Carlo sampling strategies. In NLP, Sobol helps assess which words or tokens are most
    influential for the model’s decision, including how they interact with one another.

    **Reference:**
    Fel et al. (2021). *Look at the variance! Efficient black-box explanations with Sobol-based sensitivity analysis.*
    [Paper](https://arxiv.org/abs/2111.04138)

    Examples:
        >>> from interpreto import Granularity, Sobol
        >>> from interpreto.attributions import InferenceModes
        >>> method = Sobol(model, tokenizer, batch_size=4,
        >>>                inference_mode=InferenceModes.LOGITS,
        >>>                n_token_perturbations=8,
        >>>                granularity=Granularity.WORD,
        >>>                sobol_indices_order=Sobol.sobol_indices_orders.FIRST_ORDER,
        >>>                sampler=Sobol.samplers.SOBOL))
        >>> explanations = method(text)
    """

    samplers: type[Enum] = SequenceSamplers
    sobol_indices_orders: type[Enum] = SobolIndicesOrders

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        granularity: Granularity = Granularity.WORD,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        n_token_perturbations: int = 32,
        sobol_indices_order: SobolIndicesOrders = SobolIndicesOrders.FIRST_ORDER,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        device: torch.device | None = None,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            batch_size (int): batch size for the attribution method
            granularity (Granularity): The level of granularity for the explanation (e.g., token, word, sentence).
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
            n_token_perturbations (int): the number of perturbations to generate
            sobol_indices (SobolIndicesOrders): Sobol indices order, either `FIRST_ORDER` or `TOTAL_ORDER`.
            sampler (SequenceSamplers): Sobol sequence sampler, either `SOBOL`, `HALTON` or `LatinHypercube`.
            device (torch.device): device on which the attribution method will be run
        """
        model, replace_token_id = self._set_tokenizer(model, tokenizer)

        perturbator = SobolTokenPerturbator(
            tokenizer=tokenizer,
            inputs_embedder=model.get_input_embeddings(),
            granularity=granularity,
            replace_token_id=replace_token_id,
            n_token_perturbations=n_token_perturbations,
            sampler=sampler,
        )

        aggregator = SobolAggregator(
            n_token_perturbations=n_token_perturbations,
            sobol_indices_order=sobol_indices_order,
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            perturbator=perturbator,
            aggregator=aggregator,
            batch_size=batch_size,
            granularity=granularity,
            inference_mode=inference_mode,
            device=device,
            use_gradient=False,
        )
