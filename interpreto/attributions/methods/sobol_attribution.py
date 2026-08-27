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
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor

from interpreto.attributions.aggregations.sobol_aggregation import SobolAggregator, SobolIndicesOrders
from interpreto.attributions.base import AttributionExplainer, InferenceModes, MultitaskExplainerMixin
from interpreto.attributions.perturbations import SobolPerturbator
from interpreto.attributions.perturbations.base import TextMaskPerturbator
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers
from interpreto.commons.granularity import Granularity, GranularityCombinationStrategy


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
        >>> from interpreto import TextGranularity, Sobol
        >>> from interpreto.attributions import InferenceModes
        >>> method = Sobol(model, processor, batch_size=4,
        >>>                inference_mode=InferenceModes.LOGITS,
        >>>                n_token_perturbations=8,
        >>>                granularity=TextGranularity.WORD,
        >>>                sobol_indices_order=Sobol.sobol_indices_orders.FIRST_ORDER,
        >>>                sampler=Sobol.samplers.SOBOL))
        >>> explanations = method(inputs)
    """

    samplers: type[Enum] = SequenceSamplers
    sobol_indices_orders: type[Enum] = SobolIndicesOrders

    def __init__(
        self,
        model: PreTrainedModel,
        processor: PreTrainedTokenizerBase | BaseImageProcessor,
        granularity: Granularity | None = None,
        combination_strategy: GranularityCombinationStrategy | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        device: torch.device | None = None,
        batch_size: int = 4,
        n_token_perturbations: int = 32,
        sobol_indices_order: SobolIndicesOrders = SobolIndicesOrders.TOTAL_ORDER,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        replace_value: int | float | None = None,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain.
            processor (PreTrainedTokenizerBase | BaseImageProcessor): Hugging Face tokenizer or image
                processor associated with the model.
            granularity (Granularity | None): the level of granularity for the explanation.
                Defaults to the modality's default_mask_granularity: WORD for text,
                PATCH for images.
            combination_strategy (GranularityCombinationStrategy | None): how per-token
                scores are combined into granularity scores (on the text side). how masks
                and heatmaps are resized from the granularity space to the image space
                for images.
            inference_mode (Callable[[torch.Tensor], torch.Tensor]): the mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to
                choose the appropriate mode.
            device (torch.device): device on which the attribution method will be run.
            batch_size (int): batch size for the attribution method.
            n_token_perturbations (int): the number of perturbations to generate
            sobol_indices (SobolIndicesOrders): Sobol indices order, either `FIRST_ORDER` or `TOTAL_ORDER`.
            sampler (SequenceSamplers): Sobol sequence sampler, either `SOBOL`, `HALTON` or `LatinHypercube`.
            replace_value: the id of the token used for masking in text methods, the value of the pixel
                used for masking in image methods
        """
        if granularity is None:
            granularity = self.default_mask_granularity
        if combination_strategy is None:
            combination_strategy = self.default_combination_strategy

        replace_value = self._setup_replace_value(model, processor, replace_value)

        # create the perturbator dynamically by inheriting from both the method and modality specific classes
        perturbator_class = type(
            "ModalitySpecific" + self.__class__.__name__,  # name
            (SobolPerturbator, self.base_mask_perturbator_class),  # parent classes
            {},
        )
        perturbator = perturbator_class(
            processor=processor,
            granularity=granularity,
            replace_value=replace_value,
            n_token_perturbations=n_token_perturbations,
            sampler=sampler,
            is_binarized=issubclass(self.base_mask_perturbator_class, TextMaskPerturbator),
        )

        aggregator = SobolAggregator(
            n_token_perturbations=n_token_perturbations,
            sobol_indices_order=sobol_indices_order,
        )

        super().__init__(
            model=model,
            processor=processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=aggregator,
            device=device,
            granularity=granularity,
            combination_strategy=combination_strategy,
            inference_mode=inference_mode,
            use_gradient=False,
        )
