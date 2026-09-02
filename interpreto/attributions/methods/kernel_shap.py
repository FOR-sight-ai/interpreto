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
Kernel SHAP attribution method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor

from interpreto.attributions.aggregations.linear_regression_aggregation import (
    Kernels,
    LinearRegressionAggregator,
)
from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.attributions.perturbations import ShapPerturbator
from interpreto.commons import general_bad_argument
from interpreto.commons.granularity import Granularity, GranularityCombinationStrategy
from interpreto.model_wrapping.inference_wrapper import InferenceModes


@general_bad_argument
class KernelShap(MultitaskExplainerMixin, AttributionExplainer):
    """
    KernelSHAP is a model‑agnostic Shapley value estimator that interprets predictions
    by computing Shapley values through a weighted linear regression in the space of
    feature coalitions.

    By unifying ideas from LIME and Shapley value theory, KernelSHAP provides additive
    feature attributions with strong consistency guarantees.

    **Reference:**
    Lundberg and Lee (2017). *A Unified Approach to Interpreting Model Predictions.*
    [Paper](https://arxiv.org/abs/1705.07874)

    Examples:
        >>> from interpreto import TextGranularity, KernelShap
        >>> from interpreto.attributions import InferenceModes
        >>> method = KernelShap(model, processor, batch_size=4,
        >>>                     inference_mode=InferenceModes.SOFTMAX,
        >>>                     n_perturbations=20,
        >>>                     granularity=TextGranularity.WORD)
        >>> explanations = method(inputs)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: PreTrainedTokenizerBase | BaseImageProcessor,
        granularity: Granularity | None = None,
        combination_strategy: GranularityCombinationStrategy | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        device: torch.device | None = None,
        batch_size: int = 4,
        n_perturbations: int = 100,
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
            combination_strategy (GranularityCombinationStrategy | None): how per-token or
                scores are combined into granularity scores (on the text side). how masks
                and heatmaps are resized from the granularity space to the image space
                for images.
            inference_mode (Callable[[torch.Tensor], torch.Tensor]): the mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to
                choose the appropriate mode.
            device (torch.device): device on which the attribution method will be run.
            batch_size (int): batch size for the attribution method.
            n_perturbations (int): the number of perturbations to generate
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
            (ShapPerturbator, self.base_mask_perturbator_class),  # parent classes
            {},
        )
        perturbator = perturbator_class(
            processor=processor,
            granularity=granularity,
            n_perturbations=n_perturbations,
            replace_value=replace_value,
        )

        aggregator = LinearRegressionAggregator(
            distance_function=None,  # Kernel SHAP does not use distance function
            similarity_kernel=Kernels.ONES,
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
