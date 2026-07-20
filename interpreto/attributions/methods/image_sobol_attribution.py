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
Image-side Sobol attribution method for ViT-family classification models.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.aggregations.sobol_aggregation import SobolAggregator, SobolIndicesOrders
from interpreto.attributions.base import ImageClassificationAttributionExplainer
from interpreto.attributions.perturbations import SobolImagePerturbator
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers
from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class ImageSobol(ImageClassificationAttributionExplainer):
    """
    Sobol attribution for image-classification models (ViT-family).

    Variance-based sensitivity analysis: masks granularity units (patches or
    pixels) according to quasi-Monte-Carlo (Sobol) sequences with a constant
    `replace_value` baseline, then estimates first- or total-order Sobol indices
    from the resulting output variance. Image-side analog of text `Sobol`:
    `tokenizer` -> `image_processor`, no `MultitaskExplainerMixin` (image MVP is
    classification-only), and the replacement is a per-pixel float baseline
    (`replace_value`) instead of a `replace_token_id`.

    **Reference:**
    Fel et al. (2021). *Look at the variance! Efficient black-box explanations with Sobol-based sensitivity analysis.*
    [Paper](https://arxiv.org/abs/2111.04138)

    Examples:
        >>> from interpreto import ImageSobol
        >>> method = ImageSobol(model, image_processor, batch_size=4,
        >>>                     n_token_perturbations=32,
        >>>                     sobol_indices_order=ImageSobol.sobol_indices_orders.FIRST_ORDER,
        >>>                     sampler=ImageSobol.samplers.SOBOL)
        >>> explanations = method.explain(image)
    """

    samplers: type[Enum] = SequenceSamplers
    sobol_indices_orders: type[Enum] = SobolIndicesOrders

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        resize_strategy: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        n_token_perturbations: int = 32,
        sobol_indices_order: SobolIndicesOrders = SobolIndicesOrders.TOTAL_ORDER,
        sampler: SequenceSamplers = SequenceSamplers.SOBOL,
        replace_value: float = 0.0,
        device: torch.device | None = None,
        preprocess: bool = True,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain (ViT-family).
            image_processor (BaseImageProcessor): Hugging Face image processor associated with the model.
            batch_size (int): batch size for the attribution method.
            resize_strategy (GranularityResizeStrategy, optional): how to
                aggregate per-pixel scores into per-patch scores (MEAN, MAX, MIN, SUM, SIGNED_MAX).
            inference_mode (Callable, optional): inference mode (LOGITS, SOFTMAX, LOG_SOFTMAX).
            n_token_perturbations (int): Monte-Carlo samples per granularity unit (total
                perturbations are `(g + 2) * n_token_perturbations`).
            sobol_indices_order (SobolIndicesOrders): Sobol indices order, `FIRST_ORDER` or `TOTAL_ORDER`.
            sampler (SequenceSamplers): quasi-MC sequence sampler (`SOBOL`, `HALTON`, `LatinHypercube`).
            replace_value (float): baseline value written into masked units across all
                channels. `0.0` is the per-channel mean after standard ViT normalization.
            device (torch.device, optional): device on which the attribution method will be run.
            preprocess (bool, optional): if True, raw inputs are routed through `image_processor`.
                Defaults to True.
        """
        # patch_size is reconciled from model.config by the explainer __init__.
        perturbator = SobolImagePerturbator(
            granularity=ImageGranularity.PATCH,
            n_token_perturbations=n_token_perturbations,
            sampler=sampler,
            replace_value=replace_value,
        )

        aggregator = SobolAggregator(
            n_token_perturbations=n_token_perturbations,
            sobol_indices_order=sobol_indices_order,
        )

        super().__init__(
            model=model,
            image_processor=image_processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=aggregator,
            device=device,
            granularity=ImageGranularity.PATCH,
            resize_strategy=resize_strategy,
            inference_mode=inference_mode,
            use_gradient=False,
            preprocess=preprocess,
        )
