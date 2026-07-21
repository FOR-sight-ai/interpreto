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
Image-side KernelSHAP method for ViT-family classification models.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.aggregations.linear_regression_aggregation import (
    Kernels,
    LinearRegressionAggregator,
)
from interpreto.attributions.base import ImageClassificationAttributionExplainer
from interpreto.attributions.perturbations import ShapImagePerturbator
from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class ImageKernelShap(ImageClassificationAttributionExplainer):
    """
    KernelSHAP for image-classification models (ViT-family).

    Samples granularity-unit coalitions according to the Shapley kernel,
    masking unselected units with a constant `replace_value` baseline, and
    estimates Shapley values via an unweighted linear regression over the
    coalition masks. Image-side analog of text `KernelShap`: `tokenizer` ->
    `image_processor`, no `MultitaskExplainerMixin` (image MVP is
    classification-only), and the replacement is a per-pixel float baseline
    (`replace_value`) instead of a `replace_token_id`.

    **Reference:**
    Lundberg and Lee (2017). *A Unified Approach to Interpreting Model Predictions.*
    [Paper](https://arxiv.org/abs/1705.07874)

    Examples:
        >>> from interpreto import ImageKernelShap
        >>> method = ImageKernelShap(model, image_processor, batch_size=4,
        >>>                          n_perturbations=1000)
        >>> explanations = method.explain(image)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        resize_strategy: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        n_perturbations: int = 1000,
        replace_value: float = 0.0,
        device: torch.device | None = None,
        preprocess: bool = True,
        image_mean: Sequence[float] | float | torch.Tensor | None = None,
        image_std: Sequence[float] | float | torch.Tensor | None = None,
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
            n_perturbations (int): number of coalition samples to generate.
            replace_value (float): baseline value written into masked units across all
                channels. `0.0` is the per-channel mean after standard ViT normalization.
            device (torch.device, optional): device on which the attribution method will be run.
            preprocess (bool, optional): if True, raw inputs are routed through `image_processor`.
                Defaults to True.
            image_mean / image_std (Sequence[float] | float | torch.Tensor | None, optional):
                de-normalization affine `x * image_std + image_mean`. Required when
                `preprocess=False`; must be omitted when `preprocess=True`.
        """
        # patch_size is reconciled from model.config by the explainer __init__.
        perturbator = ShapImagePerturbator(
            granularity=ImageGranularity.PATCH,
            n_perturbations=n_perturbations,
            replace_value=replace_value,
            device=device,
        )

        aggregator = LinearRegressionAggregator(
            distance_function=None,  # Kernel SHAP does not use distance function
            similarity_kernel=Kernels.ONES,
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
            image_mean=image_mean,
            image_std=image_std,
        )
