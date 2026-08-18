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
Image-side Occlusion method for ViT-family classification models.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.aggregations import OcclusionAggregator
from interpreto.attributions.base import ImageClassificationAttributionExplainer
from interpreto.attributions.perturbations import OcclusionImagePerturbator
from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class ImageOcclusion(ImageClassificationAttributionExplainer):
    """
    Occlusion for image-classification models (ViT-family).

    Perturbation-based method that masks one granularity unit (patch or pixel)
    at a time with a constant `replace_value` baseline, plus one reference where
    nothing is masked, and attributes importance from the drop in the targeted
    logit. Image-side analog of text `Occlusion`: `tokenizer` ->
    `image_processor`, no `MultitaskExplainerMixin` (image MVP is
    classification-only), and the replacement is a per-pixel float baseline
    (`replace_value`) instead of a `replace_token_id`.

    **Reference:**
    Zeiler and Fergus (2014). *Visualizing and understanding convolutional networks.*
    [Paper](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)

    Examples:
        >>> from interpreto import ImageOcclusion
        >>> method = ImageOcclusion(model, image_processor, batch_size=4)
        >>> explanations = method.explain(image)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        resize_strategy: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        replace_value: float = 0.0,
        device: torch.device | None = None,
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
            replace_value (float): baseline value written into the occluded unit across all
                channels. `0.0` is the per-channel mean after standard ViT normalization.
            device (torch.device, optional): device on which the attribution method will be run.
        """
        # patch_size is reconciled from model.config by the explainer __init__.
        perturbator = OcclusionImagePerturbator(
            granularity=ImageGranularity.PATCH,
            replace_value=replace_value,
        )
        super().__init__(
            model=model,
            image_processor=image_processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=OcclusionAggregator(),
            device=device,
            granularity=ImageGranularity.PATCH,
            resize_strategy=resize_strategy,
            inference_mode=inference_mode,
            use_gradient=False,
        )
