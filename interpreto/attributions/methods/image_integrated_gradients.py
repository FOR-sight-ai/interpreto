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
Image-side Integrated Gradients method for ViT-family classification models.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.aggregations import TrapezoidalMeanAggregator
from interpreto.attributions.base import ImageClassificationAttributionExplainer
from interpreto.attributions.perturbations import LinearInterpolationImagePerturbator
from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class ImageIntegratedGradients(ImageClassificationAttributionExplainer):
    """
    Integrated Gradients for image-classification models (ViT-family).

    Integrates the model's gradients along a straight-line path from a baseline
    to the input `pixel_values`, addressing gradient saturation. Image-side
    analog of text `IntegratedGradients`: `tokenizer` -> `image_processor`, no
    `MultitaskExplainerMixin`, and interpolation happens directly in pixel
    space (no `inputs_embedder` indirection).

    **Reference:**
    Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.*
    [Paper](http://proceedings.mlr.press/v70/sundararajan17a.html)

    Examples:
        >>> from interpreto import ImageIntegratedGradients
        >>> method = ImageIntegratedGradients(model, image_processor, batch_size=4,
        >>>                                   n_perturbations=50)
        >>> explanations = method.explain(image)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        granularity: ImageGranularity = ImageGranularity.DEFAULT,
        resize_strategy: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR,
        device: torch.device | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        input_x_gradient: bool = True,
        n_perturbations: int = 10,
        baseline: torch.Tensor | float | None = None,
        preprocess: bool = True,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain (ViT-family).
            image_processor (BaseImageProcessor): Hugging Face image processor associated with the model.
            batch_size (int): batch size for the attribution method.
            granularity (ImageGranularity, optional): granularity level (`PIXEL`, `PATCH`).
                Defaults to `ImageGranularity.DEFAULT` (= `PATCH`).
            resize_strategy (GranularityResizeStrategy, optional): how to
                aggregate per-pixel gradients into per-patch scores (MEAN, MAX, MIN, SUM, SIGNED_MAX).
            device (torch.device, optional): device on which the attribution method will be run.
            inference_mode (Callable, optional): inference mode (LOGITS, SOFTMAX, LOG_SOFTMAX).
            input_x_gradient (bool, optional): if True, multiply `pixel_values` by their gradients
                before the channel-collapse step. Defaults to True.
            n_perturbations (int): number of interpolation steps between baseline and input.
            baseline (torch.Tensor | float | None): baseline pixel values for the interpolation path.
            preprocess (bool, optional): if True, raw inputs are routed through `image_processor`.
                Defaults to True.
        """
        perturbator = LinearInterpolationImagePerturbator(baseline=baseline, n_perturbations=n_perturbations)
        super().__init__(
            model=model,
            image_processor=image_processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=TrapezoidalMeanAggregator(),
            device=device,
            granularity=granularity,
            resize_strategy=resize_strategy,
            inference_mode=inference_mode,
            use_gradient=True,
            input_x_gradient=input_x_gradient,
            preprocess=preprocess,
        )
