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
Image-side Saliency method for ViT-family classification models.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.image_base import ImageClassificationAttributionExplainer
from interpreto.commons.granularity import GranularityAggregationStrategy
from interpreto.commons.image_granularity import ImageGranularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class ImageSaliency(ImageClassificationAttributionExplainer):
    """
    Saliency maps for image-classification models (ViT-family).

    Mirrors text `Saliency` with three modality-specific changes:
    `tokenizer` -> `image_processor`, no `MultitaskExplainerMixin` (image
    MVP is classification-only — see [[project-vit-architecture-decisions]]
    for the deferred multitask-dispatch decision), and no explicit
    `EmbeddingsPerturbator`: for ViT the gradient leaf is `pixel_values`
    directly (set via `requires_grad_(True)` inside
    `ImageClassificationInferenceWrapper._prepare_inputs`), so the no-op
    `ImagePerturbator` (the default fallback) is the right choice.

    Procedure:

    - Forward the (preprocessed) image through the model to get the targeted logit.
    - Compute the gradient of that logit w.r.t. `pixel_values`.
    - Optionally multiply by `pixel_values` (input x gradient), collapse the
      channel dimension via `abs().mean(dim=1)`, and flatten H*W -> l.
    - Aggregate per granularity unit (per-pixel or per-patch) to produce
      `(t, g)` attribution scores.

    **Reference:**
    Simonyan et al. (2013). *Deep Inside Convolutional Networks: Visualising
    Image Classification Models and Saliency Maps.*
    [Paper](https://arxiv.org/abs/1312.6034)

    Examples:
        >>> from interpreto import ImageSaliency
        >>> method = ImageSaliency(model, image_processor, batch_size=4)
        >>> explanations = method.explain(image)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        granularity: ImageGranularity = ImageGranularity.DEFAULT,
        granularity_aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        device: torch.device | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        input_x_gradient: bool = True,
        preprocess: bool = True,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain (ViT-family).
            image_processor (BaseImageProcessor): Hugging Face image processor
                associated with the model.
            batch_size (int): batch size for the attribution method.
            granularity (ImageGranularity, optional): The granularity level of the
                explanation. Options: `PIXEL`, `PATCH`. Defaults to
                `ImageGranularity.DEFAULT` (= `PATCH`).
            granularity_aggregation_strategy (GranularityAggregationStrategy, optional):
                How to aggregate per-pixel gradients into per-patch scores.
                Options: MEAN, MAX, MIN, SUM, SIGNED_MAX. Ignored when `granularity`
                is `PIXEL` (no within-unit aggregation needed).
            device (torch.device, optional): device on which the attribution method
                will be run.
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The
                mode used for inference (LOGITS, SOFTMAX, LOG_SOFTMAX). Use
                `InferenceModes` to choose.
            input_x_gradient (bool, optional): If True, multiplies `pixel_values`
                by their gradients before the channel-collapse step. Defaults to True.
            preprocess (bool, optional): If True, raw inputs (PIL.Image, np.ndarray,
                torch.Tensor) are routed through `image_processor`. If False, inputs
                must already be `BatchFeature` or normalized `(1, 3, H, W)` tensors.
                Defaults to True.
        """
        super().__init__(
            model=model,
            image_processor=image_processor,
            batch_size=batch_size,
            perturbator=None,  # falls back to ImagePerturbator() (no-op)
            aggregator=None,  # falls back to default Aggregator() (squeeze p=1)
            device=device,
            granularity=granularity,
            granularity_aggregation_strategy=granularity_aggregation_strategy,
            inference_mode=inference_mode,
            use_gradient=True,
            input_x_gradient=input_x_gradient,
            preprocess=preprocess,
        )
