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
Image-side LIME method for ViT-family classification models.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.aggregations.linear_regression_aggregation import (
    DistancesFromMask,
    DistancesFromMaskProtocol,
    Kernels,
    LinearRegressionAggregator,
)
from interpreto.attributions.base import ImageClassificationAttributionExplainer
from interpreto.attributions.perturbations import RandomMaskedImagePerturbator
from interpreto.commons.granularity import GranularityResizeStrategy
from interpreto.commons.granularity import ImageGranularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class ImageLime(ImageClassificationAttributionExplainer):
    """
    LIME for image-classification models (ViT-family).

    Samples perturbed versions of the input by randomly masking granularity
    units (patches or pixels) with a constant `replace_value` baseline, then
    fits a similarity-weighted linear surrogate over the masks to obtain
    per-unit importance scores. Image-side analog of text `Lime`: `tokenizer`
    -> `image_processor`, no `MultitaskExplainerMixin` (image MVP is
    classification-only), and the replacement is a per-pixel float baseline
    (`replace_value`) instead of a `replace_token_id`.

    **Reference:**
    Ribeiro et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.*
    [Paper](https://arxiv.org/abs/1602.04938)

    Examples:
        >>> from interpreto import ImageGranularity, ImageLime
        >>> method = ImageLime(model, image_processor, batch_size=4,
        >>>                    n_perturbations=100,
        >>>                    granularity=ImageGranularity.PATCH,
        >>>                    distance_function=ImageLime.distance_functions.COSINE)
        >>> explanations = method.explain(image)
    """

    distance_functions: type[Enum] = DistancesFromMask

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        granularity: ImageGranularity = ImageGranularity.DEFAULT,
        granularity_resize: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        n_perturbations: int = 100,
        perturb_probability: float = 0.5,
        replace_value: float = 0.0,
        distance_function: DistancesFromMaskProtocol = DistancesFromMask.COSINE,
        kernel_width: float | Callable | None = None,
        device: torch.device | None = None,
        preprocess: bool = True,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain (ViT-family).
            image_processor (BaseImageProcessor): Hugging Face image processor associated with the model.
            batch_size (int): batch size for the attribution method.
            granularity (ImageGranularity, optional): unit over which masks are defined (`PIXEL`, `PATCH`).
                Defaults to `ImageGranularity.DEFAULT` (= `PATCH`).
            granularity_resize (GranularityResizeStrategy, optional): how to
                aggregate per-pixel scores into per-patch scores (MEAN, MAX, MIN, SUM, SIGNED_MAX).
            inference_mode (Callable, optional): inference mode (LOGITS, SOFTMAX, LOG_SOFTMAX).
            n_perturbations (int): number of perturbations to generate.
            perturb_probability (float): probability that a granularity unit is masked.
            replace_value (float): baseline value written into masked units across all
                channels. `0.0` is the per-channel mean after standard ViT normalization.
            distance_function (DistancesFromMaskProtocol): distance function used to compute
                similarity weights of perturbed samples in the linear surrogate.
            kernel_width (float | Callable | None): kernel width used in the similarity kernel.
                If None, computed using the `default_kernel_width_fn` function.
            device (torch.device, optional): device on which the attribution method will be run.
            preprocess (bool, optional): if True, raw inputs are routed through `image_processor`.
                Defaults to True.
        """
        # patch_size is reconciled from model.config by the explainer __init__.
        perturbator = RandomMaskedImagePerturbator(
            granularity=granularity,
            n_perturbations=n_perturbations,
            perturb_probability=perturb_probability,
            replace_value=replace_value,
        )

        aggregator = LinearRegressionAggregator(
            distance_function=distance_function,
            similarity_kernel=Kernels.EXPONENTIAL,
            kernel_width=kernel_width,
        )

        super().__init__(
            model=model,
            image_processor=image_processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=aggregator,
            device=device,
            granularity=granularity,
            granularity_resize=granularity_resize,
            inference_mode=inference_mode,
            use_gradient=False,
            preprocess=preprocess,
        )
