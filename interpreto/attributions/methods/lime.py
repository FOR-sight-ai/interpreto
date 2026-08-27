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
LIME attribution method
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor

from interpreto.attributions.aggregations.linear_regression_aggregation import (
    DistancesFromMask,
    DistancesFromMaskProtocol,
    Kernels,
    LinearRegressionAggregator,
)
from interpreto.attributions.base import AttributionExplainer, InferenceModes, MultitaskExplainerMixin
from interpreto.attributions.perturbations import RandomMaskedPerturbator
from interpreto.commons import Granularity, GranularityCombinationStrategy


class Lime(MultitaskExplainerMixin, AttributionExplainer):
    """
    Local Interpretable Model-agnostic Explanations (LIME) is a perturbation‑based approach that explains individual predictions by
    fitting a simple, interpretable surrogate model locally around the prediction
    of interest. By sampling perturbed versions of the input and weighting them by
    their proximity to the original instance, LIME learns per‑feature importance scores
    that approximate the behaviour of the underlying black‑box model in that local region.

    **Reference:**
    Ribeiro et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.*
    [Paper](https://arxiv.org/abs/1602.04938)

    Examples:
        >>> from interpreto import TextGranularity, Lime
        >>> from interpreto.attributions import InferenceModes
        >>> method = Lime(model, processor, batch_size=4,
        >>>               inference_mode=InferenceModes.LOG_SOFTMAX,
        >>>               n_perturbations=20,
        >>>               granularity=TextGranularity.WORD,
        >>>               distance_function=Lime.distance_functions.HAMMING)
        >>> explanations = method(inputs)
    """

    distance_functions: type[Enum] = DistancesFromMask

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
        perturb_probability: float = 0.5,
        replace_value: int | float | None = None,
        distance_function: DistancesFromMaskProtocol = DistancesFromMask.COSINE,
        kernel_width: float | Callable | None = None,
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
            n_perturbations (int): the number of perturbations to generate.
            perturb_probability (float): probability of perturbation.
            replace_value: the id of the token used for masking in text methods, the value of the pixel
                used for masking in image methods
            distance_function (DistancesFromMaskProtocol): distance function used to compute weights of perturbed samples in the linear model training.
            kernel_width (float | Callable | None): kernel width used in the `similarity_kernel`.
                If None, the kernel width is computed using the `default_kernel_width_fn` function.
        """
        if granularity is None:
            granularity = self.default_mask_granularity
        if combination_strategy is None:
            combination_strategy = self.default_combination_strategy

        replace_value = self._setup_replace_value(model, processor, replace_value)

        # create the perturbator dynamically by inheriting from both the method and modality specific classes
        perturbator_class = type(
            "ModalitySpecific" + self.__class__.__name__,  # name
            (RandomMaskedPerturbator, self.base_mask_perturbator_class),  # parent classes
            {},
        )
        perturbator = perturbator_class(
            processor=processor,
            granularity=granularity,
            n_perturbations=n_perturbations,
            replace_value=replace_value,
            perturb_probability=perturb_probability,
        )

        aggregator = LinearRegressionAggregator(
            distance_function=distance_function,
            similarity_kernel=Kernels.EXPONENTIAL,
            kernel_width=kernel_width,
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
