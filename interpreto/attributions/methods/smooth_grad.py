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
SmoothGrad attribution method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor

from interpreto.attributions.aggregations import MeanAggregator
from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.attributions.perturbations.gaussian_noise_perturbation import GaussianNoisePerturbator
from interpreto.commons.granularity import Granularity, GranularityCombinationStrategy
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class SmoothGrad(MultitaskExplainerMixin, AttributionExplainer):
    """
    SmoothGrad is an enhanced version of gradient-based interpretability methods, such as saliency maps.
    It reduces the noise and visual instability often seen in raw gradient attributions by averaging gradients
    over multiple noisy versions of the input. The result is a smoothed importance score for each granularity unit.

    Procedure:

    - Generate multiple perturbed versions of the input by adding Gaussian noise to the input tensor:
      the token embeddings on the text side, `pixel_values` on the image side.
    - For each noisy input, compute the gradient of the output with respect to that tensor.
    - Average the gradients across all samples.
    - Aggregate the result per granularity unit to get the final attribution scores.

    **Reference:**
    Smilkov et al. (2017). *SmoothGrad: removing noise by adding noise.*
    [Paper](https://arxiv.org/abs/1706.03825)

    Examples:
        >>> from interpreto import SmoothGrad
        >>> method = SmoothGrad(model, processor, batch_size=4,
        >>>                     n_perturbations=50, noise_std=0.01)
        >>> explanations = method.explain(inputs)
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
        input_x_gradient: bool = True,
        n_perturbations: int = 10,
        noise_std: float = 0.1,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain.
            processor (PreTrainedTokenizerBase | BaseImageProcessor): Hugging Face tokenizer or image
                processor associated with the model.
            granularity (Granularity | None): the level of granularity for the explanation.
                Defaults to the modality's `default_tensor_granularity`: `WORD` for text,
                `PIXEL` for images.
            combination_strategy (GranularityCombinationStrategy | None): how per-token or
                per-pixel gradients are combined into granularity scores. Defaults to the
                modality's `default_combination_strategy`.
            inference_mode (Callable[[torch.Tensor], torch.Tensor]): the mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to
                choose the appropriate mode.
            device (torch.device): device on which the attribution method will be run.
            batch_size (int): batch size for the attribution method.
            input_x_gradient (bool): if True, multiplies the input tensor with its gradients
                before aggregation.
            n_perturbations (int): the number of noisy samples to average over.
            noise_std (float): standard deviation of the Gaussian noise added to the input tensor.
        """
        if granularity is None:
            granularity = self.default_tensor_granularity
        if combination_strategy is None:
            combination_strategy = self.default_combination_strategy

        text_only_kwargs = self._text_only_kwargs(model)

        # create the perturbator dynamically by inheriting from both the method and modality specific classes
        perturbator_class = type(
            "ModalitySpecific" + self.__class__.__name__,  # name
            (GaussianNoisePerturbator, self.base_tensor_perturbator_class),  # parent classes
            {"__slots__": ()},
        )
        perturbator = perturbator_class(
            processor=processor,
            granularity=granularity,
            n_perturbations=n_perturbations,
            std=noise_std,
            **text_only_kwargs,
        )

        super().__init__(
            model=model,
            processor=processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=MeanAggregator(),
            device=device,
            granularity=granularity,
            combination_strategy=combination_strategy,
            inference_mode=inference_mode,
            use_gradient=True,
            input_x_gradient=input_x_gradient,
        )
