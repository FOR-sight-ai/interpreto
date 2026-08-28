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
Saliency method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor

from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.commons import general_bad_argument
from interpreto.commons.granularity import Granularity, GranularityCombinationStrategy
from interpreto.model_wrapping.inference_wrapper import InferenceModes


@general_bad_argument
class Saliency(MultitaskExplainerMixin, AttributionExplainer):
    """
    Saliency maps are a simple and widely used gradient-based method for interpreting
    neural network predictions. The idea is to compute the gradient of the model's output
    with respect to its input tensor — the token embeddings on the text side, `pixel_values`
    on the image side — to estimate which parts of the input most influence the output.

    Procedure:

    - Pass the input through the model to obtain an output (e.g., class logit, token probability).
    - Compute the gradient of the output with respect to the input tensor.
    - Reduce each gradient vector (e.g., via product with the input) to obtain a scalar score.
    - Aggregate the result per granularity unit to get the final attribution scores.

    Saliency is the only gradient method with no perturbation of its own: it runs a single
    forward/backward pass on the unaltered input, so it uses the modality's base tensor
    perturbator directly (a no-op that only converts input ids to embeddings on the text side).

    **Reference:**
    Simonyan et al. (2013). *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.*
    [Paper](https://arxiv.org/abs/1312.6034)

    Examples:
        >>> from interpreto import Saliency
        >>> method = Saliency(model, processor, batch_size=4)
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
        """
        if granularity is None:
            granularity = self.default_tensor_granularity
        if combination_strategy is None:
            combination_strategy = self.default_combination_strategy

        # No method-specific perturbator: unlike SmoothGrad or IntegratedGradients, Saliency adds
        # nothing on top of the modality base, whose `perturb_tensor` is already the identity.
        # So there is nothing to mix in and the base class is instantiated directly.
        perturbator = self.base_tensor_perturbator_class(
            processor=processor,
            granularity=granularity,
        )

        super().__init__(
            model=model,
            processor=processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=None,
            device=device,
            granularity=granularity,
            combination_strategy=combination_strategy,
            inference_mode=inference_mode,
            use_gradient=True,
            input_x_gradient=input_x_gradient,
        )
