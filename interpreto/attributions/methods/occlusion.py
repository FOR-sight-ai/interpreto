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
Occlusion attribution method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.aggregations.base import OcclusionAggregator
from interpreto.attributions.base_merged import AttributionExplainer, MultitaskExplainerMixin
from interpreto.attributions.perturbations.occlusion_perturbation_merged import OcclusionPerturbator
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class Occlusion(MultitaskExplainerMixin, AttributionExplainer):
    """
    The Occlusion method is a perturbation-based approach to interpret model behavior by analyzing
    the impact of removing or masking parts of the input text. The principle is simple: by
    systematically occluding (i.e., masking, deleting, or replacing) specific tokens or spans in the
    input and observing how the model's output changes, one can infer the relative importance of
    each part of the input to the model's behavior.

    **Reference:**
    Zeiler and Fergus (2014). *Visualizing and understanding convolutional networks.*
    [Paper](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)

    Examples:
        >>> from interpreto import TextGranularity, Occlusion
        >>> from interpreto.attributions import InferenceModes
        >>> method = Occlusion(model, processor, batch_size=4,
        >>>                    inference_mode=InferenceModes.SOFTMAX,
        >>>                    granularity=TextGranularity.WORD)
        >>> explanations = method(text)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: PreTrainedTokenizer | BaseImageProcessor,
        granularity: Granularity | None = None,
        combination_strategy: GranularityCombinationStrategy | None = None,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        device: torch.device | None = None,
        batch_size: int = 4,
        replace_value: int | float | None = None,
        preprocess: bool = True,
        image_mean: Sequence[float] | float | torch.Tensor | None = None,
        image_std: Sequence[float] | float | torch.Tensor | None = None,
    ):
        if granularity is None:
            granularity = self.default_mask_granularity
        if combination_strategy is None:
            combination_strategy = self.default_combination_strategy

        image_only_kwargs = self._image_only_kwargs(
            preprocess=preprocess,
            image_mean=image_mean,
            image_std=image_std,
        )

        replace_value = self._setup_replace_value(model, processor, replace_value)
        # create the perturbator dynamically by inheriting from both the method and modality specific classes
        perturbator_class = type(
            "ModalitySpecific" + self.__class__.__name__,  # name
            (
                OcclusionPerturbator,
                self.base_mask_perturbator_class,
            ),  # parent classes
            {"__slots__": ()},
        )
        # n_perturbations is no longer passed: OcclusionPerturbator.__init__ pins it to -1.
        perturbator = perturbator_class(
            processor=processor,
            granularity=granularity,
            replace_value=replace_value,
        )

        super().__init__(
            model=model,
            processor=processor,
            batch_size=batch_size,
            perturbator=perturbator,
            aggregator=OcclusionAggregator(),
            device=device,
            granularity=granularity,
            combination_strategy=combination_strategy,
            inference_mode=inference_mode,
            use_gradient=False,
            **image_only_kwargs,
        )
