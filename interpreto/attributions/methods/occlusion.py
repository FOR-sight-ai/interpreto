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
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from interpreto.attributions.aggregations.base import OcclusionAggregator
from interpreto.attributions.base import (
    AttributionExplainer,
    MultitaskExplainerMixin,
)
from interpreto.attributions.perturbations import OcclusionPerturbator
from interpreto.commons.granularity import Granularity
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
        >>> from interpreto import Granularity, Occlusion
        >>> from interpreto.attributions import InferenceModes
        >>> method = Occlusion(model, tokenizer, batch_size=4,
        >>>                    inference_mode=InferenceModes.SOFTMAX,
        >>>                    granularity=Granularity.WORD)
        >>> explanations = method(text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        granularity: Granularity = Granularity.WORD,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        device: torch.device | None = None,
        replace_token_id: int | None = None,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            batch_size (int): batch size for the attribution method
            granularity (Granularity): granularity level of the perturbations (token, word, sentence, etc.)
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
            device (torch.device): device on which the attribution method will be run
        """
        model, replace_token_id = self._set_tokenizer(model, tokenizer)

        perturbator = OcclusionPerturbator(
            tokenizer=tokenizer,
            inputs_embedder=model.get_input_embeddings(),
            granularity=granularity,
            replace_token_id=replace_token_id,
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            perturbator=perturbator,
            aggregator=OcclusionAggregator(),
            granularity=granularity,
            inference_mode=inference_mode,
            use_gradient=False,
        )
