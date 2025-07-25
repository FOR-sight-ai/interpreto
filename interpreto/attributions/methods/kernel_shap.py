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
Kernel SHAP attribution method
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from interpreto.attributions.aggregations.linear_regression_aggregation import (
    Kernels,
    LinearRegressionAggregator,
)
from interpreto.attributions.base import AttributionExplainer, MultitaskExplainerMixin
from interpreto.attributions.perturbations.shap_perturbation import ShapTokenPerturbator
from interpreto.commons.granularity import Granularity
from interpreto.model_wrapping.inference_wrapper import InferenceModes


class KernelShap(MultitaskExplainerMixin, AttributionExplainer):
    """
    KernelSHAP is a model‑agnostic Shapley value estimator that interprets predictions
    by computing Shapley values through a weighted linear regression in the space of
    feature coalitions.

    By unifying ideas from LIME and Shapley value theory, KernelSHAP provides additive
    feature attributions with strong consistency guarantees.

    **Reference:**
    Lundberg and Lee (2017). *A Unified Approach to Interpreting Model Predictions.*
    [Paper](https://arxiv.org/abs/1705.07874)

    Examples:
        >>> from interpreto import Granularity, KernelShap
        >>> from interpreto.attributions import InferenceModes
        >>> method = KernelShap(model, tokenizer, batch_size=4,
        >>>                     inference_mode=InferenceModes.SOFTMAX,
        >>>                     n_perturbations=20,
        >>>                     granularity=Granularity.WORD)
        >>> explanations = method(text)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        granularity: Granularity = Granularity.WORD,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        n_perturbations: int = 1000,
        device: torch.device | None = None,
    ):
        """
        Initialize the attribution method.

        Args:
            model (PreTrainedModel): model to explain
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer associated with the model
            batch_size (int): batch size for the attribution method
            granularity (Granularity, optional): The level of granularity for the explanation.
                Options are: `ALL_TOKENS`, `TOKEN`, `WORD`, or `SENTENCE`.
                Defaults to Granularity.WORD.
                To obtain it, `from interpreto import Granularity` then `Granularity.WORD`.
            inference_mode (Callable[[torch.Tensor], torch.Tensor], optional): The mode used for inference.
                It can be either one of LOGITS, SOFTMAX, or LOG_SOFTMAX. Use InferenceModes to choose the appropriate mode.
            n_perturbations (int): the number of perturbations to generate
            distance_function (DistancesFromMaskProtocol): distance function used to compute weights of perturbed samples in the linear model training.
            similarity_kernel (SimilarityKernelProtocol): similarity kernel used to compute weights of perturbed samples in the linear model training.
            kernel_width (float | Callable): kernel width used in the `similarity_kernel`
            device (torch.device): device on which the attribution method will be run
        """
        model, replace_token_id = self._set_tokenizer(model, tokenizer)

        perturbator = ShapTokenPerturbator(
            tokenizer=tokenizer,
            inputs_embedder=model.get_input_embeddings(),
            granularity=granularity,
            replace_token_id=replace_token_id,
            n_perturbations=n_perturbations,
            device=device,
        )

        aggregator = LinearRegressionAggregator(
            distance_function=None,  # Kernel SHAP does not use distance function
            similarity_kernel=Kernels.ONES,
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            perturbator=perturbator,
            aggregator=aggregator,
            batch_size=batch_size,
            granularity=granularity,
            inference_mode=inference_mode,
            device=device,
            use_gradient=False,
        )
