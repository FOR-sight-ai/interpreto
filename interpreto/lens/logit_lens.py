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
Logit Lens implementation built on top of `ModelWithSplitPoints`.
"""

from __future__ import annotations

import torch

from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints

from ._lens_base import BaseLens, PoolingStrategy


class LogitLens(BaseLens):
    """Code: [:octicons-mark-github-24: `lens/logit_lens.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/lens/logit_lens.py)

    Inspect intermediate transformer states by projecting them through the model prediction head.

    `LogitLens` follows the idea introduced by nostalgebraist in
    [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens),
    and is closely related to the vocabulary-space analysis studied by
    [Geva et al., 2022](https://aclanthology.org/2022.emnlp-main.3/).

    In Interpreto, the method is integrated with `ModelWithSplitPoints` so that the same workflow
    can be used for causal language modeling, masked language modeling, and sequence classification.
    See the `ModelWithSplitPoints` documentation for split-point selection and activation extraction.
    Raw text inputs are tokenized internally by the lens methods with the wrapped tokenizer.
    The wrapped tokenizer should already expose a pad token or an eos token, since the lens methods
    do not resize model embeddings to introduce new special tokens.

    For sequence classification, the projection path falls into three cases:
    - a model-specific pooler or transform is resolved before a vector head
    - a sequence-aware classification head consumes the 3D hidden states directly
    - a bare vector head requires an explicit `pooling_strategy`

    Args:
        model_with_split_points (ModelWithSplitPoints): Wrapped model used to extract split activations.
        head_name (str | None): Optional path to the prediction head.
            If `None`, a short list of known paths is tried.
        pre_head_name (str | None): Optional path to a module applied before the head.
            This is useful for models exposing a pooler or a dedicated transformation block.
        pooling_strategy (Literal["cls", "mean", "last"] | None): Optional pooling used for
            sequence classification when the classification head expects one vector per sample.
            Pooling is only applied when it is explicitly requested for bare vector heads.
        top_k (int): Number of labels or tokens returned per prediction.
        device (torch.device | str | None): Device used by learned translators in subclasses.
            For `LogitLens`, the projection itself always runs on the wrapped model device.

    Examples:
        >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
        >>> from interpreto import LogitLens, ModelWithSplitPoints
        >>> model = AutoModelForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-bert")
        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     split_points="bert.encoder.layer.1.output",
        ... )
        >>> lens = LogitLens(model_with_split_points, top_k=3)
        >>> explanations = lens.explain("Interpreto is useful.")

    The `explain` method returns one dictionary entry per split point.
    Each entry contains two tensor fields:
    - `top_indices`: predicted token ids or label ids
    - `top_scores`: normalized scores associated with those ids

    Human-readable token or label decoding is handled by the notebook visualization helpers.
    For sequence classification, readable class names can be passed explicitly to `lens(..., label_names=...)`.

    Raises:
        ValueError: If the projection path cannot be resolved or if the output shape is incompatible.
        NotImplementedError: If the wrapped model is a token classification model.
        RuntimeError: If the resolved projection contains meta tensors.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        head_name: str | None = None,
        pre_head_name: str | None = None,
        pooling_strategy: PoolingStrategy | None = None,
        top_k: int = 5,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            model_with_split_points=model_with_split_points,
            head_name=head_name,
            pre_head_name=pre_head_name,
            pooling_strategy=pooling_strategy,
            top_k=top_k,
            device=device,
        )
