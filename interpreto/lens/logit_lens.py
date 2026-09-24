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

"""Logit Lens implementation."""

from __future__ import annotations

from interpreto.concepts.splitters.model_with_split_points import ModelWithSplitPoints

from ._lens_base import BaseLens, PoolingStrategy


class LogitLens(BaseLens):
    """Code: [:octicons-mark-github-24: `lens/logit_lens.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/lens/logit_lens.py)

    Project a split activation through the model prediction head.

    `LogitLens` follows the idea introduced by nostalgebraist in
    [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens),
    and is closely related to the vocabulary-space analysis studied by
    [Geva et al., 2022](https://aclanthology.org/2022.emnlp-main.3/).

    For a standard Logit Lens, configure `ModelWithSplitPoints` at a transformer block output so
    the captured tensor is a residual-stream state. Projecting another compatible activation can
    still be useful, but does not have the same interpretation.

    Automatic projection is limited to model layouts whose final normalization, pooler, and output
    head are represented by reusable modules. Explicit paths can be supplied for other layouts when
    those modules faithfully reproduce the model's output path.

    Args:
        model_with_split_points (ModelWithSplitPoints): Wrapped model containing the split point to project.
        head_name (str | None): Optional path to the prediction head.
        pre_head_name (str | None): Optional path to a module applied before the prediction head.
        pooling_strategy (PoolingStrategy | None): Optional token pooling for a simple
            sequence-classification head.
        top_k (int): Number of token or class scores returned per prediction.

    Examples:
        >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
        >>> from interpreto import LogitLens, ModelWithSplitPoints
        >>> model = AutoModelForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-bert")
        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        >>> wrapped_model = ModelWithSplitPoints(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     split_point="bert.encoder.layer.1.output",
        ... )
        >>> lens = LogitLens(wrapped_model, top_k=3)
        >>> results = lens.explain("Interpreto is useful.")

    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        head_name: str | None = None,
        pre_head_name: str | None = None,
        pooling_strategy: PoolingStrategy | None = None,
        top_k: int = 5,
    ) -> None:
        super().__init__(
            model_with_split_points=model_with_split_points,
            head_name=head_name,
            pre_head_name=pre_head_name,
            pooling_strategy=pooling_strategy,
            top_k=top_k,
        )
