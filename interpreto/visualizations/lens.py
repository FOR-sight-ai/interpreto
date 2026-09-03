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

"""Lens visualization helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from html import escape

from IPython.display import HTML, display
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from interpreto.typing import LabelNames, LensResults, LensTopKOutput

from .commons import _build_html_header, _save_html

_LENS_STYLES = """
.lens-layer { margin: .8rem 0; border: 1px solid #d8dee9; border-radius: .4rem; }
.lens-layer summary { cursor: pointer; padding: .6rem; font-weight: 700; background: #f5f7fa; }
.lens-layer table { width: 100%; border-collapse: collapse; }
.lens-layer th, .lens-layer td { padding: .45rem .6rem; border-top: 1px solid #e2e8f0; text-align: left; }
.lens-token { white-space: pre; }
.lens-score { color: #526172; }
"""

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def _decode(tokenizer: Tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _predictions(output: LensTopKOutput, index: int, tokenizer: Tokenizer) -> str:
    indices = output["top_indices"][0, index].tolist()
    scores = output["top_scores"][0, index].tolist()
    return ", ".join(
        f"<span class='lens-token'>{escape(_decode(tokenizer, token_id))}</span> "
        f"<span class='lens-score'>({score:.3g})</span>"
        for token_id, score in zip(indices, scores, strict=True)
    )


def _render_language_model(results: LensResults, inputs: str, tokenizer: Tokenizer) -> str:
    token_ids = tokenizer.encode(inputs)
    sections = []
    for layer_name, output in results.items():
        rows = [
            "<tr><th>Input token</th><th>Top predictions</th></tr>",
            *(
                "<tr>"
                f"<td class='lens-token'>{escape(_decode(tokenizer, token_id))}</td>"
                f"<td>{_predictions(output, index, tokenizer)}</td>"
                "</tr>"
                for index, token_id in enumerate(token_ids)
            ),
        ]
        sections.append(
            "<details class='lens-layer' open>"
            f"<summary>{escape(layer_name)}</summary>"
            f"<table>{''.join(rows)}</table>"
            "</details>"
        )
    return "".join(sections)


def _label_name(index: int, label_names: LabelNames | None) -> str:
    if label_names is None:
        return str(index)
    if isinstance(label_names, Mapping):
        return str(label_names.get(index, label_names.get(str(index), index)))
    return str(label_names[index]) if index < len(label_names) else str(index)


def _render_classification(results: LensResults, inputs: str, label_names: LabelNames | None) -> str:
    sections = [f"<p>{escape(inputs)}</p>"]
    for layer_name, output in results.items():
        labels = output["top_indices"][0].tolist()
        scores = output["top_scores"][0].tolist()
        rows = "".join(
            f"<tr><td>{escape(_label_name(label, label_names))}</td><td class='lens-score'>{score:.3g}</td></tr>"
            for label, score in zip(labels, scores, strict=True)
        )
        sections.append(
            "<details class='lens-layer' open>"
            f"<summary>{escape(layer_name)}</summary>"
            f"<table><tr><th>Class</th><th>Score</th></tr>{rows}</table>"
            "</details>"
        )
    return "".join(sections)


def plot_lens(
    results: LensResults,
    inputs: str,
    *,
    tokenizer: Tokenizer,
    label_names: LabelNames | None = None,
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Display lens outputs and optionally save them as HTML.

    Args:
        results (LensResults): Output returned by `LogitLens.explain()` or `TunedLens.explain()`.
        inputs (str): Text used to produce `results`.
        tokenizer (Tokenizer): Tokenizer used by the lens splitter.
        label_names (LabelNames | None): Optional display names for classification labels.
        custom_css (str): Additional CSS appended to the visualization styles.
        save_path (str | os.PathLike[str] | None): Optional path for the rendered HTML.

    Returns:
        None: This function displays HTML and saves it when requested.

    Raises:
        ValueError: If `results` is empty or has an unsupported output shape.

    Examples:
        >>> results = lens.explain("Interpreto is useful.")
        >>> plot_lens(results, "Interpreto is useful.", tokenizer=tokenizer)
    """
    if not results:
        raise ValueError("`results` must contain at least one layer output.")

    output_rank = next(iter(results.values()))["top_indices"].ndim
    if output_rank == 3:
        body = _render_language_model(results, inputs, tokenizer)
    elif output_rank == 2:
        body = _render_classification(results, inputs, label_names)
    else:
        raise ValueError("Lens outputs must contain language-model or classification predictions.")

    html = _build_html_header(f"{_LENS_STYLES}\n{custom_css}", include_js=False) + body + "</body></html>"
    if save_path is not None:
        _save_html(html, save_path)
    display(HTML(html))
