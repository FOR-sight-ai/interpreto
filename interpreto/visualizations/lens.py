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
from transformers import PreTrainedTokenizerBase

from interpreto.typing import LabelNames, LensResults, LensTopKOutput

from .commons import _build_html_header, _save_html

_LENS_STYLES = """
.lens-input { margin: 0 0 .5rem; }
.lens-legend { display: flex; align-items: center; gap: .4rem; margin: .25rem 0; font-size: .85em; }
.lens-gradient { width: 7rem; height: .65rem; background: linear-gradient(90deg, rgba(31, 119, 180, .15), rgb(31, 119, 180)); }
.lens-scroll { max-width: 100%; max-height: 36rem; overflow: auto; }
.lens-grid { display: grid; width: max-content; gap: 1px; padding: 1px; background: rgba(127, 127, 127, .35); }
.lens-grid > * { padding: .25rem .45rem; }
.lens-corner, .lens-header, .lens-layer-label { background: var(--background-color); font-weight: 600; }
.lens-corner { position: sticky; top: 0; left: 0; z-index: 3; }
.lens-header { position: sticky; top: 0; z-index: 2; text-align: center; white-space: pre; }
.lens-layer-label { position: sticky; left: 0; z-index: 1; white-space: nowrap; }
.lens-cell { min-width: 3.5rem; overflow: hidden; text-align: center; text-overflow: ellipsis; white-space: pre; }
.lens-cell:hover { outline: 2px solid var(--text-color); z-index: 1; }
"""


def _decode(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    token = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if token.isspace():
        return token.replace(" ", "␠").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")
    return token


def _score_bounds(results: LensResults) -> tuple[float, float]:
    scores = [
        float(score) for output in results.values() for score in output["top_scores"][..., 0].reshape(-1).tolist()
    ]
    return min(scores), max(scores)


def _cell(label: str, score: float, title: str, score_bounds: tuple[float, float]) -> str:
    minimum, maximum = score_bounds
    normalized = 0.6 if maximum == minimum else (score - minimum) / (maximum - minimum)
    intensity = 0.15 + 0.85 * min(max(normalized, 0.0), 1.0)
    text_color = "white" if intensity >= 0.55 else "var(--text-color)"
    return (
        "<div class='lens-cell highlighted-word-style' "
        f"style='background-color: rgba(31, 119, 180, {intensity:.3f}); color: {text_color}' "
        f"title='{escape(title, quote=True)}'>"
        f"<span class='lens-prediction'>{escape(label)}</span>"
        "</div>"
    )


def _layer_label(index: int, layer_name: str) -> str:
    label = "Input" if index == 0 else f"Layer {index - 1}"
    return f"<div class='lens-layer-label' title='{escape(layer_name, quote=True)}'>{label}</div>"


def _legend() -> str:
    return (
        "<div class='lens-legend'>"
        "<span>Relative confidence</span><span>low</span>"
        "<span class='lens-gradient'></span><span>high</span>"
        "</div>"
    )


def _language_prediction(
    output: LensTopKOutput,
    index: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[str, float, str]:
    indices = output["top_indices"][0, index].tolist()
    scores = output["top_scores"][0, index].tolist()
    labels = [_decode(tokenizer, token_id) for token_id in indices]
    title = "\n".join(f"{label}: {score:.3g}" for label, score in zip(labels, scores, strict=True))
    return labels[0], scores[0], title


def _render_language_model(results: LensResults, inputs: str, tokenizer: PreTrainedTokenizerBase) -> str:
    token_ids = tokenizer.encode(inputs)
    score_bounds = _score_bounds(results)
    cells = [
        _legend(),
        "<div class='lens-scroll'>",
        f"<div class='lens-grid' style='grid-template-columns: max-content repeat({len(token_ids)}, minmax(4rem, max-content))'>",
        "<div class='lens-corner'>Layer</div>",
        *(
            f"<div class='lens-header' title='{escape(_decode(tokenizer, token_id), quote=True)}'>"
            f"{escape(_decode(tokenizer, token_id))}</div>"
            for token_id in token_ids
        ),
    ]
    for layer_index, (layer_name, output) in reversed(list(enumerate(results.items()))):
        cells.append(_layer_label(layer_index, layer_name))
        for token_index in range(len(token_ids)):
            label, score, title = _language_prediction(output, token_index, tokenizer)
            cells.append(_cell(label, score, f"{layer_name}\n{title}", score_bounds))
    cells.extend(["</div>", "</div>"])
    return "".join(cells)


def _label_name(index: int, label_names: LabelNames | None) -> str:
    if label_names is None:
        return str(index)
    if isinstance(label_names, Mapping):
        return str(label_names.get(index, label_names.get(str(index), index)))
    return str(label_names[index]) if index < len(label_names) else str(index)


def _classification_prediction(
    output: LensTopKOutput,
    label_names: LabelNames | None,
) -> tuple[str, float, str]:
    indices = output["top_indices"][0].tolist()
    scores = output["top_scores"][0].tolist()
    labels = [_label_name(index, label_names) for index in indices]
    title = "\n".join(f"{label}: {score:.3g}" for label, score in zip(labels, scores, strict=True))
    return labels[0], scores[0], title


def _render_classification(results: LensResults, inputs: str, label_names: LabelNames | None) -> str:
    score_bounds = _score_bounds(results)
    cells = [
        f"<p class='lens-input'>{escape(inputs)}</p>",
        _legend(),
        "<div class='lens-scroll'>",
        "<div class='lens-grid' style='grid-template-columns: max-content minmax(8rem, max-content)'>",
        "<div class='lens-corner'>Layer</div><div class='lens-header'>Prediction</div>",
    ]
    for layer_index, (layer_name, output) in reversed(list(enumerate(results.items()))):
        label, score, title = _classification_prediction(output, label_names)
        cells.append(_layer_label(layer_index, layer_name))
        cells.append(_cell(label, score, f"{layer_name}\n{title}", score_bounds))
    cells.extend(["</div>", "</div>"])
    return "".join(cells)


def plot_lens(
    results: LensResults,
    inputs: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    label_names: LabelNames | None = None,
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Display model-depth predictions from the final layer to the input.

    Color intensity shows relative confidence. Hover over a cell to see its
    numerical score and the remaining top-k predictions.

    Args:
        results (LensResults): Output returned by `LogitLens.explain()` or `TunedLens.explain()`.
        inputs (str): Text used to produce `results`.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used by the lens splitter.
        label_names (LabelNames | None): Optional display names for classification labels.
        custom_css (str): Additional CSS appended to the visualization styles.
        save_path (str | os.PathLike[str] | None): Optional path for the rendered HTML.

    Returns:
        None: This function displays HTML and saves it when requested.

    Raises:
        ValueError: If `results` is empty or has an unsupported output shape.

    Examples:
        >>> results = lens.explain("Interpreto is useful.")
        >>> plot_lens(results, "Interpreto is useful.", tokenizer=splitter.tokenizer)
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
