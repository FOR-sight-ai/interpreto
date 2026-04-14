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
Lens visualization helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from interpreto.typing import LabelNames, LensResults

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - optional notebook dependency
    HTML = None
    display = None

LensTask = Literal["language_model", "sequence_classification"]

_LENS_SHARED_STYLES = [
    ".lens-shell { font-family: 'Avenir Next', 'Segoe UI', sans-serif; color: #24313f; }",
    ".lens-layer { margin: 0 0 1rem 0; border: 1px solid #d8dee9; border-radius: 18px; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); box-shadow: 0 14px 40px rgba(30, 42, 56, 0.08); }",
    ".lens-layer summary { cursor: pointer; list-style: none; padding: 0.95rem 1.1rem; font-weight: 700; background: linear-gradient(90deg, #e8f2ff 0%, #f9fbff 100%); border-bottom: 1px solid #e2e8f0; }",
    ".lens-layer summary::-webkit-details-marker { display: none; }",
    ".lens-layer-body { padding: 1rem 1.1rem 1.15rem 1.1rem; }",
    ".lens-sample-card { margin: 0 0 1rem 0; padding: 0.95rem; border-radius: 16px; background: #fffdfb; border: 1px solid #eef2f7; }",
]

_LANGUAGE_MODEL_STYLES = [
    ".lens-layer { overflow: visible; }",
    ".lens-layer-body { overflow: visible; }",
    ".lens-layer-subtitle { margin: 0 0 0.8rem 0; color: #526172; font-size: 0.95rem; }",
    ".lens-sample-header { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; margin-bottom: 0.7rem; }",
    ".lens-sample-title { font-weight: 700; color: #253649; }",
    ".lens-sample-text { color: #627386; font-size: 0.92rem; }",
    ".lens-token-stream { display: flex; flex-wrap: wrap; gap: 0.15rem; align-items: flex-start; line-height: 1.85; }",
    ".lens-token { position: relative; display: inline-flex; align-items: center; border-radius: 999px; padding: 0.1rem 0.48rem; margin: 0.04rem 0; border: 1px solid rgba(72, 102, 132, 0.12); background: #ffffff; white-space: pre; cursor: pointer; transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease; z-index: 0; }",
    ".lens-token:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(34, 56, 78, 0.15); background: #fffaf2; z-index: 12; }",
    ".lens-token-label { white-space: pre; }",
    ".lens-token-confidence { width: 0.5rem; height: 0.5rem; border-radius: 999px; margin-left: 0.4rem; flex: 0 0 auto; }",
    ".lens-tooltip { display: none; position: absolute; left: 0; top: calc(100% + 0.45rem); z-index: 20; min-width: 18rem; max-width: min(28rem, 80vw); padding: 0.75rem 0.85rem; border-radius: 14px; background: rgba(255, 255, 255, 0.98); border: 1px solid rgba(110, 130, 150, 0.25); box-shadow: 0 18px 48px rgba(28, 42, 56, 0.18); }",
    ".lens-token:hover .lens-tooltip { display: block; }",
    ".lens-tooltip-title { font-weight: 700; margin-bottom: 0.45rem; color: #24313f; }",
    ".lens-tooltip-list { margin: 0; padding: 0 0.2rem 0 0; list-style: none; max-height: 12rem; overflow-y: auto; scrollbar-width: thin; }",
    ".lens-tooltip-item { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.75rem; align-items: start; padding: 0.18rem 0; }",
    ".lens-tooltip-token { font-weight: 700; white-space: pre-wrap; word-break: break-word; }",
    ".lens-tooltip-score { color: #546577; font-variant-numeric: tabular-nums; }",
]

_SEQUENCE_CLASSIFICATION_STYLES = [
    ".lens-layer { overflow: hidden; }",
    ".lens-sample-title { font-weight: 700; color: #253649; margin-bottom: 0.2rem; }",
    ".lens-sample-text { color: #627386; font-size: 0.92rem; margin-bottom: 0.8rem; }",
    ".lens-prediction-row { display: grid; grid-template-columns: minmax(7rem, auto) 1fr auto; gap: 0.8rem; align-items: center; margin: 0.45rem 0; }",
    ".lens-prediction-label { font-weight: 700; }",
    ".lens-prediction-bar { height: 0.72rem; border-radius: 999px; background: #ecf1f6; overflow: hidden; }",
    ".lens-prediction-fill { height: 100%; border-radius: 999px; }",
    ".lens-prediction-score { color: #546577; font-variant-numeric: tabular-nums; }",
    ".lens-top-choice { padding: 0.22rem 0.55rem; border-radius: 999px; background: #eef7ff; color: #23567f; font-size: 0.82rem; font-weight: 700; margin-bottom: 0.7rem; display: inline-flex; }",
]


def _build_style_block(extra_styles: list[str]) -> str:
    return "<style>" + "".join(_LENS_SHARED_STYLES + extra_styles) + "</style>"


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _format_score(score: float) -> str:
    score_value = float(score)
    if score_value >= 0.1:
        return f"{score_value:.3f}"
    if score_value >= 0.01:
        return f"{score_value:.4f}"
    if score_value >= 0.001:
        return f"{score_value:.5f}"
    return f"{score_value:.2e}"


def _format_token_for_display(token: str) -> str:
    cleaned = str(token)
    cleaned = cleaned.replace("Ġ", " ")
    cleaned = cleaned.replace("▁", " ")
    cleaned = cleaned.replace("</w>", "")
    return cleaned if cleaned != "" else str(token)


def _score_to_color(score: float) -> str:
    bounded_score = min(max(float(score), 0.0), 1.0)
    red = int(241 - (113 * bounded_score))
    green = int(91 + (105 * bounded_score))
    blue = int(181 - (76 * bounded_score))
    return f"rgb({red}, {green}, {blue})"


def _get_visible_token_indices(model_inputs: BatchEncoding) -> list[list[int]]:
    if "attention_mask" not in model_inputs:
        return [list(range(model_inputs["input_ids"].shape[1])) for _ in range(model_inputs["input_ids"].shape[0])]

    visible_indices: list[list[int]] = []
    for mask in model_inputs["attention_mask"]:
        visible_indices.append([index for index, keep in enumerate(mask.detach().cpu().tolist()) if keep == 1])
    return visible_indices


def _decode_sample_text(
    model_inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    sample_index: int,
) -> str:
    visible_indices = _get_visible_token_indices(model_inputs)[sample_index]
    token_ids = model_inputs["input_ids"][sample_index, visible_indices].detach().cpu().tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _resolve_label_name(index: int, label_names: LabelNames | None) -> str:
    if label_names is None:
        return str(index)

    if isinstance(label_names, Mapping):
        return str(label_names.get(index, label_names.get(str(index), index)))

    if 0 <= index < len(label_names):
        return str(label_names[index])

    return str(index)


def _render_language_model_html(
    results: LensResults,
    model_inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> str:
    raw_tokens = [tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().tolist()) for input_ids in model_inputs["input_ids"]]
    visible_token_indices = _get_visible_token_indices(model_inputs)

    sections: list[str] = [
        "<div class='lens-shell'>",
        _build_style_block(_LANGUAGE_MODEL_STYLES),
    ]

    for split_point, split_results in results.items():
        sections.append("<details class='lens-layer' open>")
        sections.append(f"<summary>{_escape_html(split_point)}</summary>")
        sections.append("<div class='lens-layer-body'>")
        sections.append("<p class='lens-layer-subtitle'>Hover a token to see what this layer currently predicts.</p>")
        for sample_index, sample_tokens in enumerate(raw_tokens):
            sample_text = _escape_html(_decode_sample_text(model_inputs, tokenizer, sample_index))
            sections.append("<div class='lens-sample-card'>")
            sections.append(
                "<div class='lens-sample-header'>"
                f"<span class='lens-sample-title'>Sample {sample_index + 1}</span>"
                f"<span class='lens-sample-text'>{sample_text}</span>"
                "</div>"
            )
            sections.append("<div class='lens-token-stream'>")

            for token_index in visible_token_indices[sample_index]:
                token = sample_tokens[token_index]
                top_indices = split_results["top_indices"][sample_index, token_index]
                top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
                top_scores = split_results["top_scores"][sample_index, token_index].tolist()
                token_label = _escape_html(_format_token_for_display(token))
                confidence_color = _score_to_color(top_scores[0])
                tooltip_rows = []
                for predicted_token, score in zip(top_tokens, top_scores, strict=False):
                    display_token = _escape_html(_format_token_for_display(str(predicted_token)))
                    tooltip_rows.append(
                        "<li class='lens-tooltip-item'>"
                        f"<span class='lens-tooltip-token' style='color: {_score_to_color(score)};'>{display_token}</span>"
                        f"<span class='lens-tooltip-score'>{_format_score(score)}</span>"
                        "</li>"
                    )

                sections.append(
                    "<span class='lens-token'>"
                    f"<span class='lens-token-label'>{token_label}</span>"
                    f"<span class='lens-token-confidence' style='background: {confidence_color};'></span>"
                    "<span class='lens-tooltip'>"
                    "<div class='lens-tooltip-title'>Top-k predictions</div>"
                    "<ul class='lens-tooltip-list'>"
                    + "".join(tooltip_rows)
                    + "</ul></span></span>"
                )

            sections.append("</div></div>")

        sections.append("</div></details>")

    sections.append("</div>")
    return "".join(sections)


def _render_sequence_classification_html(
    results: LensResults,
    model_inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    label_names: LabelNames | None,
) -> str:
    sections: list[str] = [
        "<div class='lens-shell'>",
        _build_style_block(_SEQUENCE_CLASSIFICATION_STYLES),
    ]

    for split_point, split_results in results.items():
        sections.append("<details class='lens-layer' open>")
        sections.append(f"<summary>{_escape_html(split_point)}</summary>")
        sections.append("<div class='lens-layer-body'>")
        for sample_index, label_row in enumerate(split_results["top_indices"].detach().cpu().tolist()):
            sample_text = _escape_html(_decode_sample_text(model_inputs, tokenizer, sample_index))
            decoded_labels = [_resolve_label_name(index, label_names) for index in label_row]
            top_label = _escape_html(decoded_labels[0])
            top_score = float(split_results["top_scores"][sample_index, 0])
            sections.append("<div class='lens-sample-card'>")
            sections.append(f"<div class='lens-sample-title'>Sample {sample_index + 1}</div>")
            sections.append(f"<div class='lens-sample-text'>{sample_text}</div>")
            sections.append(
                "<div class='lens-top-choice'>"
                f"Current top class: {top_label} ({_format_score(top_score)})"
                "</div>"
            )

            for label, score in zip(decoded_labels, split_results["top_scores"][sample_index].tolist(), strict=False):
                fill_color = _score_to_color(score)
                sections.append(
                    "<div class='lens-prediction-row'>"
                    f"<span class='lens-prediction-label'>{_escape_html(str(label))}</span>"
                    "<span class='lens-prediction-bar'>"
                    f"<span class='lens-prediction-fill' style='width: {score * 100:.2f}%; background: {fill_color};'></span>"
                    "</span>"
                    f"<span class='lens-prediction-score'>{_format_score(score)}</span>"
                    "</div>"
                )

            sections.append("</div>")

        sections.append("</div></details>")

    sections.append("</div>")
    return "".join(sections)


def render_lens_results(
    results: LensResults,
    model_inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    task: LensTask,
    label_names: LabelNames | None = None,
) -> str:
    """
    Render lens outputs into notebook-friendly HTML.

    Args:
        results (dict[str, LensTopKOutput]): Output returned by `LogitLens.explain()` or
            `TunedLens.explain()`.
        model_inputs (BatchEncoding): Tokenized inputs corresponding to `results`.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer used to decode
            input tokens and language-model predictions.
        task (Literal["language_model", "sequence_classification"]): Lens task used to select
            the appropriate renderer.
        label_names (Mapping[int | str, str] | list[str] | tuple[str, ...] | None): Optional
            display names for sequence-classification labels. If `None`, raw label ids are shown.

    Returns:
        str: HTML string ready to be displayed in a notebook.

    Examples:
        >>> results = lens.explain("Interpreto is helpful.")
        >>> model_inputs = tokenizer(["Interpreto is helpful."], return_tensors="pt", padding=True)
        >>> html = render_lens_results(
        ...     results,
        ...     model_inputs,
        ...     tokenizer=tokenizer,
        ...     task=lens.task,
        ... )
    """
    if task == "language_model":
        return _render_language_model_html(results, model_inputs, tokenizer)

    if task == "sequence_classification":
        return _render_sequence_classification_html(results, model_inputs, tokenizer, label_names)

    raise ValueError(f"Unsupported lens task: {task}.")


def display_lens_results(
    results: LensResults,
    model_inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    task: LensTask,
    label_names: LabelNames | None = None,
) -> None:
    """
    Display lens outputs in a notebook.

    Args:
        results (dict[str, LensTopKOutput]): Output returned by `LogitLens.explain()` or
            `TunedLens.explain()`.
        model_inputs (BatchEncoding): Tokenized inputs corresponding to `results`.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer used to decode
            tokens for display.
        task (Literal["language_model", "sequence_classification"]): Lens task used to select
            the appropriate renderer.
        label_names (Mapping[int | str, str] | list[str] | tuple[str, ...] | None): Optional
            display names for sequence-classification labels. If `None`, raw label ids are shown.

    Examples:
        >>> results = lens.explain(batch_encoding)
        >>> display_lens_results(results, batch_encoding, tokenizer=tokenizer, task=lens.task)
    """
    if HTML is None or display is None:
        return

    html = render_lens_results(
        results,
        model_inputs,
        tokenizer=tokenizer,
        task=task,
        label_names=label_names,
    )
    display(HTML(html))
