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
from typing import Literal

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from interpreto.typing import LabelNames, LensResults

from .commons import _build_html_header, _save_html

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - optional notebook dependency
    HTML = None
    display = None

LensTask = Literal["language_model", "sequence_classification"]

_LENS_SHARED_STYLES = [
    ".lens-shell { color: #24313f; }",
    ".lens-layer { margin: 0 0 1rem 0; border: 1px solid #d8dee9; border-radius: 0.5rem; background: #ffffff; }",
    ".lens-layer summary { cursor: pointer; list-style: none; padding: 0.75rem 0.9rem; font-weight: 700; background: #f5f7fa; border-bottom: 1px solid #e2e8f0; }",
    ".lens-layer summary::-webkit-details-marker { display: none; }",
    ".lens-layer-body { padding: 0.9rem; }",
    ".lens-sample-card { margin: 0 0 0.9rem 0; padding: 0.85rem; border-radius: 0.4rem; border: 1px solid #eef2f7; }",
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
    ".lens-token:hover, .lens-token:focus { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(34, 56, 78, 0.15); background: #fffaf2; z-index: 12; outline: none; }",
    ".lens-token:focus-visible { outline: 2px solid #23567f; outline-offset: 2px; }",
    ".lens-token-label { white-space: pre; }",
    ".lens-tooltip { display: none; position: absolute; left: 0; top: calc(100% + 0.45rem); z-index: 20; min-width: 18rem; max-width: min(28rem, 80vw); padding: 0.75rem 0.85rem; border-radius: 14px; background: rgba(255, 255, 255, 0.98); border: 1px solid rgba(110, 130, 150, 0.25); box-shadow: 0 18px 48px rgba(28, 42, 56, 0.18); white-space: normal; }",
    ".lens-token:hover .lens-tooltip, .lens-token:focus .lens-tooltip { display: block; }",
    ".lens-tooltip-title { display: block; font-weight: 700; margin-bottom: 0.45rem; color: #24313f; }",
    ".lens-tooltip-list { display: block; margin: 0; padding: 0 0.2rem 0 0; max-height: 12rem; overflow-y: auto; scrollbar-width: thin; }",
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


def _decode_token_for_display(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    token_id: int,
) -> str:
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if isinstance(decoded, str) and decoded and "\ufffd" not in decoded:
        return decoded
    return f"[token {token_id}]"


def _score_to_color(score: float) -> str:
    bounded_score = min(max(float(score), 0.0), 1.0)
    red = int(241 - (113 * bounded_score))
    green = int(91 + (105 * bounded_score))
    blue = int(181 - (76 * bounded_score))
    return f"rgb({red}, {green}, {blue})"


def _validate_lens_plot_inputs(
    results: LensResults,
    model_inputs: BatchEncoding,
    task: LensTask,
) -> None:
    if task not in {"language_model", "sequence_classification"}:
        raise ValueError(f"Unsupported lens task: {task}.")
    if not isinstance(model_inputs, BatchEncoding):
        raise TypeError("`model_inputs` must be a tensor-backed BatchEncoding.")
    if not isinstance(results, Mapping):
        raise TypeError("`results` must be a mapping of split points to lens outputs.")
    if not results:
        raise ValueError("`results` must contain at least one split-point output.")

    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("`model_inputs['input_ids']` must be a tensor.")
    if input_ids.ndim != 2 or 0 in input_ids.shape:
        raise ValueError("`model_inputs['input_ids']` must have a nonempty 2D shape.")
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("`model_inputs['attention_mask']` must be a tensor.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("`attention_mask` must have the same shape as `input_ids`.")

    expected_ndim = 3 if task == "language_model" else 2
    for split_point, split_results in results.items():
        if not isinstance(split_point, str):
            raise TypeError("Lens result keys must be split-point strings.")
        if not isinstance(split_results, Mapping):
            raise TypeError(f"Lens output for `{split_point}` must be a mapping.")
        top_indices = split_results.get("top_indices")
        top_scores = split_results.get("top_scores")
        if not isinstance(top_indices, torch.Tensor) or not isinstance(top_scores, torch.Tensor):
            raise TypeError(f"Lens output for `{split_point}` must contain tensor top-k values.")
        if top_indices.shape != top_scores.shape:
            raise ValueError(f"Top-k indices and scores for `{split_point}` must have matching shapes.")
        if top_indices.ndim != expected_ndim or top_indices.shape[-1] == 0:
            raise ValueError(f"Lens output for `{split_point}` must be a nonempty {expected_ndim}D tensor pair.")
        if top_indices.shape[0] != input_ids.shape[0]:
            raise ValueError(f"Lens output for `{split_point}` does not match the input batch size.")
        if task == "language_model" and top_indices.shape[1] != input_ids.shape[1]:
            raise ValueError(f"Lens output for `{split_point}` does not match the input sequence length.")
        if top_indices.is_floating_point() or top_indices.is_complex() or top_indices.dtype == torch.bool:
            raise TypeError(f"Top-k indices for `{split_point}` must use an integer tensor dtype.")
        if not top_scores.is_floating_point():
            raise TypeError(f"Top-k scores for `{split_point}` must use a floating-point tensor dtype.")
        if not torch.isfinite(top_scores).all() or torch.any((top_scores < 0) | (top_scores > 1)):
            raise ValueError(f"Top-k scores for `{split_point}` must be finite values between zero and one.")


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
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    if not isinstance(decoded, str):
        raise TypeError("The tokenizer must decode one token sequence to a string.")
    return decoded


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
    visible_token_indices = _get_visible_token_indices(model_inputs)

    sections: list[str] = [
        "<div class='lens-shell'>",
    ]

    for split_point, split_results in results.items():
        sections.append("<details class='lens-layer' open>")
        sections.append(f"<summary>{_escape_html(split_point)}</summary>")
        sections.append("<div class='lens-layer-body'>")
        sections.append(
            "<p class='lens-layer-subtitle'>"
            "Hover or focus a token to inspect vocabulary scores decoded from that position."
            "</p>"
        )
        for sample_index in range(model_inputs["input_ids"].shape[0]):
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
                token_id = int(model_inputs["input_ids"][sample_index, token_index])
                top_indices = split_results["top_indices"][sample_index, token_index]
                top_scores = split_results["top_scores"][sample_index, token_index].tolist()
                token_label = _escape_html(_decode_token_for_display(tokenizer, token_id))
                tooltip_rows = []
                for predicted_token_id, score in zip(top_indices.tolist(), top_scores, strict=True):
                    display_token = _escape_html(_decode_token_for_display(tokenizer, predicted_token_id))
                    tooltip_rows.append(
                        "<span class='lens-tooltip-item'>"
                        f"<span class='lens-tooltip-token'>{display_token}</span>"
                        f"<span class='lens-tooltip-score'>{_format_score(score)}</span>"
                        "</span>"
                    )

                sections.append(
                    "<span class='lens-token' tabindex='0'>"
                    f"<span class='lens-token-label'>{token_label}</span>"
                    "<span class='lens-tooltip' role='tooltip'>"
                    "<span class='lens-tooltip-title'>Top vocabulary scores</span>"
                    "<span class='lens-tooltip-list'>" + "".join(tooltip_rows) + "</span></span></span>"
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
                f"<div class='lens-top-choice'>Current top class: {top_label} ({_format_score(top_score)})</div>"
            )

            for label, score in zip(decoded_labels, split_results["top_scores"][sample_index].tolist(), strict=True):
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


def _render_lens_results_html(
    results: LensResults,
    model_inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    task: LensTask,
    label_names: LabelNames | None = None,
) -> str:
    if task == "language_model":
        return _render_language_model_html(results, model_inputs, tokenizer)

    if task == "sequence_classification":
        return _render_sequence_classification_html(results, model_inputs, tokenizer, label_names)

    raise ValueError(f"Unsupported lens task: {task}.")


def plot_lens(
    results: LensResults,
    model_inputs: BatchEncoding,
    *,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    task: LensTask,
    label_names: LabelNames | None = None,
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Display lens outputs and optionally save them as HTML.

    Args:
        results (LensResults): Output returned by `LogitLens.explain()` or `TunedLens.explain()`.
        model_inputs (BatchEncoding): Tokenized inputs corresponding to `results`.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer used to decode
            tokens for display.
        task (LensTask): Lens task used to select the appropriate renderer.
        label_names (LabelNames | None): Optional display names for sequence-classification labels.
            If `None`, raw label ids are shown.
        custom_css (str): Additional CSS appended to the visualization styles.
        save_path (str | os.PathLike[str] | None): Optional path for the rendered HTML.

    Returns:
        None: This function displays HTML when IPython is available and saves it when requested.

    Examples:
        >>> results = lens.explain(batch_encoding)
        >>> plot_lens(results, batch_encoding, tokenizer=tokenizer, task=lens.task)
    """
    _validate_lens_plot_inputs(results, model_inputs, task)
    body = _render_lens_results_html(
        results,
        model_inputs,
        tokenizer=tokenizer,
        task=task,
        label_names=label_names,
    )
    task_styles = _LANGUAGE_MODEL_STYLES if task == "language_model" else _SEQUENCE_CLASSIFICATION_STYLES
    lens_css = "\n".join(_LENS_SHARED_STYLES + task_styles)
    if custom_css:
        lens_css += f"\n{custom_css}"
    html = _build_html_header(lens_css, include_js=False) + body + "\n</body></html>\n"
    if save_path is not None:
        _save_html(html, save_path)
    if HTML is not None and display is not None:
        display(HTML(html))
