from __future__ import annotations

import json
import math
import os
import uuid
from typing import Any

import numpy as np
import torch
from IPython.display import HTML, display

from .commons import (
    _build_default_colormap,
    _build_html_header,
    _normalize_onclick_colormap,
    _save_html,
)


def _to_list(values: Any) -> list[Any]:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    raise TypeError(f"Unsupported type for list conversion: {type(values).__name__}")


def _coerce_importances(values: Any) -> list[float]:
    raw_values = _to_list(values)
    if any(isinstance(value, list) for value in raw_values):
        raise ValueError("Concept importances must be a 1D list of numbers.")
    normalized = []
    for raw_value in raw_values:
        normalized_value = float(raw_value)
        if not math.isfinite(normalized_value):
            normalized_value = 0.0
        normalized.append(normalized_value)
    return normalized


def _normalize_label(label: Any) -> str | list[str]:
    if isinstance(label, (list, tuple)):
        return [str(item) for item in label]
    return str(label)


def _normalize_label_map(label_map: dict[Any, Any]) -> dict[int, str | list[str]]:
    normalized = {}
    for key, label in label_map.items():
        normalized[int(key)] = _normalize_label(label)
    return normalized


def _normalize_concepts_labels(
    concepts_labels: list[Any] | tuple[Any, ...] | dict[Any, Any],
    nb_concepts: int,
) -> list[str | list[str]]:
    if isinstance(concepts_labels, dict):
        label_map = _normalize_label_map(concepts_labels)
    elif isinstance(concepts_labels, (list, tuple)):
        if nb_concepts and len(concepts_labels) > nb_concepts:
            raise ValueError("Concept labels length must not exceed the number of concepts in activations.")
        label_map = {index: _normalize_label(label) for index, label in enumerate(concepts_labels)}
    else:
        raise TypeError("concepts_labels must be a list, tuple, or dict.")

    labels = []
    for index in range(nb_concepts):
        labels.append(label_map.get(index, f"Concept #{index}"))
    return labels


def _coerce_matrix(
    values: Any,
    expected_rows: int,
    *,
    name: str,
    row_count_error: str,
) -> list[list[float]]:
    matrix = _to_list(values)
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be a 2D list or tensor.")
    if len(matrix) != expected_rows:
        raise ValueError(row_count_error)
    if matrix and not isinstance(matrix[0], (list, tuple)):
        raise ValueError(f"{name} must be a 2D list or tensor.")

    normalized = []
    row_length = None
    for row in matrix:
        row_values = list(row) if isinstance(row, tuple) else row
        if not isinstance(row_values, list):
            raise ValueError(f"{name} must be a 2D list or tensor.")
        if row_length is None:
            row_length = len(row_values)
        elif len(row_values) != row_length:
            raise ValueError(f"{name} rows must have consistent length.")

        normalized_row = []
        for raw_value in row_values:
            normalized_value = float(raw_value)
            if not math.isfinite(normalized_value):
                normalized_value = 0.0
            normalized_row.append(normalized_value)
        normalized.append(normalized_row)

    return normalized


def _coerce_activation_matrix(values: Any, expected_rows: int) -> list[list[float]]:
    return _coerce_matrix(
        values,
        expected_rows,
        name="concepts_activations",
        row_count_error="concepts_activations must have one row per sample element.",
    )


def _coerce_importances_matrix(values: Any, expected_rows: int) -> list[list[float]]:
    return _coerce_matrix(
        values,
        expected_rows,
        name="concepts_importances",
        row_count_error="concepts_importances must have one row per output element.",
    )


def _resolve_class_entries(
    classes_names: list[str] | dict[int, str] | None,
    class_keys: list[Any],
) -> list[tuple[Any, str]]:
    def _default_name(class_key: Any) -> str:
        if isinstance(class_key, int):
            return f"Class #{class_key}"
        return str(class_key)

    used_keys: set[Any] = set()
    entries: list[tuple[Any, str]] = []

    if classes_names is None:
        for key in class_keys:
            entries.append((key, _default_name(key)))
        return entries

    if isinstance(classes_names, dict):
        for key, name in classes_names.items():
            if key in class_keys:
                display_name = _default_name(key) if name is None else str(name)
                entries.append((key, display_name))
                used_keys.add(key)
            elif name in class_keys:
                display_name = _default_name(name) if name is None else str(name)
                entries.append((name, display_name))
                used_keys.add(name)
    elif isinstance(classes_names, (list, tuple)):
        for index, name in enumerate(classes_names):
            if index in class_keys:
                display_name = _default_name(index) if name is None else str(name)
                entries.append((index, display_name))
                used_keys.add(index)
            elif name in class_keys:
                display_name = _default_name(name) if name is None else str(name)
                entries.append((name, display_name))
                used_keys.add(name)
    else:
        raise TypeError("classes_names must be a list, dict, or None.")

    for key in class_keys:
        if key in used_keys:
            continue
        entries.append((key, _default_name(key)))

    return entries


def _coerce_class_importances(
    concepts_importances: dict[Any, Any] | torch.Tensor,
) -> dict[Any, Any]:
    if isinstance(concepts_importances, dict):
        return concepts_importances
    if isinstance(concepts_importances, torch.Tensor):
        tensor = concepts_importances.detach().cpu()
        if tensor.ndim == 1:
            return {0: tensor.tolist()}
        if tensor.ndim == 2:
            return {index: tensor[index].tolist() for index in range(tensor.shape[0])}
        raise ValueError("concepts_importances tensor must be 1D or 2D.")
    raise TypeError("concepts_importances must be a dict or torch.Tensor.")


def _coerce_class_importances_local(
    concepts_importances: Any,
) -> tuple[list[Any], dict[Any, list[float]]]:
    if isinstance(concepts_importances, dict):
        if not concepts_importances:
            return [], {}
        importances_map: dict[Any, list[float]] = {}
        expected_len = None
        for key, values in concepts_importances.items():
            row = _coerce_importances(values)
            if expected_len is None:
                expected_len = len(row)
            elif len(row) != expected_len:
                raise ValueError("concepts_importances rows must have consistent length.")
            importances_map[key] = row
        return list(concepts_importances.keys()), importances_map

    if isinstance(concepts_importances, torch.Tensor):
        importances = _coerce_class_importances(concepts_importances)
        if not importances:
            return [], {}
        importances_map: dict[Any, list[float]] = {}
        expected_len = None
        for key, values in importances.items():
            row = _coerce_importances(values)
            if expected_len is None:
                expected_len = len(row)
            elif len(row) != expected_len:
                raise ValueError("concepts_importances rows must have consistent length.")
            importances_map[key] = row
        return list(importances.keys()), importances_map

    raw_importances = _to_list(concepts_importances)
    if not isinstance(raw_importances, list):
        raise TypeError("concepts_importances must be a 2D list, dict, or tensor.")
    importances_matrix = _coerce_importances_matrix(raw_importances, len(raw_importances))
    importances_map = dict(enumerate(importances_matrix))
    return list(importances_map.keys()), importances_map


def _is_classwise_concepts(
    concepts_labels: dict[Any, Any],
    class_key_set: set[Any],
) -> bool:
    if not concepts_labels:
        return False

    def _key_matches_class(key: Any) -> bool:
        if key in class_key_set:
            return True
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            key_int = int(key)
            return key_int in class_key_set or str(key_int) in class_key_set
        if isinstance(key, str):
            try:
                return int(key) in class_key_set
            except ValueError:
                return False
        return False

    return all(_key_matches_class(key) and isinstance(value, dict) for key, value in concepts_labels.items())


def _resolve_classwise_label_map(
    concepts_labels: dict[Any, Any],
    class_key: Any,
    class_name: Any,
    class_index: int,
) -> dict[Any, Any]:
    for candidate in (
        class_key,
        class_name,
        class_index,
        str(class_key),
        str(class_index),
    ):
        if candidate in concepts_labels:
            return concepts_labels.get(candidate) or {}
    return {}


def _resolve_class_importances_row(
    importances: dict[Any, list[float]],
    class_key: Any,
    class_name: Any,
) -> list[float] | None:
    if class_key in importances:
        return importances[class_key]
    if class_name in importances:
        return importances[class_name]
    if isinstance(class_key, (int, np.integer)) and not isinstance(class_key, bool):
        key_int = int(class_key)
        key_str = str(key_int)
        if key_str in importances:
            return importances[key_str]
    if isinstance(class_key, str):
        try:
            key_int = int(class_key)
        except ValueError:
            key_int = None
        if key_int is not None and key_int in importances:
            return importances[key_int]
    return None


def _resolve_class_activations_matrix(
    activations: dict[Any, list[list[float]]],
    class_key: Any,
    class_name: Any,
    class_index: int,
) -> list[list[float]] | None:
    for candidate in (
        class_key,
        class_name,
        class_index,
        str(class_key),
        str(class_index),
    ):
        if candidate in activations:
            return activations[candidate]
    return None


def _coerce_class_activations_local(
    concepts_activations: Any,
    sample_length: int,
) -> tuple[list[list[float]], dict[Any, list[list[float]]] | None]:
    if isinstance(concepts_activations, dict):
        if not concepts_activations:
            return [], {}
        activations_map: dict[Any, list[list[float]]] = {}
        expected_rows = None
        expected_cols = None
        for key, values in concepts_activations.items():
            matrix = _to_list(values)
            if not isinstance(matrix, list):
                raise TypeError("concepts_activations must be a 2D list or tensor.")
            if len(matrix) not in (1, sample_length):
                raise ValueError("concepts_activations must have one row or one row per sample element.")
            normalized = _coerce_matrix(
                matrix,
                len(matrix),
                name="concepts_activations",
                row_count_error=("concepts_activations must have one row or one row per sample element."),
            )
            if expected_rows is None:
                expected_rows = len(normalized)
            elif len(normalized) != expected_rows:
                raise ValueError("concepts_activations rows must have consistent length.")
            cols = len(normalized[0]) if normalized else 0
            if expected_cols is None:
                expected_cols = cols
            elif cols != expected_cols:
                raise ValueError("concepts_activations rows must have consistent length.")
            activations_map[key] = normalized

        aggregated = []
        if expected_rows and expected_cols is not None:
            aggregated = [[0.0] * expected_cols for _ in range(expected_rows)]
            for matrix in activations_map.values():
                for row_index in range(expected_rows):
                    row = matrix[row_index]
                    for concept_id in range(expected_cols):
                        aggregated[row_index][concept_id] += abs(row[concept_id])

        return aggregated, activations_map

    raw_activations = _to_list(concepts_activations)
    if not isinstance(raw_activations, list):
        raise TypeError("concepts_activations must be a 2D list or tensor.")
    if len(raw_activations) not in (1, sample_length):
        raise ValueError("concepts_activations must have one row or one row per sample element.")
    activations = _coerce_matrix(
        raw_activations,
        len(raw_activations),
        name="concepts_activations",
        row_count_error=("concepts_activations must have one row or one row per sample element."),
    )
    return activations, None


def _render_visualization_html(
    body: str,
    *,
    custom_css: str,
    save_path: str | os.PathLike[str] | None,
) -> None:
    html = _build_html_header(custom_css) + body + "\n</body></html>\n"
    if save_path:
        _save_html(html, save_path)
    display(HTML(html))


def plot_concepts(
    *,
    concepts_importances: Any,
    concepts_labels: list[str | list[str]] | tuple[str | list[str], ...] | dict[Any, Any],
    sample: list[str] | None = None,
    classes_names: list[str] | dict[int, str] | None = None,
    concepts_activations: Any | None = None,
    top_k: int | None = 10,
    concept_color: str = "#f39c12",
    default_colormap: dict[int, str] | None = None,
    onclick_colormap: tuple[str, str] | list[str] | None = None,
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """
    Plot concept importance and activation visualizations for classification or generation tasks.

    The mode is inferred from the inputs:
    - classification_global: sample is None and concepts_activations is None.
    - classification_local: sample and concepts_activations provided, classes_names provided.
    - generation_local: sample and concepts_activations provided, classes_names omitted.

    Args:
        concepts_importances: Concept importance scores.
            - classification_global: tensor or dict {class_id/class_name: [scores]}.
            - classification_local: tensor/list/dict with shape (nb_classes, nb_concepts) or
              {class_id/class_name: [scores]}.
            - generation_local: tensor/list with shape (nb_outputs, nb_concepts).
        concepts_labels: Concept labels.
            - classification_global: dict {concept_id: label} or
              {class_id/class_name: {concept_id: label}}.
            - classification_local: list/tuple/dict of labels or class-wise dict as above.
            - generation_local: list/tuple/dict of labels.
        sample: Required for local modes. Full sequence tokens (inputs + outputs for generation).
        classes_names: Optional class display names. If omitted, the plot defaults to generation_local.
        concepts_activations: Token-level concept activations.
            - classification_local: tensor/list with shape (len(sample), nb_concepts) or
              (1, nb_concepts); dicts may be keyed by class id/name.
            - generation_local: tensor/list with shape (len(sample), nb_concepts).
        top_k: Number of concepts shown in classification lists, and for generation_local when no
            output token is selected.
        concept_color: Default concept color for ranking bars.
        default_colormap: Optional {concept_id: color} override for concept colors.
        onclick_colormap: (selected_color, hover_color) for active labels.
        custom_css: Additional CSS injected into the HTML visualization.
        save_path: Optional path to save the HTML visualization.

    Raises:
        ValueError: If sample and concepts_activations are not provided together, or top_k is invalid.

    Examples:
        >>> # classification global (from docs/notebooks/classification_demonstration.ipynb)
        >>> plot_concepts(
        ...     classes_names=classes_names,
        ...     concepts_importances=mean_gradients,
        ...     concepts_labels={k: list(v.keys()) for k, v in topk_words.items()},
        ... )
        >>>
        >>> # classification local (from docs/notebooks/classification_concept_tutorial.ipynb)
        >>> plot_concepts(
        ...     sample=[example],
        ...     classes_names=classes_names,
        ...     concepts_activations=concepts_activations,
        ...     concepts_importances=local_importance.squeeze(),
        ...     concepts_labels=concept_interpretations,
        ... )
        >>>
        >>> # generation local (from docs/notebooks/generation_concept_tutorial.ipynb)
        >>> plot_concepts(
        ...     concepts_activations=concepts_activations,
        ...     concepts_importances=local_importances,
        ...     concepts_labels=interpretations,
        ...     sample=sample_tokens,
        ... )
    """

    has_sample = sample is not None
    has_activations = concepts_activations is not None
    resolved_top_k = top_k if top_k is not None else 10
    if not isinstance(resolved_top_k, int) or resolved_top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if not has_sample and not has_activations:
        plot_classification_global_concepts(
            classes_names,
            concepts_labels,  # type: ignore
            concepts_importances,
            top_k=resolved_top_k,
            concept_color=concept_color,
            default_colormap=default_colormap,
            onclick_colormap=onclick_colormap,
            custom_css=custom_css,
            save_path=save_path,
        )
        return

    if has_sample != has_activations:
        raise ValueError("sample and concepts_activations must be provided together for local modes.")

    if classes_names is not None:
        plot_classification_local_concepts(
            sample,  # type: ignore
            classes_names,
            concepts_activations,
            concepts_importances,
            concepts_labels,
            top_k=resolved_top_k,
            concept_color=concept_color,
            default_colormap=default_colormap,
            onclick_colormap=onclick_colormap,
            custom_css=custom_css,
            save_path=save_path,
        )
        return

    plot_generation_local_concepts(
        concepts_labels,
        sample,  # type: ignore
        concepts_activations,
        concepts_importances,
        top_k=resolved_top_k,
        concept_color=concept_color,
        default_colormap=default_colormap,
        onclick_colormap=onclick_colormap,
        custom_css=custom_css,
        save_path=save_path,
    )


def plot_classification_global_concepts(
    classes_names: list[str] | dict[int, str] | None,
    concepts_labels: dict[int, str | list[str]] | dict[str, dict[int, str | list[str]]],
    concepts_importances: torch.Tensor | dict[str | int, torch.tensor],  # type: ignore
    *,
    top_k: int = 10,
    concept_color: str = "#f39c12",
    default_colormap: dict[int, str] | None = None,
    onclick_colormap: tuple[str, str] | list[str] | None = None,
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """
    Display global concept importances for classification tasks.

    Args:
        classes_names: Display names for classes.
        concepts_labels: Either {concept_id: label} or {class_id/class_name: {concept_id: label}}.
        concepts_importances: {class_name: [scores]} with scores in [0, 1] or a tensor.
        top_k: Number of concepts shown per class.
        default_colormap: Optional {id: color} override for class colors.
        onclick_colormap: (selected_color, hover_color) for active labels.
    """

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    concepts_importances = _coerce_class_importances(concepts_importances)
    if not isinstance(concepts_labels, dict):
        raise TypeError("concepts_labels must be a dict.")

    class_keys = list(concepts_importances.keys())
    class_entries = _resolve_class_entries(classes_names, class_keys)
    class_key_set = set(class_keys) | {name for _, name in class_entries}
    if isinstance(classes_names, dict):
        class_key_set |= set(classes_names.keys())
        class_key_set |= set(classes_names.values())
    elif isinstance(classes_names, (list, tuple)):
        class_key_set |= set(range(len(classes_names)))
        class_key_set |= set(classes_names)
    classwise = _is_classwise_concepts(concepts_labels, class_key_set)

    def _as_color_id(value: Any, fallback: int) -> int:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        return fallback

    class_color_ids = [_as_color_id(class_key, index) for index, (class_key, _) in enumerate(class_entries)]
    class_color_map = _build_default_colormap(class_color_ids, default_colormap)
    classes_descriptions = [
        {
            "name": name,
            "color": class_color_map[_as_color_id(class_key, index)],
        }
        for index, (class_key, name) in enumerate(class_entries)
    ]
    on_click_colors = list(_normalize_onclick_colormap(onclick_colormap))
    coerced_importances: dict[Any, list[float]] = {}
    for class_key in class_keys:
        importances = _coerce_importances(concepts_importances[class_key])
        coerced_importances[class_key] = importances
    concepts_by_class = []

    for class_index, (class_key, class_name) in enumerate(class_entries):
        importances = coerced_importances[class_key]

        if classwise:
            label_map = _resolve_classwise_label_map(concepts_labels, class_key, class_name, class_index)
        else:
            label_map = concepts_labels
        label_map = _normalize_label_map(label_map) if label_map else {}

        for concept_id in label_map.keys():
            if concept_id < 0 or concept_id >= len(importances):
                raise ValueError(f"Concept id {concept_id} is out of range for class {class_name!r}.")

        concept_ids = list(range(len(importances)))

        concepts = []
        for concept_id in concept_ids:
            importance = importances[concept_id]
            if importance == 0.0:
                continue
            label = label_map.get(concept_id, f"Concept #{concept_id}")
            concepts.append(
                {
                    "id": concept_id,
                    "label": label,
                    "importance": importance,
                }
            )

        concepts.sort(key=lambda item: item["importance"], reverse=True)
        if top_k:
            concepts = concepts[:top_k]
        concepts_by_class.append(concepts)

    data = {
        "classes": classes_descriptions,
        "concepts": concepts_by_class,
        "top_k": top_k,
        "concept_color": concept_color,
        "default_colormap": {},
        "onclick_colormap": on_click_colors,
    }

    json_data = json.dumps(data, ensure_ascii=True, indent=2)

    unique_id = f"{uuid.uuid4()}"
    body = (
        f"<h3>Classes</h3>"
        f"<div class='line-style'><div id='classes-{unique_id}'></div></div>\n"
        f"<div id='concepts-wrapper-{unique_id}' class='is-hidden'>"
        f"<h3>Concepts</h3>"
        f"<div id='concepts-{unique_id}'></div>"
        f"</div>\n"
    )
    body += f"""
    <script>
        var viz = new ClassificationConceptsVisualization(
            'classes-{unique_id}',
            'concepts-{unique_id}',
            'concepts-wrapper-{unique_id}',
            {json.dumps(json_data)}
        );
        window.viz = viz;
    </script>
    """
    _render_visualization_html(body, custom_css=custom_css, save_path=save_path)


def plot_classification_local_concepts(
    sample: list[str],
    classes_names: list[str] | dict[int, str] | None,
    concepts_activations: Any,
    concepts_importances: Any,
    concepts_labels: list[str | list[str]] | tuple[str | list[str], ...] | dict[int, str | list[str]],
    *,
    top_k: int = 10,
    concept_color: str = "#f39c12",
    default_colormap: dict[int, str] | None = None,
    onclick_colormap: tuple[str, str] | list[str] | None = None,
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """
    Display local concept activations for classification tasks.

    Args:
        sample: Full sequence tokens.
        classes_names: Display names for classes.
        concepts_activations: Tensor, list, or dict with shape (len(sample), nb_concepts) or
            (1, nb_concepts). Dicts may be keyed by class id/name.
        concepts_importances: Tensor, list, or dict with shape (nb_classes, nb_concepts) or
            {class_id/class_name: [scores]}.
        concepts_labels: Either {concept_id: label}, a list of labels, or
            {class_id/class_name: {concept_id: label}} for class-wise labels.
        top_k: Number of concepts shown in the concept list.
        default_colormap: Optional {concept_id: color} override for concept colors.
        onclick_colormap: (selected_color, hover_color) for active labels.
    """

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if not isinstance(sample, (list, tuple)):
        raise TypeError("sample must be a list or tuple of strings.")

    sample_tokens = [str(token) for token in sample]

    activations, activations_by_class = _coerce_class_activations_local(concepts_activations, len(sample_tokens))

    class_keys, importances_map = _coerce_class_importances_local(concepts_importances)
    class_entries = _resolve_class_entries(classes_names, class_keys)
    classes_descriptions = [{"name": name} for _, name in class_entries]
    importances: list[list[float]] = []
    expected_len = None
    for class_key, class_name in class_entries:
        row = _resolve_class_importances_row(importances_map, class_key, class_name)
        if row is None:
            raise ValueError(f"Missing importances for class {class_name!r}.")
        if expected_len is None:
            expected_len = len(row)
        elif len(row) != expected_len:
            raise ValueError("concepts_importances rows must have consistent length.")
        importances.append(row)
    nb_concepts = len(activations[0]) if activations else 0
    if importances:
        if nb_concepts and len(importances[0]) != nb_concepts:
            raise ValueError("concepts_importances must have the same number of concepts as concepts_activations.")
        if not nb_concepts:
            nb_concepts = len(importances[0])

    class_key_set = set(class_keys) | {name for _, name in class_entries}
    if isinstance(classes_names, dict):
        class_key_set |= set(classes_names.keys())
        class_key_set |= set(classes_names.values())
    elif isinstance(classes_names, (list, tuple)):
        class_key_set |= set(range(len(classes_names)))
        class_key_set |= set(classes_names)

    classwise = isinstance(concepts_labels, dict) and _is_classwise_concepts(concepts_labels, class_key_set)
    labels_by_class = None
    if classwise:
        labels_by_class = {}
        default_labels = None
        for class_index, (class_key, class_name) in enumerate(class_entries):
            label_map = _resolve_classwise_label_map(concepts_labels, class_key, class_name, class_index)  # type: ignore
            label_map = _normalize_label_map(label_map) if label_map else {}
            labels_list = _normalize_concepts_labels(label_map, nb_concepts)
            labels_by_class[class_index] = labels_list
            if default_labels is None:
                default_labels = labels_list
        labels = default_labels or _normalize_concepts_labels({}, nb_concepts)
    else:
        labels = _normalize_concepts_labels(concepts_labels, nb_concepts)

    totals = [0.0] * nb_concepts
    for row in activations:
        for concept_id, value in enumerate(row):
            totals[concept_id] += abs(value)
    ordered_concept_ids = sorted(
        range(nb_concepts),
        key=lambda concept_id: totals[concept_id],
        reverse=True,
    )
    concept_color_map = _build_default_colormap(ordered_concept_ids, default_colormap)
    on_click_colors = list(_normalize_onclick_colormap(onclick_colormap))

    activations_by_class_index = None
    if activations_by_class is not None:
        activations_by_class_index = {}
        for class_index, (class_key, class_name) in enumerate(class_entries):
            matrix = _resolve_class_activations_matrix(activations_by_class, class_key, class_name, class_index)
            if matrix is None:
                raise ValueError(f"Missing activations for class {class_name!r}.")
            activations_by_class_index[class_index] = matrix

    data = {
        "classes": classes_descriptions,
        "sample": sample_tokens,
        "labels": labels,
        "labels_by_class": labels_by_class,
        "activations": activations,
        "activations_by_class": activations_by_class_index,
        "importances": importances,
        "top_k": top_k,
        "concept_color": concept_color,
        "default_colormap": concept_color_map,
        "onclick_colormap": on_click_colors,
    }

    json_data = json.dumps(data, ensure_ascii=True, indent=2)

    unique_id = f"{uuid.uuid4()}"
    body = (
        f"<h3>Classes</h3>"
        f"<div class='line-style'><div id='classes-{unique_id}'></div></div>\n"
        f"<div id='concepts-wrapper-{unique_id}'>"
        f"<h3>Concepts</h3>"
        f"<div id='concepts-{unique_id}'></div>"
        f"</div>\n"
        f"<h3>Sample</h3><div id='sample-{unique_id}'></div>\n"
    )
    body += f"""
    <script>
        var viz = new ClassificationLocalConceptsVisualization(
            'classes-{unique_id}',
            'concepts-{unique_id}',
            'concepts-wrapper-{unique_id}',
            'sample-{unique_id}',
            {json.dumps(json_data)}
        );
        window.viz = viz;
    </script>
    """
    _render_visualization_html(body, custom_css=custom_css, save_path=save_path)


def plot_generation_local_concepts(
    concepts_labels: list[str | list[str]] | tuple[str | list[str], ...] | dict[int, str | list[str]],
    sample: list[str],
    concepts_activations: Any,
    concepts_importances: Any,
    *,
    top_k: int = 10,
    concept_color: str = "#f39c12",
    default_colormap: dict[int, str] | None = None,
    onclick_colormap: tuple[str, str] | list[str] | None = None,
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """
    Display local concept activations for a sample sequence.

    Args:
        concepts_labels: Either {concept_id: label} or a list of labels.
        sample: Full sequence (inputs + outputs) tokens.
        concepts_activations: Tensor or list with shape (len(sample), nb_concepts).
        concepts_importances: Tensor or list with shape (nb_outputs, nb_concepts).
            Outputs are assumed to be the last nb_outputs tokens in sample.
        top_k: Number of concepts kept for the sentence-level view.
        default_colormap: Optional {concept_id: color} override for concept colors.
        onclick_colormap: (selected_color, hover_color) for active labels.
    """

    if not isinstance(sample, (list, tuple)):
        raise TypeError("sample must be a list or tuple of strings.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    sample_tokens = [str(token) for token in sample]
    activations = _coerce_activation_matrix(concepts_activations, len(sample_tokens))
    raw_importances = _to_list(concepts_importances)
    importances = _coerce_importances_matrix(raw_importances, len(raw_importances))
    if len(importances) > len(sample_tokens):
        raise ValueError("concepts_importances must have at most one row per sample element.")
    nb_concepts = len(activations[0]) if activations else 0
    if importances and nb_concepts and len(importances[0]) != nb_concepts:
        raise ValueError("concepts_importances must have the same number of concepts as concepts_activations.")
    labels = _normalize_concepts_labels(concepts_labels, nb_concepts)
    totals = [0.0] * nb_concepts
    for row in importances:
        for concept_id, value in enumerate(row):
            totals[concept_id] += value
    ordered_concept_ids = sorted(
        range(nb_concepts),
        key=lambda concept_id: totals[concept_id],
        reverse=True,
    )
    concept_color_map = _build_default_colormap(ordered_concept_ids, default_colormap)
    on_click_colors = list(_normalize_onclick_colormap(onclick_colormap))

    data = {
        "sample": sample_tokens,
        "labels": labels,
        "activations": activations,
        "importances": importances,
        "top_k": top_k,
        "concept_color": concept_color,
        "default_colormap": concept_color_map,
        "onclick_colormap": on_click_colors,
    }

    json_data = json.dumps(data, ensure_ascii=True, indent=2)

    unique_id = f"{uuid.uuid4()}"
    body = (
        f"<div id='concepts-wrapper-{unique_id}'>"
        f"<h3>Concepts</h3>"
        f"<div id='concepts-{unique_id}'></div>"
        f"</div>\n"
        f"<h3>Sample</h3><div id='sample-{unique_id}'></div>\n"
    )
    body += f"""
    <script>
        var viz = new GenerationLocalConceptsVisualization(
            'sample-{unique_id}',
            'concepts-{unique_id}',
            'concepts-wrapper-{unique_id}',
            {json.dumps(json_data)}
        );
        window.viz = viz;
    </script>
    """
    _render_visualization_html(body, custom_css=custom_css, save_path=save_path)
