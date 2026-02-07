# attributions.py

from __future__ import annotations

import json
import math
import os
import uuid
import warnings

import numpy as np
import torch
from IPython.display import HTML, display

from interpreto.attributions.base import AttributionOutput, ModelTask

from .commons import (
    _build_default_colormap,
    _build_html_header,
    _normalize_onclick_colormap,
    _save_html,
)


def replace_nan_with_none(data_list):
    """Recursively replace NaN values with None in nested lists."""
    if isinstance(data_list, list):
        return [replace_nan_with_none(item) for item in data_list]
    elif isinstance(data_list, float) and (math.isnan(data_list) or not math.isfinite(data_list)):
        return None
    return data_list


def tensor_to_list(obj):
    """Convert tensors to lists."""
    if isinstance(obj, torch.Tensor):
        # Convert tensors to lists and replace NaN values with None since NaN values are not JSON serializable
        return replace_nan_with_none(obj.tolist())
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def plot_attributions(
    attribution_output: AttributionOutput,
    *,
    classes_names: dict[int, str] | list[str] | tuple[str, ...] | None = None,
    save_path: str | os.PathLike[str] | None = None,
    normalize: bool = True,
    positive_color: str = "#ff0000",
    negative_color: str = "#0000ff",
    default_colormap: dict[int, str] | None = None,
    onclick_colormap: tuple[str, str] | list[str] | None = None,
    highlight_border: bool = False,
    margin_right: str = "0.2em",
    custom_css: str = "",
) -> None:
    """
    Display token-level attribution visualizations for classification or generation tasks.

    Classification tasks (single- and multi-class) share the same visualization layout.

    Args:
        attribution_output: Attribution output returned by an attribution explainer.
        classes_names: Optional mapping or list of class display names.
        save_path: Optional path to save the HTML visualization.
        normalize: Whether to normalize attribution magnitudes for display.
        positive_color: Hex color for positive contributions.
        negative_color: Hex color for negative contributions.
        default_colormap: Optional {class_id: color} override for class colors.
        onclick_colormap: (selected_color, hover_color) for active labels.
        highlight_border: Whether to highlight the selected token span.
        margin_right: Extra right margin for generation layouts.
        custom_css: Additional CSS injected into the HTML visualization.

    Raises:
        ValueError: If the attribution output uses an unsupported model task.

    Examples:
        >>> # classification example (from docs/notebooks/classification_demonstration.ipynb)
        >>> attribution_explainer = Lime(model, tokenizer)
        >>> attributions = attribution_explainer(
        ...     model_inputs="Love and hate are two sides of the same coin.",
        ...     targets=torch.tensor([[0, 1, 2, 3, 4, 5]]),
        ... )
        >>> plot_attributions(attributions[0], classes_names=classes_names)
        >>>
        >>> # generation example (from docs/notebooks/generation_demonstration.ipynb)
        >>> attribution_explainer = KernelShap(model, tokenizer)
        >>> attributions = attribution_explainer(
        ...     model_inputs="Alice and Bob enter the bar, ",
        ...     targets="then Alice offers a drink to Bob.",
        ... )
        >>> plot_attributions(attributions[0])
    """

    if attribution_output.model_task == ModelTask.GENERATION:
        _display_generation_attributions(
            attribution_output,
            positive_color=positive_color,
            negative_color=negative_color,
            normalize=normalize,
            highlight_border=highlight_border,
            margin_right=margin_right,
            custom_css=custom_css,
            save_path=save_path,
        )
        return

    if attribution_output.model_task in (
        ModelTask.CLASSIFICATION,  # type: ignore
        ModelTask.SINGLE_CLASS_CLASSIFICATION,  # deprecated
        ModelTask.MULTI_CLASS_CLASSIFICATION,  # deprecated
    ):
        _display_classification_attributions(
            attribution_output,
            positive_color=positive_color,
            negative_color=negative_color,
            classes_names=classes_names,
            default_colormap=default_colormap,
            onclick_colormap=onclick_colormap,
            normalize=normalize,
            highlight_border=highlight_border,
            margin_right=margin_right,
            custom_css=custom_css,
            save_path=save_path,
        )
        return

    raise ValueError(f"Unsupported model task: {attribution_output.model_task!r}")


class AttributionVisualization:
    """Deprecated wrapper around plot_attributions."""

    def __init__(
        self,
        attribution_output: AttributionOutput,
        *,
        positive_color: str = "#ff0000",
        negative_color: str = "#0000ff",
        class_names: dict[int, str] | list[str] | tuple[str, ...] | None = None,
        classes_names: dict[int, str] | list[str] | tuple[str, ...] | None = None,
        default_colormap: dict[int, str] | None = None,
        onclick_colormap: tuple[str, str] | list[str] | None = None,
        normalize: bool = True,
        highlight_border: bool = False,
        margin_right: str = "0.2em",
        custom_css: str = "",
        save_path: str | os.PathLike[str] | None = None,
    ) -> None:
        warnings.warn(
            "AttributionVisualization is deprecated; use plot_attributions(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if class_names is not None and classes_names is not None:
            raise ValueError("Provide only one of class_names or classes_names.")

        self.attribution_output = attribution_output
        self.positive_color = positive_color
        self.negative_color = negative_color
        self.class_names = class_names if class_names is not None else classes_names
        self.default_colormap = default_colormap
        self.onclick_colormap = onclick_colormap
        self.normalize = normalize
        self.highlight_border = highlight_border
        self.margin_right = margin_right
        self.custom_css = custom_css
        self.save_path = save_path

    def display(self) -> None:
        plot_attributions(
            self.attribution_output,
            positive_color=self.positive_color,
            negative_color=self.negative_color,
            classes_names=self.class_names,
            default_colormap=self.default_colormap,
            onclick_colormap=self.onclick_colormap,
            normalize=self.normalize,
            highlight_border=self.highlight_border,
            margin_right=self.margin_right,
            custom_css=self.custom_css,
            save_path=self.save_path,
        )


def _display_generation_attributions(
    attribution_output: AttributionOutput,
    *,
    positive_color: str = "#ff0000",
    negative_color: str = "#0000ff",
    normalize: bool = True,
    highlight_border: bool = False,
    margin_right: str = "0.2em",
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Display attribution visualization for generation tasks."""
    custom_style = {"margin-right": margin_right} if margin_right else {}

    nb_outputs, nb_inputs_outputs = attribution_output.attributions.shape
    nb_inputs = nb_inputs_outputs - nb_outputs
    assert nb_inputs_outputs == len(attribution_output.elements), (
        f"The attribution shape ({nb_inputs_outputs}) does not match the number of elements ({len(attribution_output.elements)})"
    )

    # Reformat attribution_output to match the expected format for the js visualization
    inputs_words = attribution_output.elements[:nb_inputs]
    outputs_words = attribution_output.elements[nb_inputs:]

    # Split the attributions into input_attributions and output_attributions
    inputs_attributions = attribution_output.attributions[:, :nb_inputs].unsqueeze(-1)
    assert inputs_attributions.shape == (nb_outputs, nb_inputs, 1), (
        f"The inputs attributions shape ({inputs_attributions.shape}) "
        f"does not match the expected shape ({nb_outputs}, {nb_inputs}, 1)"
    )

    outputs_attributions = attribution_output.attributions[:, nb_inputs:].unsqueeze(-1)
    assert outputs_attributions.shape == (nb_outputs, nb_outputs, 1), (
        f"The outputs attributions shape ({outputs_attributions.shape}) "
        f"does not match the expected shape ({nb_outputs}, {nb_outputs}, 1)"
    )

    if normalize:
        attributions_np = attribution_output.attributions.cpu().detach().numpy()
        min_value = np.nanmin([np.nanmin(attributions_np), -np.nanmax(attributions_np)]).item()
        max_value = np.nanmax([np.nanmax(attributions_np), -np.nanmin(attributions_np)]).item()
        assert min_value <= max_value, f"The min value ({min_value}) should be less than the max value ({max_value})"
    else:
        min_value = -1.0
        max_value = 1.0

    classes_descriptions = [_make_class_description("None", positive_color, negative_color, min_value, max_value)]

    # Adapt data
    data = {
        "classes": classes_descriptions,
        "inputs": {"words": inputs_words, "attributions": inputs_attributions},
        "outputs": {"words": outputs_words, "attributions": outputs_attributions},
        "custom_style": custom_style,
    }

    # Build HTML
    json_data = json.dumps(data, default=tensor_to_list, indent=2)
    html = _build_html_header(custom_css)

    unique_id = f"{uuid.uuid4()}"

    html += f"<h3>Inputs</h3><div id='inputs-{unique_id}'></div>\n"
    html += f"<h3>Outputs</h3><div class='line-style'><div id='outputs-{unique_id}'></div></div>\n"
    html += f"""
    <script>
        var viz = new GenerationVisualization('inputs-{unique_id}', 'outputs-{unique_id}', '{highlight_border}', {json.dumps(json_data)});
        window.viz = viz;
    </script>
    </body></html>
    """

    if save_path:
        _save_html(html, save_path)
    display(HTML(html))


def _make_class_description(
    name: str,
    positive_color: str,
    negative_color: str,
    min_value: float,
    max_value: float,
    color: str | None = None,
    class_id: int | None = None,
) -> dict:
    """Create a structure describing a single class."""
    desc = {
        "name": name,
        "description": f"This is the description of class #{name}",
        "positive_color": positive_color,
        "negative_color": negative_color,
        "min": min_value,
        "max": max_value,
    }
    # Add color for multi-class default view
    if color is not None:
        desc["color"] = color
    if class_id is not None:
        desc["id"] = class_id
    return desc


def _display_classification_attributions(
    attribution_output: AttributionOutput,
    *,
    positive_color: str = "#ff0000",
    negative_color: str = "#0000ff",
    classes_names: dict[int, str] | list[str] | tuple[str, ...] | None = None,
    default_colormap: dict[int, str] | None = None,
    onclick_colormap: tuple[str, str] | list[str] | None = None,
    normalize: bool = True,
    highlight_border: bool = False,
    margin_right: str = "0.2em",
    custom_css: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Display attribution visualization for classification tasks."""

    custom_style = {"margin-right": margin_right} if margin_right else {}
    inputs_sentence = attribution_output.elements

    # Normalize attributions shape: for single class, reshape to match multi-class format
    attributions = attribution_output.attributions
    if attributions.shape[0] == 1:
        # Single class:
        class_ids = [int(attribution_output.targets[0])]
    # Multi-class: align class labels with provided targets when available
    elif attribution_output.targets is not None and len(attribution_output.targets) == attributions.shape[0]:
        class_ids = [int(target) for target in attribution_output.targets]
    else:
        class_ids = list(range(attributions.shape[0]))

    nb_classes = attributions.shape[0]
    inputs_attributions = attributions.T.unsqueeze(0)

    # Build class names list
    def _default_class_name(class_id: int) -> str:
        return f"class #{class_id}"

    if classes_names is None:
        classes_names_list = [_default_class_name(class_id) for class_id in class_ids]
    elif isinstance(classes_names, dict):
        classes_names_list = [
            _default_class_name(class_id) if classes_names.get(class_id) is None else str(classes_names[class_id])
            for class_id in class_ids
        ]
    elif isinstance(classes_names, (list, tuple)):
        fallback_to_positional = len(classes_names) == len(class_ids)
        classes_names_list = []
        for index, class_id in enumerate(class_ids):
            if 0 <= class_id < len(classes_names):
                name = classes_names[class_id]
            elif fallback_to_positional:
                name = classes_names[index]
            else:
                name = None
            classes_names_list.append(_default_class_name(class_id) if name is None else str(name))
    else:
        raise TypeError("classes_names must be a list, tuple, dict, or None.")

    # Generate distinct colors for each class (used in default multi-class view)
    class_color_map = _build_default_colormap(class_ids, default_colormap)

    if normalize:
        mins_list = attribution_output.attributions.min(axis=1).values  # type: ignore
        maxs_list = attribution_output.attributions.max(axis=1).values  # type: ignore
        min_values = [
            min(min_val.item(), -max_val.item()) for min_val, max_val in zip(mins_list, maxs_list, strict=False)
        ]
        max_values = [
            max(max_val.item(), -min_val.item()) for min_val, max_val in zip(mins_list, maxs_list, strict=False)
        ]
    else:
        min_values = [-1.0] * nb_classes
        max_values = [1.0] * nb_classes

    classes_descriptions = [
        _make_class_description(
            name,
            positive_color,  # Red for hover view (positive attributions)
            negative_color,  # Blue for hover view (negative attributions)
            min_val,
            max_val,
            color=class_color_map[class_id],  # Distinct color for default multi-class view
            class_id=class_id,
        )
        for class_id, name, min_val, max_val in zip(
            class_ids, classes_names_list, min_values, max_values, strict=False
        )
    ]
    header_text = "Classes"
    on_click_colors = list(_normalize_onclick_colormap(onclick_colormap))

    # Adapt data
    data = {
        "classes": classes_descriptions,
        "inputs": {"words": inputs_sentence, "attributions": inputs_attributions},
        "outputs": {"words": None, "attributions": None},
        "custom_style": custom_style,
        "onclick_colormap": on_click_colors,
    }

    # Build HTML
    json_data = json.dumps(data, default=tensor_to_list, indent=2)
    html = _build_html_header(custom_css)

    unique_id = f"{uuid.uuid4()}"

    html += f"<h3>{header_text}</h3><div class='line-style'><div id='classes-{unique_id}'></div></div>\n"
    html += f"<h3>Inputs</h3><div id='inputs-{unique_id}'></div>\n"
    html += f"""
    <script>
        var viz = new ClassificationVisualization('classes-{unique_id}', 'inputs-{unique_id}', '{highlight_border}', {json.dumps(json_data)});
        window.viz = viz;
    </script>
    </body></html>
    """

    if save_path:
        _save_html(html, save_path)
    display(HTML(html))
