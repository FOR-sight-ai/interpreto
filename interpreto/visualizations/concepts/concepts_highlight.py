"""
Base classes for concepts visualizations
"""

from __future__ import annotations

import json

from interpreto.attributions.base import AttributionOutput

from ..base import ConceptAttributionVisualization, tensor_to_list


class ConceptHighlightVisualization(ConceptAttributionVisualization):
    """
    Class for concepts visualization
    """

    def __init__(
        self,
        attribution_output: AttributionOutput,
        concepts_colors: list[tuple],
        concepts_names: list[str] = None,
        topk: int = 3,
        normalize: bool = True,
        highlight_border: bool = False,
        margin_right: str = "0.2em",
        css: str = None,
    ):
        """
        Initialize the visualization

        Args:
            attribution_output: AttributionOutput: The attribution outputs to visualize
                attribution_output.attributions is a tensor of shape (nb_inputs + nb_outputs, nb_outputs, nb_concepts)
                attribution_output.elements is a list of words (nb_inputs + nb_outputs)
            concepts_colors (List[Tuple]): List of colors for the concepts
            concepts_names (List[str], optional): List of names for the concepts. Defaults to None
            topk (int, optional): Number of top concepts to display. Defaults to 3
            normalize (bool, optional): Whether to normalize the attributions. If False, then the attributions values range will be assumed to be [0, 1]. Defaults to True
            highlight_border (bool, optional): Whether to highlight the border of the words. Defaults to False
            margin_right (str, optional): A custom CSS margin property to set the spacing between words. Defaults to '0.2em'
            css: (str, optional): A custom CSS to replace the default Interpreto CSS. Note: This CSS will be applied globally across the entire notebook. Defaults to None

        """
        super().__init__()
        nb_inputs_outputs, nb_outputs, nb_concepts = attribution_output.attributions.shape
        nb_inputs = nb_inputs_outputs - nb_outputs
        assert nb_inputs_outputs == len(attribution_output.elements), (
            f"The attribution shape ({nb_inputs_outputs}) does not match the number of elements ({len(attribution_output.elements)})"
        )
        assert nb_concepts > 0, "The number of concepts should be greater than 0"

        if concepts_names is None:
            concepts_names = [f"concept #{i}" for i in range(nb_concepts)]
        assert nb_concepts == len(concepts_names), (
            "The number of concepts should be equal to the number of concepts names"
        )

        # reformat attribution_output to match the expected format for the js visualization
        inputs_words = attribution_output.elements[:nb_inputs]
        outputs_words = attribution_output.elements[nb_inputs:]

        # split the attributions into input_attributions and output_attributions
        # attribution shape is (nb_inputs + nb_outputs, nb_outputs)
        # js expects inputs attributions of shape (nb_outputs, nb_inputs, nb_concepts)
        # and outputs attributions of shape (nb_outputs, nb_outputs, nb_concepts)
        inputs_attributions = attribution_output.attributions[:nb_inputs]
        inputs_attributions = inputs_attributions.transpose(0, 1)
        assert inputs_attributions.shape == (
            nb_outputs,
            nb_inputs,
            nb_concepts,
        ), (
            f"inputs_attributions shape {inputs_attributions.shape} does not match expected shape {(nb_outputs, nb_inputs, nb_concepts)}"
        )

        outputs_attributions = attribution_output.attributions[nb_inputs:]
        outputs_attributions = outputs_attributions.transpose(0, 1)
        assert outputs_attributions.shape == (
            nb_outputs,
            nb_outputs,
            nb_concepts,
        ), (
            f"outputs_attributions shape {outputs_attributions.shape} does not match expected shape {(nb_outputs, nb_outputs, nb_concepts)}"
        )

        # compute the min and max values for the attributions to be used for normalization
        if normalize:
            min_values = [attribution_output.attributions[:, :, c].min() for c in range(nb_concepts)]
            max_values = [attribution_output.attributions[:, :, c].max() for c in range(nb_concepts)]
        else:
            min_values = [0.0] * nb_concepts
            max_values = [1.0] * nb_concepts

        self.topk = topk
        self.highlight_border = highlight_border
        self.custom_style = {"margin-right": margin_right} if margin_right else {}
        self.custom_css = css
        self.data = self.adapt_data(
            inputs_sentences=[inputs_words],
            inputs_attributions=[inputs_attributions],
            outputs_words=outputs_words,
            outputs_attributions=outputs_attributions,
            concepts_descriptions=self.make_concepts_descriptions(
                concepts_colors, concepts_names, min_values, max_values
            ),
            custom_style=self.custom_style,
        )

    def make_concepts_descriptions(
        self,
        concepts_colors: list[tuple],
        concepts_names: list[str],
        min_values: list[float],
        max_values: list[float],
    ):
        """
        Create a structure describing the concepts

        Args:
            concepts_colors (List[Tuple]): A list of colors for each concept
            concepts_names (List[str]): A list of names for each concept
            min_value (List, optional): The minimum values for the attributions
            max_value (List, optional): The maximum values for the attributions

        Returns:
            dict: A dictionary describing the concepts
        """
        if len(concepts_colors) != len(concepts_names):
            raise ValueError("The number of colors should be equal to the number of concepts")
        return [
            {
                "name": f"{name}",
                "description": f"This is the description of concept #{name}",
                "color": color,
                "min": min_value,
                "max": max_value,
            }
            for color, name, min_value, max_value in zip(
                concepts_colors, concepts_names, min_values, max_values, strict=False
            )
        ]

    def set_concept_name(self, concept_id: int, name: str):
        """
        Set the name of a concept

        Args:
            concept_id (int): The id of the concept
            name (str): The name of the concept
        """
        self.data["concepts"][concept_id]["name"] = name

    def set_concept_color(self, concept_id: int, color: tuple):
        """
        Set the color of a concept

        Args:
            concept_id (int): The id of the concept
            color (Tuple): The color of the concept
        """
        self.data["concepts"][concept_id]["color"] = color

    def build_html(self):
        """
        Build the HTML visualization
        """
        json_data = json.dumps(self.data, default=tensor_to_list, indent=2)
        html = self.build_html_header()
        html += f"<h3>Concepts</h3><div class='line-style'><div id='{self.unique_id_concepts}'></div></div>\n"
        html += f"<h3>Inputs</h3><div id='{self.unique_id_inputs}'></div>\n"
        html += f"<h3>Outputs</h3><div class='line-style'><div id='{self.unique_id_outputs}'></div></div>\n"
        html += f"""
        <script>
            var viz = new DataVisualization('{self.unique_id_concepts}', '{self.unique_id_inputs}', '{self.unique_id_outputs}', {self.topk}, '{self.highlight_border}', {json.dumps(json_data)});
            window.viz = viz;
        </script>
        </body></html>
        """
        return html
