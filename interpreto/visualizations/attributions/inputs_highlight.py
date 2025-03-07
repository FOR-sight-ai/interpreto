"""
Base classes for attributions visualizations
"""

from __future__ import annotations

import torch
from typing import List, Tuple
import json

from interpreto.visualizations.base import WordHighlightVisualization, tensor_to_list


class AttributionHighlightVisualization(WordHighlightVisualization):
    """
    Class for attributions visualization
    """

    def __init__(
        self,
        inputs_sentences: List[List[str]],
        inputs_attributions: List[torch.Tensor],
        color: Tuple,
    ):
        # inputs: list of sentences composed of dicts:
        # - 'words': a list of words (array)
        # - 'attributions': an attribution array of the same length as the words array
        super().__init__()
        self.color = color
        self.data = self.adapt_data(
            inputs_sentences=inputs_sentences,
            inputs_attributions=inputs_attributions,
            outputs_words=None,
            outputs_attributions=None,
            concepts_descriptions=self.make_concepts_descriptions(color),
        )

    def make_concepts_descriptions(self, color: Tuple, name: str = "None"):
        """
        Create a structure describing the concepts

        Args:
            color (Tuple): A color for the concept
            name (str, optional): The name of the concept. Defaults to "None".

        Returns:
            dict: A dictionary describing the concept
        """
        return [
            {
                "name": f"concept #{name}",
                "description": f"This is the description of concept #{name}",
                "color": color,
            }
        ]

    def build_html(self):
        """
        Build the html for the visualization
        """
        json_data = json.dumps(self.data, default=tensor_to_list, indent=2)
        html = self.build_html_header()
        html += f"<h3>Inputs</h3><div id='{self.unique_id_inputs}'></div>\n"
        html += """
        <p><br><p><br><p>
        """
        html += f"""
        <script>
            var viz = new DataVisualisation(null, '{self.unique_id_inputs}', null, null, {json.dumps(json_data)});
            window.viz = viz;
        </script>
        </body></html>
        """
        return html


class MulticlassAttributionHighlightVisualization(WordHighlightVisualization):
    """
    Class for multi class attributions visualization
    """

    def __init__(
        self,
        inputs_sentences: List[List[str]],
        inputs_attributions: List[torch.Tensor],
        class_colors: List[Tuple],
        class_names: List[str] = None,
    ):
        """
        Create a multi class attribution visualization

        Args:
            inputs_sentences (List[List[str]]): A list of sentences composed of words
            inputs_attributions (List[torch.Tensor]): A list of attributions for each sentence
            class_colors (List[Tuple]): A list of colors for each class
            class_names (List[str], optional): A list of names for each class. Defaults to None

        """
        # inputs: list of sentences composed of dicts:
        # - 'words': a list of words (array)
        # - 'attributions': an attribution array of the same length as the words array
        super().__init__()
        if class_names is None:
            class_names = [f"class #{i}" for i in range(len(class_colors))]
        self.data = self.adapt_data(
            inputs_sentences=inputs_sentences,
            inputs_attributions=inputs_attributions,
            outputs_words=None,
            outputs_attributions=None,
            concepts_descriptions=self.make_concepts_descriptions(class_colors, class_names),
        )

    def make_concepts_descriptions(self, class_colors: List[Tuple], class_names: List[str]):
        """
        Create a structure describing the concepts

        Args:
            class_colors (List[Tuple]): A list of colors for each class
            class_names (List[str]): A list of names for each class

        Returns:
            dict: A dictionary describing the concepts
        """
        return [
            {
                "name": f"{name}",
                "description": f"This is the description of class #{name}",
                "color": color,
            }
            for color, name in zip(class_colors, class_names)
        ]

    def build_html(self):
        """
        Build the html for the visualization
        """
        json_data = json.dumps(self.data, default=tensor_to_list, indent=2)
        html = self.build_html_header()
        html += f"<h3>Classes</h3><div class='line-style'><div id='{self.unique_id_concepts}'></div></div>\n"
        html += f"<h3>Inputs</h3><div id='{self.unique_id_inputs}'></div>\n"
        html += """
        <p><br><p><br><p>
        """
        html += f"""
        <script>
            var viz = new DataVisualisation('{self.unique_id_concepts}', '{self.unique_id_inputs}', null, null, {json.dumps(json_data)});
            window.viz = viz;
        </script>
        </body></html>
        """
        return html
