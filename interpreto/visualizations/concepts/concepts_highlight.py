"""
Base classes for concepts visualizations
"""

from __future__ import annotations

from typing import List, Tuple
import json
import torch

from ..base import WordHighlightVisualization, tensor_to_list


class ConceptHighlightVisualization(WordHighlightVisualization):
    """
    Class for concepts visualization
    """

    def __init__(
        self,
        inputs_sentences: List[List[str]],
        inputs_attributions: List[torch.Tensor],
        outputs_words: List[str],
        outputs_attributions: torch.Tensor,
        concepts_colors: List[Tuple],
        concepts_names: List[str] = None,
        topk: int = 3,
    ):
        """
        Initialize the visualization

        Args:
            inputs_sentences (List[List[str]]): List of sentences composed of several words
            inputs_attributions (List[torch.Tensor]): List of attributions for each sentence
                (same dimension)
            outputs_words (List[str]): List of words for the output (1 sentence)
            outputs_attributions (torch.Tensor): Attributions for the output (same dimension)
            concepts_colors (List[Tuple]): List of colors for the concepts
            concepts_names (List[str], optional): List of names for the concepts. Defaults to None.
            topk (int, optional): Number of top concepts to display. Defaults to 3.
        """
        super().__init__()
        # TODO: add checks on the dimensions of the args
        if concepts_names is None:
            concepts_names = [f"concept #{i}" for i in range(len(concepts_colors))]
        self.topk = topk
        self.data = self.adapt_data(
            inputs_sentences=inputs_sentences,
            inputs_attributions=inputs_attributions,
            outputs_words=outputs_words,
            outputs_attributions=outputs_attributions,
            concepts_descriptions=self.make_concepts_descriptions(concepts_colors, concepts_names),
        )

    def make_concepts_descriptions(self, concepts_colors: List[Tuple], concepts_names: List[str]):
        """
        Create a structure describing the concepts

        Args:
            concepts_colors (List[Tuple]): A list of colors for each concept
            concepts_names (List[str]): A list of names for each concept

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
            }
            for color, name in zip(concepts_colors, concepts_names)
        ]

    def set_concept_name(self, concept_id: int, name: str):
        """
        Set the name of a concept

        Args:
            concept_id (int): The id of the concept
            name (str): The name of the concept
        """
        self.data["concepts"][concept_id]["name"] = name

    def set_concept_color(self, concept_id: int, color: Tuple):
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
        html += """
        <p><br><p><br><p>
        """
        html += f"""
        <script>
            var viz = new DataVisualisation('{self.unique_id_concepts}', '{self.unique_id_inputs}', '{self.unique_id_outputs}', {self.topk}, {json.dumps(json_data)});
            window.viz = viz;
        </script>
        </body></html>
        """
        return html
