"""
Base classes for visualizations used in the lib
"""

from __future__ import annotations

from typing import List

from abc import ABC, abstractmethod
import os
import uuid
import torch
from IPython.display import HTML, display


def tensor_to_list(obj):
    """Convert tensors to lists."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class WordHighlightVisualization(ABC):
    """
    Abstract class for words highlighting visualization
    """

    def __init__(self):
        self.unique_id_concepts = None
        self.unique_id_inputs = None
        self.unique_id_outputs = None

    def adapt_data(
        self,
        inputs_sentences: List[List[str]],
        inputs_attributions: List[torch.Tensor],
        outputs_words: List[str],
        outputs_attributions: torch.Tensor,
        concepts_descriptions: List[dict],
    ):
        """
        Adapt the data to the expected format for the visualization

        Args:
            inputs_sentences (List[List[str]]): List of sentences composed of several words
            inputs_attributions (List[torch.Tensor]): List of attributions for each sentence
                (same dimension)
            outputs_words (List[str]): List of words for the output (1 sentence)
            outputs_attributions (torch.Tensor): Attributions for the output (same dimension)
            concepts_descriptions (List[dict]): List of descriptions for the concepts

        Returns:
            dict: The adapted data
        """
        nb_inputs_sentences = len(inputs_sentences)
        nb_concepts = len(concepts_descriptions)
        nb_outputs = 0 if outputs_words is None else len(outputs_words)

        print(f"{nb_outputs = }, {nb_inputs_sentences = }, {nb_concepts = }")

        data_struct = {
            "concepts": concepts_descriptions,
            "inputs": [
                {"words": words, "attributions": attributions}
                for words, attributions in zip(inputs_sentences, inputs_attributions)
            ],
            "outputs": {"words": outputs_words, "attributions": outputs_attributions},
        }
        return data_struct

    def build_html_header(self) -> str:
        """
        Build the html header for the visualization

        Returns:
            str: The html header
        """
        # Generate unique ids for the divs so that we can have multiple visualizations on the same page
        self.unique_id_concepts = f"concepts-{uuid.uuid4()}"
        self.unique_id_inputs = f"inputs-{uuid.uuid4()}"
        self.unique_id_outputs = f"outputs-{uuid.uuid4()}"

        # Load the JS and CSS files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        js_file_path = os.path.join(current_dir, "visualisation.js")
        with open(js_file_path, "r", encoding="utf-8") as file:
            js_content = file.read()

        css_file_path = os.path.join(current_dir, "visualisation.css")
        with open(css_file_path, "r", encoding="utf-8") as file:
            css = file.read()

        html = f"""
            <head>
                <style>
                    {css}
                </style>
                <script>
                    {js_content}
                </script>
                <script>
                </script>
            </head>
            <body class="body-visualization">
        """
        return html

    @abstractmethod
    def build_html(self) -> str:
        """
        Build the html for the visualization
        """
        raise NotImplementedError

    def display(self) -> None:
        """
        Display the visualization in the notebook
        """
        html = self.build_html()
        display(HTML(html))

    def save(self, path: str) -> None:
        """
        Save the visualization to a file
        """
        html = self.build_html()
        with open(path, "w", encoding="utf-8") as file:
            file.write(html)
