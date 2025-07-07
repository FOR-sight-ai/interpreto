"""
Base classes for visualizations used in the lib
"""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod

import torch
from IPython.display import HTML, display


def tensor_to_list(obj):
    """Convert tensors to lists."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class BaseAttributionVisualization(ABC):
    """
    Abstract class for words highlighting visualization
    """

    def __init__(self):
        self.custom_css = None
        self.js_file_path = "visualization_attribution.js"
        self.css_file_path = "visualization.css"

        # Generate unique ids for the divs so that we can have multiple visualizations on the same page
        self.unique_id_classes = f"classes-{uuid.uuid4()}"
        self.unique_id_inputs = f"inputs-{uuid.uuid4()}"
        self.unique_id_outputs = f"outputs-{uuid.uuid4()}"

    def adapt_data(
        self,
        input_words: list[str],
        input_attributions: torch.Tensor,
        output_words: list[str],
        output_attributions: torch.Tensor,
        classes_descriptions: list[dict],
        custom_style: dict = None,
    ):
        """
        Adapt the data to the expected format for the visualization

        Args:
            input_words (List[str]]): list of input words (1 sentence)
            input_attributions (torch.Tensor): Attributions for the input words
                (same dimension)
            output_words (List[str]): List of output words (1 sentence)
            output_attributions (torch.Tensor): Attributions for the output (same dimension)
            classes_descriptsion (List[dict]): Description of the classes.
            custom_style (dict): Custom style to apply to the visualization

        Returns:
            dict: The adapted data
        """
        data_struct = {
            "classes": classes_descriptions,
            "inputs": {"words": input_words, "attributions": input_attributions},
            "outputs": {"words": output_words, "attributions": output_attributions},
            "custom_style": custom_style,
        }
        return data_struct

    def build_html_header(self) -> str:
        """
        Build the html header for the visualization

        Returns:
            str: The html header
        """
        # Load the JS and CSS files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        js_file_path = os.path.join(current_dir, self.js_file_path)
        with open(js_file_path, encoding="utf-8") as file:
            js_content = file.read()

        css_file_path = os.path.join(current_dir, self.css_file_path)
        with open(css_file_path, encoding="utf-8") as file:
            css = file.read()

        html = f"""
            <head>
                <style>
                    {css}
                    {self.custom_css if self.custom_css else ""}
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


class ConceptAttributionVisualization(ABC):
    """
    Abstract class for words highlighting visualization
    """

    def __init__(self):
        self.unique_id_concepts = None
        self.unique_id_inputs = None
        self.unique_id_outputs = None
        self.custom_css = None

    def adapt_data(
        self,
        inputs_sentences: list[list[str]],
        inputs_attributions: list[torch.Tensor],
        outputs_words: list[str],
        outputs_attributions: torch.Tensor,
        concepts_descriptions: list[dict],
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
        data_struct = {
            "concepts": concepts_descriptions,
            "inputs": [
                {"words": words, "attributions": attributions}
                for words, attributions in zip(inputs_sentences, inputs_attributions, strict=False)
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
        js_file_path = os.path.join(current_dir, "visualization.js")
        with open(js_file_path, encoding="utf-8") as file:
            js_content = file.read()

        css_file_path = os.path.join(current_dir, "visualization.css")
        with open(css_file_path, encoding="utf-8") as file:
            css = file.read()

        html = f"""
            <head>
                <style>
                    {css}
                    {self.custom_css if self.custom_css else ""}
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
