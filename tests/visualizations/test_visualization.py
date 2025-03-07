import os
from matplotlib import pyplot as plt

import torch
import matplotlib.colors as mcolors

from interpreto.visualizations.attributions.inputs_highlight import (
    AttributionHighlightVisualization,
    MulticlassAttributionHighlightVisualization,
)
from interpreto.visualizations.concepts.concepts_highlight import (
    ConceptHighlightVisualization,
)


def test_simple_attribution_monoclass():
    # attributions (1 classe)
    inputs_sentences = [["A", "B", "C", "one", "two", "three"], ["do", "re", "mi"]]
    nb_concepts = 1
    nb_outputs = 1
    inputs_attributions = [
        torch.rand((nb_outputs, len(sentence), nb_concepts))
        for sentence in inputs_sentences
    ]
    viz = AttributionHighlightVisualization(
        inputs_sentences=inputs_sentences,
        inputs_attributions=inputs_attributions,
        color=mcolors.to_rgb("red"),
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(
        current_dir, "test_simple_attribution_monoclass.html"
    )

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)


def test_attribution_multiclass():
    # attributions (2 classes)
    inputs_sentences = [["A", "B", "C", "one", "two", "three"], ["do", "re", "mi"]]
    nb_concepts = 2
    nb_outputs = 1  # fixed for attributions
    inputs_attributions = [
        torch.rand((nb_outputs, len(sentence), nb_concepts))
        for sentence in inputs_sentences
    ]
    print(f"1st sentence attribution shape: {inputs_attributions[0].shape}")

    viz = MulticlassAttributionHighlightVisualization(
        inputs_sentences=inputs_sentences,
        inputs_attributions=inputs_attributions,
        class_colors=[mcolors.to_rgb("red"), mcolors.to_rgb("blue")],
        class_names=["class1", "class2"],
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_attribution_multiclass.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)


def test_concepts():
    # Concepts: 9 concepts (with inputs attributions for each output word)

    inputs_sentence = ["A", "B", "C", "one", "two", "three"]
    outputs_sentence = ["do", "re", "mi"]
    nb_concepts = 9

    inputs_attributions = torch.rand(
        (len(outputs_sentence), len(inputs_sentence), nb_concepts)
    )
    outputs_attributions = torch.rand(
        (len(outputs_sentence), len(outputs_sentence), nb_concepts)
    )

    colors_set1 = plt.get_cmap("Set1").colors

    viz = ConceptHighlightVisualization(
        inputs_sentences=[inputs_sentence],
        inputs_attributions=[inputs_attributions],
        outputs_words=outputs_sentence,
        outputs_attributions=outputs_attributions,
        concepts_colors=colors_set1,
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_concepts.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)
