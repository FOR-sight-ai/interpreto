import os

import torch
from matplotlib import pyplot as plt

from interpreto.attributions.base import (
    AttributionOutput,
    ModelTask
)
from interpreto.visualizations.attributions import (
    AttributionVisualization
)
from interpreto.visualizations.concepts.concepts_highlight import (
    ConceptHighlightVisualization,
)


def test_attribution_monoclass():
    # attributions (1 classe)
    sentence = ["A", "B", "C", "one", "two", "three"]
    attributions = torch.linspace(-10, 10, steps=len(sentence))
    single_class_classification_output = AttributionOutput(elements=sentence,
                                                           attributions=attributions,
                                                           model_task=ModelTask.SINGLE_CLASS_CLASSIFICATION,
                                                           classes=[0])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_attribution_monoclass.html")

    viz = AttributionVisualization(attribution_output=single_class_classification_output)

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)

    os.remove(output_file_path)


def test_attribution_multiclass():
    # attributions (2 classes)
    sentence = ["A", "B", "C", "one", "two", "three"]
    nb_classes = 2

    attributions = torch.rand(nb_classes, len(sentence))  # (c, l)
    multi_class_classification_output = AttributionOutput(elements=sentence,
                                                          attributions=attributions,
                                                          model_task=ModelTask.MULTI_CLASS_CLASSIFICATION,
                                                          classes=[0, 1])

    viz = AttributionVisualization(
        attribution_output=multi_class_classification_output,
        class_names={0:"class1", 1:"class2"},
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

    os.remove(output_file_path)


def test_attribution_generation():
    inputs_sentence = ["A", "B", "C", "one", "two", "three"]
    outputs_sentence = ["do", "re", "mi"]

    def make_attributions_outputs(inputs, outputs):
        attributions = torch.rand(len(outputs), len(inputs) + len(outputs))  # (l_g, l)
        return AttributionOutput(elements=inputs + outputs, attributions=attributions, model_task=ModelTask.GENERATION)

    generation_output = make_attributions_outputs(inputs_sentence, outputs_sentence)

    viz = AttributionVisualization(attribution_output=generation_output)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_attribution_generation.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)

    os.remove(output_file_path)


def test_concepts():
    # Concepts: 9 concepts (with inputs attributions for each output word)

    inputs_sentence = ["A", "B", "C", "one", "two", "three"]
    outputs_sentence = ["do", "re", "mi"]
    nb_concepts = 9

    def make_attributions_outputs(inputs, outputs):
        attributions = torch.rand(len(inputs) + len(outputs), len(outputs), nb_concepts)  # (l, l_g, c)
        return AttributionOutput(elements=inputs + outputs, attributions=attributions, model_task=ModelTask.GENERATION)

    generation_output = make_attributions_outputs(inputs_sentence, outputs_sentence)
    colors_set1 = plt.get_cmap("Set1").colors

    viz = ConceptHighlightVisualization(generation_output, concepts_colors=colors_set1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "test_concepts.html")

    # remove the file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # generate the html file
    viz.save(output_file_path)

    # assert that the file has been created
    assert os.path.exists(output_file_path)

    os.remove(output_file_path)
