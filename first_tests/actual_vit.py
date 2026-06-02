from PIL import Image
from interpreto.attributions import ImageSaliency, ImageGradientShap, ImageIntegratedGradients, ImageSmoothGrad
from interpreto.attributions import *
from interpreto import ImageGranularity
from transformers import AutoModelForImageClassification, AutoImageProcessor, ViTModel
from interpreto.visualizations import plot_image_attribution, plot_image_attributions_comparison
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import functools 
matplotlib.use("Qt5Agg")
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open("../cat.jpg")

model = AutoModelForImageClassification.from_pretrained('akahana/vit-base-cats-vs-dogs')
processor = AutoImageProcessor.from_pretrained('akahana/vit-base-cats-vs-dogs')

#cat_image = Image.open("../cat.jpg")
#dog_image = Image.open("../dog.jpg")
cat_and_dog_image = Image.open("../cat and dog.jpg")
image = cat_and_dog_image
print(model.config)

test_imagekernelshap = functools.partial(ImageKernelShap, n_perturbations=20)
test_imagesobol = functools.partial(ImageSobol, n_token_perturbations=10)

methods = [ImageSaliency, ImageGradientShap, ImageIntegratedGradients, ImageSmoothGrad, test_imagekernelshap, ImageLime, test_imagesobol, ImageOcclusion, ImageSquareGrad, ImageVarGrad]


def method_name(method_cls):
    # functools.partial has no __name__; the wrapped class lives on .func
    target = getattr(method_cls, "func", method_cls)
    return target.__name__

outputs = []  # one ImageAttributionOutput per method
labels = []   # matching method names
for method_cls in methods:
    method = method_cls(
        model=model,
        image_processor=processor,
        granularity=ImageGranularity.PATCH,
    )

    output = method.explain(model_inputs=image,targets = [1])[0]
    targets = output.targets
    elements = output.elements
    attributions = output.attributions
    print(targets)
    print(elements)
    print(attributions)

    with torch.no_grad():
        logits = model(**output.model_inputs_to_explain).logits
    expected = logits.argmax(dim=-1)

    print(f"{method_name(method_cls)}: expected={expected}, got={output.targets}, "
          f"match={torch.equal(expected, output.targets)}")

    # target_idx=0 is the target the comparison plot draws; turn its class
    # index into the model's human-readable label.
    target_class = int(output.targets[0].item())
    target_label = model.config.id2label[target_class]

    # The probability isn't stored on the output (it only carries attribution
    # scores), so derive it from the logits of the explained pixel_values above.
    target_prob = logits.softmax(dim=-1)[0, target_class].item()

    outputs.append(output)
    labels.append(f"{method_name(method_cls)}\n{target_label} ({target_prob:.1%})")

# Compare every method on the same image, each panel with its own score legend.
fig, axes = plot_image_attributions_comparison(
    outputs,
    labels=labels,
    image=image,
    alpha=0.5,
)
plt.show()
