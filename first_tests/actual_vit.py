from PIL import Image
from interpreto.attributions import ImageSaliency, ImageGradientShap, ImageIntegratedGradients, ImageSmoothGrad
from interpreto.attributions import *
from interpreto import ImageGranularity
from transformers import AutoModelForImageClassification, AutoImageProcessor, ViTModel
from interpreto.visualizations import plot_image_attribution
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open("../cat.jpg")

model = AutoModelForImageClassification.from_pretrained('akahana/vit-base-cats-vs-dogs')
processor = AutoImageProcessor.from_pretrained('akahana/vit-base-cats-vs-dogs')

cat_image = Image.open("../cat.jpg")
print(model.config)

methods = [ImageSaliency, ImageGradientShap, ImageIntegratedGradients, ImageSmoothGrad, ImageKernelShap, ImageLime, ImageSobol, ImageOcclusion, ImageSquareGrad, ImageVarGrad]

rendered = []  # (name, RGB array) per method
for method_cls in methods:
    method = method_cls(
        model=model,
        image_processor=processor,
        granularity=ImageGranularity.PATCH,
    )

    output = method.explain(cat_image)[0]
    targets = output.targets
    elements = output.elements
    attributions = output.attributions
    print(targets)
    print(elements)
    print(attributions)

    with torch.no_grad():
        logits = model(**output.model_inputs_to_explain).logits
    expected = logits.argmax(dim=-1)

    print(f"{method_cls.__name__}: expected={expected}, got={output.targets}, "
          f"match={torch.equal(expected, output.targets)}")

    # Use the real plotting function, then rasterize its figure to an RGB array.
    method_fig, _ = plot_image_attribution(output, alpha=0.5)
    method_fig.canvas.draw()
    rgb = np.asarray(method_fig.canvas.buffer_rgba())
    rendered.append((method_cls.__name__, rgb))
    plt.close(method_fig)  # avoid leaving each method's figure open

# Composite every method's rendered figure side by side in one displayer.
fig, axes = plt.subplots(1, len(rendered), figsize=(4 * len(rendered), 4), squeeze=False)
for ax, (name, rgb) in zip(axes[0], rendered):
    ax.imshow(rgb)
    ax.set_title(name)
    ax.axis("off")

fig.tight_layout()
plt.show()
