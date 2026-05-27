from PIL import Image
from interpreto import *
from interpreto.attributions import ImageSaliency
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch


processor = AutoImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit")
model = AutoModelForImageClassification.from_pretrained("hf-internal-testing/tiny-random-vit")
cat_image = Image.open("../cat.jpg")
print(model.config)
method = ImageSaliency(
    model=model,
    image_processor=processor
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

print(f"expected={expected}, got={output.targets}, match={torch.equal(expected, output.targets)}")
