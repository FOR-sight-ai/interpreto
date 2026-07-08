# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including witho"ut limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from interpreto.model_wrapping.image_classification_inference_wrapper import ImageClassificationInferenceWrapper
from interpreto.typing import IncompatibilityError

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_DIR = Path(__file__).parents[1] / "fixtures" / "images"

CLASSIFICATION_MODELS = [
    "hf-internal-testing/tiny-random-vit",
    "hf-internal-testing/tiny-random-BeitForImageClassification",
    "hf-internal-testing/tiny-random-ViTForImageClassification",
    "akahana/vit-base-cats-vs-dogs",
]


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([[1.0, 3.0, 2.0]]),
        torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.0, -1.0]]),
        torch.tensor([[-1.0, -3.0, -2.0]]),
        torch.tensor([[2.0, 2.0, 2.0]]),
        torch.tensor([[0.5]]),
    ],
)
def test_extract_targets_from_logits(logits):
    targets = ImageClassificationInferenceWrapper._extract_targets_from_logits(None, logits)

    assert targets.shape == (logits.shape[0], 1), "targets should have shape (b, 1)"
    assert targets.dtype in (torch.int32, torch.int64), "targets should be an integer tensor"
    assert torch.equal(targets, logits.argmax(dim=-1, keepdim=True)), (
        "targets should be the argmax class for each sample"
    )
    for row, target in zip(logits, targets, strict=True):
        assert row[target.item()] == row.max(), "target should index into the highest logit"


def test_target_logits():
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.0, -1.0]])
    targets = torch.tensor([0, 2])

    result = ImageClassificationInferenceWrapper._target_logits(None, logits, targets)

    expected = torch.tensor([[1.0, 2.0], [5.0, -1.0]])
    assert torch.equal(result, expected), "_target_logits should return logits indexed at the given targets"


def test_prepare_inputs_rejects_wrong_pixel_values_shape():
    fake_self = SimpleNamespace(device=torch.device("cpu"))
    inputs = [{"pixel_values": torch.rand(4, 8, 8)}]  # wrong channel count

    with pytest.raises(AssertionError):
        ImageClassificationInferenceWrapper._prepare_inputs(fake_self, inputs)


@pytest.mark.parametrize("for_gradients", [False, True])
def test_prepare_inputs(for_gradients):
    fake_self = SimpleNamespace(device=torch.device("cpu"))
    b, h, w = 3, 8, 8
    inputs = [{"pixel_values": torch.rand(3, h, w), "extra_key": torch.rand(3, h, w)} for _ in range(b)]

    prepared = ImageClassificationInferenceWrapper._prepare_inputs(fake_self, inputs, for_gradients=for_gradients)

    assert isinstance(prepared, MutableMapping), "prepared inputs should be a TensorMapping"
    assert set(prepared.keys()) == set(inputs[0].keys()), "prepared should keep all the input keys"
    for key in inputs[0].keys():
        assert prepared[key].shape == (b, 3, h, w), f"{key} should have shape (b, 3, H, W)"

    if for_gradients:
        assert prepared["pixel_values"].is_leaf, "pixel_values should be a clean leaf when for_gradients=True"
        assert prepared["pixel_values"].requires_grad, "pixel_values should require grad when for_gradients=True"


def test_compute_gradients_shapes():
    fake_self = SimpleNamespace(input_x_gradient=False)
    b, t, h, w = 2, 3, 2, 2
    pixel_values = torch.rand(b, 3, h, w, requires_grad=True)
    targeted_logits_chunk = pixel_values.flatten(1).sum(dim=1, keepdim=True).repeat(1, t)
    chunk_slice = slice(0, b)

    grads = ImageClassificationInferenceWrapper._compute_gradients(
        fake_self, {"pixel_values": pixel_values}, targeted_logits_chunk, chunk_slice, last_chunk=True
    )

    assert grads.shape == (b, t, 3, h * w), "gradients should have shape (c, t, 3, H*W)"
    assert torch.equal(grads, torch.ones_like(grads)), (
        "gradient of a per-sample sum should be 1 everywhere, regardless of pixel_values"
    )


def test_compute_gradients_input_x_gradient_zero():
    fake_self = SimpleNamespace(input_x_gradient=True)
    b, t, h, w = 2, 3, 2, 2
    pixel_values = torch.zeros(b, 3, h, w, requires_grad=True)
    targeted_logits_chunk = pixel_values.flatten(1).sum(dim=1, keepdim=True).repeat(1, t)
    chunk_slice = slice(0, b)

    grads = ImageClassificationInferenceWrapper._compute_gradients(
        fake_self, {"pixel_values": pixel_values}, targeted_logits_chunk, chunk_slice, last_chunk=True
    )

    assert torch.equal(grads, torch.zeros_like(grads)), (
        "input_x_gradient with all-zero pixel_values should produce a null tensor"
    )


def test_image_classification_wrapper_fast():
    """Test classification wrapper with a single model for fast tests."""
    test_image_classification_wrapper("hf-internal-testing/tiny-random-vit")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_image_classification_wrapper(model_name):
    # images divided in two batches which we could see as different samples in interpreto
    # for each sample there are several perturbed images
    dog = Image.open(IMAGES_DIR / "dog.jpg").convert("RGB")
    cat_and_dog = Image.open(IMAGES_DIR / "equal_cat_and_dog.jpg").convert("RGB")
    images = [
        [cat_and_dog] * 2,
        [dog] * 4,
    ]

    # Model preparation
    model = AutoModelForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    inference_wrapper = ImageClassificationInferenceWrapper(model, batch_size=3, device=DEVICE)

    # Construct inputs
    with torch.no_grad():
        pixel_inputs = [image_processor(images=imgs, return_tensors="pt").to(DEVICE) for imgs in images]
        logits = [model(**p).logits for p in pixel_inputs]

    # Reference values
    expected_targets = [l.argmax(dim=-1, keepdim=True) for l in logits]
    targeted_logits = [l.gather(dim=-1, index=t) for l, t in zip(logits, expected_targets, strict=True)]

    # Compute elements with the wrapper
    test_targets = list(inference_wrapper(pixel_inputs))
    test_targeted_logits = list(inference_wrapper(pixel_inputs, expected_targets))
    inference_wrapper.gradients = True
    try:
        test_gradients = list(inference_wrapper(pixel_inputs, expected_targets))
    except IncompatibilityError:
        test_gradients = "ignore"

    for i in range(len(images)):
        assert torch.allclose(expected_targets[i].cpu(), test_targets[i], atol=1e-5), (
            "Classification targets are not argmax"
        )
        assert torch.allclose(targeted_logits[i], test_targeted_logits[i], atol=1e-5), (
            "Classification targeted logits are not correct"
        )
        grads_shape = (
            len(images[i]),
            expected_targets[i].shape[0],
            3,
            pixel_inputs[i]["pixel_values"].shape[-2] * pixel_inputs[i]["pixel_values"].shape[-1],
        )  # (b, n_targets, d=3, l=H*W)
        if test_gradients != "ignore":
            assert grads_shape == test_gradients[i].shape, "Classification gradients have wrong shape."


if __name__ == "__main__":
    test_image_classification_wrapper("hf-internal-testing/tiny-random-vit")
