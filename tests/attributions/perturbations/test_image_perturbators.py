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

from collections.abc import MutableMapping
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers.image_processing_utils import BatchFeature

from interpreto.attributions.perturbations import (
    GaussianNoisePerturbator,
    GradientShapPerturbator,
    LinearInterpolationPerturbator,
    OcclusionPerturbator,
    RandomMaskedPerturbator,
    ShapPerturbator,
    SobolPerturbator,
)
from interpreto.attributions.perturbations.base import (
    ImageMaskPerturbator,
    ImageTensorPerturbator,
    TextMaskPerturbator,
    TextTensorPerturbator,
)
from interpreto.commons import GranularityResizeStrategy

SLOW_MODELS = [
    "hf-internal-testing/tiny-random-BeitForImageClassification",
    "hf-internal-testing/tiny-random-ViTForImageClassification",
]

FIXTURE_IMAGES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "images"


def _image_variant(method_class: type, modality_base: type) -> type:
    """
    Combine a method perturbator with an image modality base, as the explainer does at
    construction time. The method class alone leaves `perturb` abstract.
    """
    return type("Image" + method_class.__name__, (method_class, modality_base), {})


image_embedding_perturbators = [
    _image_variant(GaussianNoisePerturbator, ImageTensorPerturbator),
    _image_variant(LinearInterpolationPerturbator, ImageTensorPerturbator),
    _image_variant(GradientShapPerturbator, ImageTensorPerturbator),
]

image_mask_perturbators = [
    _image_variant(OcclusionPerturbator, ImageMaskPerturbator),
    _image_variant(RandomMaskedPerturbator, ImageMaskPerturbator),
    _image_variant(ShapPerturbator, ImageMaskPerturbator),
    _image_variant(SobolPerturbator, ImageMaskPerturbator),
]


def _setup_attribution_explainer_values(
    perturbator,
    model=None,
    patch_size: int | None = None,
    resize_strategy: GranularityResizeStrategy = GranularityResizeStrategy.BILINEAR,
):
    """
    Sets values for the Perturbators that should be set by the AttributionExplainer in the normal
    functioning of the library
    """
    perturbator.patch_size = patch_size
    perturbator.granularity_combination_strategy = resize_strategy


def _check_image_embedding_perturbator(perturbator_class, image_processor, images):
    """Mirror of test_embeddings_perturbators for the image tensor-space perturbators."""
    assert issubclass(perturbator_class, ImageTensorPerturbator), (
        "image embedding perturbators must subclass ImageTensorPerturbator"
    )
    assert not issubclass(perturbator_class, ImageMaskPerturbator), (
        "image embedding perturbators must not subclass ImageMaskPerturbator"
    )
    assert not issubclass(perturbator_class, TextMaskPerturbator), (
        "image embedding perturbators must not subclass TextMaskPerturbator"
    )
    assert not issubclass(perturbator_class, TextTensorPerturbator), (
        "image embedding perturbators must not subclass TextTensorPerturbator"
    )

    p = 10
    perturbator = perturbator_class(n_perturbations=p, processor=image_processor)

    for img in images:
        processed_image = image_processor(img, return_tensors="pt")
        assert isinstance(processed_image, BatchFeature), "the image processor must return a BatchFeature"

        perturbed_inputs, _ = perturbator.perturb(processed_image)

        assert isinstance(perturbed_inputs, MutableMapping), "perturbed_inputs must be a MutableMapping"

        assert "pixel_values" in perturbed_inputs.keys(), "perturbed_inputs must have the 'pixel_values' key"
        assert isinstance(perturbed_inputs["pixel_values"], torch.Tensor), (
            'perturbed_inputs["pixel_values"] must be a torch.Tensor'
        )

        _, _, h, w = processed_image["pixel_values"].shape
        assert perturbed_inputs["pixel_values"].shape == (p, 3, h, w), (
            'perturbed_inputs["pixel_values"] must have shape (n_perturbations, 3, H, W). Expected '
            f"{(p, 3, h, w)}, got {tuple(perturbed_inputs['pixel_values'].shape)}"
        )


def _check_image_mask_perturbator(perturbator_class, model, image_processor, images):
    """Mirror of test_token_perturbators for the image mask-based perturbators."""
    assert issubclass(perturbator_class, ImageMaskPerturbator), (
        "image mask perturbators must subclass ImageMaskPerturbator"
    )
    assert not issubclass(perturbator_class, ImageTensorPerturbator), (
        "image mask perturbators must not subclass ImageTensorPerturbator"
    )
    assert not issubclass(perturbator_class, TextMaskPerturbator), (
        "image mask perturbators must not subclass TextMaskPerturbator"
    )
    assert not issubclass(perturbator_class, TextTensorPerturbator), (
        "image mask perturbators must not subclass TextTensorPerturbator"
    )

    patch_size = 2
    p = 15
    perturbator = perturbator_class(processor=image_processor)
    perturbator.n_perturbations = p
    _setup_attribution_explainer_values(perturbator, model=model, patch_size=patch_size)

    for img in images:
        processed_image = image_processor(img, return_tensors="pt")
        assert isinstance(processed_image, BatchFeature), "the image processor must return a BatchFeature"

        perturbed_inputs, masks = perturbator.perturb(processed_image)

        assert isinstance(perturbed_inputs, BatchFeature), "perturbed_inputs must be a BatchFeature"
        assert "pixel_values" in perturbed_inputs.keys(), "perturbed_inputs must have the 'pixel_values' key"
        assert isinstance(perturbed_inputs["pixel_values"], torch.Tensor), (
            'perturbed_inputs["pixel_values"] must be a torch.Tensor'
        )

        _, _, h, w = processed_image["pixel_values"].shape
        g = (h // patch_size) * (w // patch_size)
        if isinstance(perturbator, OcclusionPerturbator):
            real_p = g + 1
        elif isinstance(perturbator, SobolPerturbator):
            k = perturbator.n_granularity_perturbations
            real_p = (g + 2) * k
        else:
            real_p = perturbator.n_perturbations

        assert perturbed_inputs["pixel_values"].shape == (real_p, 3, h, w), (
            'perturbed_inputs["pixel_values"] must have shape (real_p, 3, H, W), where real_p accounts for the '
            "method's actual number of masks (g+1 for Occlusion, (g+2)*k for Sobol, n_perturbations otherwise). "
            f"Expected {(real_p, 3, h, w)}, got {tuple(perturbed_inputs['pixel_values'].shape)}"
        )

        assert isinstance(masks, torch.Tensor), "masks must be a torch.Tensor"
        assert masks.shape[0] == perturbed_inputs["pixel_values"].shape[0], (
            "there must be exactly one mask per perturbed sample ie the shapes of the masks and the inputs "
            "should match"
        )


@pytest.fixture(scope="module")
def images():
    return [Image.open(p).convert("RGB") for p in sorted(FIXTURE_IMAGES_DIR.glob("*.jpg"))]


@pytest.mark.parametrize("perturbator_class", image_embedding_perturbators)
def test_image_embedding_perturbator(perturbator_class, vit_image_processor, images):
    _check_image_embedding_perturbator(perturbator_class, vit_image_processor, images)


@pytest.mark.parametrize("perturbator_class", image_mask_perturbators)
def test_image_mask_perturbator(perturbator_class, vit_model, vit_image_processor, images):
    _check_image_mask_perturbator(perturbator_class, vit_model, vit_image_processor, images)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", SLOW_MODELS)
@pytest.mark.parametrize("perturbator_class", image_embedding_perturbators)
def test_slow_image_embedding_perturbator(perturbator_class, model_name, images):
    _check_image_embedding_perturbator(perturbator_class, AutoImageProcessor.from_pretrained(model_name), images)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", SLOW_MODELS)
@pytest.mark.parametrize("perturbator_class", image_mask_perturbators)
def test_slow_image_mask_perturbator(perturbator_class, model_name, images):
    _check_image_mask_perturbator(
        perturbator_class, model_name, AutoImageProcessor.from_pretrained(model_name), images
    )
