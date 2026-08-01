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

from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers.image_processing_utils import BatchFeature

from interpreto.attributions.perturbations.base_merged import (
    ImageMaskPerturbator,
    ImageTensorPerturbator,
    TextMaskPerturbator,
    TextTensorPerturbator,
)
from interpreto.attributions.perturbations.occlusion_perturbation_merged import (
    OcclusionPerturbator as OcclusionImagePerturbator,
)

# from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers

CLASSIFICATION_MODELS = [
    "hf-internal-testing/tiny-random-vit",
    "hf-internal-testing/tiny-random-BeitForImageClassification",
    "hf-internal-testing/tiny-random-ViTForImageClassification",
]

SLOW_MODELS = ["akahana/vit-base-cats-vs-dogs"]

FIXTURE_IMAGES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "images"

# image_embedding_perturbators = [
#     GaussianNoiseImagePerturbator,
#     LinearInterpolationImagePerturbator,
#     GradientShapImagePerturbator,
# ]

image_mask_perturbators = [
    OcclusionImagePerturbator,
    # RandomMaskedImagePerturbator,
    # ShapImagePerturbator,
    # SobolImagePerturbator,
]


@pytest.fixture(scope="module")
def images():
    return [Image.open(p).convert("RGB") for p in sorted(FIXTURE_IMAGES_DIR.glob("*.jpg"))]


# @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
# @pytest.mark.parametrize("perturbator_class", image_embedding_perturbators)
# def test_image_embedding_perturbator(perturbator_class, model_name, images):
#     """Mirror of test_embeddings_perturbators for the image tensor-space perturbators."""
#     assert issubclass(perturbator_class, ImageTensorPerturbator)
#     assert not issubclass(perturbator_class, ImageMaskPerturbator)
#     assert not issubclass(perturbator_class, TextMaskPerturbator)
#     assert not issubclass(perturbator_class, TextTensorPerturbator)

#     p = 10
#     perturbator = perturbator_class(n_perturbations=p)

#     image_processor = AutoImageProcessor.from_pretrained(model_name)

#     for img in images:
#         processed_image = image_processor(img, return_tensors="pt")
#         assert isinstance(processed_image, BatchFeature)

#         perturbed_inputs, _ = perturbator.perturb(processed_image)

#         assert isinstance(perturbed_inputs, MutableMapping)

#         assert "pixel_values" in perturbed_inputs.keys()
#         assert isinstance(perturbed_inputs["pixel_values"], torch.Tensor)

#         _, _, h, w = processed_image["pixel_values"].shape
#         assert perturbed_inputs["pixel_values"].shape == (p, 3, h, w)


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
@pytest.mark.parametrize("perturbator_class", image_mask_perturbators)
def test_image_mask_perturbator(perturbator_class, model_name, images):
    """Mirror of test_token_perturbators for the image mask-based perturbators."""
    assert issubclass(perturbator_class, ImageMaskPerturbator)
    assert not issubclass(perturbator_class, ImageTensorPerturbator)
    assert not issubclass(perturbator_class, TextMaskPerturbator)
    assert not issubclass(perturbator_class, TextTensorPerturbator)

    patch_size = 2
    p = 15

    perturbator = perturbator_class(patch_size=patch_size)
    perturbator.n_perturbations = p

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    for img in images:
        processed_image = image_processor(img, return_tensors="pt")
        assert isinstance(processed_image, BatchFeature)

        perturbed_inputs, masks = perturbator.perturb(processed_image)

        assert isinstance(perturbed_inputs, BatchFeature)
        assert "pixel_values" in perturbed_inputs.keys()
        assert isinstance(perturbed_inputs["pixel_values"], torch.Tensor)

        _, _, h, w = processed_image["pixel_values"].shape
        g = (h // patch_size) * (w // patch_size)
        if isinstance(perturbator, OcclusionImagePerturbator):
            real_p = g + 1
        elif isinstance(perturbator, SobolImagePerturbator):
            k = perturbator.n_token_perturbations
            real_p = (g + 2) * k
        else:
            real_p = perturbator.n_perturbations

        assert perturbed_inputs["pixel_values"].shape == (real_p, 3, h, w)

        assert isinstance(masks, torch.Tensor)
        assert masks.shape[0] == perturbed_inputs["pixel_values"].shape[0]


# @pytest.mark.slow
# @pytest.mark.parametrize("model_name", SLOW_MODELS)
# @pytest.mark.parametrize("perturbator_class", image_embedding_perturbators)
# def test_slow_image_embedding_perturbator(perturbator_class, model_name, images):
#     """Mirror of test_embeddings_perturbators for the image tensor-space perturbators."""
#     assert issubclass(perturbator_class, ImageTensorPerturbator)
#     assert not issubclass(perturbator_class, ImageMaskPerturbator)
#     assert not issubclass(perturbator_class, TextMaskPerturbator)
#     assert not issubclass(perturbator_class, TextTensorPerturbator)

#     p = 10
#     perturbator = perturbator_class(n_perturbations=p)

#     image_processor = AutoImageProcessor.from_pretrained(model_name)

#     for img in images:
#         processed_image = image_processor(img, return_tensors="pt")
#         assert isinstance(processed_image, BatchFeature)

#         perturbed_inputs, _ = perturbator.perturb(processed_image)

#         assert isinstance(perturbed_inputs, MutableMapping)

#         assert "pixel_values" in perturbed_inputs.keys()
#         assert isinstance(perturbed_inputs["pixel_values"], torch.Tensor)

#         _, _, h, w = processed_image["pixel_values"].shape
#         assert perturbed_inputs["pixel_values"].shape == (p, 3, h, w)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", SLOW_MODELS)
@pytest.mark.parametrize("perturbator_class", image_mask_perturbators)
def test_slow_image_mask_perturbator(perturbator_class, model_name, images):
    """Mirror of test_token_perturbators for the image mask-based perturbators."""
    assert issubclass(perturbator_class, ImageMaskPerturbator)
    assert not issubclass(perturbator_class, ImageTensorPerturbator)
    assert not issubclass(perturbator_class, TextMaskPerturbator)
    assert not issubclass(perturbator_class, TextTensorPerturbator)

    patch_size = 16
    p = 15

    perturbator = perturbator_class(patch_size=patch_size)
    perturbator.n_perturbations = p

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    for img in images:
        processed_image = image_processor(img, return_tensors="pt")
        assert isinstance(processed_image, BatchFeature)

        perturbed_inputs, masks = perturbator.perturb(processed_image)

        assert isinstance(perturbed_inputs, BatchFeature)
        assert "pixel_values" in perturbed_inputs.keys()
        assert isinstance(perturbed_inputs["pixel_values"], torch.Tensor)

        _, _, h, w = processed_image["pixel_values"].shape
        g = (h // patch_size) * (w // patch_size)
        if isinstance(perturbator, OcclusionImagePerturbator):
            real_p = g + 1
        elif isinstance(perturbator, SobolImagePerturbator):
            k = perturbator.n_token_perturbations
            real_p = (g + 2) * k
        else:
            real_p = perturbator.n_perturbations

        assert perturbed_inputs["pixel_values"].shape == (real_p, 3, h, w)

        assert isinstance(masks, torch.Tensor)
        assert masks.shape[0] == perturbed_inputs["pixel_values"].shape[0]


# def test_linear_interpolation_image_perturbation_adjust_baseline():
#     inputs = torch.randn(1, 3, 8, 8)

#     # Test with None baseline
#     baseline = LinearInterpolationImagePerturbator.adjust_baseline(None, inputs)
#     assert torch.all(baseline.abs() < 1e-5)
#     assert baseline.shape == inputs.shape[1:]

#     # Test with float baseline
#     baseline = LinearInterpolationImagePerturbator.adjust_baseline(0.5, inputs)
#     assert torch.all(baseline == 0.5)
#     assert baseline.shape == inputs.shape[1:]

#     # Test with tensor baseline
#     baseline_tensor = torch.randn(3, 8, 8)
#     baseline = LinearInterpolationImagePerturbator.adjust_baseline(baseline_tensor, inputs)
#     assert torch.all(baseline == baseline_tensor)
#     assert baseline.shape == inputs.shape[1:]


# def test_linear_interpolation_image_perturbation_adjust_baseline_invalid():
#     inputs = torch.randn(1, 3, 8, 8)

#     # Test with invalid baseline type
#     with pytest.raises(TypeError, match="Type-check error"):
#         LinearInterpolationImagePerturbator.adjust_baseline("invalid", inputs)  # type: ignore

#     # Test with mismatched tensor shape
#     baseline_tensor = torch.randn(2, 8, 8)
#     with pytest.raises(ValueError, match="does not match expected shape"):
#         LinearInterpolationImagePerturbator.adjust_baseline(baseline_tensor, inputs)

#     # Test with mismatched tensor dtype
#     baseline_tensor = torch.randn(3, 8, 8, dtype=torch.float64)
#     with pytest.raises(ValueError, match="does not match expected dtype"):
#         LinearInterpolationImagePerturbator.adjust_baseline(baseline_tensor, inputs)


# @pytest.mark.parametrize(
#     "sampler",
#     [SequenceSamplers.SOBOL, SequenceSamplers.HALTON, SequenceSamplers.LatinHypercube],
# )
# def test_image_sobol_masks(sampler):
#     k = 10
#     perturbator = SobolImagePerturbator(
#         n_token_perturbations=k,
#         sampler=sampler,
#     )

#     for l in range(2, 20, 3):
#         mask = perturbator.get_mask(l)
#         assert mask.shape == ((l + 2) * k, l)
#         A = mask[:k]
#         B = mask[k : 2 * k]
#         C = mask[2 * k :].view(l, k, l)

#         # verify token-wise mask compared to the initial mask
#         for i in range(l):
#             assert torch.all(torch.isclose(C[i, :, i], B[:, i], atol=1e-5))
#             if i != 0:
#                 assert torch.all(torch.isclose(C[i, :, :i], A[:, :i], atol=1e-5))
#             if i != l - 1:
#                 assert torch.all(torch.isclose(C[i, :, i + 1 :], A[:, i + 1 :], atol=1e-5))


def test_image_occlusion_masks():
    perturbator = OcclusionImagePerturbator()
    for l in range(2, 20, 3):
        mask = perturbator.get_mask(l)
        assert torch.equal(mask, torch.cat([torch.zeros(1, l), torch.eye(l)], dim=0))
