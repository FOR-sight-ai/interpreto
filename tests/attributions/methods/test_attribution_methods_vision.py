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

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.image_processing_utils import BatchFeature

from interpreto.attributions import (
    ImageGradientShap,
    ImageIntegratedGradients,
    ImageKernelShap,
    ImageOcclusion,
    ImageSaliency,
    ImageSmoothGrad,
    ImageSquareGrad,
    ImageVarGrad,
)
from interpreto.attributions.aggregations.base import (
    Aggregator,
    MeanAggregator,
    OcclusionAggregator,
    SquaredMeanAggregator,
    TrapezoidalMeanAggregator,
    VarianceAggregator,
)
from interpreto.attributions.aggregations.linear_regression_aggregation import LinearRegressionAggregator
from interpreto.attributions.base import ImageAttributionOutput
from interpreto.attributions.perturbations import (
    GaussianNoiseImagePerturbator,
    GradientShapImagePerturbator,
    ImageTensorPerturbator,
    LinearInterpolationImagePerturbator,
    OcclusionImagePerturbator,
    ShapImagePerturbator,
)
from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFICATION_MODELS = [
    "hf-internal-testing/tiny-random-vit",
    "hf-internal-testing/tiny-random-BeitForImageClassification",
    "hf-internal-testing/tiny-random-ViTForImageClassification",
]

SLOW_MODELS = ["akahana/vit-base-cats-vs-dogs"]

FIXTURE_IMAGES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "images"


@pytest.fixture(scope="module")
def image1() -> Image.Image:
    return Image.open(sorted(FIXTURE_IMAGES_DIR.glob("*.jpg"))[0]).convert("RGB")


@pytest.fixture(scope="module")
def image2() -> Image.Image:
    return Image.open(sorted(FIXTURE_IMAGES_DIR.glob("*.jpg"))[1]).convert("RGB")


@pytest.fixture(scope="module")
def image_list(image1, image2) -> list[Image.Image]:
    return [image1, image2]


@pytest.fixture(scope="module")
def small_tensor_1() -> torch.Tensor:
    return torch.rand(3, 30, 30)


@pytest.fixture(scope="module")
def small_tensor_2() -> torch.Tensor:
    return torch.rand(3, 30, 30)


@pytest.fixture(scope="module")
def small_tensor_list(small_tensor_1, small_tensor_2) -> list[torch.Tensor]:
    return [small_tensor_1, small_tensor_2]


@pytest.fixture(scope="module")
def small_batch_feature_1() -> BatchFeature:
    return BatchFeature({"pixel_values": torch.rand(3, 30, 30)})


@pytest.fixture(scope="module")
def small_batch_feature_2() -> BatchFeature:
    return BatchFeature({"pixel_values": torch.rand(3, 30, 30)})


@pytest.fixture(scope="module")
def small_batch_feature_list(small_batch_feature_1, small_batch_feature_2) -> list[BatchFeature]:
    return [small_batch_feature_1, small_batch_feature_2]


@pytest.fixture(scope="module")
def small_ndarray_1() -> np.ndarray:
    return np.random.rand(3, 30, 30).astype(np.float32)


@pytest.fixture(scope="module")
def small_ndarray_2() -> np.ndarray:
    return np.random.rand(3, 30, 30).astype(np.float32)


@pytest.fixture(scope="module")
def small_ndarray_list(small_ndarray_1, small_ndarray_2) -> list[np.ndarray]:
    return [small_ndarray_1, small_ndarray_2]


@pytest.fixture(scope="module")
def model_and_processor():
    model_name = CLASSIFICATION_MODELS[0]
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


FAST_METHOD_SPECS = [
    (ImageGradientShap, GradientShapImagePerturbator, MeanAggregator, ImageGranularity.PIXEL),
    (ImageIntegratedGradients, LinearInterpolationImagePerturbator, TrapezoidalMeanAggregator, ImageGranularity.PIXEL),
    (ImageKernelShap, ShapImagePerturbator, LinearRegressionAggregator, ImageGranularity.PATCH),
    (ImageOcclusion, OcclusionImagePerturbator, OcclusionAggregator, ImageGranularity.PATCH),
    (ImageSaliency, ImageTensorPerturbator, Aggregator, ImageGranularity.PIXEL),
    (ImageSmoothGrad, GaussianNoiseImagePerturbator, MeanAggregator, ImageGranularity.PIXEL),
    (ImageSquareGrad, GaussianNoiseImagePerturbator, SquaredMeanAggregator, ImageGranularity.PIXEL),
    (ImageVarGrad, GaussianNoiseImagePerturbator, VarianceAggregator, ImageGranularity.PIXEL),
]


# (fixture_name, targets): single-unit inputs get one target, list inputs get one target per item.
INPUT_FIXTURES = [
    ("image1", 0),
    ("image_list", [0, 1]),
    ("small_tensor_1", 0),
    ("small_tensor_list", [0, 1]),
    ("small_batch_feature_1", 0),
    ("small_batch_feature_list", [0, 1]),
    ("small_ndarray_1", 0),
    ("small_ndarray_list", [0, 1]),
]


@pytest.mark.parametrize("input_fixture, targets", INPUT_FIXTURES)
@pytest.mark.parametrize("preprocess, image_mean, image_std", [(True, None, None), (False, 2.0, 0.75)])
@pytest.mark.parametrize("resize_strategy", list(GranularityResizeStrategy))
@pytest.mark.parametrize("attribution_method, perturbator, aggregator, granularity", FAST_METHOD_SPECS)
def test_vision_attribution_methods_fast(
    request,
    model_and_processor,
    attribution_method,
    perturbator,
    aggregator,
    granularity,
    resize_strategy,
    preprocess,
    image_mean,
    image_std,
    input_fixture,
    targets,
):
    model, processor = model_and_processor
    inputs = request.getfixturevalue(input_fixture)

    if attribution_method in (ImageOcclusion, ImageSaliency):
        explainer = attribution_method(
            model,
            processor,
            resize_strategy=resize_strategy,
            preprocess=preprocess,
            image_mean=image_mean,
            image_std=image_std,
        )
    else:
        explainer = attribution_method(
            model,
            processor,
            n_perturbations=5,
            resize_strategy=resize_strategy,
            preprocess=preprocess,
            image_mean=image_mean,
            image_std=image_std,
        )

    assert isinstance(explainer.perturbator, perturbator)
    assert isinstance(explainer.aggregator, aggregator)

    # preprocess=False rejects raw PIL/ndarray: they can only be normalized by the processor.
    items = inputs if isinstance(inputs, list) else [inputs]
    if not preprocess and any(isinstance(i, (Image.Image, np.ndarray)) for i in items):
        with pytest.raises(ValueError, match="raw inputs must be torch.Tensor"):
            explainer.explain(inputs, targets)
        return

    output = explainer.explain(inputs, targets)

    expected_len = len(inputs) if isinstance(inputs, list) else 1
    assert len(output) == expected_len
    assert all(isinstance(o, ImageAttributionOutput) for o in output)
    assert all(o.granularity is granularity for o in output)
    assert all(o.granularity_resize is resize_strategy for o in output)
    assert all(o.attributions.device.type == "cpu" for o in output)
    assert all(o.targets.device.type == "cpu" for o in output)

    # Reassemble the per-output model inputs (scattered one-per-ImageAttributionOutput) and check
    # they match a fresh, deterministic re-run of process_model_inputs on the same raw inputs.
    stored_inputs = [o.model_inputs_to_explain for o in output]
    expected_inputs = explainer.process_model_inputs(inputs)
    for stored, expected in zip(stored_inputs, expected_inputs, strict=True):
        assert stored.keys() == expected.keys()
        assert all(torch.equal(stored[key], expected[key]) for key in expected)

    if preprocess:
        if getattr(processor, "do_normalize", False):
            expected_mean = torch.as_tensor(processor.image_mean, dtype=torch.float32).view(-1, 1, 1)
            expected_std = torch.as_tensor(processor.image_std, dtype=torch.float32).view(-1, 1, 1)
        else:
            expected_mean = torch.as_tensor(0.0, dtype=torch.float32).view(-1, 1, 1)
            expected_std = torch.as_tensor(1.0, dtype=torch.float32).view(-1, 1, 1)
    else:
        expected_mean = torch.as_tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
        expected_std = torch.as_tensor(image_std, dtype=torch.float32).view(-1, 1, 1)

    for o in output:
        assert torch.equal(o.image_mean, expected_mean)
        assert torch.equal(o.image_std, expected_std)
