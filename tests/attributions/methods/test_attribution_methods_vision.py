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

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.image_processing_utils import BatchFeature

from interpreto.attributions.aggregations.base import MeanAggregator, OcclusionAggregator
from interpreto.attributions.base_merged import ImageAttributionOutput
from interpreto.attributions.methods.occlusion_merge import Occlusion
from interpreto.attributions.methods.smoothgrad_merged import SmoothGrad
from interpreto.attributions.perturbations.gaussian_noise_perturbation_merged import GaussianNoisePerturbator
from interpreto.attributions.perturbations.occlusion_perturbation_merged import OcclusionPerturbator
from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity
from interpreto.visualizations.image_attributions_merge import plot_image_attribution

plt.switch_backend("Agg")  # headless: render into a buffer, never open a window.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_CLASSIFICATION_MODELS = [
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
    model_name = IMAGE_CLASSIFICATION_MODELS[0]
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


FAST_METHOD_SPECS = [
    (Occlusion, OcclusionPerturbator, OcclusionAggregator, ImageGranularity.PATCH),
    (SmoothGrad, GaussianNoisePerturbator, MeanAggregator, ImageGranularity.PIXEL),
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


def _assert_explains_and_plots(
    explainer,
    processor,
    perturbator_cls,
    aggregator_cls,
    inputs,
    targets,
    granularity,
    resize_strategy,
):
    """
    Shared body for the vision method tests. Given an already-built explainer, check that its
    perturbator/aggregator are the expected types, run `explain`, and check the outputs: length,
    type, granularity/resize strategy, that everything is on the CPU, that the stored per-output
    model inputs match a fresh deterministic re-run, and that the de-normalization stats are the
    expected ones. Finally confirm each output is plottable on the Agg backend.

    Method-specific knobs (Sobol's sampler/order, LIME's kernel_width, ...) are *not* checked
    here — those stay in each method's test. This only covers what every method shares.

    Returns the list of `ImageAttributionOutput`.
    """
    assert isinstance(explainer.perturbator, perturbator_cls), (
        f"explainer.perturbator must be an instance of {perturbator_cls}, got {type(explainer.perturbator)}"
    )
    assert isinstance(explainer.aggregator, aggregator_cls), (
        f"explainer.aggregator must be an instance of {aggregator_cls}, got {type(explainer.aggregator)}"
    )

    output = explainer.explain(inputs, targets)

    expected_len = len(inputs) if isinstance(inputs, list) else 1
    assert len(output) == expected_len, (
        "explain() must return one ImageAttributionOutput per input for list inputs, or exactly one for a single "
        f"input; expected {expected_len}, got {len(output)}"
    )
    assert all(isinstance(o, ImageAttributionOutput) for o in output), (
        "every element of output must be an ImageAttributionOutput"
    )
    assert all(o.granularity is granularity for o in output), (
        "every output must carry the granularity the explainer was built with"
    )
    assert all(o.granularity_resize is resize_strategy for o in output), (
        "every output must carry the resize strategy the explainer was built with"
    )
    assert all(o.attributions.device.type == "cpu" for o in output), (
        "attributions must be moved back to CPU regardless of the compute device"
    )
    assert all(o.targets.device.type == "cpu" for o in output), (
        "targets must be moved back to CPU regardless of the compute device"
    )

    # Reassemble the per-output model inputs (scattered one-per-ImageAttributionOutput) and check
    # they match a fresh, deterministic re-run of process_model_inputs on the same raw inputs.
    stored_inputs = [o.model_inputs_to_explain for o in output]
    expected_inputs = explainer.process_model_inputs(inputs)
    for stored, expected in zip(stored_inputs, expected_inputs, strict=True):
        assert stored.keys() == expected.keys(), (
            "the model inputs stored on each output must have the same keys as a fresh process_model_inputs "
            "call on the same raw inputs"
        )
        assert all(torch.equal(stored[key], expected[key]) for key in expected), (
            "the model inputs stored on each output must equal, tensor for tensor, a fresh process_model_inputs "
            "call on the same raw inputs"
        )

    if getattr(processor, "do_normalize", False):
        expected_mean = torch.as_tensor(processor.image_mean, dtype=torch.float32).view(-1, 1, 1)
        expected_std = torch.as_tensor(processor.image_std, dtype=torch.float32).view(-1, 1, 1)
    else:
        expected_mean = torch.as_tensor(0.0, dtype=torch.float32).view(-1, 1, 1)
        expected_std = torch.as_tensor(1.0, dtype=torch.float32).view(-1, 1, 1)

    for o in output:
        assert torch.equal(o.image_mean, expected_mean), (
            "each output must store the image_mean actually used to normalize it"
        )
        assert torch.equal(o.image_std, expected_std), (
            "each output must store the image_std actually used to normalize it"
        )

    # The produced outputs must be plottable: run the viz end-to-end on the Agg backend.
    for o in output:
        fig, _ = plot_image_attribution(o)
        plt.close(fig)

    return output


@pytest.mark.parametrize("input_fixture, targets", INPUT_FIXTURES)
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
    input_fixture,
    targets,
):
    model, processor = model_and_processor
    inputs = request.getfixturevalue(input_fixture)

    #    if attribution_method in (Occlusion, ImageSaliency):
    if attribution_method in (Occlusion,):
        explainer = attribution_method(
            model,
            processor,
            combination_strategy=resize_strategy,
        )
    else:
        explainer = attribution_method(
            model,
            processor,
            n_perturbations=5,
            combination_strategy=resize_strategy,
        )

    _assert_explains_and_plots(
        explainer,
        processor,
        perturbator,
        aggregator,
        inputs,
        targets,
        granularity,
        resize_strategy,
    )


# Sobol carries its own machinery (quasi-MC SequenceSampler + variance-based aggregation),
# so it gets its own test rather than a FAST_METHOD_SPECS row: it takes `n_token_perturbations`
# (not `n_perturbations`) and two extra knobs, the indices `order` and the `sampler`, both worth
# sweeping. Kept to a small representative set: both orders, all three samplers.
# SOBOL_SPECS = [
#     (5, SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.SOBOL),
#     (5, SobolIndicesOrders.TOTAL_ORDER, SequenceSamplers.HALTON),
#     (5, SobolIndicesOrders.FIRST_ORDER, SequenceSamplers.LatinHypercube),
# ]


# @pytest.mark.parametrize("input_fixture, targets", INPUT_FIXTURES)
# @pytest.mark.parametrize("resize_strategy", list(GranularityResizeStrategy))
# @pytest.mark.parametrize("n_token_perturbations, order, sampler", SOBOL_SPECS)
# def test_image_sobol(
#     request,
#     model_and_processor,
#     n_token_perturbations,
#     order,
#     sampler,
#     resize_strategy,
#     input_fixture,
#     targets,
# ):
#     model, processor = model_and_processor
#     inputs = request.getfixturevalue(input_fixture)

#     explainer = ImageSobol(
#         model,
#         processor,
#         n_token_perturbations=n_token_perturbations,
#         sobol_indices_order=order,
#         sampler=sampler,
#         resize_strategy=resize_strategy,
#     )

#     # The two Sobol-specific knobs must land on the right objects (stored as their `.value`).
#     assert explainer.perturbator.sampler_class == sampler.value, (
#         "explainer.perturbator.sampler_class should be the sampler value passed as input. "
#         f"Expected {sampler.value}, got {explainer.perturbator.sampler_class}"
#     )
#     assert explainer.aggregator.sobol_indices_order == order.value, (
#         "explainer.aggregator.sobol_indices_order should be the order value passed as input. "
#         f"Expected {order.value}, got {explainer.aggregator.sobol_indices_order}"
#     )

#     _assert_explains_and_plots(
#         explainer,
#         processor,
#         SobolImagePerturbator,
#         SobolAggregator,
#         inputs,
#         targets,
#         ImageGranularity.PATCH,
#         resize_strategy,
#     )

#     # The Sobol perturbator builds ((g + 2) * k, g) masks, where g = gh * gw is the number of
#     # patches and k = n_token_perturbations. Derive g from the processed pixel grid and the
#     # reconciled patch_size, then check the mask directly (mirrors the text-side Sobol test).
#     _, _, height, width = explainer.process_model_inputs(inputs)[0]["pixel_values"].shape
#     patch_size = explainer.perturbator.patch_size
#     seq_len = (height // patch_size) * (width // patch_size)
#     mask = explainer.perturbator.get_mask(seq_len)
#     assert isinstance(mask, torch.Tensor), "get_mask must return a torch.Tensor"
#     assert mask.shape == ((seq_len + 2) * n_token_perturbations, seq_len), (
#         "Sobol mask must have shape ((seq_len + 2) * n_token_perturbations, seq_len). Expected "
#         f"{((seq_len + 2) * n_token_perturbations, seq_len)}, got {tuple(mask.shape)}"
#     )
#     assert mask.dtype == torch.float32, "Sobol mask.dtype must be torch.float32"


# LIME carries its own machinery (random masking + a similarity-weighted linear surrogate), so
# like Sobol it gets its own test rather than a FAST_METHOD_SPECS row: on top of `n_perturbations`
# it takes `perturb_probability`, the `distance_function`, and the `kernel_width`, all worth
# sweeping. Values mirror the text-side LIME test: the three distance functions paired with the
# three kernel_width forms (None -> default fn, an int, a float).
# LIME_SPECS = [
#     (5, 0.5, DistancesFromMask.HAMMING, None),
#     (5, 0.8, DistancesFromMask.EUCLIDEAN, 5),
#     (5, 0.5, DistancesFromMask.COSINE, 0.5),
# ]


# @pytest.mark.parametrize("input_fixture, targets", INPUT_FIXTURES)
# @pytest.mark.parametrize("resize_strategy", list(GranularityResizeStrategy))
# @pytest.mark.parametrize("n_perturbations, perturb_probability, distance_function, kernel_width", LIME_SPECS)
# def test_image_lime(
#     request,
#     model_and_processor,
#     n_perturbations,
#     perturb_probability,
#     distance_function,
#     kernel_width,
#     resize_strategy,
#     input_fixture,
#     targets,
# ):
#     model, processor = model_and_processor
#     inputs = request.getfixturevalue(input_fixture)

#     explainer = ImageLime(
#         model,
#         processor,
#         n_perturbations=n_perturbations,
#         perturb_probability=perturb_probability,
#         distance_function=distance_function,
#         kernel_width=kernel_width,
#         resize_strategy=resize_strategy,
#     )

#     # The LIME-specific knobs must land on the right objects.
#     assert explainer.perturbator.n_perturbations == n_perturbations, (
#         "explainer.perturbator.n_perturbations should be the n_perturbations passed as input. "
#         f"Expected {n_perturbations}, got {explainer.perturbator.n_perturbations}"
#     )
#     assert pytest.approx(explainer.perturbator.perturb_probability, rel=1e-6) == perturb_probability, (
#         "explainer.perturbator.perturb_probability should be the perturb_probability passed as input. "
#         f"Expected {perturb_probability}, got {explainer.perturbator.perturb_probability}"
#     )
#     assert explainer.aggregator.distance_function == distance_function, (
#         "explainer.aggregator.distance_function should be the distance_function passed as input. "
#         f"Expected {distance_function}, got {explainer.aggregator.distance_function}"
#     )
#     assert explainer.aggregator.similarity_kernel == Kernels.EXPONENTIAL, (
#         "explainer.aggregator.similarity_kernel should be the exponential kernel LIME always uses. "
#         f"Expected {Kernels.EXPONENTIAL}, got {explainer.aggregator.similarity_kernel}"
#     )
#     # kernel_width=None falls back to the default kernel-width function; otherwise it is stored as-is.
#     if kernel_width is None:
#         assert explainer.aggregator.kernel_width == default_kernel_width_fn, (
#             "explainer.aggregator.kernel_width should fall back to default_kernel_width_fn when kernel_width=None. "
#             f"Got {explainer.aggregator.kernel_width}"
#         )
#     else:
#         assert explainer.aggregator.kernel_width == kernel_width, (
#             "explainer.aggregator.kernel_width should be the kernel_width passed as input. "
#             f"Expected {kernel_width}, got {explainer.aggregator.kernel_width}"
#         )

#     _assert_explains_and_plots(
#         explainer,
#         processor,
#         RandomMaskedImagePerturbator,
#         LinearRegressionAggregator,
#         inputs,
#         targets,
#         ImageGranularity.PATCH,
#         resize_strategy,
#     )

#     # The LIME perturbator builds (n_perturbations, g) masks, where g = gh * gw is the number of
#     # patches. Derive g from the processed pixel grid and the reconciled patch_size, then check
#     # the mask directly (mirrors the text-side LIME test).
#     _, _, height, width = explainer.process_model_inputs(inputs)[0]["pixel_values"].shape
#     patch_size = explainer.perturbator.patch_size
#     seq_len = (height // patch_size) * (width // patch_size)
#     mask = explainer.perturbator.get_mask(seq_len)
#     assert isinstance(mask, torch.Tensor), "get_mask must return a torch.Tensor"
#     assert mask.shape == (n_perturbations, seq_len), (
#         f"LIME mask must have shape (n_perturbations, seq_len). Expected {(n_perturbations, seq_len)}, got "
#         f"{tuple(mask.shape)}"
#     )
#     assert mask.dtype == torch.float32, "LIME mask.dtype must be torch.float32"


# End-to-end smoke test against a *real* ViT (not the tiny random one). The point is to catch
# any shape/dtype drift between what `explain` returns and what the viz consumes that the tiny
# model can hide. Everything else is deliberately kept small: one resize strategy (BILINEAR)
# and a handful of representative inputs. All ten methods run in the same
# function — Sobol and LIME use their default parameters here, so they need none of their extra
# knobs and construct exactly like the others. The raw tensor/ndarray are deliberately given a
# real, non-square size (3, 340, 270) so the processor's resize/crop path is actually exercised.
@pytest.fixture(scope="module")
def large_tensor() -> torch.Tensor:
    return torch.rand(3, 340, 270)


@pytest.fixture(scope="module")
def large_ndarray() -> np.ndarray:
    return np.random.rand(3, 340, 270).astype(np.float32)


SLOW_INPUT_FIXTURES = [
    ("image_list", [0, 1]),
    ("large_tensor", 0),
    ("large_ndarray", 0),
]

# SLOW_METHOD_SPECS = FAST_METHOD_SPECS + [
#     (ImageSobol, SobolImagePerturbator, SobolAggregator, ImageGranularity.PATCH),
#     (ImageLime, RandomMaskedImagePerturbator, LinearRegressionAggregator, ImageGranularity.PATCH),
# ]


# @pytest.mark.slow
# @pytest.mark.parametrize("input_fixture, targets", SLOW_INPUT_FIXTURES)
# @pytest.mark.parametrize("attribution_method, perturbator, aggregator, granularity", SLOW_METHOD_SPECS)
# def test_vision_attribution_methods_slow(
#     request,
#     attribution_method,
#     perturbator,
#     aggregator,
#     granularity,
#     input_fixture,
#     targets,
# ):
#     model = AutoModelForImageClassification.from_pretrained(SLOW_MODELS[0])
#     processor = AutoImageProcessor.from_pretrained(SLOW_MODELS[0])
#     inputs = request.getfixturevalue(input_fixture)

# Default parameters for every method (Sobol/LIME included), so construction is uniform.
# explainer = attribution_method(
#     model,
#     processor,
#     resize_strategy=GranularityResizeStrategy.BILINEAR,
# )

# _assert_explains_and_plots(
#     explainer,
#     processor,
#     perturbator,
#     aggregator,
#     inputs,
#     targets,
#     granularity,
#     GranularityResizeStrategy.BILINEAR,
# )
