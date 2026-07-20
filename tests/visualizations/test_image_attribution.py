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

"""
Test file to test the visualization functions
"""

import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor

from interpreto.attributions.base import ImageAttributionOutput
from interpreto.commons.granularity import ImageGranularity
from interpreto.visualizations.image_attributions import (
    _clip_percentile,
    _color_limits,
    _denormalize,
    _prepare_heatmap,
    _to_grayscale,
    plot_image_attribution,
)

matplotlib.use("Agg")  # headless: build figures in memory, never try to open a window


def test_clip_percentile_constant_is_unchanged():
    # A constant array is a fixed point: lo == hi == 10, clip is a no-op.
    arr = np.full(20, 10.0)
    out = _clip_percentile(arr, percentile=np.random.uniform(0, 50))
    assert np.array_equal(out, arr)


def test_clip_percentile_pulls_tails_to_bounds():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = _clip_percentile(arr, percentile=0.2)
    expected = np.array([1.008, 2.0, 3.0, 4.0, 4.992])
    assert np.allclose(out, expected)


def test_prepare_heatmap():
    # _prepare_heatmap only reads `.attributions_image`, so a SimpleNamespace stand-in is enough.
    # (1, 2, 2): t=1 target, 2x2 spatial. Mixed signs + a value >1 on purpose.
    output = SimpleNamespace(attributions_image=torch.tensor([[[10.0, 20.0], [-3.0, 0.0]]]))

    # absolute_value=True -> no negative values
    abs_out = _prepare_heatmap(output, target_idx=0, clip_percentile=None, absolute_value=True)
    assert (abs_out >= 0).all()

    # absolute_value=False -> floating dtype, and scores kept in real scale (NOT rescaled to [0,1])
    raw_out = _prepare_heatmap(output, target_idx=0, clip_percentile=None, absolute_value=False)
    assert issubclass(raw_out.dtype.type, np.floating)
    assert raw_out.max() > 1.0 or raw_out.min() < 0.0


@pytest.mark.parametrize(
    "values, expected",
    [
        ([-3.0, 1.0, 20.0], (-20.0, 20.0)),  # straddles zero -> +/- m, m = max(|lo|, |hi|) = 20
        ([1.0, 2.0, 20.0], (-20.0, 20.0)),  # all positive -> still centered, uses upper half only
        ([-5.0, -2.0, -1.0], (-5.0, 5.0)),  # all negative -> still centered, uses lower half only
    ],
)
def test_color_limits(values, expected):
    out = _color_limits(np.array(values, dtype=np.float32))
    assert out == expected


def _denorm_output(pixel_values, image_mean=0.5, image_std=0.5):
    # _denormalize only reads these three attributes. Stats are viewed as (-1, 1, 1) to
    # broadcast over (C, H, W), exactly as _resolve_normalization_stats builds them.
    return SimpleNamespace(
        model_inputs_to_explain={"pixel_values": pixel_values},
        image_mean=torch.as_tensor(image_mean, dtype=torch.float32).view(-1, 1, 1),
        image_std=torch.as_tensor(image_std, dtype=torch.float32).view(-1, 1, 1),
    )


@pytest.mark.parametrize(
    "image_mean, image_std",
    [
        (None, torch.full((3, 1, 1), 0.5)),  # per-channel std, as the processor gives it
        (torch.as_tensor(0.5).view(-1, 1, 1), None),  # scalar mean, as _resolve_normalization_stats views it
        (None, None),
    ],
)
def test_denormalize_raises_without_stats(image_mean, image_std):
    # A None stat means the output was built by hand without saying how it was normalized.
    # Guessing an identity would display a normalized tensor as if it were the real image.
    output = SimpleNamespace(
        model_inputs_to_explain={"pixel_values": torch.rand(1, 3, 8, 8) * 2.0 - 1.0},
        image_mean=image_mean,
        image_std=image_std,
    )

    with pytest.raises(ValueError):
        _denormalize(output)


@pytest.mark.parametrize(
    "image_mean, image_std",
    [
        (0.5, 0.5),  # scalar -> view(-1, 1, 1) is (1, 1, 1), broadcasts over C
        ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # per-channel -> (3, 1, 1), what image_processor holds
        ([0.1, 0.5, 0.9], [0.2, 0.3, 0.4]),  # distinct per channel -> catches a channel-axis mix-up
    ],
)
def test_denormalize_inverts_the_normalization(image_mean, image_std):
    mean = torch.as_tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.as_tensor(image_std, dtype=torch.float32).view(-1, 1, 1)

    # Start from the displayable image we expect back, and normalize it the way the processor
    # would. It is in [0, 1] by construction, so the clamp never fires and _denormalize must
    # return it unchanged.
    image = torch.rand(1, 3, 8, 8)
    output = _denorm_output((image - mean) / std, image_mean, image_std)

    out = _denormalize(output)

    assert np.allclose(out, image[0].permute(1, 2, 0).numpy())


def test_denormalize_returns_channel_last_float_array():
    output = _denorm_output(torch.rand(1, 3, 8, 6) * 2.0 - 1.0)

    out = _denormalize(output)

    assert isinstance(out, np.ndarray)
    assert issubclass(out.dtype.type, np.floating)
    assert out.shape == (8, 6, 3)  # (C, H, W) -> (H, W, C) for imshow


@pytest.mark.parametrize(
    "pixel_values",
    [
        torch.rand(3, 8, 8),  # unbatched: process_model_inputs always leaves the batch dim on
        torch.rand(2, 3, 8, 8),  # a real batch: [0] would silently display sample 0 as the whole input
    ],
)
def test_denormalize_throws_on_wrong_image_shape(pixel_values):
    match = re.escape(str(tuple(pixel_values.shape)))

    with pytest.raises(ValueError, match=match):
        _denormalize(_denorm_output(pixel_values))


@pytest.mark.parametrize(
    "pixel_value, expected_span",
    [
        (5.0, "3.000"),  # 5.0 * 0.5 + 0.5 = 3.0, above 1
        (-5.0, "-2.000"),  # -5.0 * 0.5 + 0.5 = -2.0, below 0
    ],
)
def test_denormalize_warns_and_clamps_when_out_of_range(pixel_value, expected_span):
    # Stats that do not match how the tensor was normalized push the result outside [0, 1].
    # The range is reported, not repaired: fitting it would be a min-max stretch that
    # silently produces a plausible, wrong image.
    pixel_values = torch.full((1, 3, 4, 4), pixel_value)
    output = _denorm_output(pixel_values)

    with pytest.warns(UserWarning, match=expected_span):
        out = _denormalize(output)

    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "error_value",
    [
        1e-4,  # above float32 eps (~1.2e-7) so it survives, below _RANGE_TOLERANCE so it stays quiet
        -1e-4,
    ],
)
def test_denormalize_does_not_warn_inside_the_tolerance(error_value):
    # x * std + mean rarely lands exactly on 0.0/1.0; float error must not trip the warning.
    # Any warning is promoted to an error by the filterwarnings mark, so reaching the end passes.
    pixel_values = torch.tensor([-1.0, 1.0]).view(1, 1, 1, 2).expand(1, 3, 1, 2).contiguous() + error_value

    # the de-normalized image really does leave [0, 1] — only the tolerance keeps it quiet,
    # so the test is not passing just because the values landed in range.
    denormalized = pixel_values * 0.5 + 0.5
    assert denormalized.min() < 0.0 or denormalized.max() > 1.0

    _denormalize(_denorm_output(pixel_values))


@pytest.mark.parametrize(
    "shape",
    [
        (8, 8),  # 2D: no channel axis at all
        (1, 8, 8, 3),  # 4D: batch dim still on
        (8, 8, 2),  # C = 2: neither grayscale nor RGB
        (8, 8, 4),  # C = 4: RGBA is not accepted, the alpha is not silently dropped
    ],
)
def test_to_grayscale_raises_on_wrong_shape(shape):
    with pytest.raises(ValueError):
        _to_grayscale(np.random.rand(*shape).astype(np.float32))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_to_grayscale_matches_the_rec601_formula(dtype):
    # The `@` must contract the channel axis, not a spatial one: an axis mix-up still
    # returns a plausible 2D map, so compare against the weights applied by hand.
    img = np.random.rand(8, 6, 3).astype(dtype)

    out = _to_grayscale(img)

    expected = 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    assert out.shape == (8, 6)
    assert np.allclose(out, expected)


@pytest.fixture
def vit_processor():
    # The real ViT pipeline, constructed rather than downloaded: the class defaults already
    # are google/vit-base-patch16-224's (rescale 1/255, normalize with mean/std 0.5), so this
    # needs no network. 32x32 keeps it cheap, and matching the source image size makes the
    # resize a no-op, which is what lets the de-normalization round-trip exactly.
    return ViTImageProcessor(size={"height": 32, "width": 32})


@pytest.fixture
def image():
    # Deterministic; the values matter only in that they span the full uint8 range.
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))


def _processed_output(processor, image, fill_value, n_targets=1):
    # A genuinely processed image, with a constant attribution map tailored to its size.
    inputs = processor(image, return_tensors="pt")
    h, w = inputs["pixel_values"].shape[-2:]
    return ImageAttributionOutput(
        attributions=torch.full((n_targets, 4), fill_value),
        attributions_image=torch.full((n_targets, h, w), fill_value),
        granularity=ImageGranularity.PIXEL,
        model_inputs_to_explain=inputs,
        targets=torch.arange(n_targets),
        image_mean=torch.as_tensor(processor.image_mean, dtype=torch.float32).view(-1, 1, 1),
        image_std=torch.as_tensor(processor.image_std, dtype=torch.float32).view(-1, 1, 1),
    )


def test_denormalize_round_trips_a_real_processed_image(vit_processor, image):
    # The processor rescales by 1/255 and then normalizes; _denormalize undoes the normalize,
    # so it must land back on the rescaled image — not the raw [0, 255] one.
    output = _processed_output(vit_processor, image, 1.0)

    out = _denormalize(output)

    assert np.allclose(out, np.asarray(image) / 255.0, atol=1e-6)


@pytest.mark.parametrize("fill_value", [1.0, 0.0, -1.0])
def test_plot_image_attribution_centers_the_clim_on_zero(vit_processor, image, fill_value):
    # A constant map is a fixed point of clip_percentile, so m = max|score| = |fill_value|.
    # All three cases give (-1, 1): +/-1 through max|.|, and 0 through the `or 1.0` guard
    # that stops vmin == vmax.
    output = _processed_output(vit_processor, image, fill_value)

    fig, axes = plot_image_attribution(output)
    try:
        backdrop, heatmap = axes[0][0].images
        assert backdrop.get_clim() == (0.0, 1.0)  # true luminance, not autoscaled contrast
        assert heatmap.get_clim() == (-1.0, 1.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("target_idx", [None, 0, [0, 2]])
def test_plot_image_attribution_runs_on_a_fake_image(vit_processor, image, target_idx):
    # Smoke test: the whole path (de-normalize, grayscale, heatmap, colorbar) must draw a
    # real processed image without raising.
    output = _processed_output(vit_processor, image, 1.0, n_targets=3)

    fig, _ = plot_image_attribution(output, target_idx=target_idx)
    plt.close(fig)


def test_plot_image_attribution_raises_on_empty_target_idx(vit_processor, image):
    # Without the guard this is a ZeroDivisionError from the grid layout: min(cols, 0) == 0.
    output = _processed_output(vit_processor, image, 1.0)

    with pytest.raises(ValueError, match="empty"):
        plot_image_attribution(output, target_idx=[])


def test_plot_image_attribution_raises_on_out_of_range_target(vit_processor, image):
    output = _processed_output(vit_processor, image, 1.0, n_targets=3)

    with pytest.raises(IndexError, match="out of range"):
        plot_image_attribution(output, target_idx=3)
