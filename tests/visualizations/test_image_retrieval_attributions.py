"""Check numeric-to-image fidelity and self-contained retrieval HTML output."""

import base64
import io
import json
import re

import numpy as np
import pytest
import torch
from PIL import Image

from interpreto import plot_image_retrieval_attribution
from interpreto.attributions.base import ImageRetrievalAttributionOutput
from interpreto.commons.granularity import GranularityResizeStrategy, ImageGranularity
from interpreto.visualizations.image_retrieval_attributions import _normalize_l1_signed, _normalize_signed
from interpreto.visualizations.image_utils import _to_displayable_image


def _output(scores, similarities=None, granularity=ImageGranularity.PIXEL, patch_size=1, input_image=None):
    """A non-square image with known normalization, without loading a model."""
    if similarities is None:
        similarities = [1.0] * len(scores)
    return ImageRetrievalAttributionOutput(
        attributions=torch.tensor(scores, dtype=torch.bfloat16),
        targets=["first", "</script><b>second</b>"][: len(scores)],
        similarities=torch.tensor(similarities, dtype=torch.float32),
        model_inputs_to_explain={"pixel_values": torch.zeros(1, 3, 2, 4)},
        image_mean=torch.tensor(0.5),
        image_std=torch.tensor(0.25),
        granularity=granularity,
        granularity_resize=GranularityResizeStrategy.NEAREST,
        patch_size=patch_size,
        input_image=input_image,
    )


def _render(tmp_path, monkeypatch, output, **kwargs):
    displayed = []
    monkeypatch.setattr("interpreto.visualizations.image_retrieval_attributions.display", displayed.append)
    path = tmp_path / "retrieval.html"
    plot_image_retrieval_attribution(output, save_path=path, **kwargs)
    html = path.read_text()
    assert displayed[0].data == html
    match = re.search(r'new ImageRetrievalVisualization\("[^"]+", (.*?)\);</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match[1]), html


def _decode(uri):
    return np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))


@pytest.mark.parametrize("granularity", [ImageGranularity.PIXEL, ImageGranularity.PATCH])
def test_similarity_weighted_aggregate_and_query_highlights(tmp_path, monkeypatch, granularity):
    """Query highlights come from similarities; the aggregate weights L1-normalized maps."""
    scores = [[4, -2], [-1, -1]]
    if granularity is ImageGranularity.PIXEL:
        scores = [[4, 4, -2, -2] * 2, [-1] * 8]
    output = _output(scores, [3.0, 1.0], granularity, patch_size=2)
    data, html = _render(
        tmp_path,
        monkeypatch,
        output,
        alpha=1,
        grayscale_background=False,
        resize_strategy=GranularityResizeStrategy.NEAREST,
    )
    assert data["token_scores"] == [3.0, 1.0]
    # Display highlights are max-normalized (strongest token at full opacity),
    # while the aggregate itself uses L1-normalized similarity weights.
    assert data["token_highlights"] == [1.0, pytest.approx(1 / 3)]
    # Patch and pixel inputs both reconstruct two rows of the same 4-pixel map.
    aggregate = _decode(data["aggregate"])
    assert aggregate.shape == (2, 4, 4)
    # Weighted aggregate of L1-normalized rows: first half positive, second half
    # negative, with equal display magnitudes after zero-anchored normalization.
    np.testing.assert_array_equal(aggregate[0], [[255, 0, 0, 255]] * 2 + [[0, 0, 255, 255]] * 2)
    np.testing.assert_array_equal(_decode(data["maps"][0])[0, :, 3], [255, 255, 128, 128])
    assert np.all(_decode(data["maps"][1])[..., 3] == 255)
    assert np.all(_decode(data["image"]) == 128)
    assert data["tokens"] == output.targets
    assert "</script><b>second</b>" not in html
    assert "window.ImageRetrievalVisualization" in html
    assert "<script src=" not in html


@pytest.mark.parametrize(
    ("similarities", "expected_index"),
    [([1.0, 0.0], 0), ([0.0, 2.0], 1)],
)
def test_zero_similarity_token_does_not_impact_aggregate(tmp_path, monkeypatch, similarities, expected_index):
    """A zero-similarity token is invisible in the default aggregate view."""
    scores = [[4, 4, -2, -2] * 2, [-1] * 8]
    data, _ = _render(
        tmp_path,
        monkeypatch,
        _output(scores, similarities),
        alpha=1,
        resize_strategy=GranularityResizeStrategy.NEAREST,
    )
    assert data["token_highlights"][1 - expected_index] == 0
    # Display normalization cancels the global weight scale, so the aggregate
    # matches the surviving token's own map pixel-for-pixel.
    np.testing.assert_array_equal(_decode(data["aggregate"]), _decode(data["maps"][expected_index]))


def test_zero_maps_are_transparent(tmp_path, monkeypatch):
    data, _ = _render(tmp_path, monkeypatch, _output([[0] * 8], [1.0]))
    for image_map in [data["aggregate"], *data["maps"]]:
        assert not _decode(image_map)[..., 3].any()
    assert data["token_scores"] == [1.0]
    assert _decode(data["image"]).shape == (2, 4)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 0.5], [1.0, 0.5]),
        ([2.0, -4.0, 6.0], [1 / 3, -2 / 3, 1.0]),
        ([-2.0, -4.0, -6.0], [-1 / 3, -2 / 3, -1.0]),
        ([0.0, -2.0, 4.0], [0.0, -0.5, 1.0]),
        ([3.0, -3.0], [1.0, -1.0]),
    ],
)
def test_signed_magnitude_minmax(values, expected):
    torch.testing.assert_close(_normalize_signed(torch.tensor(values)), torch.tensor(expected))


def test_l1_signed_normalization():
    torch.testing.assert_close(_normalize_l1_signed(torch.tensor([3.0, -1.0])), torch.tensor([0.75, -0.25]))
    torch.testing.assert_close(_normalize_l1_signed(torch.tensor([0.0, 0.0])), torch.tensor([0.0, 0.0]))


def test_bicubic_patch_display_is_smooth_and_preserves_sign(tmp_path, monkeypatch):
    """A positive patch next to zero gains intermediate opacity, not blue ringing."""
    output = _output([[0, 1]], [1.0], ImageGranularity.PATCH, patch_size=2)
    smooth, _ = _render(tmp_path, monkeypatch, output, alpha=1)
    crisp, _ = _render(tmp_path, monkeypatch, output, alpha=1, resize_strategy=GranularityResizeStrategy.NEAREST)
    rgba = _decode(smooth["maps"][0])
    assert np.any((rgba[..., 3] > 0) & (rgba[..., 3] < 255))
    assert not rgba[..., 2].any()
    assert set(np.unique(_decode(crisp["maps"][0])[..., 3])) == {0, 255}
    assert rgba[..., 3].min() == 0 and rgba[..., 3].max() == 255


@pytest.mark.parametrize("scores", [[], [[float("nan")] * 8], [[float("inf")] * 8]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError, match="attributions|finite scores"):
        plot_image_retrieval_attribution(_output(scores))


def test_token_alignment_rejected():
    output = _output([[1] * 8])
    output.targets = []
    with pytest.raises(ValueError, match="one row per query token"):
        plot_image_retrieval_attribution(output)


@pytest.mark.parametrize("bad_side", [0, -4, 7.5, "768", True])
def test_max_image_side_rejected(bad_side):
    with pytest.raises(ValueError, match="max_image_side"):
        plot_image_retrieval_attribution(_output([[1] * 8]), max_image_side=bad_side)


def test_original_image_display_and_exact_nearest_warp(tmp_path, monkeypatch):
    """The base image keeps the original aspect; overlays warp onto its dims.

    Integer 2x upscale with NEAREST replicates each model-grid cell exactly,
    so the overlay content is fully determined.
    """
    raw = Image.new("RGB", (8, 4), color=(0, 0, 0))  # PIL size is (W, H)
    data, _ = _render(
        tmp_path,
        monkeypatch,
        _output([[1, 2, 3, 4, 5, 6, 7, 8]], [1.0], input_image=raw),
        alpha=1,
        grayscale_background=False,
        resize_strategy=GranularityResizeStrategy.NEAREST,
        max_image_side=None,
    )
    base = _decode(data["image"])
    assert base.shape == (4, 8, 3)
    assert not base.any()
    overlay = _decode(data["aggregate"])
    assert overlay.shape == (4, 8, 4)
    # Single token with weight 1: the aggregate matches the token map exactly.
    np.testing.assert_array_equal(overlay, _decode(data["maps"][0]))
    alphas = np.rint(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32) / 8 * 255).astype(np.uint8)
    expected_alpha = np.repeat(np.repeat(alphas.reshape(2, 4), 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(overlay[..., 3], expected_alpha)
    assert np.all(overlay[..., :3][..., 0] == 255)  # all positive -> red
    assert not overlay[..., 1:3].any()  # no green/blue, only the alpha channel varies


def test_max_image_side_downscales_base_and_overlays(tmp_path, monkeypatch):
    raw = Image.new("RGB", (12, 8), color=(255, 255, 255))
    data, _ = _render(
        tmp_path,
        monkeypatch,
        _output([[1] * 8], [1.0], input_image=raw),
        grayscale_background=False,
        max_image_side=6,
    )
    assert _decode(data["image"]).shape == (4, 6, 3)
    assert _decode(data["aggregate"]).shape == (4, 6, 4)
    assert _decode(data["maps"][0]).shape == (4, 6, 4)


@pytest.mark.parametrize(
    ("raw", "expected_shape", "expected_max"),
    [
        (Image.new("RGB", (4, 6), color=(255, 0, 0)), (6, 4, 3), 1.0),
        (Image.new("L", (4, 6), color=128), (6, 4, 1), 128 / 255),
        (Image.new("RGBA", (4, 6), color=(255, 0, 0, 0)), (6, 4, 3), 1.0),
        (np.full((6, 4, 3), 255, dtype=np.uint8), (6, 4, 3), 1.0),
        (np.zeros((6, 4)), (6, 4, 1), 0.0),
        (np.full((6, 4, 3), 0.5, dtype=np.float32), (6, 4, 3), 0.5),
        (torch.ones(3, 6, 4), (6, 4, 3), 1.0),
        (torch.zeros(6, 4), (6, 4, 1), 0.0),
        (torch.full((1, 3, 6, 4), 0.25), (6, 4, 3), 0.25),
        (torch.full((3, 6, 4), 255, dtype=torch.uint8), (6, 4, 3), 1.0),
    ],
)
def test_to_displayable_image_conversions(raw, expected_shape, expected_max):
    image = _to_displayable_image(raw)
    assert image.shape == expected_shape
    assert image.dtype == np.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert image.max() == pytest.approx(expected_max)


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros((6, 4, 2)),
        np.zeros((6, 4, 3, 1)),
        np.zeros((6, 4), dtype=np.int32),
        torch.zeros(2, 3, 6, 4),
        torch.zeros(6, 4, 3),  # channel-last tensors are rejected: torch means channel-first
        "not an image",
    ],
)
def test_to_displayable_image_rejections(raw):
    with pytest.raises(ValueError, match="image|Image"):
        _to_displayable_image(raw)


def test_to_displayable_image_clamps_floats_and_detaches():
    raw = torch.tensor([[[2.0, -1.0]]], requires_grad=True)
    image = _to_displayable_image(raw)
    np.testing.assert_array_equal(image[..., 0], [[1.0, 0.0]])


@pytest.mark.parametrize(
    "similarities",
    [[1.0], [[1.0, 2.0], [3.0, 4.0]], [float("nan"), 1.0], [float("inf"), 1.0]],
)
def test_invalid_similarities_rejected(similarities):
    output = _output([[1] * 8, [2] * 8], similarities)
    with pytest.raises(ValueError, match="[Ss]imilarit"):
        plot_image_retrieval_attribution(output)
