"""Self-contained HTML retrieval explanations with precomputed PNG heatmaps.

Scores stay in canonical `(query tokens, image units)` form until expansion by
ImageGranularity. Only display artifacts cross the Python/JavaScript boundary:
one base image and independently normalized overlays.
"""

from __future__ import annotations

import base64
import io
import json
import os
import uuid

import numpy as np
import torch
from IPython.display import HTML, display
from PIL import Image, ImageColor

from interpreto.attributions.base import ImageRetrievalAttributionOutput
from interpreto.commons.granularity import GranularityResizeStrategy

from .commons import _build_html_header, _save_html
from .image_utils import _denormalize, _to_displayable_image


def _png_uri(pixels: np.ndarray) -> str:
    """Encode uint8 grayscale, RGB or RGBA pixels without external resources."""
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _normalize_signed(values: torch.Tensor) -> torch.Tensor:
    """Normalize magnitudes over [0, max(abs(values))], then restore signs.

    Including zero keeps weaker nonzero scores visible even if every value is
    nonzero. Only an all-zero map stays transparent. Each token-wise image map
    is normalized independently.
    """
    magnitudes = values.abs()
    hi = magnitudes.max()
    if hi == 0:
        return torch.zeros_like(values)
    return values.sign() * (magnitudes / hi)


def _normalize_l1_signed(values: torch.Tensor) -> torch.Tensor:
    """Scale signed values so their absolute values sum to 1.

    Signs are preserved, so negative entries stay negative. An all-zero input
    maps to all zeros.
    """
    total = values.abs().sum()
    if total == 0:
        return torch.zeros_like(values)
    return values / total


def _heatmap_overlay(heatmap: torch.Tensor, colors: np.ndarray, alpha: float) -> str:
    """Encode signed min–max scores as a transparent, two-color PNG overlay."""
    values = _normalize_signed(heatmap).numpy()
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.where((values >= 0)[..., None], colors[0], colors[1])
    rgba[..., 3] = np.rint(np.abs(values) * alpha * 255).astype(np.uint8)
    return _png_uri(rgba)


def plot_image_retrieval_attribution(
    attribution_output: ImageRetrievalAttributionOutput,
    *,
    save_path: str | os.PathLike[str] | None = None,
    alpha: float = 0.5,
    positive_color: str = "#ff0000",
    negative_color: str = "#0000ff",
    grayscale_background: bool = True,
    resize_strategy: GranularityResizeStrategy = GranularityResizeStrategy.BICUBIC,
    max_image_side: int | None = 768,
    custom_css: str = "",
) -> None:
    """Display a query above its image with interactive token-wise heatmaps.

    The displayed image is the raw input image (`input_image`, before any
    processor resize or normalization), so its aspect is preserved. Heatmaps
    live on the model's resized grid and are warped onto the displayed image
    for rendering only — the underlying scores are untouched. This warp is the
    exact inverse of preprocessing only for resize-only processors (e.g. the
    SigLIP processor behind ColPali); a processor that crops or pads would
    misalign the overlay. Without `input_image` (e.g. hand-built outputs), the
    de-normalized model input is shown instead.

    The default image explanation is a similarity-weighted sum of the token
    attribution maps: each token map is first L1-normalized (absolute values
    summing to 1, so methods with different score scales contribute
    comparably), then weighted by that token's similarity. Tokens with low
    similarity therefore barely affect the default view. Hover a token to
    preview its own map; click to pin one token, and click it again to return
    to the weighted sum. Hovering another token temporarily overrides the
    pinned view. Keyboard focus previews tokens and Enter/Space selects them;
    Escape clears selection.

    Every token-wise image heatmap uses zero-anchored min–max normalization of
    absolute values, with signs restored afterward, yielding scores in
    [-1, 1]; the weighted aggregate is normalized the same way for display, so
    the strongest displayed value always reaches full opacity. Query highlights
    come from the per-token **similarities**, max-normalized the same way (the
    strongest token reaches full opacity); the L1-normalized weights are only
    used to combine the maps, never for display. The tooltip on each token
    shows its raw similarity. Red/blue encode sign; opacity encodes normalized
    magnitude.

    Patch maps are smoothly expanded with bicubic interpolation by default.
    Interpolated scores are clamped to the original map's range before
    normalization to avoid overshoot introducing contributions of a new sign.

    Args:
        attribution_output: One image retrieval explanation, with `(t, g)`
            attributions, `t` token strings in `targets`, and `t` per-token
            similarities.
        save_path: Optional path for a self-contained, offline HTML export.
        alpha: Maximum image heatmap opacity, between 0 and 1.
        positive_color: Positive contribution color (Pillow-compatible RGB color).
        negative_color: Negative contribution color (Pillow-compatible RGB color).
        grayscale_background: Display the image in grayscale for better contrast.
        resize_strategy: Display interpolation for patch maps, defaulting to
            BICUBIC. Use NEAREST for crisp patch boundaries. Pixel maps are
            already at image resolution and are not interpolated. The same
            strategy warps heatmaps from the model grid onto the displayed image.
        max_image_side: Longest displayed image side in pixels, to keep saved
            HTML light. `None` keeps the native resolution. Attribution scores
            are never affected, only the rendering.
        custom_css: Additional CSS for the HTML visualization.

    Raises:
        ValueError: If scores or similarities are empty, non-finite or
            misaligned with tokens, alpha is outside [0, 1], max_image_side is
            not a positive int, or the image cannot be prepared for display.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if max_image_side is not None and (
        isinstance(max_image_side, bool) or not isinstance(max_image_side, int) or max_image_side < 1
    ):
        raise ValueError(f"max_image_side must be a positive int or None, got {max_image_side!r}.")
    scores = attribution_output.attributions.detach().to(device="cpu", dtype=torch.float32)
    if scores.ndim != 2 or not scores.numel() or scores.shape[0] != len(attribution_output.targets):
        raise ValueError("Expected non-empty (t, g) attributions with one row per query token in targets.")
    if not torch.isfinite(scores).all():
        raise ValueError("Image retrieval attributions must contain only finite scores.")
    similarities = attribution_output.similarities.detach().to(device="cpu", dtype=torch.float32).flatten()
    if similarities.numel() != scores.shape[0]:
        raise ValueError(
            "Expected one similarity per query token: "
            f"got {similarities.numel()} similarities for {scores.shape[0]} tokens."
        )
    if not torch.isfinite(similarities).all():
        raise ValueError("Image retrieval similarities must contain only finite scores.")
    colors = np.array([ImageColor.getrgb(positive_color), ImageColor.getrgb(negative_color)], dtype=np.uint8)
    if colors.shape != (2, 3):
        raise ValueError("positive_color and negative_color must be RGB colors without alpha.")

    if attribution_output.input_image is not None:
        image = _to_displayable_image(attribution_output.input_image)
    else:
        image = _denormalize(attribution_output)
    if image.shape[-1] == 1:
        image = image[..., 0]
    elif image.shape[-1] != 3:
        raise ValueError("The image must be grayscale (C = 1) or RGB (C = 3).")
    elif grayscale_background:
        image = image @ np.array([0.2989, 0.587, 0.114], dtype=np.float32)

    # Downscale for display only: base image and warped overlays share these
    # dims, so alignment is exact by construction. Scores are never affected.
    display_size = (image.shape[0], image.shape[1])
    if max_image_side is not None and max(display_size) > max_image_side:
        scale = max_image_side / max(display_size)
        display_size = (max(1, round(image.shape[0] * scale)), max(1, round(image.shape[1] * scale)))
        image = (
            np.asarray(
                Image.fromarray(np.rint(image * 255).astype(np.uint8)).resize(
                    (display_size[1], display_size[0]), Image.Resampling.BICUBIC
                ),
                dtype=np.float32,
            )
            / 255.0
        )

    # Token weights preserve similarity signs and sum to 1 in absolute value,
    # so low-similarity tokens barely affect the default aggregate view.
    weights = _normalize_l1_signed(similarities)
    # L1-normalize each token map on the original `(t, g)` scores — before any
    # interpolation — so tokens contribute according to similarity, not to the
    # raw scale of their attribution method. Zero maps stay zero.
    row_totals = scores.abs().sum(dim=1, keepdim=True)
    normalized_scores = torch.where(row_totals > 0, scores / row_totals.clamp_min(1e-12), torch.zeros_like(scores))
    aggregate_contribution = (weights.unsqueeze(1) * normalized_scores).sum(dim=0, keepdim=True)

    # Expand one map at a time to avoid retaining a full (t, H, W) float tensor:
    # canonical (t, g) -> model grid (Hm, Wm) -> displayed image (Hd, Wd).
    def prepare_map(contribution: torch.Tensor) -> str:
        heatmap = attribution_output.granularity.resize_to_image(
            contribution=contribution,
            resize_strategy=resize_strategy,
            inputs=attribution_output.model_inputs_to_explain,
            patch_size=attribution_output.patch_size,
        )
        heatmap = resize_strategy.resize(heatmap, output_size=display_size)
        heatmap = heatmap.clamp(contribution.min(), contribution.max())
        return _heatmap_overlay(heatmap[0], colors, alpha)

    data = {
        "tokens": list(attribution_output.targets),
        "token_scores": similarities.tolist(),
        # Display-only scale: max-normalized so highlights stay visible no
        # matter how many tokens share the L1 weight mass used above.
        "token_highlights": _normalize_signed(similarities).tolist(),
        "token_style": {
            "positive_color": "#" + "".join(f"{channel:02x}" for channel in colors[0]),
            "negative_color": "#" + "".join(f"{channel:02x}" for channel in colors[1]),
            "min": -1,
            "max": 1,
        },
        "image": _png_uri(np.rint(image * 255).astype(np.uint8)),
        "aggregate": prepare_map(aggregate_contribution),
        "maps": [prepare_map(row.unsqueeze(0)) for row in scores],
    }
    # Escape '<' even inside JSON strings: HTML parsers recognize </script>
    # regardless of JavaScript quoting. Token labels are rendered as textContent.
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    unique_id = f"retrieval-{uuid.uuid4()}"
    html = _build_html_header(custom_css)
    html += f'<div id="{unique_id}" class="image-retrieval-visualization"></div>'
    html += f'<script>new ImageRetrievalVisualization("{unique_id}", {payload});</script></body></html>'
    if save_path is not None:
        _save_html(html, save_path)
    display(HTML(html))
