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
Matplotlib visualization for `ImageAttributionOutput`.

The normalization / percentile-clipping idea follows xplique's `plots/image.py`
(Apache 2.0, DEEL / Université Paul Sabatier). Reimplemented here to handle
Interpreto's flat-attribution storage (shape `(t, l)`), reshaped to a 2D heatmap
at render time using the per-unit `(row, col)` coordinates in
`ImageAttributionOutput.elements`.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL.Image import Image as PILImage

from interpreto.attributions.base import ImageAttributionOutput


def _to_displayable(image: PILImage | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert PIL / ndarray / Tensor to a float HxWxC (or HxW) array in [0, 1]."""
    if isinstance(image, PILImage):
        arr = np.array(image)
    elif isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.dtype is torch.bfloat16:
            t = t.to(torch.float32)
        arr = t.numpy()
    else:
        arr = np.asarray(image)

    arr = arr.astype(np.float32)
    # (C, H, W) -> (H, W, C) when channel-first is detectable.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def _clip_percentile(arr: np.ndarray, percentile: float) -> np.ndarray:
    """Clip at (percentile, 100-percentile) to suppress outliers in the heatmap."""
    lo = np.percentile(arr, percentile)
    hi = np.percentile(arr, 100.0 - percentile)
    return np.clip(arr, lo, hi)


def _prepare_heatmap(
    attribution_output: ImageAttributionOutput,
    target_idx: int,
    clip_percentile: float | None,
    absolute_value: bool,
) -> np.ndarray:
    """
    Reshape flat `attributions[target_idx]` (shape `(l,)`) into a 2D heatmap using
    `elements` to recover the grid shape.

    The heatmap is returned in the **real attribution-score scale** (after the
    optional abs + percentile clip). It is deliberately NOT rescaled to [0, 1]:
    color normalization is left to `imshow`'s vmin/vmax at draw time so each
    panel's `colorbar` shows genuine scores. Rescaling here would collapse every
    method's legend to a meaningless 0->1 axis and make cross-method comparison
    impossible.
    """
    row = attribution_output.attributions[target_idx].detach().cpu()
    if row.dtype is torch.bfloat16:
        row = row.to(torch.float32)
    flat = row.numpy().astype(np.float32)

    elements = attribution_output.elements
    #for a 30*30 image with patch_size = 2, elements stores [(0,0),(0,1),...,(14,14)] so the heatmap is effectively a 15*15 array, which is what we want.
    #TODO: This works for rectangular Granularity but may break if we introduce other type of Granularities. Check if another solution is needed.
    rows = max(e[0] for e in elements) + 1
    cols = max(e[1] for e in elements) + 1
    heatmap = flat.reshape(rows, cols)

    if absolute_value:
        heatmap = np.abs(heatmap)
    if clip_percentile is not None:
        heatmap = _clip_percentile(heatmap, clip_percentile)

    return heatmap


def _draw_attribution_on_ax(
    ax,
    img_disp: np.ndarray,
    heatmap: np.ndarray,
    *,
    cmap: str,
    alpha: float,
    interpolation: str,
    colorbar: bool,
    fig,
    **plot_kwargs,
):
    """
    Draw one image + heatmap pair onto `ax` as a live mappable and (optionally)
    attach a per-axes colorbar showing the heatmap's real score range.

    Centralizing the draw here keeps the single-method and multi-method entry
    points pixel-for-pixel identical, and — crucially — keeps each panel's
    colorbar tied to that panel's own vmin/vmax, so every method keeps its own
    legend.
    """
    H_img, W_img = img_disp.shape[:2]
    ax.imshow(img_disp)
    im = ax.imshow(
        heatmap,
        extent=(0, W_img, H_img, 0),
        cmap=cmap,
        alpha=alpha,
        interpolation=interpolation,
        vmin=float(heatmap.min()),
        vmax=float(heatmap.max()),
        **plot_kwargs,
    )
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_image_attribution(
    attribution_output: ImageAttributionOutput,
    image: PILImage | np.ndarray | torch.Tensor | None = None,
    target_idx: int | Iterable[int] | None = None,
    cmap: str = "jet",
    alpha: float = 0.5,
    clip_percentile: float | None = 0.1,
    absolute_value: bool = False,
    interpolation: str = "nearest",
    img_size: float = 3.0,
    cols: int = 4,
    colorbar: bool = True,
    **plot_kwargs,
):
    """
    Display attribution heatmap(s) over the underlying image.

    The flat `(t, l)` attributions are reshaped to a 2D grid using `elements`
    (max row/col + 1). For `ImageGranularity.PATCH` this gives e.g. (14, 14),
    which matplotlib stretches to the image's (H, W) via `extent`; the
    `interpolation` kwarg controls whether each patch is shown as a solid block
    (`'nearest'`, the honest default) or smoothed (`'bilinear'`).

    Args:
        attribution_output: Output from an image-classification attribution method.
        image: Underlying image (PIL.Image / ndarray / Tensor) to overlay. If None,
            falls back to `attribution_output.raw_image`. If that's also None,
            only the heatmap is shown.
        target_idx: If int, plot only that target. If an iterable of ints, plot
            that subset. If None, one subplot per target.
        cmap: Matplotlib colormap for the heatmap.
        alpha: Heatmap opacity over the image (0-1).
        clip_percentile: Clip at (p, 100-p) to suppress outliers. None disables.
        absolute_value: If True, take abs() before plotting (magnitude only).
        interpolation: Passed to `imshow` for the heatmap.
        img_size: Subplot side length in inches.
        cols: Max columns when laying out multiple targets in a grid.
        colorbar: If True, attach a per-target colorbar showing the real score range.
        **plot_kwargs: Extra kwargs forwarded to the heatmap `imshow`.

    Returns:
        (fig, axes) — matplotlib Figure and 2D Axes array. Call `plt.show()`
        from a script, or just keep the reference in a notebook.
    """
    if image is None:
        image = attribution_output.raw_image

    n_targets = attribution_output.attributions.shape[0]
    if target_idx is None:
        target_indices = list(range(n_targets))
    elif isinstance(target_idx, int):
        target_indices = [target_idx]
    else:
        target_indices = list(target_idx)

    if not target_indices:
        raise ValueError("target_idx is empty — pass None to plot all targets.")
    for t_idx in target_indices:
        if not 0 <= t_idx < n_targets:
            raise IndexError(
                f"target_idx {t_idx} out of range for {n_targets} targets."
            )
    n_plots = len(target_indices)

    actual_cols = min(cols, n_plots)
    n_rows = ceil(n_plots / actual_cols)
    fig, axes = plt.subplots(
        n_rows,
        actual_cols,
        figsize=(actual_cols * img_size, n_rows * img_size),
        squeeze=False,
    )

    if image is None:
        raise ValueError("There is no image in ImageAttributionOutput, which should not happen")
    img_disp = _to_displayable(image)
    targets_tensor = attribution_output.targets

    for i, t_idx in enumerate(target_indices):
        ax = axes[i // actual_cols][i % actual_cols]
        heatmap = _prepare_heatmap(attribution_output, t_idx, clip_percentile, absolute_value)

        _draw_attribution_on_ax(
            ax,
            img_disp,
            heatmap,
            cmap=cmap,
            alpha=alpha,
            interpolation=interpolation,
            colorbar=colorbar,
            fig=fig,
            **plot_kwargs,
        )

        # TODO: pair class index with its human-readable label (model.config.id2label).
        # Needs the id2label mapping plumbed in — either as a kwarg to this function
        # or stored on ImageAttributionOutput at explain() time.
        ax.set_title(f"target {int(targets_tensor[t_idx].item())}")
        ax.axis("off")

    # Hide unused cells in the last row.
    for j in range(n_plots, n_rows * actual_cols):
        axes[j // actual_cols][j % actual_cols].axis("off")

    fig.tight_layout()
    return fig, axes


def plot_image_attributions_comparison(
    attribution_outputs: Iterable[ImageAttributionOutput],
    labels: Iterable[str] | None = None,
    image: PILImage | np.ndarray | torch.Tensor | None = None,
    target_idx: int = 0,
    cmap: str = "jet",
    alpha: float = 0.5,
    clip_percentile: float | None = 0.1,
    absolute_value: bool = False,
    interpolation: str = "nearest",
    img_size: float = 3.0,
    cols: int = 4,
    colorbar: bool = True,
    **plot_kwargs,
):
    """
    Compare several attribution methods on the SAME image, side by side.

    Each method's heatmap is drawn as a **live mappable** into a shared grid of
    subplots — not a rasterized snapshot of a separate figure. This is what lets
    every panel carry its own colorbar in its own real score scale, so you can
    read and compare the magnitude each method assigns, not just the spatial
    pattern. (Rasterizing each method to RGB and tiling, the old approach, threw
    the score scale away and produced static images with no legend.)

    All outputs are expected to come from explaining the same image at the same
    granularity, so their heatmaps share a grid shape; only the scores differ.

    Args:
        attribution_outputs: One `ImageAttributionOutput` per method to compare.
        labels: Optional per-method titles (e.g. method names). If None, panels
            are titled "method 0", "method 1", ... Must match the number of outputs.
        image: Underlying image to overlay. If None, falls back to each output's
            `raw_image`.
        target_idx: Which target to plot for every method (single int — the point
            is to compare methods, holding the target fixed).
        cmap: Matplotlib colormap for the heatmaps.
        alpha: Heatmap opacity over the image (0-1).
        clip_percentile: Clip at (p, 100-p) to suppress outliers. None disables.
        absolute_value: If True, take abs() before plotting (magnitude only).
        interpolation: Passed to `imshow` for the heatmap.
        img_size: Subplot side length in inches.
        cols: Max columns when laying out the methods in a grid.
        colorbar: If True, attach a per-method colorbar showing its real score range.
        **plot_kwargs: Extra kwargs forwarded to the heatmap `imshow`.

    Returns:
        (fig, axes) — matplotlib Figure and 2D Axes array.
    """
    outputs = list(attribution_outputs)
    if not outputs:
        raise ValueError("attribution_outputs is empty — pass at least one output.")

    if labels is None:
        label_list: list[str] = [f"method {i}" for i in range(len(outputs))]
    else:
        label_list = list(labels)
        if len(label_list) != len(outputs):
            raise ValueError(
                f"labels has {len(label_list)} entries but there are {len(outputs)} outputs."
            )

    n_plots = len(outputs)
    actual_cols = min(cols, n_plots)
    n_rows = ceil(n_plots / actual_cols)
    fig, axes = plt.subplots(
        n_rows,
        actual_cols,
        figsize=(actual_cols * img_size, n_rows * img_size),
        squeeze=False,
    )

    for i, (output, label) in enumerate(zip(outputs, label_list, strict=True)):
        ax = axes[i // actual_cols][i % actual_cols]

        img = image if image is not None else output.raw_image
        if img is None:
            raise ValueError(
                f"No image to overlay for '{label}': pass `image=` or ensure the "
                "output carries a `raw_image`."
            )
        img_disp = _to_displayable(img)

        n_targets = output.attributions.shape[0]
        if not 0 <= target_idx < n_targets:
            raise IndexError(
                f"target_idx {target_idx} out of range for '{label}' ({n_targets} targets)."
            )

        heatmap = _prepare_heatmap(output, target_idx, clip_percentile, absolute_value)
        _draw_attribution_on_ax(
            ax,
            img_disp,
            heatmap,
            cmap=cmap,
            alpha=alpha,
            interpolation=interpolation,
            colorbar=colorbar,
            fig=fig,
            **plot_kwargs,
        )
        ax.set_title(label)
        ax.axis("off")

    # Hide unused cells in the last row.
    for j in range(n_plots, n_rows * actual_cols):
        axes[j // actual_cols][j % actual_cols].axis("off")

    fig.tight_layout()
    return fig, axes
