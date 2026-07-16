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
(Apache 2.0, DEEL / Université Paul Sabatier). The heatmap is read directly from
the display-ready `(t, H, W)` `ImageAttributionOutput.attributions_image` map;
`explain` already did the g->l interpolation, so the viz never reshapes or
interpolates itself.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from math import ceil

import numpy as np
import torch
from matplotlib import pyplot as plt

from interpreto.attributions.base import ImageAttributionOutput

# Slack on the [0, 1] range check, to absorb float error in x * std + mean rather than warn
# about a display that is fine. Wide enough to stay quiet on rounding, far too tight to hide
# a genuine scale mismatch (those land orders of magnitude out).
_RANGE_TOLERANCE = 1e-3


def _denormalize(attribution_output: ImageAttributionOutput) -> np.ndarray:
    """
    Recover a displayable image from the pre-processed `pixel_values`.

    `x * image_std + image_mean` inverts the processor's normalization exactly, but nothing
    guarantees the result lands in [0, 1] — that depends on the scale the tensor was in
    before normalization (`rescale_factor`, the input dtype), which the stats alone do not
    pin down. So the range is checked rather than assumed, and a violation is reported, not
    repaired: fitting the observed range would be the min-max stretch this replaced, which
    silently produces a plausible, wrong image.
    """
    if attribution_output.image_mean is None or attribution_output.image_std is None:
        raise ValueError(
            "ImageAttributionOutput carries no image_mean/image_std, so pixel_values cannot be "
            "de-normalized for display. `explain` always sets them; an output built by hand must "
            "pass them too (image_mean=0, image_std=1 if no normalization was applied)."
        )

    pixel_values = attribution_output.model_inputs_to_explain["pixel_values"]
    image = pixel_values.detach().cpu().to(torch.float32)
    if image.ndim == 4:
        image = image[0]

    image = image * attribution_output.image_std + attribution_output.image_mean

    lo, hi = float(image.min()), float(image.max())
    if lo < -_RANGE_TOLERANCE or hi > 1.0 + _RANGE_TOLERANCE:
        warnings.warn(
            f"De-normalized image spans [{lo:.3f}, {hi:.3f}], outside the displayable [0, 1]; "
            "it will be clamped and may look wrong. The image_mean/image_std do not match how "
            "this tensor was normalized, or it was never in [0, 1] before normalization. With "
            "preprocess=False, pass the affine that de-normalizes it.",
            stacklevel=3,
        )

    return image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


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
    Read the 2D heatmap `(H, W)` directly from `attributions_image[target_idx]`
    (already expanded to pixel resolution by `explain`).

    The heatmap is returned in the **real attribution-score scale** (after the
    optional abs + percentile clip). It is deliberately NOT rescaled to [0, 1]:
    color normalization is left to `imshow`'s vmin/vmax at draw time so each
    panel's `colorbar` shows genuine scores. Rescaling here would collapse every
    method's legend to a meaningless 0->1 axis and make cross-method comparison
    impossible.
    """
    # `attributions_image` is already the display-ready (t, H, W) map: `explain` did the
    # g->l interpolation (NEAREST for gradient methods, the mask strategy for masking methods),
    # so the viz just reads the 2-D slice and never reshapes or interpolates itself.
    row = attribution_output.attributions_image[target_idx].detach().cpu()
    if row.dtype is torch.bfloat16:
        row = row.to(torch.float32)
    heatmap = row.numpy().astype(np.float32)

    if absolute_value:
        heatmap = np.abs(heatmap)
    if clip_percentile is not None:
        heatmap = _clip_percentile(heatmap, clip_percentile)

    return heatmap


def _color_limits(heatmap: np.ndarray, center_zero: bool | None) -> tuple[float, float, bool]:
    """
    Decide the colorbar range for one heatmap.

    Attribution scores are now **signed** for the mean/trapezoid gradient methods
    (Saliency, SmoothGrad, GradientShap, IntegratedGradients) and for the mask
    methods (Occlusion, LIME, KernelSHAP, Sobol): the sign tells whether a pixel
    pushes the target logit up or down. For such maps a sequential scale anchored
    at min/max would put 0 at an arbitrary color; instead we center the range at 0
    (`[-m, +m]`) so the neutral color means "no contribution" and +/- are readable.

    VarGrad/SquareGrad (and any abs'd map) are non-negative, so we keep the plain
    min/max range there.

    Returns (vmin, vmax, centered). `center_zero=None` auto-detects (center iff the
    map straddles zero); pass True/False to force.
    """
    lo, hi = float(heatmap.min()), float(heatmap.max())
    centered = (lo < 0.0 < hi) if center_zero is None else center_zero
    if centered:
        m = max(abs(lo), abs(hi)) or 1.0
        return -m, m, True
    return lo, hi, False


def _to_grayscale(img_disp: np.ndarray) -> np.ndarray:
    """
    Collapse a displayable image to a 2D luminance map so it can be drawn under the
    heatmap as a neutral backdrop (Rec. 601 weights). Already-2D images pass through.
    """
    if img_disp.ndim == 2:
        return img_disp
    if img_disp.shape[-1] == 1:
        return img_disp[..., 0]
    return img_disp[..., :3] @ np.array([0.2989, 0.587, 0.114], dtype=img_disp.dtype)


def _draw_attribution_on_ax(
    ax,
    img_disp: np.ndarray,
    heatmap: np.ndarray,
    *,
    cmap: str | None,
    alpha: float,
    colorbar: bool,
    center_zero: bool | None,
    grayscale_background: bool,
    fig,
    **plot_kwargs,
):
    """
    Draw one image + heatmap pair onto `ax` as a live mappable and (optionally)
    attach a per-axes colorbar showing the heatmap's real score range.

    Centralizing the draw here keeps the single-method and multi-method entry
    points pixel-for-pixel identical, and it crucially keeps each panel's
    colorbar tied to that panel's own vmin/vmax, so every method keeps its own
    legend.

    Unless the caller forces a colormap, every panel uses the diverging `vanimo`
    map — including non-negative (sequential) heatmaps — so all methods share the
    same color language and stay visually comparable side by side.

    When `grayscale_background` is True the underlying image is drawn in grayscale so
    the colored heatmap stands out against a neutral backdrop.
    """
    vmin, vmax, centered = _color_limits(heatmap, center_zero)
    resolved_cmap = cmap if cmap is not None else "coolwarm"

    H_img, W_img = img_disp.shape[:2]
    if grayscale_background:
        ax.imshow(_to_grayscale(img_disp), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(img_disp)
    im = ax.imshow(
        heatmap,
        extent=(0, W_img, H_img, 0),
        cmap=resolved_cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        **plot_kwargs,
    )
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_image_attribution(
    attribution_output: ImageAttributionOutput,
    target_idx: int | Iterable[int] | None = None,
    cmap: str | None = None,
    alpha: float = 0.5,
    clip_percentile: float | None = 0.1,
    absolute_value: bool = False,
    img_size: float = 3.0,
    cols: int = 4,
    colorbar: bool = True,
    center_zero: bool | None = None,
    grayscale_background: bool = True,
    **plot_kwargs,
):
    """
    Display attribution heatmap(s) over the underlying image.

    The heatmap is the display-ready `(t, H, W)` `attributions_image` map produced
    by `explain` (the g->l interpolation — NEAREST for gradient methods, the mask
    strategy for masking methods — is already baked in). It is drawn over the image
    via `extent`; since it already matches pixel resolution, no imshow-level
    interpolation is applied — the heatmap's sharpness is fully decided by the
    method's `granularity_resize` strategy.

    The underlying image is de-normalized from the output's own `pixel_values`, so it is
    always at the model's input resolution — the same grid `attributions_image` lives on.

    Args:
        attribution_output: Output from an image-classification attribution method.
        target_idx: If int, plot only that target. If an iterable of ints, plot
            that subset. If None, one subplot per target.
        cmap: Matplotlib colormap for the heatmap. If None (default), `coolwarm`
            (diverging) is used for every panel, sequential or signed, so methods
            stay visually comparable.
        alpha: Heatmap opacity over the image (0-1).
        clip_percentile: Clip at (p, 100-p) to suppress outliers. None disables.
        absolute_value: If True, take abs() before plotting (magnitude only).
        img_size: Subplot side length in inches.
        cols: Max columns when laying out multiple targets in a grid.
        colorbar: If True, attach a per-target colorbar showing the real score range.
        center_zero: Center the color scale at 0 with a symmetric range. None
            (default) auto-detects (centers iff the map has both signs); True/False
            forces it.
        grayscale_background: If True, draw the underlying image in grayscale so
            the colored heatmap stands out. Defaults to True.
        **plot_kwargs: Extra kwargs forwarded to the heatmap `imshow`.

    Returns:
        (fig, axes) — matplotlib Figure and 2D Axes array. Call `plt.show()`
        from a script, or just keep the reference in a notebook.
    """
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

    img_disp = _denormalize(attribution_output)
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
            colorbar=colorbar,
            center_zero=center_zero,
            grayscale_background=grayscale_background,
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
    target_idx: int = 0,
    cmap: str | None = None,
    alpha: float = 0.5,
    clip_percentile: float | None = 0.1,
    absolute_value: bool = False,
    img_size: float = 3.0,
    cols: int = 4,
    colorbar: bool = True,
    center_zero: bool | None = None,
    grayscale_background: bool = True,
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
        target_idx: Which target to plot for every method (single int — the point
            is to compare methods, holding the target fixed).
        cmap: Matplotlib colormap for the heatmaps. If None (default), `coolwarm`
            (diverging) is used for every panel — signed or sequential alike — so
            every method in the grid shares the same color language and is
            actually comparable.
        alpha: Heatmap opacity over the image (0-1).
        clip_percentile: Clip at (p, 100-p) to suppress outliers. None disables.
        absolute_value: If True, take abs() before plotting (magnitude only).
        img_size: Subplot side length in inches.
        cols: Max columns when laying out the methods in a grid.
        colorbar: If True, attach a per-method colorbar showing its real score range.
        center_zero: Center the color scale at 0 with a symmetric range. None
            (default) auto-detects per panel; True/False forces it.
        grayscale_background: If True, draw the underlying image in grayscale so
            the colored heatmap stands out. Defaults to True.
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

        img_disp = _denormalize(output)

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
            colorbar=colorbar,
            center_zero=center_zero,
            grayscale_background=grayscale_background,
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
