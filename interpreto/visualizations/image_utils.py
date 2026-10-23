# MIT License
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III.
# DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry, CRIAQ and ANITI.

"""Image preparation shared by HTML and Matplotlib attribution renderers."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from PIL.Image import Image as PILImage

from interpreto.attributions.base import ImageAttributionOutput, ImageRetrievalAttributionOutput


def _denormalize(
    attribution_output: ImageAttributionOutput | ImageRetrievalAttributionOutput,
    range_tolerance: float = 1e-3,
) -> np.ndarray:
    """Invert processor normalization and return a displayable `(H, W, C)` image.

    Small floating-point excursions outside [0, 1] are tolerated; larger ones
    warn before clamping, since they indicate a preprocessing scale mismatch.
    """
    if attribution_output.image_mean is None or attribution_output.image_std is None:
        raise ValueError(
            "ImageAttributionOutput carries no image_mean/image_std, so pixel_values cannot be "
            "de-normalized for display. `explain` always sets them; an output built by hand must "
            "pass them too (image_mean=0, image_std=1 if no normalization was applied)."
        )

    pixel_values = attribution_output.model_inputs_to_explain["pixel_values"]
    image = pixel_values.detach().cpu().to(torch.float32)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            f"expected pixel_values of shape (1, 3, H, W), got {tuple(image.shape)}. "
            "Explain images one at a time — `explain` returns one output per image."
        )

    image = image[0]
    image = image * attribution_output.image_std + attribution_output.image_mean

    lo, hi = float(image.min()), float(image.max())
    if lo < -range_tolerance or hi > 1.0 + range_tolerance:
        warnings.warn(
            f"De-normalized image spans [{lo:.3f}, {hi:.3f}], outside the displayable [0, 1]; "
            "it will be clamped and may look wrong. This may be because of a problem with the "
            "pre-processing pipeline or because your image was not in [0, 1] before "
            "normalization. If this is the case, consider changing the rescale factor of the "
            "image_processor so that the rescaled image falls within the [0, 1] range.",
            stacklevel=3,
        )

    return image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _to_displayable_image(raw: PILImage | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a raw input image to a displayable `(H, W, C)` float array in [0, 1].

    Single funnel for the heterogeneous raw types `explain` accepts, so the
    visualization supports hand-built and explainer-produced outputs through one
    code path. Conventions follow each library's native layout: channel-last
    for PIL/`numpy`, channel-first for `torch`.

    Accepted layouts:
    - `PIL.Image`: modes `L` (kept 1-channel), `RGB`, and `RGBA` (alpha dropped).
    - `np.ndarray`: `(H, W)`, `(H, W, 1)`, `(H, W, 3)`; `uint8` is rescaled by
      1/255, floats are clamped to [0, 1].
    - `torch.Tensor`: `(H, W)`, `(C, H, W)` and `(1, C, H, W)` with C in
      {1, 3}; detached, moved to CPU, `uint8` rescaled by 1/255, floats clamped
      to [0, 1]. Gradient tracking is never enabled by this conversion.

    Anything else (batched tensors, channel-ambiguous shapes, other channel
    counts or dtypes) raises `ValueError` instead of guessing.
    """
    if isinstance(raw, PILImage):
        if raw.mode == "L":
            array = np.asarray(raw, dtype=np.float32) / 255.0
            return array[..., None]
        converted = raw.convert("RGB")
        return np.asarray(converted, dtype=np.float32) / 255.0

    if isinstance(raw, np.ndarray):
        array = raw
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3 or array.shape[-1] not in (1, 3):
            raise ValueError(
                f"expected an ndarray image of shape (H, W) or (H, W, C) with C in (1, 3), got {tuple(array.shape)}."
            )
        return _rescale_array(array)

    if isinstance(raw, torch.Tensor):
        tensor = raw.detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"expected a single image tensor, got a batch of {tensor.shape[0]}. "
                    "Explain images one at a time — `explain` returns one output per image."
                )
            tensor = tensor[0]
        if tensor.ndim == 2:
            return _rescale_array(tensor.to(torch.float32).numpy()[..., None])
        if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
            raise ValueError(
                f"expected a tensor image of shape (H, W) or (C, H, W) with C in (1, 3), got {tuple(raw.shape)}."
            )
        array = tensor.to(torch.float32).permute(1, 2, 0).numpy()
        return _rescale_array(array)

    raise ValueError(
        f"unsupported raw image type {type(raw).__name__}: expected PIL.Image, np.ndarray or torch.Tensor."
    )


def _rescale_array(array: np.ndarray) -> np.ndarray:
    """Map a `(H, W, C)` array to float32 in [0, 1] by dtype convention."""
    if array.dtype == np.uint8:
        return (array.astype(np.float32) / 255.0).clip(0.0, 1.0)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32).clip(0.0, 1.0)
    raise ValueError(
        f"unsupported image dtype {array.dtype}: expected uint8 or a floating dtype with values in [0, 1]."
    )
