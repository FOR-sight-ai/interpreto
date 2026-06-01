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
Image-side perturbators. `ImagePerturbator` is the no-op default used by
`ImageClassificationAttributionExplainer`, mirroring the text `Perturbator`
no-op but keyed on `pixel_values` instead of `input_ids`.
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from interpreto.attributions.perturbations.base import Perturbator
from interpreto.commons.image_granularity import ImageGranularity
from interpreto.typing import TensorMapping


class ImagePerturbator(Perturbator):
    """
    No-op image-side perturbator.

    Mirrors the text-side `Perturbator` no-op with `pixel_values` substituted
    for `input_ids`. Returns the input unchanged with `mask=None`; the default
    `Aggregator` ignores the mask, which is the correct placeholder for
    gradient-based methods like Saliency.

    The perturbation dimension is the leading axis of `pixel_values`, which
    `BatchFeature` from a ViT `image_processor` already provides as `1`
    (shape `(1, 3, H, W)`). Unlike text — where the tokenizer can yield 1D
    `input_ids` that need unsqueezing — no shape massage is required here.

    For perturbation-based methods (Occlusion, LIME, KernelShap, Sobol),
    subclass and override `perturb` to produce a `(p, 3, H, W)` batch of
    perturbed pixel_values and a matching `(p, g)` mask, mirroring the role
    of `IdsPerturbator` on the text side.
    """

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        return model_inputs, None


class ImageEmbeddingsPerturbator(ImagePerturbator):
    """
    Image-side analog of `EmbeddingsPerturbator`.

    Operates directly on `pixel_values` of shape `(1, 3, H, W)`. Despite the
    "Embeddings" name (kept for naming symmetry with the text side), this
    works in raw pixel space — there is no IDs-to-embeddings indirection like
    `inputs_embedder` on the text side, because `pixel_values` is already a
    float tensor straight from the image processor.

    Subclasses override `perturb_embeds` to produce a `(p, 3, H, W)` batch.
    The returned mask is `None`; granularity is applied post-hoc by the
    aggregator via `granularity_score_aggregation(..., aggregate_inputs=True)`.
    Used by gradient-style methods (Saliency, SmoothGrad, IntegratedGradient).
    """

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        if "pixel_values" not in model_inputs:
            raise ValueError("model_inputs should contain 'pixel_values'")

        inputs = deepcopy(model_inputs)
        pixel_values: Float[torch.Tensor, "1 3 H W"] = inputs["pixel_values"]

        perturbed_embeds: Float[torch.Tensor, "p 3 H W"]
        mask: Float[torch.Tensor, "p g"] | None
        perturbed_embeds, mask = self.perturb_embeds(pixel_values)

        inputs["pixel_values"] = perturbed_embeds
        return inputs, mask

    def perturb_embeds(
        self, pixel_values: Float[torch.Tensor, "1 3 H W"]
    ) -> tuple[Float[torch.Tensor, "p 3 H W"], Float[torch.Tensor, "p g"] | None]:
        """
        Default no-op: subclasses override to apply noise / interpolation / etc.

        Args:
            pixel_values: Shape (1, 3, H, W).
        Returns:
            perturbed_embeds: Shape (p, 3, H, W).
            mask: (p, g) or None.
        """
        return pixel_values, None


class ImageMaskPerturbator(ImagePerturbator):
    """
    Image-side analog of `IdsPerturbator`.

    Masking-based perturbator: each perturbation hides a subset of granularity
    units by overwriting their pixel positions with a
    constant `replace_value` baseline. This is the basis for perturbation
    methods (Occlusion, LIME, KernelShap, Sobol), mirroring the role of
    `IdsPerturbator` for the text side.

    Flow (mirrors `IdsPerturbator.perturb`):
      1. Build the `(g, l)` association matrix from the granularity, where
         `g` = number of granularity units and `l = H*W` pixel positions.
      2. Ask the subclass for a granularity-wise mask `gran_mask (p, g)` via
         `get_mask`. `1` = masked (replaced), `0` = kept — same convention as text.
      3. Expand to pixel space: `real_mask = gran_mask @ assoc -> (p, l)`.
      4. Overwrite masked pixel positions with `replace_value`, broadcasting
         across the 3 channels, and reshape back to `(p, 3, H, W)`.
      5. Return `(perturbed_inputs, gran_mask)`. The aggregator consumes
         `gran_mask`; `granularity_score_aggregation(..., aggregate_inputs=False)`
         then returns scores unchanged because granularity is already encoded
         in the masks.

    Differences from `IdsPerturbator`:
      - Replacement is a per-position float baseline (`replace_value`) applied
        across all 3 channels, not an integer `replace_token_id`.
        TODO: extend `replace_value` to support a per-channel `(3,)` tensor and
        a full `(1, 3, H, W)` baseline image (e.g. blurred input), captum-style.
        Kept scalar for the MVP.
      - `pixel_values` is spatially flattened to `(1, 3, l)` to apply the mask,
        then reshaped, instead of operating on a 1D token axis.
    """

    __slots__ = ("granularity", "n_perturbations", "replace_value", "patch_size")

    def __init__(
        self,
        granularity: ImageGranularity = ImageGranularity.PATCH,
        n_perturbations: int = 1,
        replace_value: float = 0.0,
        patch_size: int | None = None,
    ):
        """
        Args:
            granularity (ImageGranularity): unit over which masks are defined
                (PATCH or PIXEL). PATCH is the default and the cheap choice
                (`g = num_patches` instead of `g = H*W`).
            n_perturbations (int): number of perturbations produced by `perturb`.
            replace_value (float): baseline value written into masked pixel
                positions across all channels. `0.0` is the per-channel mean
                after standard ViT normalization (a neutral "grey" baseline).
            patch_size (int | None): patch side length, used to build the association
                matrix for PATCH granularity (ignored for PIXEL). Defaults to `None`,
                NOT a number: the explainer is the source of truth and overwrites it
                from `model.config.patch_size` at construction. Leaving it `None`
                makes a PATCH perturb that never went through that reconcile fail
                loudly rather than silently mask with a wrong patch size.
        """
        self.granularity = granularity
        self.n_perturbations = n_perturbations
        self.replace_value = replace_value
        self.patch_size = patch_size

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def get_mask(self, mask_dim: int, **kwargs) -> Float[torch.Tensor, "{self.n_perturbations} {mask_dim}"]:
        """
        Return the granularity-wise perturbation mask, of shape `(n_perturbations, g)`.

        `mask_dim` is `g`, the number of granularity units. `1` marks a masked
        unit (replaced by the baseline), `0` marks a kept unit. Implemented by
        subclasses (random masking for LIME, single-unit masking for Occlusion, etc.).

        Args:
            mask_dim (int): number of granularity units `g`.
            kwargs: extra arguments for specific mask strategies.

        Returns:
            torch.Tensor: mask of shape `(n_perturbations, g)`.
        """
        raise NotImplementedError()

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        if "pixel_values" not in model_inputs:
            raise ValueError("model_inputs should contain 'pixel_values'")

        inputs = deepcopy(model_inputs)
        pixel_values: Float[torch.Tensor, "1 3 H W"] = inputs["pixel_values"]

        if pixel_values.shape[0] != 1:
            raise ValueError(
                "Inputs are treated one by one in the perturbator, "
                f"but received pixel_values of shape {tuple(pixel_values.shape)} "
                "- expected shape (1, 3, H, W)."
            )

        _, c, h, w = pixel_values.shape
        l = h * w

        # PATCH needs a real patch_size; it should have been set by the explainer.
        # Fail loudly instead of falling back to a wrong default. (PIXEL ignores it.)
        if self.granularity == ImageGranularity.PATCH and self.patch_size is None:
            raise ValueError(
                "patch_size is None. It must be set "
                "from the model config — normally the explainer does this at construction. "
                "If using the perturbator standalone, pass patch_size explicitly."
            )

        # association matrix mapping granularity units to flat pixel positions: (g, l)
        association_matrix: Float[torch.Tensor, "g l"] = self.granularity.get_association_matrix(
            inputs,
            patch_size=self.patch_size,
        )[0].float()

        # granularity-wise mask from the subclass: (p, g)
        gran_mask: Float[torch.Tensor, "p g"] = self.get_mask(association_matrix.shape[0])

        # expand to pixel space: (p, g) @ (g, l) -> (p, l)
        real_mask: Float[torch.Tensor, "p l"] = gran_mask @ association_matrix

        # apply the mask in flattened spatial space, broadcasting across channels
        flat: Float[torch.Tensor, "1 3 l"] = pixel_values.reshape(1, c, l)
        spatial_mask: Float[torch.Tensor, "p 1 l"] = real_mask.unsqueeze(1)
        perturbed_flat: Float[torch.Tensor, "p 3 l"] = (
            flat * (1 - spatial_mask) + self.replace_value * spatial_mask
        )
        perturbed_pixel_values: Float[torch.Tensor, "p 3 H W"] = perturbed_flat.reshape(-1, c, h, w)

        inputs["pixel_values"] = perturbed_pixel_values
        return inputs, gran_mask
