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

from copy import deepcopy

import torch
from jaxtyping import Float

from interpreto.attributions.perturbations.base import Perturbator
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
