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
Definition of granularity levels for image-classification explainers (pixel, patch).
"""

from __future__ import annotations

from enum import Enum

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

from interpreto.commons.granularity import GranularityAggregationStrategy
from interpreto.typing import TensorMapping


class ImageGranularity(Enum):
    """
    Enumeration of granularity levels for image classification explainers.

    Standalone enum (not a subclass of `Granularity`) because Python forbids
    extending enums that already have members. Duck-typed with text `Granularity`
    via the same method names (`get_indices`, `get_association_matrix`,
    `granularity_score_aggregation`, `get_decomposition`).

    Positions are flattened row-major: index `i` <-> `(row=i//W, col=i%W)`.
    For `vit-base-patch16-224`: PIXEL has `H*W = 50176` singleton units,
    PATCH has `196` units of `patch_size*patch_size = 256` pixel positions each.
    """

    PIXEL = "pixel"
    PATCH = "patch"
    DEFAULT = PATCH

    def get_indices(
        self,
        inputs: TensorMapping,
        patch_size: int = 16,
    ) -> list[list[list[int]]]:
        """
        Return indices of the pixel positions composing each granularity unit, per sample.

        Args:
            inputs (TensorMapping): Batched inputs with `pixel_values` of shape `(b, 3, H, W)`.
            patch_size (int): Patch side length used for PATCH granularity. Ignored for PIXEL.

        Returns:
            list[list[list[int]]]: Outer list indexed by sample, middle list indexed by
                granularity unit, inner list = pixel positions (flat row-major) inside the unit.
        """
        pixel_values = inputs["pixel_values"]
        n_samples = pixel_values.shape[0]
        h, w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])

        match self:
            case ImageGranularity.PIXEL:
                per_sample: list[list[int]] = [[i] for i in range(h * w)]
            case ImageGranularity.PATCH:
                per_sample = []
                # rows of patches, then cols of patches
                for row_start in range(0, h, patch_size):
                    for col_start in range(0, w, patch_size):
                        patch = []
                        for i in range(patch_size):
                            for j in range(patch_size):
                                # pixel positions in this patch, starting from row_start and col_start and moving by patch_size pixels
                                patch.append((i+row_start) * w + j + col_start)
                                
                        per_sample.append(patch)
            case _:
                raise NotImplementedError(f"Granularity {self} not implemented.")

        # per-sample structure is identical across the batch (unlike text, where it depends on tokens)
        return [per_sample for _ in range(n_samples)]

    @jaxtyped(typechecker=beartype)
    def get_association_matrix(
        self,
        inputs: TensorMapping,
        patch_size: int = 16,
        indices_list: list[list[list[int]]] | None = None,
    ) -> list[Bool[torch.Tensor, "g l"]]:
        """
        Build the boolean matrix that maps from this granularity to flat pixel-position space.

        Mirrors `Granularity.get_association_matrix` with `l = H*W` instead of padded token count.

        Args:
            inputs (TensorMapping): Batched inputs with `pixel_values` of shape `(b, 3, H, W)`.
            patch_size (int): Patch side length used when computing indices for PATCH.
            indices_list (list[list[list[int]]] | None): Precomputed `get_indices` output.

        Returns:
            list[Bool[Tensor, "g l"]]: One matrix per sample. `g` = number of granularity units,
                `l = H*W` = total pixel positions (no padding for ViT).
        """
        if indices_list is None:
            indices_list = self.get_indices(inputs, patch_size)

        pixel_values = inputs["pixel_values"]
        l = int(pixel_values.shape[-1]) * int(pixel_values.shape[-2])  # H * W

        assoc_matrix_list: list[Bool[torch.Tensor, "g l"]] = []
        for indices in indices_list:
            g = len(indices)
            assoc_matrix: Bool[torch.Tensor, "g l"] = torch.zeros((g, l), dtype=torch.bool)
            for j, gran_indices in enumerate(indices):
                assoc_matrix[j, gran_indices] = True
            assoc_matrix_list.append(assoc_matrix)

        return assoc_matrix_list

    def granularity_score_aggregation(
        self,
        contribution: torch.Tensor,
        granularity_aggregation_strategy: GranularityAggregationStrategy | None = None,
        inputs: TensorMapping | None = None,
        patch_size: int = 16,
        aggregate_inputs: bool = False,
        indices_list: list[list[list[int]]] | None = None,
    ) -> Float[torch.Tensor, "t g"]:
        """
        Aggregate `contribution` of shape `(t, l=H*W)` to the chosen granularity.

        Mirrors `Granularity.granularity_score_aggregation` minus the generation
        (`aggregate_targets`) branch — image classification only.

        PIXEL behaves like text `TOKEN` (direct indexing).
        PATCH behaves like text `WORD` (within-unit aggregation via `GranularityAggregationStrategy`).

        Args:
            contribution (torch.Tensor): Per-pixel attribution scores of shape `(t, l)`.
            granularity_aggregation_strategy (GranularityAggregationStrategy | None):
                Aggregation method for PATCH. Required for PATCH, ignored for PIXEL.
            inputs (TensorMapping | None): Required for non-default granularity to read H, W.
            patch_size (int): Patch side length, used to compute indices for PATCH.
            aggregate_inputs (bool): If False (perturbation methods), returns `contribution`
                unchanged — granularity is already encoded in the perturbation masks.
                If True (gradient methods), aggregates over pixel positions to granularity units.
            indices_list (list[list[list[int]]] | None): Precomputed `get_indices` output.

        Returns:
            torch.Tensor: Aggregated contribution of shape `(t, g)`.
        """
        # Perturbation methods: granularity is already in the masks, return as-is
        if not aggregate_inputs:
            return contribution

        if inputs is None:
            raise ValueError("Inputs are required for non-default granularity aggregation.")

        if indices_list is None:
            indices_list = self.get_indices(inputs, patch_size)

        if len(indices_list) > 1:
            raise ValueError(
                "`granularity_score_aggregation` does not support batched inputs. Please provide a single input."
            )
        sample_indices: list[list[int]] = indices_list[0]

        # Gradient methods: aggregate per granularity unit
        match self:
            case ImageGranularity.PIXEL:
                # PIXEL behaves like text TOKEN — direct indexing, singleton units
                indices = torch.tensor(sample_indices).squeeze(1)
                return contribution[:, indices]
            case ImageGranularity.PATCH:
                # PATCH behaves like text WORD — within-unit aggregation
                if granularity_aggregation_strategy is None:
                    raise ValueError(
                        "granularity_aggregation_strategy must be provided for PATCH granularity."
                    )
                aggregated_contribution: Float[torch.Tensor, "t g"] = torch.zeros(
                    (contribution.shape[0], len(sample_indices)),
                    dtype=contribution.dtype,
                    device=contribution.device,
                )
                for j, unit_indices in enumerate(sample_indices):
                    unit_contribution: Float[torch.Tensor, "t gi"] = contribution[:, unit_indices]
                    assert unit_contribution.dim() == 2 and unit_contribution.shape[1] >= 2, (
                        f"PATCH units must contain >=2 pixels; got shape {tuple(unit_contribution.shape)}. "
                        "Use ImageGranularity.PIXEL for pixel-level granularity."
                    )
                    aggregated_contribution[:, [j]] = granularity_aggregation_strategy.aggregate(
                        unit_contribution, dim=1
                    )
                return aggregated_contribution
            case _:
                raise NotImplementedError(f"Granularity {self} not implemented for aggregation.")

    def get_decomposition(
        self,
        inputs: TensorMapping,
        patch_size: int = 16,
        return_coordinates: bool = False,
        indices_list: list[list[list[int]]] | None = None,
        **kwargs,
    ) -> list[list[list[int]]] | list[list[tuple[int, int]]]:
        """
        Return per-unit labels for visualization or downstream consumption.

        If `return_coordinates=True`, returns `(row, col)` int tuples per granularity unit
        (pixel coordinates for PIXEL, patch-grid coordinates for PATCH).
        Otherwise returns the same nested-int structure as `get_indices`.

        Args:
            inputs (TensorMapping): Batched inputs with `pixel_values` of shape `(b, 3, H, W)`.
            patch_size (int): Patch side length, used for PATCH.
            return_coordinates (bool): Tuple-coordinate output vs. raw index nesting.
            indices_list (list[list[list[int]]] | None): Precomputed `get_indices` output.
            **kwargs: Absorbs text-side caller args (e.g., `tokenizer=...`) that do not apply here.

        Returns:
            list[list[list[int]]] | list[list[tuple[int, int]]]: Outer list indexed by sample.
        """
        del kwargs  # tokenizer etc. — explicitly discarded for text-API parity

        if indices_list is None:
            indices_list = self.get_indices(inputs, patch_size)

        if not return_coordinates:
            return indices_list

        pixel_values = inputs["pixel_values"]
        w = int(pixel_values.shape[-1])
        n_samples = len(indices_list)

        # all samples share the same per-sample structure (image processor produces fixed shape),
        # so we compute the decomposition once and replicate.
        sample_indices = indices_list[0]
        decomposition: list[tuple[int, int]] = []
        match self:
            case ImageGranularity.PIXEL:
                # one singleton index per unit; recover (row, col) from flat position
                for unit in sample_indices:
                    idx = unit[0]
                    decomposition.append((idx // w, idx % w))
            case ImageGranularity.PATCH:
                # patch-grid coords: (patch_row, patch_col), order matches get_indices traversal
                n_patches_per_row = w // patch_size
                for j in range(len(sample_indices)):
                    decomposition.append((j // n_patches_per_row, j % n_patches_per_row))
            case _:
                raise NotImplementedError(f"Granularity {self} not implemented for decomposition.")

        all_decompositions: list[list[tuple[int, int]]] = [decomposition for _ in range(n_samples)]

        return all_decompositions
