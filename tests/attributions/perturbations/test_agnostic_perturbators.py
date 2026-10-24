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
Tests for the perturbator functions that do not depend on the modality: `adjust_baseline` and the
`get_mask` of Sobol and Occlusion.

`get_mask` only reads `mask_dim` and the method's own knobs, but the method classes leave `perturb`
abstract, so they still need a modality base to be instantiated. That base is parametrized only to
make the class instantiable; it has no effect on the mask. `get_mask` never touches the processor,
hence `processor=None`.
"""

import pytest
import torch

from interpreto.attributions.perturbations import (
    LinearInterpolationPerturbator,
    OcclusionPerturbator,
    SobolPerturbator,
)
from interpreto.attributions.perturbations.base import ImageMaskPerturbator, TextMaskPerturbator
from interpreto.attributions.perturbations.sobol_perturbation import SequenceSamplers

MODALITY_BASES = [TextMaskPerturbator, ImageMaskPerturbator]
MODALITY_IDS = ["text", "vision"]


def _variant(method_class: type, modality_base: type) -> type:
    """
    Combine a method perturbator with a modality base, as the explainer does at construction
    time. The method class alone leaves `perturb` abstract.
    """
    return type(modality_base.__name__ + method_class.__name__, (method_class, modality_base), {})


def test_linear_interpolation_perturbation_adjust_baseline():
    inputs = torch.randn(1, 3, 8, 8)

    # Test with None baseline
    baseline = LinearInterpolationPerturbator.adjust_baseline(None, inputs)
    assert torch.all(baseline.abs() < 1e-5), (
        "adjust_baseline with baseline=None must return a baseline near zero (the 1e-6 nudge used to avoid NaN gradients)"
    )
    assert baseline.shape == inputs.shape[1:], (
        f"adjust_baseline must return a baseline shaped like inputs.shape[1:]. Expected {tuple(inputs.shape[1:])}, "
        f"got {tuple(baseline.shape)}"
    )

    # Test with float baseline
    baseline = LinearInterpolationPerturbator.adjust_baseline(0.5, inputs)
    assert torch.all(baseline == 0.5), (
        "adjust_baseline with a single float as input must return a tensor filled with only this float's value"
    )
    assert baseline.shape == inputs.shape[1:], (
        f"adjust_baseline must return a baseline shaped like inputs.shape[1:]. Expected {tuple(inputs.shape[1:])}, "
        f"got {tuple(baseline.shape)}"
    )

    # Test with tensor baseline
    baseline_tensor = torch.randn(3, 8, 8)
    baseline = LinearInterpolationPerturbator.adjust_baseline(baseline_tensor, inputs)
    assert torch.all(baseline == baseline_tensor), (
        "adjust_baseline with a tensor baseline already shaped like inputs.shape[1:] must return it unchanged"
    )
    assert baseline.shape == inputs.shape[1:], (
        f"adjust_baseline must return a baseline shaped like inputs.shape[1:]. Expected {tuple(inputs.shape[1:])}, "
        f"got {tuple(baseline.shape)}"
    )


def test_linear_interpolation_perturbation_adjust_baseline_invalid():
    inputs = torch.randn(1, 3, 8, 8)

    # Test with invalid baseline type
    with pytest.raises(TypeError, match="Type-check error"):
        LinearInterpolationPerturbator.adjust_baseline("invalid", inputs)  # type: ignore

    # Test with mismatched tensor shape
    baseline_tensor = torch.randn(2, 8, 8)
    with pytest.raises(ValueError, match="does not match expected shape"):
        LinearInterpolationPerturbator.adjust_baseline(baseline_tensor, inputs)

    # Test with mismatched tensor dtype
    baseline_tensor = torch.randn(3, 8, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="does not match expected dtype"):
        LinearInterpolationPerturbator.adjust_baseline(baseline_tensor, inputs)


@pytest.mark.parametrize("modality_base", MODALITY_BASES, ids=MODALITY_IDS)
@pytest.mark.parametrize(
    "sampler",
    [SequenceSamplers.SOBOL, SequenceSamplers.HALTON, SequenceSamplers.LatinHypercube],
)
def test_sobol_masks(sampler, modality_base):
    k = 4
    perturbator = _variant(SobolPerturbator, modality_base)(
        n_granularity_perturbations=k,
        sampler=sampler,
        processor=None,
    )

    for l in range(2, 20, 3):
        mask = perturbator.get_mask(l)
        assert mask.shape == ((l + 2) * k, l), (
            f"Sobol mask must have shape ((l + 2) * k, l). Expected {((l + 2) * k, l)}, got {tuple(mask.shape)}"
        )
        A = mask[:k]
        B = mask[k : 2 * k]
        C = mask[2 * k :].view(l, k, l)

        # verify token-wise mask compared to the initial mask
        for i in range(l):
            assert torch.all(torch.isclose(C[i, :, i], B[:, i], atol=1e-5)), (
                "each C_i's i-th column must equal B's i-th column (C_i is A with column i swapped for B's column i)"
            )
            if i != 0:
                assert torch.all(torch.isclose(C[i, :, :i], A[:, :i], atol=1e-5)), (
                    "each C_i's columns before i must equal A's corresponding columns (only column i is swapped for B)"
                )
            if i != l - 1:
                assert torch.all(torch.isclose(C[i, :, i + 1 :], A[:, i + 1 :], atol=1e-5)), (
                    "each C_i's columns after i must equal A's corresponding columns (only column i is swapped for B)"
                )


@pytest.mark.parametrize("modality_base", MODALITY_BASES, ids=MODALITY_IDS)
def test_occlusion_masks(modality_base):
    perturbator = _variant(OcclusionPerturbator, modality_base)(processor=None)

    for l in range(2, 20, 3):
        mask = perturbator.get_mask(l)
        assert torch.equal(mask, torch.cat([torch.zeros(1, l), torch.eye(l)], dim=0)), (
            "Occlusion's mask must be an all-zeros reference row (nothing occluded, for the baseline output "
            "to diff against) stacked with the identity matrix (each row occludes exactly one unit)"
        )
