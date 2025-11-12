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

import pytest
import torch

from interpreto.attributions.perturbations import (
    DeletionPerturbator,
    InsertionPerturbator,
)


@pytest.mark.parametrize("PerturbatorClass, expected_value", [(DeletionPerturbator, 0), (InsertionPerturbator, 1)])
def test_baseline_mask(PerturbatorClass, expected_value):
    """Asserts that the baseline mask is initialized correctly for both insertion and
    deletion perturbators.
    """
    pert = PerturbatorClass()
    mask = pert._baseline_mask((5, 12))
    assert mask.shape == (5, 12)
    assert torch.all(mask == expected_value)


@pytest.mark.parametrize("PerturbatorClass", [DeletionPerturbator, InsertionPerturbator])
def test_get_mask_single_target(PerturbatorClass):
    """Asserts that the get_mask method generates the correct mask for insertion and deletion perturbators.

    This test focuses on perturbations for a single target.
    Several scenarios are tested: different numbers of perturbations and values of max_percentage_perturbed.
    """

    attributions = torch.tensor([0.1, 0.4, 0.2, 0.8, 0.3])  # length: 5
    mask_dim = len(attributions)

    # Test 1: the number of perturbations is equal mask_dim (+ 1 for the baseline).
    perturbator = PerturbatorClass(n_perturbations=mask_dim)

    mask = perturbator.get_mask(mask_dim, attributions)

    expected_mask = torch.Tensor(
        [
            [0, 0, 0, 0, 0],  # baseline
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    if PerturbatorClass.__name__ == "InsertionPerturbator":
        expected_mask = 1 - expected_mask

    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (mask_dim + 1, mask_dim)  # (mask_dim + 1) for the baseline mask
    assert (mask == expected_mask).all()

    # Test 2: the number of perturbations is more than mask_dim + 1.
    perturbator = PerturbatorClass(n_perturbations=10)
    mask = perturbator.get_mask(mask_dim, attributions)
    assert mask.shape == (mask_dim + 1, mask_dim)

    # Test 3: the number of perturbations is less than mask_dim + 1.
    perturbator = PerturbatorClass(n_perturbations=3)
    mask = perturbator.get_mask(mask_dim, attributions)
    assert mask.shape == (3 + 1, mask_dim)

    # Test 4: the number of perturbations is equal to 1.
    perturbator = PerturbatorClass(n_perturbations=1)
    mask = perturbator.get_mask(mask_dim, attributions)
    assert mask.shape == (1 + 1, mask_dim)
    assert (mask[0] == expected_mask[0]).all(), "First perturbation should be the baseline mask."
    assert (mask[1] == expected_mask[-1]).all(), "Second perturbation should be the full mask."

    # Test 5: max percentage perturbed at 50% (i.e. half of mask_dim).
    perturbator = PerturbatorClass(n_perturbations=10, max_percentage_perturbed=0.5)
    mask = perturbator.get_mask(mask_dim, attributions)
    assert mask.shape == (4, mask_dim)  # ceil(0.5 * mask_dim) + 1 = 4 perturbations


@pytest.mark.parametrize("PerturbatorClass", [DeletionPerturbator, InsertionPerturbator])
def test_get_mask_multiple_targets(PerturbatorClass):
    """Asserts that the get_mask method generates the correct masks for insertion and deletion perturbators
    with multiple targets.

    Three scenarios are tested: 1) the number of perturbations is equal to the number of elements (+ 1 for the
    baseline), 2) the number of perturbations is less than the number of elements, and 3) perturb only 50% of the
    elements.
    """

    attributions = torch.tensor(  # shape: (2, 6) => 2 targets, 6 elements
        [
            [0.1, 0.3, 0.2, 0.9, 0.4, 0.8],  # order: [3, 5, 4, 1, 2, 0]
            [-0.1, -0.8, -0.9, -0.2, -1.5, -1.6],  # order: [0, 3, 1, 2, 4, 5]
        ]
    )
    num_targets, mask_dim = attributions.shape

    # Test: multiple targets are not supported and must raise a ValueError.
    perturbator = PerturbatorClass(n_perturbations=mask_dim)
    with pytest.raises(ValueError, match="Only single target attributions are supported."):
        perturbator.get_mask(mask_dim, attributions)


def test_invalid_perturbation_count():
    """Asserts that a ValueError is raised when the number of perturbations is less than 1."""
    with pytest.raises(ValueError, match="The number of perturbations must be at least 1."):
        DeletionPerturbator(n_perturbations=0)


def test_max_percentage_perturbed_zero():
    """Asserts that the max_percentage_perturbed parameter raises an error when too few elements are perturbed."""
    # Max 0% -> still generates at least 2 perturbations
    perturbator = InsertionPerturbator(n_perturbations=10, max_percentage_perturbed=0.0)

    with pytest.raises(RuntimeError, match="The number of perturbed elements is too low"):
        perturbator.get_mask(mask_dim=10, attributions=torch.rand(10))


def test_too_short_sequence_length():
    """Asserts that a ValueError is raised when the sequence length is less than 2."""
    perturbator = DeletionPerturbator(n_perturbations=5)

    with pytest.raises(ValueError, match="The mask dimension .* must be greater than 1"):
        perturbator.get_mask(mask_dim=1, attributions=torch.tensor([0.1]))
