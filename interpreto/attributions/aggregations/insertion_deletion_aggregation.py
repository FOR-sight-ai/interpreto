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
Aggregator for Insertion and Deletion metrics
"""

import torch
from jaxtyping import Float
from torch import Tensor

from interpreto.attributions.aggregations.base import Aggregator


class InsertionDeletionAggregator(Aggregator):
    """
    Aggregator for Insertion and Deletion metrics.

    This class aggregates the results of the insertion and deletion metrics by filtering the results based on the
    number of perturbations per target. The results are expected to be in the shape of (p * t, t), where p is the
    number of perturbations per target and t is the number of targets. The aggregation is done by reshaping the results
    to (p, t) where each row corresponds to a target and each column corresponds to a perturbation. For each target,
    the results are aggregated by taking the values corresponding to the perturbations for that target.

    Example:
    Below is an example with t=3 targets (columns) and p=2 perturbations per target (rows). The interesting values are
    represented by 1, 2, and 3, for each target. The "*" values are not considered in the aggregation. The aggregation
    only keeps the scores of the perturbations corresponding to each target, resulting in a tensor of shape (p, t).
    ```
    results =
    [
        1a * *
        1b * *
        * 2a *
        * 2b *
        * * 3a
        * * 3b
    ]

    aggregation =
    [
        1a 2a 3a
        1b 2b 3b
    ]
    ```

    """

    def aggregate(
        self,
        results: Float[Tensor, "p*t t"],
        mask: torch.Tensor | None = None,
    ) -> Float[Tensor, "p t"]:
        """
        Aggregate the results of the insertion/deletion metric.

        Args:
            results (Float[Tensor, "p*t t"]): The results of the perturbation. Here, the results are expected to be
                of shape (p * t, t).

        Returns:
            Float[Tensor, "t l"]: The aggregated results.
        """
        if results.shape[0] % results.shape[1] != 0:
            raise ValueError(
                f"The total number of perturbations ({results.shape[0]}) must be a multiple of "
                f"the number of targets ({results.shape[1]})."
            )

        num_perturb_per_target = results.shape[0] // results.shape[1]
        # Build a tensor of shape (num_perturb_per_target, num_targets) where for each target column, we only keep
        # the perturbations corresponding to that target.
        indices = torch.arange(results.shape[1]).repeat_interleave(num_perturb_per_target)  # shape (p * t,)
        return results[torch.arange(results.shape[0]), indices].reshape(results.shape[1], num_perturb_per_target).T
