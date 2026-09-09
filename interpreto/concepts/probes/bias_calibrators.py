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
Bias calibration functions for probe models.

After a probe computes raw concept scores, a bias calibrator determines an
additive per-concept intercept `b` such that `score + b` produces a
well-calibrated decision boundary.

All calibrators share the same signature::

    (scores: Tensor[n, c], y: Tensor[n, c]) -> bias: Tensor[c]

Available strategies:
    - [prevalence_bias][interpreto.concepts.probes.bias_calibrators.prevalence_bias] — logit of the class prevalence.
    - [midpoint_bias][interpreto.concepts.probes.bias_calibrators.midpoint_bias] — midpoint between positive and negative means.
    - [fpr_bias][interpreto.concepts.probes.bias_calibrators.fpr_bias] — threshold controlling false positive rate.
    - [bce_bias][interpreto.concepts.probes.bias_calibrators.bce_bias] — L-BFGS minimization of BCE loss on the intercept.
    - [lda_shared_var_bias][interpreto.concepts.probes.bias_calibrators.lda_shared_var_bias] — 1-D LDA optimal threshold with shared variance.
"""

from collections.abc import Callable
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn

# Type alias for bias calibrator functions.
# Signature: (scores: Float[Tensor, "n c"], y: Float[Tensor, "n c"]) -> bias: Float[Tensor, "c"]
BiasCalibrator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# String literals for easy selection (optional convenience)
BiasCalibatorName = Literal["prevalence", "midpoint", "fpr", "bce", "lda"]


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def prevalence_bias(
    scores: Float[torch.Tensor, "n c"], y: Float[torch.Tensor, "n c"], eps: float = 1e-6
) -> Float[torch.Tensor, "c"]:
    """Prevalence-based bias: `bias_j = logit(mean(y_j))`.

    Sets the decision threshold at the class prior, which is optimal under
    a uniform score distribution. Ignores the actual scores.

    Args:
        scores: Raw concept scores (unused, kept for API consistency).
        y: Binary multi-label targets.
        eps (float): Clamping value for numerical stability.

    Returns:
        Per-concept bias.
    """
    y01: Float[torch.Tensor, "n c"] = (y > 0.5).to(dtype=scores.dtype)
    p: Float[torch.Tensor, "c"] = y01.mean(dim=0).clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def midpoint_bias(
    scores: Float[torch.Tensor, "n c"], y: Float[torch.Tensor, "n c"], eps: float = 1e-12
) -> Float[torch.Tensor, "c"]:
    """Midpoint bias between positive and negative score means.

    `threshold_j = 0.5 * (mean(score|y=1) + mean(score|y=0))`
    `bias_j = -threshold_j`

    Args:
        scores: Raw concept scores.
        y: Binary multi-label targets.
        eps (float): Floor for count denominators.

    Returns:
        Per-concept bias.
    """
    y01: Float[torch.Tensor, "n c"] = (y > 0.5).to(dtype=scores.dtype)
    n1: Float[torch.Tensor, "c"] = y01.sum(dim=0).clamp_min(eps)
    n0: Float[torch.Tensor, "c"] = (1.0 - y01).sum(dim=0).clamp_min(eps)
    mu1: Float[torch.Tensor, "c"] = (scores * y01).sum(dim=0) / n1
    mu0: Float[torch.Tensor, "c"] = (scores * (1.0 - y01)).sum(dim=0) / n0
    return -0.5 * (mu1 + mu0)


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def fpr_bias(
    scores: Float[torch.Tensor, "n c"], y: Float[torch.Tensor, "n c"], target_fpr: float = 1e-2
) -> Float[torch.Tensor, "c"]:
    """False-positive-rate controlled bias.

    Sets the threshold at the `(1 - target_fpr)` quantile of the negative
    score distribution, giving approximately `target_fpr` false positive rate.

    `threshold_j = quantile_{1 - target_fpr}(scores | y=0)`
    `bias_j = -threshold_j`

    Args:
        scores: Raw concept scores.
        y: Binary multi-label targets.
        target_fpr (float): Desired false positive rate (default 1%).

    Returns:
        Per-concept bias.
    """
    y01 = y > 0.5  # bool
    neg_scores: Float[torch.Tensor, "n c"] = scores.masked_fill(y01, float("inf"))
    sorted_neg: Float[torch.Tensor, "n c"] = neg_scores.sort(dim=0).values
    m: Float[torch.Tensor, "c"] = (~y01).to(dtype=scores.dtype).sum(dim=0).clamp_min(1.0)

    q = 1.0 - target_fpr
    idx = torch.floor((m - 1.0) * q).to(torch.long)
    t: Float[torch.Tensor, "c"] = sorted_neg.gather(0, idx.unsqueeze(0)).squeeze(0)
    return -t


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def bce_bias(
    scores: Float[torch.Tensor, "n c"], y: Float[torch.Tensor, "n c"], max_iter: int = 50, eps: float = 1e-6
) -> Float[torch.Tensor, "c"]:
    """BCE-optimal bias via L-BFGS.

    Fits a per-class intercept `b` to minimize
    `BCEWithLogitsLoss(scores + b, y)` with scores held fixed.

    Args:
        scores: Raw concept scores (treated as fixed).
        y: Binary multi-label targets.
        max_iter (int): Maximum L-BFGS iterations.
        eps (float): Clamping for initial prevalence estimate.

    Returns:
        Per-concept bias.
    """
    y01 = (y > 0.5).to(dtype=scores.dtype)

    p = y01.mean(dim=0).clamp(eps, 1.0 - eps)
    b0 = torch.log(p / (1.0 - p))

    b = b0.clone().requires_grad_(True)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(scores + b, y01)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)

    return b.detach()


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def lda_shared_var_bias(
    scores: Float[torch.Tensor, "n c"],
    y: Float[torch.Tensor, "n c"],
    eps: float = 1e-12,
    var_floor: float = 1e-6,
) -> Float[torch.Tensor, "c"]:
    """Closed-form 1-D LDA threshold with shared variance and empirical priors.

    Computes the Bayes-optimal threshold assuming Gaussian class-conditional
    distributions with a shared (pooled) variance::

        t = 0.5*(mu0 + mu1) + (var / (mu1 - mu0)) * log(pi0 / pi1)
        bias = -t

    Args:
        scores: Raw concept scores.
        y: Binary multi-label targets.
        eps (float): Floor for count and denominator stability.
        var_floor (float): Minimum variance to avoid division by zero.

    Returns:
        Per-concept bias.
    """
    y01: Float[torch.Tensor, "n c"] = (y > 0.5).to(dtype=scores.dtype)

    n1: Float[torch.Tensor, "c"] = y01.sum(dim=0).clamp_min(eps)
    n0: Float[torch.Tensor, "c"] = (1.0 - y01).sum(dim=0).clamp_min(eps)

    mu1: Float[torch.Tensor, "c"] = (scores * y01).sum(dim=0) / n1
    mu0: Float[torch.Tensor, "c"] = (scores * (1.0 - y01)).sum(dim=0) / n0

    var: Float[torch.Tensor, "c"] = scores.var(dim=0, unbiased=False).clamp_min(var_floor)

    pi1: Float[torch.Tensor, "c"] = (n1 / (n0 + n1)).clamp(eps, 1.0 - eps)
    pi0: Float[torch.Tensor, "c"] = 1.0 - pi1

    denom: Float[torch.Tensor, "c"] = mu1 - mu0
    denom = denom.sign() * denom.abs().clamp_min(eps)

    t: Float[torch.Tensor, "c"] = 0.5 * (mu0 + mu1) + (var / denom) * torch.log(pi0 / pi1)
    return -t
