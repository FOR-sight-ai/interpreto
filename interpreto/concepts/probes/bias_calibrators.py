from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

# Type alias for bias calibrator functions
# signature: (scores: Tensor, y: Tensor) -> bias: Tensor
BiasCalibrator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# String literals for easy selection (optional convenience)
BiasCalibatorName = Literal["prevalence", "midpoint", "fpr", "bce", "lda"]


@torch.no_grad()
def prevalence_bias(scores: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """bias_j = logit(mean(y_j))"""
    y01 = (y > 0.5).to(dtype=scores.dtype)  # (n,c)
    p = y01.mean(dim=0).clamp(eps, 1.0 - eps)  # (c,)
    return torch.log(p / (1.0 - p))


@torch.no_grad()
def midpoint_bias(scores: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    threshold_j = 0.5 * (mean(score|y=1) + mean(score|y=0))
    bias_j = -threshold_j
    """
    y01 = (y > 0.5).to(dtype=scores.dtype)  # (n,c)
    n1 = y01.sum(dim=0).clamp_min(eps)  # (c,)
    n0 = (1.0 - y01).sum(dim=0).clamp_min(eps)  # (c,)
    mu1 = (scores * y01).sum(dim=0) / n1
    mu0 = (scores * (1.0 - y01)).sum(dim=0) / n0
    return -0.5 * (mu1 + mu0)


@torch.no_grad()
def fpr_bias(scores: torch.Tensor, y: torch.Tensor, target_fpr: float = 1e-2) -> torch.Tensor:
    """
    Control FPR per class using negatives:
        threshold_j = quantile_{1 - target_fpr}(scores|y=0)
        bias_j = -threshold_j
    """
    y01 = y > 0.5  # bool (n,c)
    neg_scores = scores.masked_fill(y01, float("inf"))  # (n,c)
    sorted_neg = neg_scores.sort(dim=0).values  # (n,c)
    m = (~y01).to(dtype=scores.dtype).sum(dim=0).clamp_min(1.0)  # (c,)

    q = 1.0 - target_fpr
    idx = torch.floor((m - 1.0) * q).to(torch.long)  # (c,)
    t = sorted_neg.gather(0, idx.unsqueeze(0)).squeeze(0)  # (c,)
    return -t


@torch.no_grad()
def bce_bias(scores: torch.Tensor, y: torch.Tensor, max_iter: int = 50, eps: float = 1e-6) -> torch.Tensor:
    """
    Fit per-class intercept b to minimize BCEWithLogitsLoss(scores + b, y),
    with scores fixed.
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
def lda_shared_var_bias(
    scores: torch.Tensor, y: torch.Tensor, eps: float = 1e-12, var_floor: float = 1e-6
) -> torch.Tensor:
    """
    Closed-form (per class) 1D LDA with shared variance + empirical priors:
        t = 0.5*(mu0+mu1) + (var/(mu1-mu0))*log(pi0/pi1)
        bias = -t
    """
    y01 = (y > 0.5).to(dtype=scores.dtype)  # (n,c)

    n1 = y01.sum(dim=0).clamp_min(eps)
    n0 = (1.0 - y01).sum(dim=0).clamp_min(eps)

    mu1 = (scores * y01).sum(dim=0) / n1
    mu0 = (scores * (1.0 - y01)).sum(dim=0) / n0

    var = scores.var(dim=0, unbiased=False).clamp_min(var_floor)

    pi1 = (n1 / (n0 + n1)).clamp(eps, 1.0 - eps)
    pi0 = 1.0 - pi1

    denom = mu1 - mu0
    denom = denom.sign() * denom.abs().clamp_min(eps)

    t = 0.5 * (mu0 + mu1) + (var / denom) * torch.log(pi0 / pi1)
    return -t


# Registry for string-based lookup (uses default params)
BIAS_CALIBRATORS: dict[BiasCalibatorName, BiasCalibrator] = {
    "prevalence": prevalence_bias,
    "midpoint": midpoint_bias,
    "fpr": fpr_bias,
    "bce": bce_bias,
    "lda": lda_shared_var_bias,
}


def get_bias_calibrator(name: BiasCalibatorName) -> BiasCalibrator:
    """Get a bias calibrator function by name."""
    if name not in BIAS_CALIBRATORS:
        raise ValueError(f"Unknown bias calibrator: {name}. Available: {list(BIAS_CALIBRATORS.keys())}")
    return BIAS_CALIBRATORS[name]
