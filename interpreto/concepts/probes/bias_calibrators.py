import torch
import torch.nn as nn


class BiasCalibratorBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.empty(0))

    @torch.no_grad()
    def fit(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PrevalenceBias(BiasCalibratorBase):
    """
    bias_j = logit(mean(y_j))
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    @torch.no_grad()
    def fit(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y01 = (y > 0.5).to(dtype=scores.dtype)  # (n,c)
        p = y01.mean(dim=0).clamp(self.eps, 1.0 - self.eps)  # (c,)
        b = torch.log(p / (1.0 - p))
        return b.detach()


class MidpointBias(BiasCalibratorBase):
    """
    threshold_j = 0.5 * (mean(score|y=1) + mean(score|y=0))
    bias_j = -threshold_j
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    @torch.no_grad()
    def fit(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y01 = (y > 0.5).to(dtype=scores.dtype)  # (n,c)
        n1 = y01.sum(dim=0).clamp_min(self.eps)  # (c,)
        n0 = (1.0 - y01).sum(dim=0).clamp_min(self.eps)  # (c,)
        mu1 = (scores * y01).sum(dim=0) / n1
        mu0 = (scores * (1.0 - y01)).sum(dim=0) / n0
        t = 0.5 * (mu1 + mu0)
        return (-t).detach()


class FPRBias(BiasCalibratorBase):
    """
    Control FPR per class using negatives:
        threshold_j = quantile_{1 - target_fpr}(scores|y=0)
        bias_j = -threshold_j
    """

    def __init__(self, target_fpr: float = 1e-2, eps: float = 1e-12):
        super().__init__()
        self.target_fpr = float(target_fpr)
        self.eps = float(eps)

    @torch.no_grad()
    def fit(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y01 = y > 0.5  # bool (n,c)
        neg_scores = scores.masked_fill(y01, float("inf"))  # (n,c)
        sorted_neg = neg_scores.sort(dim=0).values  # (n,c)
        m = (~y01).to(dtype=scores.dtype).sum(dim=0).clamp_min(1.0)  # (c,)

        q = 1.0 - self.target_fpr
        idx = torch.floor((m - 1.0) * q).to(torch.long)  # (c,)
        t = sorted_neg.gather(0, idx.unsqueeze(0)).squeeze(0)  # (c,)

        return = (-t).detach()


class BCEBias(BiasCalibratorBase):
    """
    Fit per-class intercept b to minimize BCEWithLogitsLoss(scores + b, y),
    with scores fixed.
    """

    def __init__(self, max_iter: int = 50, eps: float = 1e-6):
        super().__init__()
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self._loss = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def fit(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y01 = (y > 0.5).to(dtype=scores.dtype)

        p = y01.mean(dim=0).clamp(self.eps, 1.0 - self.eps)
        b0 = torch.log(p / (1.0 - p))

        b = b0.clone().detach().requires_grad_(True)
        opt = torch.optim.LBFGS([b], max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._loss(scores + b, y01)
            loss.backward()
            return loss

        opt.step(closure)

        return b.detach()


class LDASharedVarBias(BiasCalibratorBase):
    """
    Closed-form (per class) 1D LDA with shared variance + empirical priors:
        t = 0.5*(mu0+mu1) + (var/(mu1-mu0))*log(pi0/pi1)
        bias = -t
    """

    def __init__(self, eps: float = 1e-12, var_floor: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.var_floor = float(var_floor)

    @torch.no_grad()
    def fit(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y01 = (y > 0.5).to(dtype=scores.dtype)  # (n,c)

        n1 = y01.sum(dim=0).clamp_min(self.eps)
        n0 = (1.0 - y01).sum(dim=0).clamp_min(self.eps)

        mu1 = (scores * y01).sum(dim=0) / n1
        mu0 = (scores * (1.0 - y01)).sum(dim=0) / n0

        var = scores.var(dim=0, unbiased=False).clamp_min(self.var_floor)

        pi1 = (n1 / (n0 + n1)).clamp(self.eps, 1.0 - self.eps)
        pi0 = 1.0 - pi1

        denom = mu1 - mu0
        denom = denom.sign() * denom.abs().clamp_min(self.eps)

        t = 0.5 * (mu0 + mu1) + (var / denom) * torch.log(pi0 / pi1)
        return (-t).detach()
