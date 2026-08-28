"""
The four-component training objective, Eqs. (21)-(25) of the paper.

    L = L_pred + l1 * L_recon + l2 * L_orth + l3 * L_rel

    L_pred  = 1/(KH) sum_{k,h} || m_hat_k(t+h) - m_k(t+h) ||^2
    L_recon = || f_hat - f ||^2
    L_orth  = || G - I_K ||_F^2                with G_ij = <m_i, m_j>
    L_rel   = 1/(KH) sum_{k,h} |m_hat - m| / max(|m|, tau),   tau = 0.01

    l1 = 0.1,  l2 = 0.01,  l3 = 0.05
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["LossWeights", "MDALoss", "gram", "orthogonality_loss"]


@dataclass
class LossWeights:
    lambda1: float = 0.1     # reconstruction
    lambda2: float = 0.01    # orthogonality
    lambda3: float = 0.05    # relative error
    tau: float = 0.01        # floor of the relative-error denominator
    normalize_gram: bool = False
    """If True, G is computed from L2-normalised modes so that ||G - I||_F
    measures *correlation* only.  The paper writes G_ij = <m_i, m_j> without
    normalisation, which is the default (False); with unnormalised modes the
    term also pulls each mode towards unit energy.  Kept as a switch because
    the distinction matters when comparing against the paper."""


def gram(m: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Gram matrix of a batch of mode sets.  (B, K, H) -> (B, K, K)."""
    if normalize:
        m = m / (m.norm(dim=-1, keepdim=True) + 1e-12)
    return m @ m.transpose(-1, -2)


def orthogonality_loss(m: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """|| G - I_K ||_F^2, Eq. (24)."""
    G = gram(m, normalize)
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    return ((G - eye) ** 2).sum(dim=(-1, -2)).mean()


class MDALoss(nn.Module):
    """Bilateral version of Eq. (21): the four terms are evaluated on the human
    and the robot branch and summed."""

    def __init__(self, w: LossWeights | None = None):
        super().__init__()
        self.w = w or LossWeights()

    def _side(self, m_hat, m_true):
        w = self.w
        # Eq. (22) -- mean over modes and horizon
        l_pred = ((m_hat - m_true) ** 2).mean(dim=(-1, -2)).mean()
        # Eq. (23) -- reconstruction of the full signal
        l_recon = ((m_hat.sum(1) - m_true.sum(1)) ** 2).sum(-1).mean()
        # Eq. (24)
        l_orth = orthogonality_loss(m_hat, w.normalize_gram)
        # Eq. (25)
        denom = torch.clamp(m_true.abs(), min=w.tau)
        l_rel = ((m_hat - m_true).abs() / denom).mean(dim=(-1, -2)).mean()
        return l_pred, l_recon, l_orth, l_rel

    def forward(self, out: dict, tgt_h: torch.Tensor, tgt_r: torch.Tensor):
        ph, rh, oh, xh = self._side(out["modes_h"], tgt_h)
        pr, rr, orr, xr = self._side(out["modes_r"], tgt_r)
        w = self.w
        parts = {
            "pred": ph + pr,
            "recon": rh + rr,
            "orth": oh + orr,
            "rel": xh + xr,
        }
        total = (parts["pred"] + w.lambda1 * parts["recon"]
                 + w.lambda2 * parts["orth"] + w.lambda3 * parts["rel"])
        parts["total"] = total
        return total, {k: float(v.detach()) for k, v in parts.items()}
