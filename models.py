"""
Mode-Domain Architecture (MDA), PyTorch.

The module layout follows Fig. 1 and Fig. 2 of the paper one-to-one:

    Fig. 2(a)  TCNEncoder            dilations 1, 2, 4  + residual skip
    Fig. 2(b)  TCNDecoder            dilations 4, 2, 1  + Linear R^d -> R^H
    Fig. 2(c)  CrossSideCrossAttention   Q from own side, K/V from the other,
                                         parallel Linear-Coupling residual
    Fig. 2(d)  CrossModeSelfAttention    K mode latents as tokens
    Fig. 1     MDA                   the two-branch bilateral network

Hyper-parameters the paper does not state (channel width, number of heads,
kernel size, FFN expansion) are collected in :class:`MDAConfig` with sensible
defaults and are flagged in the README, so nothing is silently assumed to be
"the paper's" value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MDAConfig", "MDA", "TCNEncoder", "TCNDecoder",
           "CrossModeSelfAttention", "CrossSideCrossAttention",
           "TransformerFourierBaseline", "MambaBaseline", "build_model"]


@dataclass
class MDAConfig:
    K: int = 3            # number of modes (paper: optimal K = 3 for C-OMD)
    W: int = 5            # input window length in samples
    H: int = 100          # prediction horizon in samples
    d_model: int = 64     # latent width d          (not specified in the paper)
    channels: int = 64    # TCN channel width       (not specified in the paper)
    kernel_size: int = 3  # TCN kernel              (not specified in the paper)
    n_heads: int = 4      # attention heads         (not specified in the paper)
    ffn_mult: int = 2     # FFN expansion           (not specified in the paper)
    dropout: float = 0.0
    share_encoder_across_sides: bool = True   # Fig. 1: "shared weights w/ human"


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, L) tensor."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _CausalConvBlock(nn.Module):
    """Dilated causal convolution + ReLU + LayerNorm (Fig. 2(a)/(b) blocks)."""

    def __init__(self, c_in, c_out, kernel_size, dilation, causal=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.causal = causal
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
        self.norm = _ChannelLayerNorm(c_out)

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.pad, 0))           # left padding only
        else:
            left = self.pad // 2
            x = F.pad(x, (left, self.pad - left))
        return self.norm(F.relu(self.conv(x)))


class TCNEncoder(nn.Module):
    """Fig. 2(a): per-mode TCN encoder, dilations d = 1, 2, 4, residual skip.

    (B, W) -> (B, d_model)
    """

    def __init__(self, cfg: MDAConfig):
        super().__init__()
        c = cfg.channels
        self.b1 = _CausalConvBlock(1, c, cfg.kernel_size, 1)
        self.b2 = _CausalConvBlock(c, c, cfg.kernel_size, 2)
        self.b4 = _CausalConvBlock(c, c, cfg.kernel_size, 4)
        self.to_latent = nn.Linear(c, cfg.d_model)
        self.skip = nn.Linear(cfg.W, cfg.d_model)      # dotted skip in Fig. 2(a)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):                              # x: (B, W)
        h = self.b4(self.b2(self.b1(x.unsqueeze(1))))  # (B, C, W)
        z = self.to_latent(h[:, :, -1])                # last (causal) step
        return self.drop(z + self.skip(x))


class TCNDecoder(nn.Module):
    """Fig. 2(b): per-mode TCN decoder, dilations d = 4, 2, 1, then R^d -> R^H.

    (B, d_model) -> (B, H)
    """

    def __init__(self, cfg: MDAConfig):
        super().__init__()
        c = cfg.channels
        # The fused latent is a vector in R^d, so the decoder convolutions run
        # along the latent axis and are non-causal (there is no time ordering
        # inside a latent vector).
        self.b4 = _CausalConvBlock(1, c, cfg.kernel_size, 4, causal=False)
        self.b2 = _CausalConvBlock(c, c, cfg.kernel_size, 2, causal=False)
        self.b1 = _CausalConvBlock(c, c, cfg.kernel_size, 1, causal=False)
        self.to_one = nn.Conv1d(c, 1, 1)
        self.proj = nn.Linear(cfg.d_model, cfg.H)      # "Linear Proj R^d -> R^H"
        self.skip = nn.Conv1d(1, c, 1)                 # dotted skip in Fig. 2(b)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, z):                              # z: (B, d_model)
        x = z.unsqueeze(1)                             # (B, 1, d)
        h = self.b2(self.b4(x))
        h = self.b1(h + self.skip(x))                  # skip enters the d=1 block
        h = self.to_one(h).squeeze(1)                  # (B, d)
        return self.proj(self.drop(h))                 # (B, H)


class _FFN(nn.Module):
    def __init__(self, d, mult, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, mult * d), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(mult * d, d))
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.drop(self.net(x)))


class CrossModeSelfAttention(nn.Module):
    """Fig. 2(d): self-attention over the K mode latents as tokens.

    (B, K, d) -> (B, K, d)
    """

    def __init__(self, cfg: MDAConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ffn = _FFN(cfg.d_model, cfg.ffn_mult, cfg.dropout)

    def forward(self, z):
        a, _ = self.attn(z, z, z, need_weights=False)
        return self.ffn(self.norm(z + a))              # skip -> Add & LayerNorm


class CrossSideCrossAttention(nn.Module):
    """Fig. 2(c): cross-side cross-attention with the Linear-Coupling residual.

    Q comes from this side, K/V from the other side; in parallel a linear
    coupling branch produces z_lc = W_c z_own.  The two are summed and
    layer-normalised, then passed through the feed-forward block.
    """

    def __init__(self, cfg: MDAConfig):
        super().__init__()
        d = cfg.d_model
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout,
                                          batch_first=True)
        self.coupling = nn.Linear(d, d, bias=False)    # W_c
        self.norm = nn.LayerNorm(d)
        self.ffn = _FFN(d, cfg.ffn_mult, cfg.dropout)

    def forward(self, z_own, z_other):
        a, _ = self.attn(z_own, z_other, z_other, need_weights=False)
        z_lc = self.coupling(z_own)                    # z_lc = W_c z~
        return self.ffn(self.norm(a + z_lc))           # Add & LayerNorm


# --------------------------------------------------------------------------- #
#  MDA
# --------------------------------------------------------------------------- #
class MDA(nn.Module):
    """Fig. 1: bilateral Mode-Domain Architecture.

    forward(x_h, x_r) with x_* of shape (B, K, W) returns a dict with the
    per-mode predictions (B, K, H) and the reconstructed signals (B, H) for
    both sides.
    """

    def __init__(self, cfg: MDAConfig):
        super().__init__()
        self.cfg = cfg
        self.enc_h = nn.ModuleList([TCNEncoder(cfg) for _ in range(cfg.K)])
        self.enc_r = self.enc_h if cfg.share_encoder_across_sides else \
            nn.ModuleList([TCNEncoder(cfg) for _ in range(cfg.K)])
        self.self_h = CrossModeSelfAttention(cfg)
        self.self_r = CrossModeSelfAttention(cfg)
        self.cross_h = CrossSideCrossAttention(cfg)
        self.cross_r = CrossSideCrossAttention(cfg)
        self.dec_h = nn.ModuleList([TCNDecoder(cfg) for _ in range(cfg.K)])
        self.dec_r = nn.ModuleList([TCNDecoder(cfg) for _ in range(cfg.K)])

    @staticmethod
    def _encode(encoders, x):
        return torch.stack([enc(x[:, k]) for k, enc in enumerate(encoders)], 1)

    @staticmethod
    def _decode(decoders, z):
        return torch.stack([dec(z[:, k]) for k, dec in enumerate(decoders)], 1)

    def forward(self, x_h, x_r):
        z_h = self._encode(self.enc_h, x_h)            # (B, K, d)
        z_r = self._encode(self.enc_r, x_r)

        zt_h = self.self_h(z_h)                        # cross-mode self-attention
        zt_r = self.self_r(z_r)

        zh_h = self.cross_h(zt_h, zt_r)                # K/V exchange between sides
        zh_r = self.cross_r(zt_r, zt_h)

        m_h = self._decode(self.dec_h, zh_h)           # (B, K, H)
        m_r = self._decode(self.dec_r, zh_r)
        return {"modes_h": m_h, "modes_r": m_r,
                "signal_h": m_h.sum(1), "signal_r": m_r.sum(1)}


# --------------------------------------------------------------------------- #
#  Baselines
# --------------------------------------------------------------------------- #
class TransformerFourierBaseline(nn.Module):
    """Transformer backbone with an auxiliary Fourier-feature branch.

    The paper names this baseline "Transformer + Fourier" but does not define
    the Fourier branch, so the concrete choice here -- concatenating the
    rFFT magnitude and phase of the input window to the token embedding -- is
    this repository's interpretation, not a claim about the paper.
    Single-side model: it is instantiated once per side.
    """

    def __init__(self, cfg: MDAConfig, n_layers: int = 3):
        super().__init__()
        self.cfg = cfg
        n_freq = cfg.W // 2 + 1
        self.embed = nn.Linear(cfg.W, cfg.d_model)
        self.fourier = nn.Linear(2 * n_freq, cfg.d_model)
        self.mode_emb = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads,
                                           cfg.ffn_mult * cfg.d_model,
                                           dropout=cfg.dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(cfg.d_model, cfg.H)

    def forward(self, x):                              # (B, K, W)
        spec = torch.fft.rfft(x, dim=-1)
        ff = torch.cat([spec.abs(), torch.angle(spec)], dim=-1)
        z = self.embed(x) + self.fourier(ff) + self.mode_emb
        m = self.head(self.enc(z))
        return {"modes": m, "signal": m.sum(1)}


class MambaBaseline(nn.Module):
    """Mamba backbone.  Requires the optional `mamba-ssm` package."""

    def __init__(self, cfg: MDAConfig, n_layers: int = 3):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "The Mamba baseline needs the external `mamba-ssm` package "
                "(pip install mamba-ssm, CUDA required). It is not vendored "
                "here because no faithful CPU reference implementation exists."
            ) from e
        self.embed = nn.Linear(cfg.W, cfg.d_model)
        self.mode_emb = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))
        self.blocks = nn.ModuleList([Mamba(d_model=cfg.d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.H)

    def forward(self, x):
        z = self.embed(x) + self.mode_emb
        for b in self.blocks:
            z = z + b(self.norm(z))
        m = self.head(self.norm(z))
        return {"modes": m, "signal": m.sum(1)}


class _BilateralWrapper(nn.Module):
    """Runs a single-side baseline independently on both sides.

    This is what makes the comparison in the paper meaningful: the baselines
    have no mechanism to exchange information between the human and the robot
    branch, which is exactly the ablation MDA's cross-side attention targets.
    """

    def __init__(self, make):
        super().__init__()
        self.h, self.r = make(), make()

    def forward(self, x_h, x_r):
        o_h, o_r = self.h(x_h), self.r(x_r)
        return {"modes_h": o_h["modes"], "modes_r": o_r["modes"],
                "signal_h": o_h["signal"], "signal_r": o_r["signal"]}


def build_model(name: str, cfg: MDAConfig) -> nn.Module:
    name = name.lower()
    if name == "mda":
        return MDA(cfg)
    if name in ("transformer", "transformer_fourier"):
        return _BilateralWrapper(lambda: TransformerFourierBaseline(cfg))
    if name == "mamba":
        return _BilateralWrapper(lambda: MambaBaseline(cfg))
    raise ValueError(f"unknown model '{name}' (mda | transformer | mamba)")
