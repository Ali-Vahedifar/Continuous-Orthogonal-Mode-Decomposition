"""
Inference: autoregressive restoration of lost haptic packets, plus the two
cost measurements reported in the paper (inference time in ms, and GFLOPs per
forward pass).

Nothing here returns a number taken from the paper -- every value is measured on
the machine you run it on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .decomposition import COMDConfig, comd, vmd

__all__ = ["autoregressive_restore", "stream_restore", "measure_latency",
           "model_gflops", "decomposition_gflops"]


# --------------------------------------------------------------------------- #
#  Restoration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def autoregressive_restore(model, buf_h, buf_r, n_steps: int):
    """Roll the model forward `n_steps` blocks using only its own output.

    This is the mode the paper describes for a packet that is lost or misses its
    deadline: the previous prediction is concatenated with the available history
    and fed back as the encoder input, with no ground truth in the loop.

    buf_h, buf_r : (B, K, W) tensors of the most recent modes.
    Returns (B, K, n_steps * H) for each side.
    """
    model.eval()
    W = buf_h.shape[-1]
    out_h, out_r = [], []
    for _ in range(n_steps):
        o = model(buf_h, buf_r)
        ph, pr = o["modes_h"], o["modes_r"]
        out_h.append(ph)
        out_r.append(pr)
        buf_h = torch.cat([buf_h, ph], dim=-1)[..., -W:]
        buf_r = torch.cat([buf_r, pr], dim=-1)[..., -W:]
    return torch.cat(out_h, dim=-1), torch.cat(out_r, dim=-1)


@torch.no_grad()
def stream_restore(model, modes_h, modes_r, mask, W: int, H: int, device="cpu"):
    """Replay a trace with a packet-loss mask.

    modes_h, modes_r : (K, T) ground-truth modes.
    mask : (T,) boolean, True where the packet ARRIVED.  Wherever it is False the
        sample is unavailable to the receiver and is filled by the model.
    Returns the restored modes (K, T) for both sides and the receiver-side view
    that was actually used as input, so you can score only the restored samples.
    """
    model.eval()
    K, T = modes_h.shape
    recv_h = np.array(modes_h, dtype=np.float32, copy=True)
    recv_r = np.array(modes_r, dtype=np.float32, copy=True)
    mask = np.asarray(mask, dtype=bool)
    t = W
    while t < T:
        if mask[t]:
            t += 1
            continue
        gap_end = t
        while gap_end < T and not mask[gap_end]:
            gap_end += 1
        need = gap_end - t
        x_h = torch.from_numpy(recv_h[:, t - W:t]).unsqueeze(0).to(device)
        x_r = torch.from_numpy(recv_r[:, t - W:t]).unsqueeze(0).to(device)
        steps = int(np.ceil(need / H))
        ph, pr = autoregressive_restore(model, x_h, x_r, steps)
        recv_h[:, t:gap_end] = ph[0, :, :need].cpu().numpy()
        recv_r[:, t:gap_end] = pr[0, :, :need].cpu().numpy()
        t = gap_end
    return recv_h, recv_r


# --------------------------------------------------------------------------- #
#  Latency
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_latency(model, K: int, W: int, device="cpu", batch: int = 1,
                    n_warmup: int = 20, n_runs: int = 200,
                    decomposition: str | None = None,
                    buffer: int = 256, comd_cfg: COMDConfig | None = None):
    """Per-forward-pass latency in milliseconds.

    decomposition : None | "comd" | "vmd"
        When set, the time to decompose one `buffer`-sample history window is
        measured too and added, which is what the paper's "Inf. (ms)" column
        accounts for (decomposition + network).
    Returns a dict with mean / std / median / p99 in ms.
    """
    model = model.to(device).eval()
    x_h = torch.randn(batch, K, W, device=device)
    x_r = torch.randn(batch, K, W, device=device)
    sig = np.random.default_rng(0).standard_normal(buffer)
    cfg = comd_cfg or COMDConfig(K=K)

    def once():
        t0 = time.perf_counter()
        if decomposition == "comd":
            comd(sig, cfg)
        elif decomposition == "vmd":
            vmd(sig, K=K)
        model(x_h, x_r)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(n_warmup):
        once()
    ts = np.array([once() for _ in range(n_runs)])
    return {"mean_ms": float(ts.mean()), "std_ms": float(ts.std()),
            "median_ms": float(np.median(ts)), "p99_ms": float(np.percentile(ts, 99)),
            "n_runs": n_runs, "device": device, "decomposition": decomposition}


# --------------------------------------------------------------------------- #
#  FLOPs
# --------------------------------------------------------------------------- #
def model_gflops(model, K: int, W: int, device="cpu") -> float:
    """Measured GFLOPs of one forward pass, via torch's FlopCounterMode.

    Counts multiply-accumulate-heavy ops (matmul, conv, attention); elementwise
    ops are not counted, which is the usual convention for this kind of table.
    """
    from torch.utils.flop_counter import FlopCounterMode
    model = model.to(device).eval()
    x_h = torch.randn(1, K, W, device=device)
    x_r = torch.randn(1, K, W, device=device)
    ctr = FlopCounterMode(display=False)
    with ctr, torch.no_grad():
        model(x_h, x_r)
    return ctr.get_total_flops() / 1e9


def decomposition_gflops(T: int, K: int, n_iter: int, ns_iters: int = 0,
                         method: str = "comd") -> float:
    """Analytic FLOP estimate for one decomposition of a T-sample buffer.

    The count is written out rather than hidden so it can be checked:

      * one FFT of the mirror-extended signal          5 * T' * log2(T')
      * per ADMM iteration and per mode:
          - Wiener filter               ~6 * T'   (subtract, add, divide)
          - orthogonality correction    ~6 * K * T'   (C-OMD only)
          - centre frequency            ~4 * T'
      * per Newton-Schulz step:  Gram 8*K^2*T'  +  update 8*K^2*T'
      * K inverse FFTs                                 5 * K * T' * log2(T')

    with T' = 2T for the mirror extension.  SVMD, being sequential, performs K
    such solves one after another; that is why `method="svmd"` multiplies by K.
    `method="none"` is the no-decomposition baseline: it does no work, 0 GFLOPs.
    """
    if method == "none":
        return 0.0
    Tp = 2 * T
    logT = np.log2(max(Tp, 2))
    flops = 5 * Tp * logT                               # forward FFT
    per_iter = K * (6 * Tp + 4 * Tp)
    if method == "comd":
        per_iter += K * 6 * K * Tp
        per_iter += ns_iters * (16 * K * K * Tp)
    flops += n_iter * per_iter
    flops += 5 * K * Tp * logT                          # inverse FFTs
    if method == "svmd":
        flops *= K
    return flops / 1e9
