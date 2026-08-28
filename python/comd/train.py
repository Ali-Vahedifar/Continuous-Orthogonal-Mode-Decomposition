"""
Training loop.

Optimiser settings are the ones stated in Sec. V of the paper:
Adam (beta1 = 0.9, beta2 = 0.999), weight decay 1e-4, gradient clipping at
norm 1.0, 300 epochs, lr = 5e-4 for MDA and 3e-4 for the Mamba / Transformer
baselines.

Teacher forcing: with `eps_e = 1 - e/E` at epoch e of E, the decoder is fed the
ground-truth block; otherwise it is fed its own prediction from the previous
step.  This only has an effect when `rollout > 1`, i.e. when a training sample
covers several consecutive H-blocks -- with rollout = 1 there is no previous
step to feed back and the schedule is inactive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import MDALoss, LossWeights
from .metrics import accuracy

__all__ = ["TrainConfig", "fit", "run_epoch", "teacher_forcing_ratio"]


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 5e-4                 # 3e-4 for the baselines
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0
    batch_size: int = 64
    rollout: int = 1
    detach_feedback: bool = True     # stop gradients through fed-back predictions
    accuracy_mode: str = "nrmse"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    log_every: int = 10


def teacher_forcing_ratio(epoch: int, epochs: int) -> float:
    """eps_e = 1 - e/E."""
    return max(0.0, 1.0 - epoch / max(1, epochs))


def _rollout(model, x_h, x_r, y_h, y_r, H, R, eps, detach):
    """Run R consecutive H-blocks with scheduled sampling; returns stacked preds."""
    buf_h, buf_r = x_h, x_r
    W = x_h.shape[-1]
    ph_all, pr_all = [], []
    for r in range(R):
        out = model(buf_h, buf_r)
        ph, pr = out["modes_h"], out["modes_r"]
        ph_all.append(ph)
        pr_all.append(pr)
        if r == R - 1:
            break
        gt_h = y_h[..., r * H:(r + 1) * H]
        gt_r = y_r[..., r * H:(r + 1) * H]
        fb_h = ph.detach() if detach else ph
        fb_r = pr.detach() if detach else pr
        m = (torch.rand(x_h.shape[0], 1, 1, device=x_h.device) < eps)
        nxt_h = torch.where(m, gt_h, fb_h)
        nxt_r = torch.where(m, gt_r, fb_r)
        buf_h = torch.cat([buf_h, nxt_h], dim=-1)[..., -W:]
        buf_r = torch.cat([buf_r, nxt_r], dim=-1)[..., -W:]
    return torch.cat(ph_all, dim=-1), torch.cat(pr_all, dim=-1)


def run_epoch(model, loader, loss_fn, cfg: TrainConfig, H: int, eps: float,
              optimiser=None):
    train = optimiser is not None
    model.train(train)
    tot, n, accs = 0.0, 0, []
    parts_sum = {}
    for x_h, x_r, y_h, y_r in loader:
        x_h, x_r = x_h.to(cfg.device), x_r.to(cfg.device)
        y_h, y_r = y_h.to(cfg.device), y_r.to(cfg.device)
        with torch.set_grad_enabled(train):
            ph, pr = _rollout(model, x_h, x_r, y_h, y_r, H, cfg.rollout, eps,
                              cfg.detach_feedback)
            loss, parts = loss_fn({"modes_h": ph, "modes_r": pr}, y_h, y_r)
        if train:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
        b = x_h.shape[0]
        tot += float(loss.detach()) * b
        n += b
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v * b
        accs.append(accuracy(ph.sum(1), y_h.sum(1), cfg.accuracy_mode, axis=-1).mean())
    return tot / max(n, 1), {k: v / max(n, 1) for k, v in parts_sum.items()}, float(np.mean(accs))


def fit(model, train_ds, val_ds, H: int, cfg: TrainConfig | None = None,
        weights: LossWeights | None = None, verbose: bool = True):
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(cfg.device)
    loss_fn = MDALoss(weights)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                           weight_decay=cfg.weight_decay)
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    vl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds else None

    history, best, best_state = [], -np.inf, None
    for e in range(cfg.epochs):
        eps = teacher_forcing_ratio(e, cfg.epochs)
        t0 = time.perf_counter()
        tr_loss, tr_parts, tr_acc = run_epoch(model, tl, loss_fn, cfg, H, eps, opt)
        rec = {"epoch": e, "eps": eps, "train_loss": tr_loss, "train_acc": tr_acc,
               "secs": time.perf_counter() - t0, **{f"train_{k}": v for k, v in tr_parts.items()}}
        if vl is not None:
            va_loss, va_parts, va_acc = run_epoch(model, vl, loss_fn, cfg, H, 0.0)
            rec.update({"val_loss": va_loss, "val_acc": va_acc})
            if va_acc > best:
                best = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(rec)
        if verbose and (e % cfg.log_every == 0 or e == cfg.epochs - 1):
            msg = (f"epoch {e:4d}  eps={eps:.2f}  train_loss={tr_loss:.5f}  "
                   f"train_acc={tr_acc:6.2f}%")
            if vl is not None:
                msg += f"  val_acc={rec['val_acc']:6.2f}%"
            print(msg + f"  ({rec['secs']:.1f}s)")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
