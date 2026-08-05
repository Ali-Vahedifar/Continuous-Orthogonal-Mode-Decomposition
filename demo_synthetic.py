"""
End-to-end smoke test on a synthetic bilateral haptic trace.

This runs the whole pipeline -- decomposition, windowing, training, evaluation,
latency and FLOP measurement -- without needing the real dataset, so you can
check the install in a couple of minutes.

The numbers it prints are for THIS synthetic signal.  They are not the paper's
results and should not be compared with them; use `run_experiment.py` on the
real traces for that.

    python scripts/demo_synthetic.py --epochs 20
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comd.data import ModeWindowDataset, chronological_split, decompose_channel
from comd.decomposition import COMDConfig
from comd.infer import decomposition_gflops, measure_latency, model_gflops
from comd.losses import LossWeights
from comd.metrics import accuracy
from comd.models import MDAConfig, build_model
from comd.train import TrainConfig, fit


def synthetic_bilateral(T=6000, fs=1000.0, seed=0):
    """A crude stand-in for a teleoperation force trace.

    Low-frequency operator intent + a mid-band texture + contact transients; the
    robot side is a delayed, damped and slightly noisier version of the human
    side, so the two are correlated but not identical -- which is the structure
    the cross-side attention is supposed to exploit.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    intent = 1.0 * np.sin(2 * np.pi * 1.5 * t) + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    texture = 0.30 * np.sin(2 * np.pi * 37.0 * t + 0.5)
    contact = np.zeros(T)
    for c in rng.integers(200, T - 200, size=8):
        n = np.arange(T - c)
        contact[c:] += 0.6 * np.exp(-n / 60.0) * np.sin(2 * np.pi * 95.0 * n / fs)
    human = intent + texture + contact + 0.02 * rng.standard_normal(T)

    d = 12                                     # round-trip delay, samples
    robot = np.concatenate([np.zeros(d), human[:-d]])
    robot = 0.85 * robot + 0.05 * rng.standard_normal(T)
    return human, robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--H", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--model", default="mda", choices=["mda", "transformer"])
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=" * 72)
    print("DEMO on synthetic data -- these numbers are not the paper's results")
    print("=" * 72)

    human, robot = synthetic_bilateral(args.T)
    cfg_d = COMDConfig(K=args.K)

    t0 = time.perf_counter()
    # offline decomposition: fine for a demo, see comd/data.py on why the real
    # experiments use mode="online"
    mh = decompose_channel(human, cfg_d, method="comd", mode="offline")
    mr = decompose_channel(robot, cfg_d, method="comd", mode="offline")
    print(f"\ndecomposed 2 x {args.T} samples in {time.perf_counter()-t0:.2f}s")
    G = mh @ mh.T
    d = np.sqrt(np.diag(G))
    C = np.abs(G / np.outer(d, d))
    np.fill_diagonal(C, 0)
    print(f"human modes: centre-frequency-sorted, max off-diagonal correlation "
          f"{C.max():.2e}, reconstruction error "
          f"{np.linalg.norm(mh.sum(0)-human)/np.linalg.norm(human):.2e}")

    tr, va, te = chronological_split(args.T)
    mk = lambda sl: ModeWindowDataset(mh, mr, args.W, args.H, indices=sl)
    ds_tr, ds_va, ds_te = mk(tr), mk(va), mk(te)
    print(f"windows: train {len(ds_tr)}, val {len(ds_va)}, test {len(ds_te)}")

    cfg_m = MDAConfig(K=args.K, W=args.W, H=args.H)
    model = build_model(args.model, cfg_m)
    cfg_t = TrainConfig(epochs=args.epochs, batch_size=64, device=args.device,
                        lr=5e-4 if args.model == "mda" else 3e-4, log_every=5)
    model, hist = fit(model, ds_tr, ds_va, args.H, cfg_t, LossWeights())

    # ---- test accuracy ---------------------------------------------------
    model.eval()
    ph, pr, th, tr_ = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds_te)):
            x_h, x_r, y_h, y_r = ds_te[i]
            o = model(x_h[None].to(args.device), x_r[None].to(args.device))
            ph.append(o["signal_h"][0].cpu().numpy())
            pr.append(o["signal_r"][0].cpu().numpy())
            th.append(y_h.sum(0).numpy())
            tr_.append(y_r.sum(0).numpy())
    acc_h = accuracy(np.array(ph), np.array(th), "nrmse", axis=-1).mean()
    acc_r = accuracy(np.array(pr), np.array(tr_), "nrmse", axis=-1).mean()
    print(f"\ntest accuracy (nrmse convention): human {acc_h:.2f}%  robot {acc_r:.2f}%")

    # ---- cost ------------------------------------------------------------
    lat_net = measure_latency(model, args.K, args.W, args.device, n_runs=100)
    lat_all = measure_latency(model, args.K, args.W, args.device, n_runs=20,
                              decomposition="comd", buffer=256, comd_cfg=cfg_d)
    print(f"latency, network only          : {lat_net['mean_ms']:.4f} ms "
          f"(median {lat_net['median_ms']:.4f})")
    print(f"latency, + C-OMD of 256 samples: {lat_all['mean_ms']:.4f} ms "
          f"(median {lat_all['median_ms']:.4f})")
    print(f"network GFLOPs/forward         : {model_gflops(model, args.K, args.W):.4f}")
    print(f"C-OMD GFLOPs (256, K={args.K}, 50 it): "
          f"{decomposition_gflops(256, args.K, 50, 20, 'comd'):.4f}")
    print("\nDone. Reminder: synthetic data, short training -- not paper numbers.")


if __name__ == "__main__":
    main()
