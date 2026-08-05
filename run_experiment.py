"""
Run one experiment on a real haptic trace.

Examples
--------
# 1. look at the file and get a proposed column mapping to check
python scripts/run_experiment.py --data trace.csv --autodetect --spec-out spec.json

# 2. train on the force channels with C-OMD modes, W=5, H=100
python scripts/run_experiment.py --data trace.csv --spec spec.json \
       --signals F --decomp comd --W 5 --H 100 --epochs 300 --out results.json

# 3. sweep the window sizes of the paper and both decompositions
python scripts/run_experiment.py --data trace.csv --spec spec.json \
       --W 1 5 10 25 50 100 --decomp comd vmd --epochs 300 --out sweep.json

Everything reported is measured on your data and your machine.  No result is
read back from the paper.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comd.data import (AXES, ChannelSpec, ModeWindowDataset, autodetect_spec,
                       chronological_split, decompose_channel, load_trace)
from comd.decomposition import COMDConfig
from comd.infer import decomposition_gflops, measure_latency, model_gflops
from comd.losses import LossWeights
from comd.metrics import accuracy
from comd.models import MDAConfig, build_model
from comd.train import TrainConfig, fit


def add_awgn(x, snr_db, rng):
    """Additive white Gaussian noise at a given SNR, used for the SNR sweep."""
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    return x + np.sqrt(p_noise) * rng.standard_normal(x.shape)


def read_header(path):
    with open(path) as fh:
        line = fh.readline().strip()
    delim = "\t" if "\t" in line else ","
    return [c.strip() for c in line.split(delim)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--spec", help="ChannelSpec json (see --autodetect)")
    p.add_argument("--autodetect", action="store_true")
    p.add_argument("--spec-out", default="spec.json")
    p.add_argument("--fs", type=float, default=1000.0)
    p.add_argument("--signals", nargs="+", default=["F"], choices=["F", "V", "P"])
    p.add_argument("--axes", nargs="+", default=list(AXES))
    p.add_argument("--decomp", nargs="+", default=["comd"], choices=["comd", "vmd"])
    p.add_argument("--K", type=int, nargs="+", default=[3])
    p.add_argument("--W", type=int, nargs="+", default=[5])
    p.add_argument("--H", type=int, default=100)
    p.add_argument("--snr", type=float, nargs="+", default=[None])
    p.add_argument("--model", default="mda", choices=["mda", "transformer", "mamba"])
    p.add_argument("--decomp-mode", default="online", choices=["online", "offline"])
    p.add_argument("--buffer", type=int, default=256)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rollout", type=int, default=1)
    p.add_argument("--accuracy-mode", default="nrmse",
                   choices=["nrmse", "fit", "smape", "r2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    # ---- column mapping --------------------------------------------------
    if args.autodetect:
        spec, missing = autodetect_spec(read_header(args.data), args.fs)
        spec.save(args.spec_out)
        print(f"\nwrote proposal to {args.spec_out}. "
              f"CHECK IT (and fill in {len(missing)} unmatched channels) "
              f"before running the experiment.")
        return
    if not args.spec:
        p.error("--spec is required (run once with --autodetect to get a draft)")
    spec = ChannelSpec.load(args.spec)
    trace = load_trace(args.data, spec)
    T = len(next(iter(trace.values())))
    print(f"loaded {args.data}: {T} samples, {len(trace)} channels, fs={spec.fs}")

    results = []
    for sig in args.signals:
        for ax in args.axes:
            key_h, key_r = ("human", sig, ax), ("robot", sig, ax)
            if key_h not in trace or key_r not in trace:
                print(f"skip {sig}-{ax}: not in the spec")
                continue
            for snr in args.snr:
                rng = np.random.default_rng(0)
                xh = np.asarray(trace[key_h], float)
                xr = np.asarray(trace[key_r], float)
                if snr is not None:
                    xh, xr = add_awgn(xh, snr, rng), add_awgn(xr, snr, rng)
                for dec in args.decomp:
                    for K in args.K:
                        cfg_d = COMDConfig(K=K)
                        t0 = time.perf_counter()
                        mh = decompose_channel(xh, cfg_d, dec, args.decomp_mode,
                                               args.buffer, args.stride)
                        mr = decompose_channel(xr, cfg_d, dec, args.decomp_mode,
                                               args.buffer, args.stride)
                        t_dec = time.perf_counter() - t0
                        for W in args.W:
                            for seed in args.seeds:
                                r = run_one(mh, mr, xh, xr, K, W, args, dec, sig,
                                            ax, snr, seed, cfg_d, t_dec)
                                results.append(r)
                                print(json.dumps(r))
                                with open(args.out, "w") as fh:
                                    json.dump(results, fh, indent=2)
    print(f"\nwrote {len(results)} records to {args.out}")


def run_one(mh, mr, xh, xr, K, W, args, dec, sig, ax, snr, seed, cfg_d, t_dec):
    T = mh.shape[1]
    tr, va, te = chronological_split(T)
    mk = lambda sl: ModeWindowDataset(mh, mr, W, args.H, args.rollout, indices=sl)
    ds_tr, ds_va, ds_te = mk(tr), mk(va), mk(te)

    cfg_m = MDAConfig(K=K, W=W, H=args.H)
    model = build_model(args.model, cfg_m)
    cfg_t = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                        rollout=args.rollout, device=args.device, seed=seed,
                        accuracy_mode=args.accuracy_mode,
                        lr=5e-4 if args.model == "mda" else 3e-4)
    model, _ = fit(model, ds_tr, ds_va, args.H, cfg_t, LossWeights(), verbose=True)

    model.eval()
    P_h, P_r, Y_h, Y_r = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds_te)):
            x_h, x_r, y_h, y_r = ds_te[i]
            o = model(x_h[None].to(args.device), x_r[None].to(args.device))
            P_h.append(o["signal_h"][0].cpu().numpy())
            P_r.append(o["signal_r"][0].cpu().numpy())
            Y_h.append(y_h.sum(0).numpy())
            Y_r.append(y_r.sum(0).numpy())
    am = args.accuracy_mode
    acc_h = float(accuracy(np.array(P_h), np.array(Y_h), am, axis=-1).mean())
    acc_r = float(accuracy(np.array(P_r), np.array(Y_r), am, axis=-1).mean())

    lat = measure_latency(model, K, W, args.device, n_runs=100)
    lat_full = measure_latency(model, K, W, args.device, n_runs=20,
                               decomposition=dec, buffer=args.buffer,
                               comd_cfg=cfg_d)
    return {
        "signal": sig, "axis": ax, "snr_db": snr, "decomp": dec, "model": args.model,
        "K": K, "W": W, "H": args.H, "seed": seed, "accuracy_mode": am,
        "acc_human": acc_h, "acc_robot": acc_r,
        "latency_net_ms": lat["mean_ms"], "latency_total_ms": lat_full["mean_ms"],
        "gflops_net": model_gflops(model, K, W),
        "gflops_decomp": decomposition_gflops(args.buffer, K, 50, 20, dec),
        "decompose_secs_full_trace": t_dec,
        "n_test_windows": len(ds_te), "device": args.device,
    }


if __name__ == "__main__":
    main()
