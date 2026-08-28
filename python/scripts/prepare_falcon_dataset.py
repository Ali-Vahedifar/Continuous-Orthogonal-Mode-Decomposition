"""
Convert the raw Novint-Falcon/Chai3D Client+Server CSV pairs into merged,
timestamp-aligned .npz traces that `run_experiment.py --data ... --spec ...`
can load directly.

The archive (Zenodo 10.5281/zenodo.14924062) stores each recording as two
files, one per side, with independent, offset clocks -- see
`comd.data.load_paired_falcon_trace` for why row-index alignment would be
wrong and what this does instead.

    python scripts/prepare_falcon_dataset.py \
        --dataset-dir ../Dataset --out-dir ../Dataset_prepared
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comd.data import AXES, SIDES, SIGNALS, ChannelSpec, load_paired_falcon_trace


def find_pairs(dataset_dir: str):
    pairs = []
    for root, _dirs, files in os.walk(dataset_dir):
        client = [f for f in files if f.lower().endswith("client.csv")]
        server = [f for f in files if f.lower().endswith("server.csv")]
        if len(client) == 1 and len(server) == 1:
            pairs.append((root, client[0], server[0]))
    return sorted(pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fs", type=float, default=1000.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = find_pairs(args.dataset_dir)
    if not pairs:
        raise SystemExit(f"no Client/Server csv pairs found under {args.dataset_dir}")

    # One channel-name mapping works for every merged file, so it is written
    # once rather than autodetected per-file.
    key = lambda side, sig, ax: f"{side}_{sig}_{ax}"
    spec_cols = {(side, sig, ax): key(side, sig, ax)
                 for side in SIDES for sig in SIGNALS for ax in AXES}
    spec = ChannelSpec(columns=spec_cols, fs=args.fs)
    spec_path = os.path.join(args.out_dir, "spec.json")
    spec.save(spec_path)
    print(f"wrote {spec_path} ({len(spec_cols)} channels)\n")

    manifest = []
    for root, client_fn, server_fn in pairs:
        condition = os.path.basename(os.path.dirname(root))
        task = os.path.basename(root)
        name = f"{condition}__{task}"
        trace = load_paired_falcon_trace(
            os.path.join(root, client_fn), os.path.join(root, server_fn), fs=args.fs)
        out = {key(*k): v for k, v in trace.items()}
        out_path = os.path.join(args.out_dir, f"{name}.npz")
        np.savez(out_path, **out)
        n = int(next(iter(out.values())).size)
        manifest.append({"name": name, "condition": condition, "task": task,
                         "path": out_path, "samples": n,
                         "duration_s": n / args.fs})

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    total = sum(m["samples"] for m in manifest)
    print(f"\n{len(manifest)} traces written to {args.out_dir} "
          f"({total} samples total, {total / args.fs / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
