# C-OMD + MDA

Reference implementation of **Continuous-Orthogonal Mode Decomposition (C-OMD)** and the
**Mode-Domain Architecture (MDA)** for bilateral haptic signal prediction over the Tactile
Internet.

Two languages, on purpose:

- **MATLAB** for the decomposition, because that's where the VMD/VME/SVMD family lives and
  where you probably want to poke at modes interactively.
- **PyTorch** for the decomposition *and* the network, so the whole pipeline can run on a GPU
  without a MATLAB round-trip.

Both implementations of the decomposition are written from the same equations and produce the
same numbers to machine precision (parity check further down).

---

## What's actually in here

The idea, in one paragraph: VMD gives you modes that *happen* to be orthogonal when their
frequency bands don't overlap, and stops being orthogonal the moment they do. Overlap is exactly
what haptic signals do — operator intent, surface texture, and contact transients aren't
politely separated in frequency. C-OMD adds an explicit orthogonality constraint to the
variational problem and enforces it with a Newton–Schulz projection that runs after each
Wiener-filter step, so the modes come out orthogonal by construction rather than by luck. MDA
then predicts each mode separately and lets the human and robot branches exchange keys/values
through cross-attention.

Alongside C-OMD and plain VMD, the repo also has **SVMD** (successive/sequential VMD) and a
**no-decomposition baseline** (raw signal, no modes at all) as ablations — each method lives in
its own folder, in both languages.

```
comd-mda/
├── matlab/
│   ├── comd/
│   │   ├── comd.m                  C-OMD, the whole ADMM loop
│   │   └── newton_schulz_ortho.m   the projection step (Eqs. 17, 20)
│   ├── vmd/
│   │   └── vmd_baseline.m          classical VMD, same machinery, beta = 0, no projection
│   ├── svmd/
│   │   └── svmd.m                  successive VMD: one mode at a time on the residual
│   ├── none/
│   │   └── none_baseline.m         no-decomposition baseline (K=1, raw signal)
│   ├── demo_comd.m                 run this first
│   └── export_modes.m              decompose a trace and hand the modes to Python
├── python/
│   ├── comd/
│   │   ├── methods/
│   │   │   ├── comd/comd.py        C-OMD (NumPy only, no torch needed)
│   │   │   ├── vmd/vmd.py          classical VMD
│   │   │   ├── svmd/svmd.py        successive VMD
│   │   │   └── none/none.py        no-decomposition baseline
│   │   ├── decomposition.py        re-exports the four methods above
│   │   ├── models.py               TCN encoder/decoder, both attention blocks, MDA, baselines
│   │   ├── losses.py               the four-term objective
│   │   ├── metrics.py              accuracy definitions (read this one, see the caveat below)
│   │   ├── data.py                 column mapping, causal decomposition, windows, splits
│   │   ├── train.py                training loop + teacher-forcing schedule
│   │   └── infer.py                packet-loss restoration, latency, FLOPs
│   └── scripts/
│       ├── demo_synthetic.py           end-to-end smoke test, no dataset required
│       ├── prepare_falcon_dataset.py   turn the raw Falcon Client/Server CSV pairs into
│       │                               timestamp-aligned traces run_experiment.py can read
│       └── run_experiment.py           the real thing
└── tests/
    └── test_decomposition.py       8 property tests
```

---

## Quick start

### Python

```bash
pip install -r requirements.txt
cd python
python ../tests/test_decomposition.py      # ~10 s, should print 8 passes
python scripts/demo_synthetic.py --epochs 20
```

`demo_synthetic.py` builds a fake bilateral trace (low-frequency intent + texture + damped
contact transients, with the robot side delayed and damped), decomposes it, trains MDA briefly,
and prints accuracy, measured latency, and measured FLOPs. It exists to prove the install works.
**Its numbers are not the paper's numbers** and are printed with that warning attached.

### MATLAB / Octave

```matlab
cd matlab
demo_comd
```

```
method          iters  recon rel err    max |corr_ij|   centre freqs (Hz)
------------------------------------------------------------------------------
VMD                16      4.539e-03        8.680e-04   [8 40 52.01]
C-OMD ns=1         16      4.539e-03        1.694e-04   [8 40 52.01]
C-OMD ns=5         16      4.540e-03        1.632e-06   [8 40 52.01]
C-OMD ns=20        17      4.540e-03        1.009e-14   [8 40 52.01]
```

That's the whole argument of the paper in four lines: same signal, same initialization, same
reconstruction error, and the residual inter-mode correlation drops by ten orders of magnitude
once the projection is switched on. The 40 Hz and 52 Hz components are close enough that their
VMD bands overlap, which is why VMD leaves 8.7e-4 behind.

### Using it on your own signal

```python
from comd import comd, COMDConfig

modes, omega, info = comd(x, COMDConfig(K=3))
# modes: (K, N)   omega: centre frequencies in cycles/sample -> multiply by fs for Hz
print(info["recon_rel_error"], info["iterations"])
```

```matlab
[modes, omega, info] = comd(x, struct('K', 3));
```

Swap `comd`/`comd.m` for `vmd`/`vmd_baseline.m`, `svmd`/`svmd.m`, or
`none_decompose`/`none_baseline.m` to run one of the other three methods instead.

---

## Running on your own dataset

If your data looks like the Novint Falcon / Chai3D archive this was built against — one CSV per
side (`*Client.csv` / `*Server.csv`), independent clocks, not pre-aligned — start with:

```bash
python scripts/prepare_falcon_dataset.py --dataset-dir path/to/Dataset --out-dir Dataset_prepared
```

It timestamp-aligns each Client/Server pair (they don't start at the same instant — the archive
has a few seconds of clock offset between sides) onto a common grid via linear interpolation, and
writes one `.npz` per trace plus a shared column spec. Then:

```bash
python scripts/run_experiment.py \
    --data Dataset_prepared/<condition>__<task>.npz --spec Dataset_prepared/spec.json \
    --signals F --decomp comd vmd --W 5 --H 100 --epochs 300 --out results.json
```

For any other CSV layout, `run_experiment.py --data trace.csv --autodetect --spec-out spec.json`
proposes a column mapping and prints what it couldn't resolve, so nothing is guessed silently.

---


## Citation

```bibtex
@misc{vahedifar2026continuousorthogonalmodedecomposition,
      title={Continuous Orthogonal Mode Decomposition: Haptic Signal Prediction in Tactile Internet},
      author={Mohammad Ali Vahedifar and Mojtaba Nazari and Qi Zhang},
      year={2026},
      eprint={2604.09446},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2604.09446},
}
```

The VMD baseline follows Dragomiretskiy & Zosso, *Variational Mode Decomposition*, IEEE TSP
62(3), 2014 — reimplemented from the published update equations, not adapted from the authors'
code. SVMD follows Nazari & Sakhaei, *Successive Variational Mode Decomposition*, Signal
Processing 174:107610, 2020 (see the caveat above on what's literal vs. not).
