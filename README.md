# C-OMD + MDA

Reference implementation of **Continuous-Orthogonal Mode Decomposition (C-OMD)** and the
**Mode-Domain Architecture (MDA)** for bilateral haptic signal prediction over the Tactile
Internet.

Two languages, on purpose:

- **MATLAB** for the decomposition, because that is where the VMD/VME/SVMD family lives and
  where you probably want to poke at modes interactively.
- **PyTorch** for the decomposition *and* the network, so the whole pipeline can run on a GPU
  without a MATLAB round-trip.

Both implementations of the decomposition were written from the same equations and produce the
same numbers to machine precision (there is a parity check further down).

---

## What is actually in here

The idea behind the method, in one paragraph: VMD gives you modes that *happen* to be
orthogonal when their frequency bands do not overlap, and stops being orthogonal the moment they
do. Overlap is exactly what haptic signals do — operator intent, surface texture and contact
transients are not politely separated in frequency. C-OMD adds an explicit orthogonality
constraint to the variational problem and enforces it with a Newton–Schulz projection that runs
after each Wiener-filter step, so the modes come out orthogonal by construction rather than by
luck. MDA then predicts each mode separately and lets the human and robot branches exchange
keys/values through cross-attention.

```
comd-mda/
├── matlab/
│   ├── comd.m                  C-OMD, the whole ADMM loop
│   ├── newton_schulz_ortho.m   the projection step (Eqs. 17, 20)
│   ├── vmd_baseline.m          classical VMD, same machinery, beta = 0, no projection
│   ├── demo_comd.m             run this first
│   └── export_modes.m          decompose a trace and hand the modes to Python
├── python/
│   ├── comd/
│   │   ├── decomposition.py    C-OMD + VMD (NumPy only, no torch needed)
│   │   ├── models.py           TCN encoder/decoder, both attention blocks, MDA, baselines
│   │   ├── losses.py           the four-term objective
│   │   ├── metrics.py          accuracy definitions (read this one, see the caveat below)
│   │   ├── data.py             column mapping, causal decomposition, windows, splits
│   │   ├── train.py            training loop + teacher-forcing schedule
│   │   └── infer.py            packet-loss restoration, latency, FLOPs
│   └── scripts/
│       ├── demo_synthetic.py   end-to-end smoke test, no dataset required
│       └── run_experiment.py   the real thing
└── tests/
    └── test_decomposition.py   8 property tests
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
and prints accuracy, measured latency and measured FLOPs. It exists to prove the install works.
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

That is the whole argument of the paper in four lines: same signal, same initialisation, same
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

---

## Equation → code map

Everything below is a literal transcription, and the comments in the source carry the same
equation numbers so you can diff them against the LaTeX.

| Paper | What it is | Python | MATLAB |
|---|---|---|---|
| (8) | augmented Lagrangian | `decomposition.py` (structure of the loop) | `comd.m` |
| (11) | mode update / Wiener filter | `comd()` | `comd.m` |
| (12) | numerator **A** with the orthogonality correction | `comd()` | `comd.m` |
| (13) | centre-frequency update | `comd()` | `comd.m` |
| (14), (15) | dual ascent on λ and Γ | `comd()` | `comd.m` |
| (16) | Gram matrix | `gram_matrix()` | inline + `newton_schulz_ortho.m` |
| (17) | Newton–Schulz normalise / coefficient / update | `newton_schulz_orthogonalize()` | `newton_schulz_ortho.m` |
| (18)–(20) | per-frequency orthogonalisation `v⊥ = P v` | `newton_schulz_orthogonalize()` | `newton_schulz_ortho.m` |
| Prop. 1 | band-limitedness preservation | `tests/test_decomposition.py` | — |
| Fig. 2(a) | per-mode TCN encoder, d = 1,2,4 + skip | `models.TCNEncoder` | — |
| Fig. 2(b) | per-mode TCN decoder, d = 4,2,1 + `R^d → R^H` | `models.TCNDecoder` | — |
| Fig. 2(c) | cross-side cross-attention + linear coupling | `models.CrossSideCrossAttention` | — |
| Fig. 2(d) | cross-mode self-attention | `models.CrossModeSelfAttention` | — |
| Fig. 1 | bilateral MDA | `models.MDA` | — |
| (21)–(25) | the four-term loss, λ = 0.1 / 0.01 / 0.05, τ = 0.01 | `losses.MDALoss` | — |
| Sec. V | teacher forcing ε_e = 1 − e/E | `train.teacher_forcing_ratio` | — |
| Sec. V | autoregressive restoration | `infer.autoregressive_restore` | — |
| Table I | GFLOPs, inference time | `infer.model_gflops`, `infer.measure_latency` | — |

One detail worth flagging, since it bites everyone who implements Eq. (20): the projection is
*one* K×K matrix applied identically at every frequency, not a separate solve per frequency bin.
Both implementations accumulate the Newton–Schulz coefficient matrices into a single `P` and
assert `V⊥ == P V` (there is a test for it).

---

## Things the paper does not pin down, and what this repo does instead

This is the section to read before you compare any number here against any number there. None of
these are bugs, they are genuine gaps, and each one is a switch rather than a silent default.

**1. The meaning of α.** In Eq. (8) the penalty multiplies the *reconstruction* term and the
bandwidth term has unit weight; classical VMD does it the other way round. The two conventions
are reciprocal, `alpha_paper = 1 / alpha_vmd`, so the familiar `alpha_vmd = 2000` is `alpha =
5e-4` here. `COMDConfig.from_vmd_alpha(2000)` if you prefer VMD units.

**2. The scale of β.** β multiplies `|G_ij|²`, i.e. squared *energies*, while α multiplies a
squared amplitude — so the pair is only meaningful once the signal has a fixed scale. The
decomposition normalises the input to unit L2 norm internally and rescales the modes on the way
out (exact, it is a linear operation), and `beta` defaults to `alpha`. With `normalize=False` and
β from a different scale, the iteration diverges; that is arithmetic, not a bug.

**3. Newton–Schulz and energy.** Eq. (17a) divides by `(Σ_j ||m_j||²)^{1/2}` and the iteration
drives `G → I`, so taken verbatim the projection returns *orthonormal* modes that no longer sum
to `f`. The alternation in Sec. IV-D only works if the projection stays on the same scale as the
Wiener step, so by default the same orthogonalising map is applied and each mode's original L2
norm is restored afterwards (`preserve_energy=True`). Set it to `False` for the literal
equations. This changes scaling only, not the directions being orthogonalised.

**4. Parallel vs sequential mode updates.** The paper stresses that C-OMD keeps VMD's parallel
structure. A fully parallel (Jacobi) sweep is implemented (`update="jacobi"`) and it converges
fine for well-separated modes — but for overlapping bands it needs damping, `relax ≤ 0.7`, or it
oscillates. The default is `update="gauss_seidel"` because it is unconditionally stable here;
switch to damped Jacobi when you want the GPU-parallel form.

**5. How many Newton–Schulz steps.** Not specified. Default is 20. On clean synthetic signals 8
is already enough to hit 1e-14; on noisy broadband modes (K=5, 0.2 σ noise) 5 steps leave 4.3e-2
residual correlation and 20 steps bring it to 5.8e-6 at identical reconstruction error. Cheap
either way — it is K×K work.

**6. "Accuracy (%)".** The paper reports accuracy for a regression task without defining it.
`metrics.py` implements four conventions (`nrmse`, `fit`, `smape`, `r2`) and makes you pick one.
Default is `nrmse` = `100·(1 − ‖ŷ−y‖₂/‖y‖₂)`. The *ranking* of methods is stable across the four;
the absolute values are not, so do not mix them within one table.

**7. Network hyper-parameters.** Channel width, latent width, kernel size, head count and FFN
expansion are not stated. Defaults are d = 64, 64 channels, kernel 3, 4 heads, FFN ×2, all in
`MDAConfig` and all marked in the source as unspecified-by-the-paper.

**8. "Transformer + Fourier".** The baseline is named but the Fourier part is not defined. The
implementation concatenates rFFT magnitude and phase of the input window into the token
embedding, and says so in its docstring. Treat it as this repo's interpretation.

**9. Mamba.** Not vendored. `mamba-ssm` is an optional dependency and the baseline raises a
clear ImportError instead of quietly substituting something else.

**10. `L_recon` is a sum, `L_pred` is a mean.** That is what Eqs. (22)–(23) say, so that is what
is implemented, but it does mean the two terms enter on very different scales before λ₁ = 0.1 is
applied. Worth knowing when you look at the loss components.

**11. Causality.** VMD-family decompositions are not causal: decomposing a whole trace and then
cutting windows lets post-boundary information into the input modes. Since the paper charges the
decomposition's FLOPs and latency to inference, the receiver must be decomposing its history
buffer at run time, so `decompose_channel(..., mode="online")` is the default — it decomposes the
last `buffer` samples at each step and keeps only the newest ones. `mode="offline"` is available
because it is much faster to prepare, and it is flagged as leaky everywhere it appears.

**12. The dataset's column layout.** The traces are the Novint Falcon / Chai3D kinaesthetic
interaction recordings ([Zenodo 14924062](https://doi.org/10.5281/zenodo.14924062), 3-DoF
position/velocity/force at 1 kHz, human and robot side, five tasks). The exact column names are
*not* hard-coded here, because guessing them silently is the easiest way to produce confidently
wrong results. Run:

```bash
python scripts/run_experiment.py --data trace.csv --autodetect --spec-out spec.json
```

which proposes a mapping, prints it, and lists whatever it could not resolve for you to fill in
by hand. Then pass `--spec spec.json`.

---

## Running the experiments

```bash
# window-size sweep, both decompositions, force channels
python scripts/run_experiment.py \
    --data trace.csv --spec spec.json \
    --signals F --W 1 5 10 25 50 100 --decomp comd vmd \
    --K 3 --H 100 --epochs 300 --seeds 0 1 2 3 4 \
    --out sweep_W.json

# number of modes
python scripts/run_experiment.py --data trace.csv --spec spec.json \
    --K 2 3 4 5 6 7 8 --W 5 --epochs 300 --out sweep_K.json

# SNR robustness
python scripts/run_experiment.py --data trace.csv --spec spec.json \
    --snr 30 20 10 5 0 --W 5 --epochs 300 --out sweep_snr.json
```

Every record in the output JSON carries the accuracy for both sides, the *measured* latency (with
and without the decomposition in the loop), the *measured* network GFLOPs from
`torch.utils.flop_counter`, and an analytic decomposition FLOP count whose formula is written out
in `infer.decomposition_gflops` so you can check it rather than trust it.

Optimiser settings follow Sec. V as stated: Adam (0.9, 0.999), weight decay 1e-4, gradient
clipping at 1.0, 300 epochs, lr 5e-4 for MDA and 3e-4 for the baselines, and a 70/10/20 split —
contiguous rather than random, so that near-identical neighbouring windows cannot straddle it.

---

## What has been verified

Run on Python 3.12 / NumPy 2.4.4 / SciPy 1.17.1 / PyTorch 2.13.0 and GNU Octave 8.4.0.

- `tests/test_decomposition.py` — 8/8 passing: exact reconstruction; correct centre-frequency
  recovery on the standard 2/24/288 Hz test signal; orthogonality on overlapping bands strictly
  better than VMD by >2 orders of magnitude at no reconstruction cost; Newton–Schulz converging
  to `G = I`; the projection being a single per-frequency linear map; Proposition 1
  (no spectral content created where the mode set was silent); modes staying spectrally compact
  (>95 % of each mode's energy within ±10 Hz of its own centre after projection).
- **Python ↔ MATLAB parity** on the overlap benchmark, same digits:

  | | VMD | ns=1 | ns=5 | ns=8 | ns=20 |
  |---|---|---|---|---|---|
  | Python `max\|corr_ij\|` | 8.680e-04 | 1.694e-04 | 1.632e-06 | 1.009e-14 | 1.009e-14 |
  | MATLAB `max\|corr_ij\|` | 8.680e-04 | 1.694e-04 | 1.632e-06 | 1.009e-14 | 1.009e-14 |

- Forward/backward pass, loss and gradient flow for MDA and the Transformer baseline at
  W ∈ {1, 5, 100}.
- The full pipeline end-to-end (autodetect → load → causal online decomposition → windows →
  train → evaluate → latency → FLOPs → JSON) on a synthetic CSV.

**What has not been verified:** the paper's reported accuracies and timings. This repository
ships no precomputed results and no numbers copied out of the paper. Nothing here has been tuned
against the real dataset, so treat every hyper-parameter that the paper did not state as a
starting point and not as a reproduction target. If your numbers differ, work through the twelve
points above before assuming either side is wrong.

---

## Citation

Fill in the venue and year once they are final:

```bibtex
@inproceedings{vahedifar_comd,
  title     = {Continuous Orthogonal Mode Decomposition: Haptic Signal
               Prediction in Tactile Internet},
  author    = {Vahedifar, Mohammad Ali and Nazari, Mojtaba and Zhang, Qi},
  booktitle = {TODO},
  year      = {TODO}
}
```

The VMD baseline follows Dragomiretskiy & Zosso, *Variational Mode Decomposition*, IEEE TSP
62(3), 2014. It is reimplemented from the published update equations rather than adapted from
the authors' code.
