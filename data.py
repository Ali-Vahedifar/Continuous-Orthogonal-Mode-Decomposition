"""
Data pipeline: raw haptic traces -> C-OMD modes -> sliding windows.

About the dataset
-----------------
The paper uses the Novint-Falcon / Chai3D kinaesthetic-interaction traces
(Rodriguez-Guevara & Hernandez Gobertti, Zenodo 10.5281/zenodo.14924062):
3-DoF position, velocity and force at 1 kHz, on both the human-operator and the
teleoperated-robot side, over five manipulation tasks.

This repository does **not** hard-code the file layout of that archive, because
column names differ between the released files and guessing them silently would
be the easiest way to produce wrong results.  Point :class:`ChannelSpec` at the
right columns once (or let :func:`autodetect_spec` propose a mapping, which it
prints for you to check) and everything downstream is generic.

About leakage
-------------
VMD-family decompositions are **not causal**: decomposing a whole trace and then
cutting windows out of it lets information from after the prediction boundary
leak into the input modes.  The paper's own cost accounting (it charges the
decomposition FLOPs and latency to inference) implies the decomposition runs on
the receiver's history buffer at run time, so `mode="online"` is the default
here.  `mode="offline"` is available because it is much faster to prepare, but
it is only appropriate for exploratory work -- it is flagged everywhere it is
used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .decomposition import COMDConfig, comd, vmd

try:                                    # torch is optional for the data prep
    import torch
    from torch.utils.data import Dataset
except ImportError:                     # pragma: no cover
    torch = None
    Dataset = object

__all__ = ["ChannelSpec", "load_trace", "autodetect_spec", "decompose_channel",
           "ModeWindowDataset", "chronological_split", "SIDES", "SIGNALS", "AXES"]

SIDES = ("human", "robot")
SIGNALS = ("F", "V", "P")
AXES = ("x", "y", "z")


# --------------------------------------------------------------------------- #
#  Column mapping
# --------------------------------------------------------------------------- #
@dataclass
class ChannelSpec:
    """Maps (side, signal, axis) to a column of the raw file.

    ``columns[("human", "F", "x")] = "master_force_x"`` (name) or ``3`` (index).
    """

    columns: dict = field(default_factory=dict)
    fs: float = 1000.0

    def key(self, side, sig, ax):
        return self.columns[(side, sig, ax)]

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"fs": self.fs,
                       "columns": {"|".join(k): v for k, v in self.columns.items()}},
                      fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(columns={tuple(k.split("|")): v for k, v in d["columns"].items()},
                   fs=d.get("fs", 1000.0))


_SIDE_HINTS = {"human": ("human", "master", "operator", "m_", "op_", "user"),
               "robot": ("robot", "slave", "follower", "teleoperator", "s_", "rob")}
_SIG_HINTS = {"F": ("force", "_f", "frc"), "V": ("vel", "_v", "speed"),
              "P": ("pos", "_p", "position")}


def autodetect_spec(header: Sequence[str], fs: float = 1000.0, verbose: bool = True):
    """Propose a :class:`ChannelSpec` from a CSV header.

    Returns ``(spec, unmatched)``.  It never guesses silently: the proposal is
    printed and any (side, signal, axis) it could not resolve is reported so you
    can fill it in by hand.
    """
    cols = {}
    low = [h.strip().lower() for h in header]
    for side, s_hints in _SIDE_HINTS.items():
        for sig, g_hints in _SIG_HINTS.items():
            for ax in AXES:
                cand = [h for h in low
                        if any(sh in h for sh in s_hints)
                        and any(gh in h for gh in g_hints)
                        and (h.endswith(ax) or f"_{ax}" in h or f".{ax}" in h)]
                if len(cand) == 1:
                    cols[(side, sig, ax)] = header[low.index(cand[0])]
    spec = ChannelSpec(columns=cols, fs=fs)
    missing = [(s, g, a) for s in SIDES for g in SIGNALS for a in AXES
               if (s, g, a) not in cols]
    if verbose:
        print(f"[autodetect] matched {len(cols)}/18 channels")
        for k, v in sorted(cols.items()):
            print(f"    {k} -> {v}")
        if missing:
            print("[autodetect] UNMATCHED (fill these in by hand):")
            for k in missing:
                print(f"    {k}")
    return spec, missing


def load_trace(path: str, spec: ChannelSpec) -> dict:
    """Load one trace file into ``{(side, signal, axis): 1-D array}``.

    Supports .csv/.tsv (delimiter sniffed), .npz and .mat (v7 via scipy).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt", ".tsv"):
        with open(path) as fh:
            first = fh.readline()
        delim = "\t" if "\t" in first else ("," if "," in first else None)
        raw = np.genfromtxt(path, delimiter=delim, names=True, dtype=float,
                            deletechars="", replace_space="_")
        get = lambda c: np.asarray(raw[c], dtype=float) if isinstance(c, str) \
            else np.asarray(raw[raw.dtype.names[c]], dtype=float)
    elif ext == ".npz":
        raw = np.load(path)
        get = lambda c: np.asarray(raw[c], dtype=float)
    elif ext == ".mat":
        from scipy.io import loadmat
        raw = loadmat(path)
        get = lambda c: np.asarray(raw[c], dtype=float).ravel()
    else:
        raise ValueError(f"unsupported file type: {ext}")

    out = {}
    for k, col in spec.columns.items():
        out[tuple(k)] = get(col)
    lens = {v.size for v in out.values()}
    if len(lens) != 1:
        raise ValueError(f"channels have different lengths: {sorted(lens)}")
    return out


# --------------------------------------------------------------------------- #
#  Decomposition into modes
# --------------------------------------------------------------------------- #
def decompose_channel(x: np.ndarray, cfg: COMDConfig, method: str = "comd",
                      mode: str = "online", buffer: int = 256, stride: int = 1,
                      alpha_vmd: float = 2000.0, verbose: bool = False):
    """Decompose one 1-D channel into K modes.

    mode="offline": one decomposition of the whole trace.  Fast, but the modes
        at time t depend on samples after t (see the module docstring).
        Returns an array of shape (K, T).
    mode="online": at every stride-th sample t the last `buffer` samples are
        decomposed and only the modes at the end of that buffer are kept.  This
        is causal and matches the latency/FLOP accounting reported in the paper.
        Returns (K, T) with the first `buffer` samples filled from the first
        buffer's decomposition.
    """
    x = np.asarray(x, dtype=float).ravel()
    T = x.size
    run = (lambda seg: comd(seg, cfg)[0]) if method == "comd" else \
          (lambda seg: vmd(seg, K=cfg.K, alpha_vmd=alpha_vmd)[0])

    if mode == "offline":
        return run(x)

    if mode != "online":
        raise ValueError("mode must be 'offline' or 'online'")

    out = np.zeros((cfg.K, T))
    first = run(x[:buffer])
    out[:, :buffer] = first
    t = buffer
    while t < T:
        end = min(t + stride, T)
        seg = x[end - buffer:end]
        m = run(seg)
        out[:, t:end] = m[:, -(end - t):]
        t = end
        if verbose and (t // stride) % 100 == 0:
            print(f"  online decomposition {t}/{T}", end="\r")
    return out


# --------------------------------------------------------------------------- #
#  Windowing
# --------------------------------------------------------------------------- #
def chronological_split(T: int, train: float = 0.7, val: float = 0.1,
                        test: float = 0.2):
    """Contiguous 70/10/20 split (paper: 70% train, 10% val, 20% test).

    Contiguous rather than random so that neighbouring, almost identical windows
    cannot end up on both sides of the split.
    """
    assert abs(train + val + test - 1.0) < 1e-9
    i1 = int(round(T * train))
    i2 = int(round(T * (train + val)))
    return slice(0, i1), slice(i1, i2), slice(i2, T)


class ModeWindowDataset(Dataset):
    """Sliding windows over pre-computed modes.

    Parameters
    ----------
    modes_h, modes_r : (K, T) arrays of human / robot modes for one channel.
    W : input window length.  H : prediction horizon.
    rollout : number of consecutive H-blocks returned as target.  rollout > 1
        enables the scheduled-sampling loop in `train.py` (Sec. V, teacher
        forcing ratio eps_e = 1 - e/E).
    """

    def __init__(self, modes_h, modes_r, W: int, H: int, rollout: int = 1,
                 stride: int = 1, indices: slice | None = None):
        if torch is None:
            raise ImportError("PyTorch is required for ModeWindowDataset")
        mh = np.asarray(modes_h, dtype=np.float32)
        mr = np.asarray(modes_r, dtype=np.float32)
        assert mh.shape == mr.shape, "human and robot modes must align"
        if indices is not None:
            mh, mr = mh[:, indices], mr[:, indices]
        self.mh, self.mr = mh, mr
        self.W, self.H, self.rollout = W, H, rollout
        span = W + rollout * H
        self.starts = np.arange(0, mh.shape[1] - span + 1, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        W, H, R = self.W, self.H, self.rollout
        x_h = self.mh[:, s:s + W]
        x_r = self.mr[:, s:s + W]
        y_h = self.mh[:, s + W:s + W + R * H]
        y_r = self.mr[:, s + W:s + W + R * H]
        return (torch.from_numpy(x_h.copy()), torch.from_numpy(x_r.copy()),
                torch.from_numpy(y_h.copy()), torch.from_numpy(y_r.copy()))
