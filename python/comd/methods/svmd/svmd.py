"""
Successive Variational Mode Decomposition (SVMD).

Reference: Nazari & Sakhaei, "Successive Variational Mode Decomposition",
Signal Processing 174:107610, 2020.

HONESTY NOTE (read before trusting this as a literal transcription): the
published SVMD extracts modes one at a time, and each step's objective adds a
*second* penalty term beyond the usual VMD bandwidth term -- a constraint on
the leftover residual that discourages it from also being narrowband around
the just-claimed center frequency, so each step's Wiener filter differs from
a plain single-mode VMD solve. That residual-shaping term is not reproduced
here verbatim: reconstructing its exact closed form from memory, without the
paper's equations in front of me, risks silently encoding a wrong formula and
calling it faithful -- worse than being explicit about the gap.

What IS unambiguous, and is what this file implements, is the "successive"
structure itself: unlike VMD/C-OMD, which solve for all K modes jointly,
SVMD solves K independent single-mode problems in sequence, each on the
residual left over from the previous step. That is implemented exactly, by
reusing this repo's own exact VMD solver (`comd.methods.vmd.vmd`) with K=1
at each step -- itself a valid single-mode VMD/C-OMD solve, since a 1-mode
system has no cross-mode terms to get wrong.

If you have the paper's exact per-step equations, paste them and this becomes
a literal transcription the same way `comd()` is for the C-OMD paper.
"""

from __future__ import annotations

import numpy as np

from ..vmd.vmd import vmd

__all__ = ["svmd"]


def svmd(f, K: int = 3, alpha_vmd: float = 2000.0, tau: float = 0.0,
         n_iter: int = 500, tol: float = 1e-7, mirror: bool = True,
         seed: int | None = 0):
    """Successive Variational Mode Decomposition (see module docstring).

    Extracts modes one at a time: mode k is the single-mode VMD (K=1) solution
    on the residual left over from the previous step, then it is subtracted
    before extracting mode k+1.

    Returns
    -------
    modes  : (K, N) real array, ordered by ascending centre frequency
    omegas : (K,)   centre frequencies (cycles/sample)
    info   : dict with per-step iteration counts and the final reconstruction error
    """
    f = np.asarray(f, dtype=float).ravel()
    residual = f.copy()
    modes = np.zeros((K, f.size))
    omegas = np.zeros(K)
    iters = np.zeros(K, dtype=int)

    for k in range(K):
        m, w, step_info = vmd(residual, K=1, alpha_vmd=alpha_vmd, tau=tau,
                              n_iter=n_iter, tol=tol, mirror=mirror, seed=seed)
        modes[k] = m[0]
        omegas[k] = w[0]
        iters[k] = step_info["iterations"]
        residual = residual - m[0]

    order = np.argsort(omegas)
    modes, omegas, iters = modes[order], omegas[order], iters[order]
    info = {
        "iterations": iters,                     # per-mode iteration count
        "residual_energy": float(np.sum(residual ** 2)),
        "recon_rel_error": float(np.linalg.norm(f - modes.sum(0)) /
                                 (np.linalg.norm(f) + 1e-30)),
    }
    return modes, omegas, info
