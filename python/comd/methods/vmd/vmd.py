"""
VMD baseline (Dragomiretskiy & Zosso, 2014) -- used for the ablations.

Same machinery as C-OMD (`comd.methods.comd.comd`) with beta = 0 and no
Newton-Schulz projection step: this is the C-OMD ablation "minus the
orthogonality constraint".
"""

from __future__ import annotations

from ..comd.comd import COMDConfig, comd

__all__ = ["vmd"]


def vmd(f, K: int = 3, alpha_vmd: float = 2000.0, tau: float = 0.0,
        n_iter: int = 500, tol: float = 1e-7, dc: bool = False,
        init: str = "uniform", mirror: bool = True, seed: int | None = 0):
    """Classical VMD.

    Identical machinery to :func:`comd` with ``beta = 0`` and no projection
    step, written out separately so the baseline cannot accidentally inherit
    any C-OMD behaviour.  ``alpha_vmd`` follows the original convention
    (weight on the bandwidth term).
    """
    cfg = COMDConfig(K=K, alpha=1.0 / float(alpha_vmd), beta=0.0, normalize=True,
                     tau_lambda=tau, tau_gamma=0.0, n_iter=n_iter, tol=tol,
                     ns_iters=0, dc=dc, init=init, mirror=mirror, seed=seed,
                     update="gauss_seidel")
    return comd(f, cfg)
