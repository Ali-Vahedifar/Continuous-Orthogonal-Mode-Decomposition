"""
No-decomposition baseline.

Tests whether decomposing into modes (comd/vmd/svmd) helps at all, versus
feeding the network the raw signal directly. There is nothing to decompose
into multiple channels here by construction, so K is forced to 1.
"""

from __future__ import annotations

import numpy as np

__all__ = ["none_decompose"]


def none_decompose(f, K: int = 1):
    """Return the raw signal, unchanged, as a single 'mode'.

    Same ``(modes, omegas, info)`` return shape as :func:`comd`/:func:`vmd`/
    :func:`svmd` so it is a drop-in replacement wherever those are used.
    """
    if K != 1:
        raise ValueError("the no-decomposition baseline only makes sense with K=1")
    f = np.asarray(f, dtype=float).ravel()
    modes = f[None, :].copy()
    omegas = np.zeros(1)
    info = {"iterations": 0, "recon_rel_error": 0.0}
    return modes, omegas, info
