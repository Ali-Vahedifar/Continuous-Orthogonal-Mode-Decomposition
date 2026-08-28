"""
Aggregator for the decomposition methods, each implemented in its own folder
under `methods/`: `methods/comd/`, `methods/vmd/`, `methods/svmd/`,
`methods/none/`. This module just re-exports them so the rest of the
codebase (`data.py`, `infer.py`, the scripts, the tests) can keep doing
`from .decomposition import COMDConfig, comd, vmd, ...` without caring how
the methods are laid out on disk.
"""

from __future__ import annotations

from .methods.comd.comd import COMDConfig, comd, gram_matrix, newton_schulz_orthogonalize
from .methods.vmd.vmd import vmd
from .methods.svmd.svmd import svmd
from .methods.none.none import none_decompose

__all__ = ["COMDConfig", "comd", "vmd", "svmd", "none_decompose",
           "newton_schulz_orthogonalize", "gram_matrix"]
