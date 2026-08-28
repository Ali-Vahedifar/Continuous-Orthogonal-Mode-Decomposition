"""
Verification tests for C-OMD.  Run with `pytest tests/` or directly:
`python tests/test_decomposition.py`.

These check the properties the paper actually claims, not just that the code
runs: exact reconstruction, orthogonality strictly better than VMD when the
modes overlap in frequency, Newton-Schulz convergence, and Proposition 1
(band-limitedness preservation).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from comd.decomposition import (COMDConfig, comd, vmd, gram_matrix,
                                newton_schulz_orthogonalize)

FS = 1000.0
N = 1000
T_AX = np.arange(N) / FS


def separated_signal():
    return (np.cos(2 * np.pi * 2 * T_AX) + 0.25 * np.cos(2 * np.pi * 24 * T_AX)
            + (1 / 16) * np.cos(2 * np.pi * 288 * T_AX))


def overlapping_signal():
    """Two components close enough that their VMD bands overlap."""
    return (np.cos(2 * np.pi * 8 * T_AX) + 0.6 * np.cos(2 * np.pi * 40 * T_AX)
            + 0.35 * np.cos(2 * np.pi * 52 * T_AX))


def max_offdiag_correlation(modes):
    G = modes @ modes.T
    d = np.sqrt(np.diag(G))
    C = G / np.outer(d, d)
    off = np.abs(C - np.eye(len(d)))
    np.fill_diagonal(off, 0.0)
    return off.max()


def test_reconstruction():
    for f in (separated_signal(), overlapping_signal()):
        modes, _, info = comd(f, COMDConfig(K=3))
        assert info["recon_rel_error"] < 1e-2, info["recon_rel_error"]
        assert np.allclose(modes.sum(0), f, atol=0.05 * np.abs(f).max())


def test_centre_frequencies():
    modes, omega, _ = comd(separated_signal(), COMDConfig(K=3))
    hz = omega * FS
    assert np.allclose(hz, [2, 24, 288], rtol=0.02), hz


def test_orthogonality_beats_vmd_on_overlapping_modes():
    """The paper's core claim: VMD is only orthogonal as a side effect of
    disjoint spectral support, C-OMD enforces it explicitly."""
    f = overlapping_signal()
    oi = (8 / FS, 40 / FS, 52 / FS)          # same init for both, fair comparison
    base = dict(K=3, init="manual", omega_init=oi, update="gauss_seidel")
    m_vmd, _, i_vmd = comd(f, COMDConfig(alpha=1 / 2000., beta=0.0, ns_iters=0, **base))
    m_omd, _, i_omd = comd(f, COMDConfig(ns_iters=20, **base))

    c_vmd = max_offdiag_correlation(m_vmd)
    c_omd = max_offdiag_correlation(m_omd)
    assert c_omd < c_vmd / 100, (c_vmd, c_omd)
    # and it does not cost reconstruction quality
    assert i_omd["recon_rel_error"] <= i_vmd["recon_rel_error"] * 1.05


def test_newton_schulz_orthonormalises():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((4, 200)) + 1j * rng.standard_normal((4, 200))
    V[1] += 0.9 * V[0]                        # strongly correlated on purpose
    Vp, P = newton_schulz_orthogonalize(V, n_iters=60, scale=1.0,
                                        preserve_energy=False)
    G = gram_matrix(Vp)
    assert np.allclose(G, np.eye(4), atol=1e-6), np.abs(G - np.eye(4)).max()


def test_newton_schulz_is_a_linear_map():
    """Eq. (20): one K x K matrix P applied identically at every frequency."""
    rng = np.random.default_rng(1)
    V = rng.standard_normal((3, 64)) + 1j * rng.standard_normal((3, 64))
    Vp, P = newton_schulz_orthogonalize(V, n_iters=10, scale=1.0)
    assert np.allclose(Vp, P @ V, atol=1e-10)


def test_band_limitedness_preserved():
    """Proposition 1: the projection redistributes energy between modes but
    creates no spectral content where the mode set had none."""
    rng = np.random.default_rng(2)
    F = 256
    V = rng.standard_normal((3, F)) + 1j * rng.standard_normal((3, F))
    dead = np.zeros(F, dtype=bool)
    dead[100:140] = True                      # a band where every mode is silent
    V[:, dead] = 0
    Vp, _ = newton_schulz_orthogonalize(V, n_iters=20, scale=1.0)
    assert np.abs(Vp[:, dead]).max() == 0.0


def test_modes_stay_spectrally_compact():
    """After orthogonalisation each mode still keeps its energy around its own
    centre frequency (the Wiener band), i.e. the projection does not smear the
    modes across the spectrum."""
    f = overlapping_signal()
    modes, omega, _ = comd(f, COMDConfig(K=3, ns_iters=20))
    spec = np.abs(np.fft.rfft(modes, axis=-1)) ** 2
    fax = np.fft.rfftfreq(modes.shape[1], d=1 / FS)
    for k in range(3):
        band = np.abs(fax - omega[k] * FS) <= 10.0        # +/- 10 Hz
        share = spec[k, band].sum() / spec[k].sum()
        assert share > 0.95, (k, share)


def test_vmd_matches_known_solution():
    """Sanity check of the baseline against the textbook VMD test signal."""
    modes, omega, info = vmd(separated_signal(), K=3, alpha_vmd=2000.0)
    assert np.allclose(omega * FS, [2, 24, 288], rtol=0.02)
    assert info["recon_rel_error"] < 1e-2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
