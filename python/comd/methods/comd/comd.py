"""
Continuous-Orthogonal Mode Decomposition (C-OMD).

Everything in this file is a direct, literal transcription of the equations in
the paper.  Equation numbers in the comments refer to the LaTeX source:

    (8)  augmented Lagrangian
    (11) mode update (Wiener filter)     m_hat_k = A / (1 + 2(w - w_k)^2 / alpha)
    (12) numerator A
    (13) center-frequency update
    (14) lambda dual ascent
    (15) gamma  dual ascent
    (16) Gram matrix
    (17) Newton-Schulz normalisation / coefficient / function update
    (20) per-frequency orthogonalisation  v_perp = P v

Any place where the paper is silent about a numerical detail is marked with
`IMPLEMENTATION CHOICE` and is exposed as an explicit option, so nothing is
silently invented.

Only NumPy is required here; the decomposition is deliberately framework-free
so the same modes can be fed to the PyTorch model or exported to MATLAB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = ["COMDConfig", "comd", "newton_schulz_orthogonalize", "gram_matrix"]


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class COMDConfig:
    """Hyper-parameters of C-OMD.

    Note on ``alpha``
    -----------------
    In the paper the quadratic penalty multiplies the *reconstruction* term,
    ``alpha * ||f - sum_k m_k||^2`` (Eq. 8), and the bandwidth term has unit
    weight.  Classical VMD uses the opposite convention (``alpha`` weights the
    bandwidth).  The two are reciprocal:

        alpha_paper = 1 / alpha_vmd

    so the familiar VMD setting ``alpha_vmd = 2000`` corresponds to
    ``alpha = 5e-4`` here.  Use :meth:`COMDConfig.from_vmd_alpha` if you prefer
    to think in VMD units.
    """

    K: int = 3                      # number of modes
    alpha: float = 5e-4             # reconstruction fidelity weight  (Eq. 8)
    beta: float | None = None       # orthogonality penalty (Eq. 8); None -> beta = alpha
    tau_lambda: float = 0.0         # dual step size for lambda       (Eq. 14)
    tau_gamma: float = 0.0          # dual step size for Gamma        (Eq. 15)
    n_iter: int = 500               # max ADMM iterations
    tol: float = 1e-7               # relative change stopping rule
    ns_iters: int = 20              # Newton-Schulz steps per ADMM iteration
    ortho_every: int = 1            # run the projection every n-th iteration
    # "gauss_seidel" is the default because the fully parallel ("jacobi")
    # sweep only converges with damping when the modes overlap in frequency:
    # use update="jacobi", relax<=0.7 for the GPU-parallel variant.
    update: Literal["jacobi", "gauss_seidel"] = "gauss_seidel"
    relax: float = 1.0              # damping of the Wiener step, 1.0 = none
    init: Literal["uniform", "random", "zero", "logspace", "manual"] = "uniform"
    omega_init: tuple | None = None  # used when init == "manual" (cycles/sample)
    normalize: bool = True          # scale f to unit L2 norm internally
    dc: bool = False                # keep the first mode centred at DC
    mirror: bool = True             # mirror-extend the signal (as in VMD)
    preserve_energy: bool = True    # see `newton_schulz_orthogonalize`
    seed: int | None = 0

    @classmethod
    def from_vmd_alpha(cls, alpha_vmd: float, **kw) -> "COMDConfig":
        return cls(alpha=1.0 / float(alpha_vmd), **kw)


# --------------------------------------------------------------------------- #
#  Gram matrix (Eq. 16) and Newton-Schulz orthogonalisation (Eq. 17 / Eq. 20)
# --------------------------------------------------------------------------- #
def gram_matrix(V: np.ndarray, scale: float = 1.0) -> np.ndarray:
    r"""Gram matrix of a set of spectra.

    ``G_ij = \int \hat m_i(w) \overline{\hat m_j(w)} dw``  (Eq. 16, Parseval form)

    Parameters
    ----------
    V : (K, F) complex array
        One-sided spectra of the K modes, i.e. the vectors ``v(w)`` of Eq. (18)
        stacked over frequency.
    scale : float
        Quadrature weight ``dw``.  Only the relative scaling matters, because
        the Newton-Schulz step normalises by ``trace(G)`` anyway.
    """
    return (V @ V.conj().T) * scale


def newton_schulz_orthogonalize(
    V: np.ndarray,
    n_iters: int = 5,
    scale: float = 1.0,
    preserve_energy: bool = True,
    eps: float = 1e-30,
):
    r"""Per-frequency orthogonalisation, Eqs. (17) and (20).

    The paper builds one ``K x K`` matrix ``P`` from the frequency-integrated
    Gram matrix and applies it identically at every frequency
    (``v_perp(w) = P v(w)``, Eq. 20).  That is exactly what happens here: the
    Newton-Schulz coefficient matrices ``C^{(m)} = 3/2 I - 1/2 G^{(m)}`` are
    accumulated into ``P = C^{(M-1)} ... C^{(0)}`` and left-multiplied onto the
    whole spectral matrix, which is the Fourier-domain statement of the
    functional update ``m_k^{(m+1)} = sum_j C_kj m_j^{(m)}`` (Eq. 17c).

    Parameters
    ----------
    V : (K, F) complex array   -- one-sided spectra of the K modes.
    n_iters : int              -- number of Newton-Schulz steps.
    preserve_energy : bool
        ``False`` reproduces the equations verbatim: Eq. (17a) divides the modes
        by ``(sum_j ||m_j||^2)^(1/2)`` and the iteration drives ``G -> I``, so the
        returned modes are *orthonormal* and no longer sum to ``f``.
        ``True`` (default, IMPLEMENTATION CHOICE) applies the very same
        orthogonalising map but afterwards restores each mode's original
        L2 norm.  This keeps the projection on the same scale as the Wiener
        step, which is what makes the alternation of Sec. IV-D
        ("each step correcting the constraint violation introduced by the
        other") numerically stable.  It changes only the *scaling* of the modes,
        not the directions that are orthogonalised.

    Returns
    -------
    V_perp : (K, F) complex array
    P      : (K, K) complex array -- the accumulated transformation actually applied.
    """
    K = V.shape[0]
    G = gram_matrix(V, scale)
    d = np.sqrt(np.real(np.diag(G)))                       # per-mode L2 norms
    total = np.sqrt(np.real(np.trace(G)))                  # Eq. (17a)
    if total < np.sqrt(eps):
        return V.copy(), np.eye(K, dtype=V.dtype)

    P = np.eye(K, dtype=np.complex128) / total             # normalisation folded into P
    Vn = V / total
    for _ in range(int(n_iters)):
        Gm = gram_matrix(Vn, scale)                        # Eq. (17d)
        C = 1.5 * np.eye(K) - 0.5 * Gm                     # Eq. (17b)
        Vn = C @ Vn                                        # Eq. (17c)
        P = C @ P

    if preserve_energy:
        d_new = np.sqrt(np.real(np.diag(gram_matrix(Vn, scale))))
        s = np.where(d_new > np.sqrt(eps), d / np.maximum(d_new, np.sqrt(eps)), 1.0)
        Vn = Vn * s[:, None]
        P = s[:, None] * P
    return Vn, P


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _mirror(f: np.ndarray) -> np.ndarray:
    """Mirror extension used by VMD to suppress boundary effects."""
    N = f.size
    h = N // 2
    return np.concatenate((f[:h][::-1], f, f[N - h:][::-1]))


def _init_omegas(cfg: COMDConfig) -> np.ndarray:
    K = cfg.K
    if cfg.init == "manual":
        w = np.asarray(cfg.omega_init, dtype=float).copy()
        assert w.size == K, "omega_init must have K entries"
    elif cfg.init == "uniform":
        # same convention as the reference VMD implementation: 0, .., (K-1)/2K
        w = (0.5 / K) * np.arange(K)
    elif cfg.init == "logspace":
        w = np.exp(np.log(1 / 800.0) + (np.log(0.5) - np.log(1 / 800.0)) *
                   np.sort(np.random.default_rng(cfg.seed).random(K)))
    elif cfg.init == "random":
        rng = np.random.default_rng(cfg.seed)
        w = np.sort(0.5 * rng.random(K))
    else:
        w = np.zeros(K)
    if cfg.dc:
        w[0] = 0.0
    return w.astype(float)


def _spectra_to_signals(u_hat: np.ndarray, T: int) -> np.ndarray:
    """One-sided (analytic) spectra -> real time-domain modes."""
    K = u_hat.shape[0]
    full = np.zeros((K, T), dtype=complex)
    full[:, T // 2:] = u_hat[:, T // 2:]
    full[:, 1:T // 2 + 1] = np.conj(u_hat[:, T - 1:T // 2 - 1:-1])
    full[:, 0] = np.conj(full[:, -1])
    return np.real(np.fft.ifft(np.fft.ifftshift(full, axes=-1), axis=-1))


# --------------------------------------------------------------------------- #
#  C-OMD
# --------------------------------------------------------------------------- #
def comd(f: np.ndarray, cfg: COMDConfig | None = None, **kw):
    """Continuous-Orthogonal Mode Decomposition.

    Returns
    -------
    modes  : (K, N) real array, ordered by ascending centre frequency
    omegas : (K,)   centre frequencies in normalised units (cycles/sample;
             multiply by fs to get Hz)
    info   : dict with the iteration history and the final diagnostics
    """
    cfg = cfg or COMDConfig(**kw)
    f = np.asarray(f, dtype=float).ravel()
    N = f.size
    K, alpha = cfg.K, float(cfg.alpha)
    # The orthogonality penalty multiplies |G_ij|^2, i.e. squared *energies*,
    # while alpha multiplies a squared amplitude.  The two are only comparable
    # if the signal is on a fixed scale, hence `normalize` below, and if beta is
    # measured in the same units as alpha -- the default beta = alpha.
    beta = float(cfg.alpha if cfg.beta is None else cfg.beta)
    fscale = float(np.linalg.norm(f)) if cfg.normalize else 1.0
    if fscale <= 0:
        fscale = 1.0
    f = f / fscale
    x = _mirror(f) if cfg.mirror else f.copy()
    T = x.size

    # normalised frequency axis in [-0.5, 0.5)
    freqs = np.arange(1, T + 1) / T - 0.5 - 1.0 / T
    dw = 1.0 / T                                   # quadrature weight (Parseval)

    f_hat = np.fft.fftshift(np.fft.fft(x))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T // 2] = 0.0                      # analytic signal: kill w < 0

    u_hat = np.zeros((K, T), dtype=complex)
    lambda_hat = np.zeros(T, dtype=complex)
    Gamma = np.zeros((K, K), dtype=float)          # symmetric multipliers, Eq. (8)
    omega = _init_omegas(cfg)

    pos = slice(T // 2, T)                         # non-negative frequencies
    hist = {"omega": [], "res": [], "offdiag": []}

    for it in range(cfg.n_iter):
        u_prev = u_hat.copy()

        # ---- Eq. (16): Gram matrix of the current iterate -------------------
        G = gram_matrix(u_hat[:, pos], dw)

        # ---- Eq. (11) + (12): Wiener filter with orthogonality correction ---
        u_new = np.zeros_like(u_hat)
        if cfg.update == "jacobi":                 # fully parallel over k
            tot = u_hat.sum(axis=0)
            for k in range(K):
                corr = np.zeros(T, dtype=complex)
                for j in range(K):
                    if j == k:
                        continue
                    corr += (beta * G[k, j] + 0.5 * Gamma[k, j]) * u_hat[j]
                A = (f_hat_plus - (tot - u_hat[k])
                     + lambda_hat / (2.0 * alpha)
                     - corr / alpha)                                  # Eq. (12)
                u_new[k] = A / (1.0 + 2.0 * (freqs - omega[k]) ** 2 / alpha)  # Eq. (11)
        else:                                       # Gauss-Seidel (as in VMD code)
            for k in range(K):
                others = u_new[:k].sum(axis=0) + u_hat[k + 1:].sum(axis=0)
                corr = np.zeros(T, dtype=complex)
                for j in range(K):
                    if j == k:
                        continue
                    src = u_new[j] if j < k else u_hat[j]
                    corr += (beta * G[k, j] + 0.5 * Gamma[k, j]) * src
                A = (f_hat_plus - others
                     + lambda_hat / (2.0 * alpha)
                     - corr / alpha)
                u_new[k] = A / (1.0 + 2.0 * (freqs - omega[k]) ** 2 / alpha)

        if cfg.relax != 1.0:                        # optional damping
            u_new = (1.0 - cfg.relax) * u_hat + cfg.relax * u_new
        u_new[:, :T // 2] = 0.0                     # stay analytic

        # ---- Eq. (13): centre-frequency update ------------------------------
        for k in range(K):
            p = np.abs(u_new[k, pos]) ** 2
            s = p.sum()
            if s > 0 and not (cfg.dc and k == 0):
                omega[k] = float((freqs[pos] * p).sum() / s)
            elif cfg.dc and k == 0:
                omega[k] = 0.0

        # ---- Eq. (17)/(20): explicit orthogonal projection -------------------
        if cfg.ns_iters > 0 and (it % cfg.ortho_every == 0):
            Vp, _ = newton_schulz_orthogonalize(
                u_new[:, pos], cfg.ns_iters, dw, cfg.preserve_energy)
            u_new[:, pos] = Vp

        # ---- Eqs. (14)/(15): dual ascent ------------------------------------
        # IMPLEMENTATION CHOICE: the multipliers are updated from the current
        # iterate, i.e. *after* the projection, because that is the state the
        # next Wiener step will see.  Set tau_* = 0 for the noise-tolerant
        # variant (the usual VMD default).
        if cfg.tau_lambda:
            resid = f_hat_plus - u_new.sum(axis=0)
            lambda_hat = lambda_hat + cfg.tau_lambda * resid
            lambda_hat[:T // 2] = 0.0
        if cfg.tau_gamma:
            Gn = np.real(gram_matrix(u_new[:, pos], dw))
            off = Gn - np.diag(np.diag(Gn))
            Gamma = Gamma + cfg.tau_gamma * off

        u_hat = u_new

        # ---- convergence ----------------------------------------------------
        num = np.sum(np.abs(u_hat - u_prev) ** 2)
        den = np.sum(np.abs(u_prev) ** 2) + 1e-30
        res = float(num / den)
        Gd = gram_matrix(u_hat[:, pos], dw)
        offd = float(np.sum(np.abs(Gd - np.diag(np.diag(Gd))) ** 2))
        hist["omega"].append(omega.copy())
        hist["res"].append(res)
        hist["offdiag"].append(offd)
        if it > 0 and res < cfg.tol:
            break

    order = np.argsort(omega)
    omega = omega[order]
    u_hat = u_hat[order]

    modes = _spectra_to_signals(u_hat, T)
    if cfg.mirror:
        h = N // 2
        modes = modes[:, h:h + N]

    recon = modes.sum(axis=0)
    rec_err = float(np.linalg.norm(recon - f) / (np.linalg.norm(f) + 1e-30))
    modes = modes * fscale                      # undo the internal normalisation
    f = f * fscale
    info = {
        "iterations": it + 1,
        "residual": hist["res"][-1],
        "omega_history": np.array(hist["omega"]),
        "offdiag_history": np.array(hist["offdiag"]),
        "recon_rel_error": rec_err,
        "gram_time": modes @ modes.T,
        "u_hat": u_hat,
        "freqs": freqs,
    }
    return modes, omega, info
