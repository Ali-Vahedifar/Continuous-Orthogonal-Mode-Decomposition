"""
Prediction metrics.

IMPORTANT -- read before comparing numbers with the paper
---------------------------------------------------------
The paper reports "Accuracy (%)" for a regression task but does not define the
formula.  Several conventions are in circulation in the haptic-prediction
literature and they do *not* agree, so this file implements the common ones
explicitly and forces the choice to be made in the config instead of hiding it:

    "nrmse"  100 * (1 - ||y_hat - y||_2 / ||y||_2)
             normalised by the signal energy.  This is the default here.

    "fit"    100 * (1 - ||y_hat - y||_2 / ||y - mean(y)||_2)
             the "fit percentage" of MATLAB's `compare`/`goodnessOfFit`;
             identical to nrmse for zero-mean signals, harsher otherwise.

    "smape"  100 * (1 - mean( |y_hat - y| / ((|y|+|y_hat|)/2 + eps) ))

    "r2"     100 * R^2 (coefficient of determination), clipped at 0.

Whichever you pick, use the same one everywhere; the ranking of methods is
stable across them but the absolute values are not.
"""

from __future__ import annotations

import numpy as np

__all__ = ["accuracy", "ACCURACY_MODES", "rmse", "mae"]

ACCURACY_MODES = ("nrmse", "fit", "smape", "r2")


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def accuracy(y_hat, y, mode: str = "nrmse", eps: float = 1e-12,
             axis: int | tuple | None = None) -> np.ndarray | float:
    """Accuracy in percent.  See the module docstring for the definitions."""
    y_hat, y = _to_np(y_hat), _to_np(y)
    if mode == "nrmse":
        num = np.sqrt(np.sum((y_hat - y) ** 2, axis=axis))
        den = np.sqrt(np.sum(y ** 2, axis=axis)) + eps
        val = 100.0 * (1.0 - num / den)
    elif mode == "fit":
        m = np.mean(y, axis=axis, keepdims=True)
        num = np.sqrt(np.sum((y_hat - y) ** 2, axis=axis))
        den = np.sqrt(np.sum((y - m) ** 2, axis=axis)) + eps
        val = 100.0 * (1.0 - num / den)
    elif mode == "smape":
        d = (np.abs(y) + np.abs(y_hat)) / 2.0 + eps
        val = 100.0 * (1.0 - np.mean(np.abs(y_hat - y) / d, axis=axis))
    elif mode == "r2":
        m = np.mean(y, axis=axis, keepdims=True)
        ss_res = np.sum((y - y_hat) ** 2, axis=axis)
        ss_tot = np.sum((y - m) ** 2, axis=axis) + eps
        val = 100.0 * (1.0 - ss_res / ss_tot)
    else:
        raise ValueError(f"unknown accuracy mode '{mode}', pick one of {ACCURACY_MODES}")
    return np.clip(val, 0.0, 100.0)


def rmse(y_hat, y, axis=None):
    y_hat, y = _to_np(y_hat), _to_np(y)
    return np.sqrt(np.mean((y_hat - y) ** 2, axis=axis))


def mae(y_hat, y, axis=None):
    y_hat, y = _to_np(y_hat), _to_np(y)
    return np.mean(np.abs(y_hat - y), axis=axis)
