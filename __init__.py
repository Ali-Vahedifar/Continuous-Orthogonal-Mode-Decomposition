"""C-OMD + MDA: orthogonal mode decomposition and bilateral haptic prediction."""

from .decomposition import (COMDConfig, comd, vmd, newton_schulz_orthogonalize,
                            gram_matrix)
from .metrics import accuracy, rmse, mae

__all__ = ["COMDConfig", "comd", "vmd", "newton_schulz_orthogonalize",
           "gram_matrix", "accuracy", "rmse", "mae"]
__version__ = "0.1.0"


def __getattr__(name):
    """Torch-dependent parts are imported lazily so the decomposition can be
    used (and the MATLAB export prepared) without PyTorch installed."""
    if name in ("MDA", "MDAConfig", "build_model"):
        from . import models
        return getattr(models, name)
    if name in ("MDALoss", "LossWeights"):
        from . import losses
        return getattr(losses, name)
    if name in ("fit", "TrainConfig"):
        from . import train
        return getattr(train, name)
    raise AttributeError(name)
