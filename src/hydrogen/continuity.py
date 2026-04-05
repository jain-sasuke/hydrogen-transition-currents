from __future__ import annotations

import numpy as np


def divergence_cartesian(Jx, Jy, Jz, x, y, z):
    """
    Compute div J on a uniform Cartesian grid.
    """
    dJx_dx = np.gradient(Jx, x, axis=0, edge_order=2)
    dJy_dy = np.gradient(Jy, y, axis=1, edge_order=2)
    dJz_dz = np.gradient(Jz, z, axis=2, edge_order=2)
    return dJx_dx + dJy_dy + dJz_dz


def continuity_residual(drho_dt, divJ, eps: float = 1e-12):
    """
    Relative continuity residual:
        |drho_dt + divJ| / max(|drho_dt| + |divJ|, eps)
    """
    denom = np.maximum(np.abs(drho_dt) + np.abs(divJ), eps)
    return np.abs(drho_dt + divJ) / denom