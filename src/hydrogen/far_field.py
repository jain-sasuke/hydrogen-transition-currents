from __future__ import annotations

import numpy as np


def observation_unit_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Returns array shape (ntheta, nphi, 3)
    """
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    nx = np.sin(tt) * np.cos(pp)
    ny = np.sin(tt) * np.sin(pp)
    nz = np.cos(tt)
    return np.stack([nx, ny, nz], axis=-1)


def dipole_intensity_map(dipole: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    I(theta, phi) ∝ | n x (n x d) |^2
    """
    nvec = observation_unit_vectors(theta, phi)  # (ntheta, nphi, 3)
    d = np.asarray(dipole, dtype=np.complex128)[None, None, :]  # (1,1,3)

    ndotd = np.sum(nvec * d, axis=-1, keepdims=True)
    transverse = d - nvec * ndotd
    intensity = np.sum(np.abs(transverse) ** 2, axis=-1)
    return intensity.real


def normalize_by_peak(arr: np.ndarray) -> np.ndarray:
    peak = float(np.max(arr))
    if peak <= 0.0:
        return arr.copy()
    return arr / peak