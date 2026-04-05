from __future__ import annotations

import numpy as np


def volume_element(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    return dx * dy * dz


def integrate_scalar(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> complex:
    dv = volume_element(x, y, z)
    return np.sum(field) * dv


def overlap_element(
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> complex:
    return integrate_scalar(np.conjugate(psi_a) * psi_b, x, y, z)


def norm_on_grid(
    psi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    val = overlap_element(psi, psi, x, y, z)
    return float(np.real_if_close(val))


def dipole_matrix_element(
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    Returns the length-gauge coordinate matrix element:
        <a| r |b> = (<a|x|b>, <a|y|b>, <a|z|b>)
    """
    overlap = np.conjugate(psi_a) * psi_b
    dx = integrate_scalar(overlap * X, x, y, z)
    dy = integrate_scalar(overlap * Y, x, y, z)
    dz = integrate_scalar(overlap * Z, x, y, z)
    return np.array([dx, dy, dz], dtype=np.complex128)