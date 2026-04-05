from __future__ import annotations

from typing import Callable

import numpy as np

from .constants import A0, E_CHARGE


Wavefunction = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


def spherical_grids(
    r_max: float = 30.0,
    nr: int = 600,
    ntheta: int = 240,
    nphi: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(0.0, r_max, nr)
    theta = np.linspace(0.0, np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    return r, theta, phi


def _trapz3(integrand: np.ndarray, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> complex:
    """
    Integrate array of shape (nr, ntheta, nphi) over (r, theta, phi).
    """
    out_phi = np.trapezoid(integrand, phi, axis=2)
    out_theta = np.trapezoid(out_phi, theta, axis=1)
    out_r = np.trapezoid(out_theta, r, axis=0)
    return complex(out_r)


def dipole_vector(
    psi_a: Wavefunction,
    psi_b: Wavefunction,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    a0: float = A0,
    e_charge: float = E_CHARGE,
) -> np.ndarray:
    """
    Compute dipole matrix element
        d_ab = -e ∫ psi_a^*(r) * r_vec * psi_b(r) d^3r
    in spherical coordinates.
    """
    rr, tt, pp = np.meshgrid(r, theta, phi, indexing="ij")

    psi_a_vals = psi_a(rr, tt, pp, a0=a0)
    psi_b_vals = psi_b(rr, tt, pp, a0=a0)

    x = rr * np.sin(tt) * np.cos(pp)
    y = rr * np.sin(tt) * np.sin(pp)
    z = rr * np.cos(tt)

    jac = rr**2 * np.sin(tt)
    overlap = np.conjugate(psi_a_vals) * psi_b_vals * jac

    dx = -e_charge * _trapz3(overlap * x, r, theta, phi)
    dy = -e_charge * _trapz3(overlap * y, r, theta, phi)
    dz = -e_charge * _trapz3(overlap * z, r, theta, phi)

    return np.array([dx, dy, dz], dtype=np.complex128)