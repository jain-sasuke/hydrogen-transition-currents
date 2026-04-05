from __future__ import annotations

import numpy as np

from .constants import PI


def cartesian_grid(xmax: float = 14.0, n: int = 101):
    """
    Uniform cubic grid in atomic units.
    Returns x, y, z 1D arrays and X, Y, Z meshgrids.
    """
    x = np.linspace(-xmax, xmax, n)
    y = np.linspace(-xmax, xmax, n)
    z = np.linspace(-xmax, xmax, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


def radius_xyz(X, Y, Z):
    return np.sqrt(X**2 + Y**2 + Z**2)


def psi_100_xyz(X, Y, Z):
    """
    Hydrogen 1s orbital in Cartesian coordinates, atomic units:
        psi_100 = exp(-r) / sqrt(pi)
    """
    r = radius_xyz(X, Y, Z)
    return np.exp(-r) / np.sqrt(PI)


def psi_210_xyz(X, Y, Z):
    """
    Hydrogen 2p_z orbital in Cartesian coordinates, atomic units:
        psi_210 = z exp(-r/2) / (4 sqrt(2 pi))
    """
    r = radius_xyz(X, Y, Z)
    return Z * np.exp(-r / 2.0) / (4.0 * np.sqrt(2.0 * PI))


def grad_psi_100_xyz(X, Y, Z):
    """
    Analytic gradient of 1s orbital.
    """
    r = radius_xyz(X, Y, Z)
    psi = psi_100_xyz(X, Y, Z)

    gx = np.zeros_like(X, dtype=np.float64)
    gy = np.zeros_like(Y, dtype=np.float64)
    gz = np.zeros_like(Z, dtype=np.float64)

    mask = r > 1e-12
    gx[mask] = -psi[mask] * X[mask] / r[mask]
    gy[mask] = -psi[mask] * Y[mask] / r[mask]
    gz[mask] = -psi[mask] * Z[mask] / r[mask]

    return gx, gy, gz


def grad_psi_210_xyz(X, Y, Z):
    """
    Analytic gradient of 2p_z orbital:
        psi = C z exp(-r/2),  C = 1/(4 sqrt(2 pi))
    """
    C = 1.0 / (4.0 * np.sqrt(2.0 * PI))
    r = radius_xyz(X, Y, Z)
    expfac = np.exp(-r / 2.0)

    gx = np.zeros_like(X, dtype=np.float64)
    gy = np.zeros_like(Y, dtype=np.float64)
    gz = C * expfac  # r=0 limit handled correctly here

    mask = r > 1e-12
    gx[mask] = -C * Z[mask] * expfac[mask] * X[mask] / (2.0 * r[mask])
    gy[mask] = -C * Z[mask] * expfac[mask] * Y[mask] / (2.0 * r[mask])
    gz[mask] = C * expfac[mask] * (1.0 - (Z[mask] ** 2) / (2.0 * r[mask]))

    return gx, gy, gz


def equal_superposition_phase(omega: float, t: float) -> float:
    return omega * t


def probability_density_equal_superposition(psi_a, psi_b, phase):
    """
    |Psi|^2 for Psi = (psi_a + e^{-i phase} psi_b) / sqrt(2),
    assuming real spatial orbitals.
    """
    return 0.5 * (psi_a**2 + psi_b**2 + 2.0 * psi_a * psi_b * np.cos(phase))


def charge_density_equal_superposition(psi_a, psi_b, phase):
    """
    Charge density for electron charge q = -1 (atomic units):
        rho = - |Psi|^2
    """
    return -probability_density_equal_superposition(psi_a, psi_b, phase)


def oscillating_charge_density_equal_superposition(psi_a, psi_b, phase):
    """
    Only the oscillating interference piece of charge density:
        rho_osc = - psi_a psi_b cos(phase)
    """
    return -psi_a * psi_b * np.cos(phase)


def time_derivative_charge_density_equal_superposition(psi_a, psi_b, omega, phase):
    """
    d rho / dt for the full charge density:
        drho/dt = + omega psi_a psi_b sin(phase)
    """
    return omega * psi_a * psi_b * np.sin(phase)


def electric_current_density_equal_superposition(psi_a, psi_b, grad_a, grad_b, phase):
    """
    Electric current density for equal-amplitude superposition of real orbitals,
    in atomic units, with electron charge included.

    For real orbitals:
        J = + 1/2 sin(phase) [ psi_a grad psi_b - psi_b grad psi_a ]
    """
    gax, gay, gaz = grad_a
    gbx, gby, gbz = grad_b

    pref = 0.5 * np.sin(phase)

    Jx = pref * (psi_a * gbx - psi_b * gax)
    Jy = pref * (psi_a * gby - psi_b * gay)
    Jz = pref * (psi_a * gbz - psi_b * gaz)

    return Jx, Jy, Jz


def dipole_moment_from_charge_density(rho, X, Y, Z, x, y, z):
    """
    p = ∫ r rho d^3r
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    px = np.sum(X * rho) * dx * dy * dz
    py = np.sum(Y * rho) * dx * dy * dz
    pz = np.sum(Z * rho) * dx * dy * dz

    return np.array([px, py, pz], dtype=np.float64)