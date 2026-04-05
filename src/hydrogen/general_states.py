from __future__ import annotations

import re
from math import factorial
from typing import Union

import numpy as np
from scipy.special import eval_genlaguerre, lpmv

PI = np.pi

StateSpec = Union[str, tuple[int, int, int]]


def _validate_nlm(n: int, l: int, m: int) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if l < 0 or l >= n:
        raise ValueError("Require 0 <= l <= n-1")
    if abs(m) > l:
        raise ValueError("Require |m| <= l")


def cartesian_to_spherical(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.sqrt(X**2 + Y**2 + Z**2)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(rho, Z)  # [0, pi]
    phi = np.mod(np.arctan2(Y, X), 2.0 * PI)
    return r, theta, phi


def radial_hydrogen(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """
    Hydrogen radial wavefunction R_{nl}(r) in atomic units (a0 = 1).

    Convention:
        psi_{nlm}(r,theta,phi) = R_{nl}(r) Y_{lm}(theta,phi)
    """
    _validate_nlm(n, l, 0)

    rho = 2.0 * r / float(n)

    pref = np.sqrt(
        (2.0 / n) ** 3
        * factorial(n - l - 1)
        / (2.0 * n * factorial(n + l))
    )

    lag = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return pref * np.exp(-rho / 2.0) * rho**l * lag


def complex_spherical_harmonic(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """
    Complex spherical harmonic Y_{lm}(theta,phi), including Condon-Shortley phase.
    Implemented directly via associated Legendre functions for robustness.
    """
    if l < 0 or abs(m) > l:
        raise ValueError("Invalid (l,m)")

    abs_m = abs(m)
    norm = np.sqrt(
        (2.0 * l + 1.0)
        / (4.0 * PI)
        * factorial(l - abs_m)
        / factorial(l + abs_m)
    )

    P = lpmv(abs_m, l, np.cos(theta))
    Y_abs = norm * P * np.exp(1j * abs_m * phi)

    if m >= 0:
        return Y_abs

    # Y_{l,-m} = (-1)^m Y_{lm}^*
    return ((-1) ** abs_m) * np.conjugate(Y_abs)


def psi_nlm_spherical(
    n: int, l: int, m: int, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    _validate_nlm(n, l, m)
    return radial_hydrogen(n, l, r) * complex_spherical_harmonic(l, m, theta, phi)


def psi_nlm_xyz(n: int, l: int, m: int, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    r, theta, phi = cartesian_to_spherical(X, Y, Z)
    return psi_nlm_spherical(n, l, m, r, theta, phi)


def real_p_orbital_xyz(
    n: int, label: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """
    Real p orbitals from complex m = +/-1,0 combinations.

    Supported labels:
        'px', 'py', 'pz'
    """
    label = label.lower().strip()
    if label not in {"px", "py", "pz"}:
        raise ValueError("Supported real p labels are: px, py, pz")

    if n < 2:
        raise ValueError("p orbitals require n >= 2")

    if label == "pz":
        return np.real_if_close(psi_nlm_xyz(n, 1, 0, X, Y, Z))

    psi_p1 = psi_nlm_xyz(n, 1, +1, X, Y, Z)
    psi_m1 = psi_nlm_xyz(n, 1, -1, X, Y, Z)

    if label == "px":
        # proportional to sin(theta) cos(phi)
        psi = (psi_m1 - psi_p1) / np.sqrt(2.0)
        return np.real_if_close(psi)

    # label == "py"
    # proportional to sin(theta) sin(phi)
    psi = 1j * (psi_m1 + psi_p1) / np.sqrt(2.0)
    return np.real_if_close(psi)


_STATE_RE = re.compile(r"^\s*(\d+)\s*(s|px|py|pz)\s*$", re.IGNORECASE)


def hydrogen_state_xyz(
    state: StateSpec,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    General hydrogen state evaluator.

    Supported inputs:
    1) tuple (n,l,m): returns complex spherical-basis state psi_{nlm}
    2) strings:
         '1s', '2s', '2px', '2py', '2pz', ...
       These return real orbital forms for s/p labels.
    """
    if isinstance(state, tuple):
        if len(state) != 3:
            raise ValueError("Tuple state must be (n,l,m)")
        n, l, m = state
        return psi_nlm_xyz(int(n), int(l), int(m), X, Y, Z)

    if isinstance(state, str):
        match = _STATE_RE.match(state)
        if match is None:
            raise ValueError(
                "String states must look like '1s', '2s', '2px', '2py', '2pz'"
            )
        n = int(match.group(1))
        label = match.group(2).lower()

        if label == "s":
            return np.real_if_close(psi_nlm_xyz(n, 0, 0, X, Y, Z))

        return real_p_orbital_xyz(n, label, X, Y, Z)

    raise TypeError("state must be either a tuple (n,l,m) or a string like '2pz'")


def gradient_hydrogen_state_xyz(
    state: StateSpec,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    h: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerical Cartesian gradient of a general hydrogen state.

    This is intentionally generic and correct.
    It is not the final optimized version for large scans.
    """
    psi_xp = hydrogen_state_xyz(state, X + h, Y, Z)
    psi_xm = hydrogen_state_xyz(state, X - h, Y, Z)
    dpsi_dx = (psi_xp - psi_xm) / (2.0 * h)

    psi_yp = hydrogen_state_xyz(state, X, Y + h, Z)
    psi_ym = hydrogen_state_xyz(state, X, Y - h, Z)
    dpsi_dy = (psi_yp - psi_ym) / (2.0 * h)

    psi_zp = hydrogen_state_xyz(state, X, Y, Z + h)
    psi_zm = hydrogen_state_xyz(state, X, Y, Z - h)
    dpsi_dz = (psi_zp - psi_zm) / (2.0 * h)

    return dpsi_dx, dpsi_dy, dpsi_dz