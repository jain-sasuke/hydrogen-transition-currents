from __future__ import annotations

import numpy as np

from .general_states import hydrogen_state_xyz, gradient_hydrogen_state_xyz


def hydrogen_energy_au(n: int) -> float:
    return -0.5 / (n * n)


def parse_n_from_state(state: str) -> int:
    s = state.strip().lower()
    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        raise ValueError(f"Could not parse n from state '{state}'")
    return int("".join(digits))


def coherent_state_three_component(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    ground_state: str = "1s",
    excited_x: str = "2px",
    excited_z: str = "2pz",
    a: float = 1.0 / np.sqrt(2.0),
    b_x: float = 0.5,
    b_z: float = 0.5,
    delta: float = 0.0,
    t: float = 0.0,
):
    """
    Build the coherent state

        Psi(r,t) = a psi_g e^{-i E_g t}
                 + b_x psi_x e^{-i E_e t}
                 + b_z e^{i delta} psi_z e^{-i E_e t}

    where psi_g, psi_x, psi_z are spatial hydrogen orbitals.
    """
    psi_g = hydrogen_state_xyz(ground_state, X, Y, Z)
    psi_x = hydrogen_state_xyz(excited_x, X, Y, Z)
    psi_z = hydrogen_state_xyz(excited_z, X, Y, Z)

    E_g = hydrogen_energy_au(parse_n_from_state(ground_state))
    E_e_x = hydrogen_energy_au(parse_n_from_state(excited_x))
    E_e_z = hydrogen_energy_au(parse_n_from_state(excited_z))

    if abs(E_e_x - E_e_z) > 1.0e-15:
        raise ValueError("excited_x and excited_z must be degenerate for this task")

    phase_g = np.exp(-1j * E_g * t)
    phase_e = np.exp(-1j * E_e_x * t)

    psi = (
        a * psi_g * phase_g
        + b_x * psi_x * phase_e
        + b_z * np.exp(1j * delta) * psi_z * phase_e
    )
    return psi


def coherent_density(
    psi: np.ndarray,
):
    return np.abs(psi) ** 2


def coherent_current_density(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    ground_state: str = "1s",
    excited_x: str = "2px",
    excited_z: str = "2pz",
    a: float = 1.0 / np.sqrt(2.0),
    b_x: float = 0.5,
    b_z: float = 0.5,
    delta: float = 0.0,
    t: float = 0.0,
    h: float = 1.0e-4,
):
    """
    Full probability current density for the coherent 1s + 2px + 2pz state:

        J = Im( psi* grad psi )

    in atomic units with m = hbar = 1.
    """
    psi_g = hydrogen_state_xyz(ground_state, X, Y, Z)
    psi_x = hydrogen_state_xyz(excited_x, X, Y, Z)
    psi_z = hydrogen_state_xyz(excited_z, X, Y, Z)

    dg_dx, dg_dy, dg_dz = gradient_hydrogen_state_xyz(ground_state, X, Y, Z, h=h)
    dx_dx, dx_dy, dx_dz = gradient_hydrogen_state_xyz(excited_x, X, Y, Z, h=h)
    dz_dx, dz_dy, dz_dz = gradient_hydrogen_state_xyz(excited_z, X, Y, Z, h=h)

    E_g = hydrogen_energy_au(parse_n_from_state(ground_state))
    E_e_x = hydrogen_energy_au(parse_n_from_state(excited_x))
    E_e_z = hydrogen_energy_au(parse_n_from_state(excited_z))

    if abs(E_e_x - E_e_z) > 1.0e-15:
        raise ValueError("excited_x and excited_z must be degenerate for this task")

    phase_g = np.exp(-1j * E_g * t)
    phase_e = np.exp(-1j * E_e_x * t)

    psi = (
        a * psi_g * phase_g
        + b_x * psi_x * phase_e
        + b_z * np.exp(1j * delta) * psi_z * phase_e
    )

    dpsi_dx = (
        a * dg_dx * phase_g
        + b_x * dx_dx * phase_e
        + b_z * np.exp(1j * delta) * dz_dx * phase_e
    )
    dpsi_dy = (
        a * dg_dy * phase_g
        + b_x * dx_dy * phase_e
        + b_z * np.exp(1j * delta) * dz_dy * phase_e
    )
    dpsi_dz = (
        a * dg_dz * phase_g
        + b_x * dx_dz * phase_e
        + b_z * np.exp(1j * delta) * dz_dz * phase_e
    )

    Jx = np.imag(np.conjugate(psi) * dpsi_dx)
    Jy = np.imag(np.conjugate(psi) * dpsi_dy)
    Jz = np.imag(np.conjugate(psi) * dpsi_dz)

    return psi, Jx, Jy, Jz


def normalize_coefficients(a: float, b_x: float, b_z: float):
    norm = np.sqrt(a * a + b_x * b_x + b_z * b_z)
    if norm == 0.0:
        raise ValueError("All coefficients cannot be zero")
    return a / norm, b_x / norm, b_z / norm