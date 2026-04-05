from __future__ import annotations

import numpy as np


# Speed of light in atomic units
C_AU = 137.035999084


def wave_number_au(omega_au: float) -> float:
    """
    k = omega / c in atomic units.
    """
    return omega_au / C_AU


def volume_element(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    return dx * dy * dz


def integrated_current_vector(
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    ∫ J(r) d^3r
    """
    dv = volume_element(x, y, z)
    return np.array(
        [
            np.sum(Jx) * dv,
            np.sum(Jy) * dv,
            np.sum(Jz) * dv,
        ],
        dtype=np.complex128,
    )


def direction_unit_vector(theta: float, phi: float) -> np.ndarray:
    st = np.sin(theta)
    return np.array(
        [
            st * np.cos(phi),
            st * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )


def fourier_current_in_direction(
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: float,
    theta: float,
    phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spatial Fourier transform of the current at q = k n_hat:

        J_tilde(k n_hat) = ∫ d^3r J(r) exp[-i k n_hat · r]

    Returns:
        J_tilde (complex 3-vector), n_hat
    """
    n_hat = direction_unit_vector(theta, phi)
    phase = np.exp(-1j * k * (n_hat[0] * X + n_hat[1] * Y + n_hat[2] * Z))
    dv = volume_element(x, y, z)

    Jtx = np.sum(Jx * phase) * dv
    Jty = np.sum(Jy * phase) * dv
    Jtz = np.sum(Jz * phase) * dv

    return np.array([Jtx, Jty, Jtz], dtype=np.complex128), n_hat


def transverse_projection(vec: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    """
    n x (n x vec) = vec - n (n·vec)
    """
    return vec - n_hat * np.dot(n_hat, vec)


def far_field_amplitude_from_current(
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    """
    Up to an overall prefactor, the far-field electric amplitude is

        E(n_hat) ∝ n_hat x [ n_hat x J_tilde(k n_hat) ].

    Returns the complex transverse amplitude vector.
    """
    J_tilde, n_hat = fourier_current_in_direction(
        Jx, Jy, Jz, X, Y, Z, x, y, z, k, theta, phi
    )
    return transverse_projection(J_tilde, n_hat)


def far_field_intensity_from_current(
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: float,
    theta: float,
    phi: float,
) -> float:
    """
    Intensity ∝ |E|^2.
    """
    amp = far_field_amplitude_from_current(
        Jx, Jy, Jz, X, Y, Z, x, y, z, k, theta, phi
    )
    return float(np.real(np.vdot(amp, amp)))


def dipole_pattern_z(theta: np.ndarray) -> np.ndarray:
    """
    Dipole-limit radiation pattern for a z-directed dipole:
        I(theta) ∝ sin^2(theta)
    """
    return np.sin(theta) ** 2