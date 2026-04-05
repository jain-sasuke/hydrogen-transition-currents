from __future__ import annotations

import numpy as np

from .general_states import hydrogen_state_xyz, gradient_hydrogen_state_xyz


def transition_current_amplitude_equal_superposition(
    state_a,
    state_b,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    h: float = 1.0e-4,
):
    """
    Current-amplitude field for an equal-weight superposition of two real hydrogen states.

    For
        Psi = (psi_a e^{-i E_a t} + psi_b e^{-i E_b t}) / sqrt(2)

    and real orbitals, the oscillating current is

        J(r,t) = sin(omega t) * J_amp(r)

    with

        J_amp = 1/2 * [ psi_a grad psi_b - psi_b grad psi_a ].

    This function returns:
        psi_a, psi_b, Jx_amp, Jy_amp, Jz_amp
    """
    psi_a = np.real_if_close(hydrogen_state_xyz(state_a, X, Y, Z))
    psi_b = np.real_if_close(hydrogen_state_xyz(state_b, X, Y, Z))

    da_dx, da_dy, da_dz = gradient_hydrogen_state_xyz(state_a, X, Y, Z, h=h)
    db_dx, db_dy, db_dz = gradient_hydrogen_state_xyz(state_b, X, Y, Z, h=h)

    da_dx = np.real_if_close(da_dx)
    da_dy = np.real_if_close(da_dy)
    da_dz = np.real_if_close(da_dz)

    db_dx = np.real_if_close(db_dx)
    db_dy = np.real_if_close(db_dy)
    db_dz = np.real_if_close(db_dz)

    Jx_amp = 0.5 * (psi_a * db_dx - psi_b * da_dx)
    Jy_amp = 0.5 * (psi_a * db_dy - psi_b * da_dy)
    Jz_amp = 0.5 * (psi_a * db_dz - psi_b * da_dz)

    return psi_a, psi_b, Jx_amp, Jy_amp, Jz_amp