from __future__ import annotations


def hydrogen_energy_au(n: int) -> float:
    """
    Hydrogen bound-state energy in atomic units:
        E_n = -1 / (2 n^2)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return -0.5 / (n * n)


def transition_omega_au(n_initial: int, n_final: int) -> float:
    """
    Transition angular frequency in atomic units:
        omega = E_final - E_initial
    For 1s -> 2p, this is positive: 0.375 a.u.
    """
    return hydrogen_energy_au(n_final) - hydrogen_energy_au(n_initial)