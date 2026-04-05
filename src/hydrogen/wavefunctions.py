from __future__ import annotations

import numpy as np

from .constants import A0, PI


def psi_100(r: np.ndarray, theta: np.ndarray, phi: np.ndarray, a0: float = A0) -> np.ndarray:
    """
    Hydrogen 1s orbital:
        psi_100 = (1/sqrt(pi a0^3)) exp(-r/a0)
    """
    pref = 1.0 / np.sqrt(PI * a0**3)
    return pref * np.exp(-r / a0)


def psi_200(r: np.ndarray, theta: np.ndarray, phi: np.ndarray, a0: float = A0) -> np.ndarray:
    """
    Hydrogen 2s orbital:
        psi_200 = (1/(4 sqrt(2 pi a0^3))) (2 - r/a0) exp(-r/(2a0))
    """
    pref = 1.0 / (4.0 * np.sqrt(2.0 * PI * a0**3))
    return pref * (2.0 - r / a0) * np.exp(-r / (2.0 * a0))


def psi_210(r: np.ndarray, theta: np.ndarray, phi: np.ndarray, a0: float = A0) -> np.ndarray:
    """
    Hydrogen 2p_z orbital (2,1,0):
        psi_210 = (1/(4 sqrt(2 pi a0^3))) (r/a0) exp(-r/(2a0)) cos(theta)
    """
    pref = 1.0 / (4.0 * np.sqrt(2.0 * PI * a0**3))
    return pref * (r / a0) * np.exp(-r / (2.0 * a0)) * np.cos(theta)