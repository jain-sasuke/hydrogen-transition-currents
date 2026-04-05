#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrogen.dipole import dipole_vector, spherical_grids
from hydrogen.far_field import dipole_intensity_map, normalize_by_peak
from hydrogen.wavefunctions import psi_100, psi_200, psi_210


FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

R_MAX = 30.0
NR = 600
NTHETA_INT = 240
NPHI_INT = 256

NTHETA_FF = 361
NPHI_FF = 361


def save_bar(components: np.ndarray, labels: list[str], title: str, filename: str) -> None:
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(labels, np.abs(components))
    plt.ylabel("Absolute value")
    plt.title(title)
    plt.tight_layout()
    path = FIG_DIR / filename
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_profile(theta: np.ndarray, i_num: np.ndarray, i_ana: np.ndarray, filename: str) -> None:
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(theta, i_num, label="numerical")
    plt.plot(theta, i_ana, "--", label=r"analytic $\sin^2\theta$")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Normalized intensity")
    plt.title(r"Hydrogen benchmark: $1s \leftrightarrow 2p_z$")
    plt.legend()
    plt.tight_layout()
    path = FIG_DIR / filename
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def save_map(theta: np.ndarray, phi: np.ndarray, intensity: np.ndarray, filename: str) -> None:
    plt.figure(figsize=(7.2, 5.4))
    plt.imshow(
        intensity,
        origin="lower",
        aspect="auto",
        extent=[phi[0], phi[-1], theta[0], theta[-1]],
    )
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.colorbar(label="Normalized intensity")
    plt.title(r"Far-field map for $1s \leftrightarrow 2p_z$")
    plt.tight_layout()
    path = FIG_DIR / filename
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"[Saved] {path}")


def main() -> None:
    print("Starting Task 1 hydrogen transition benchmark...", flush=True)

    r, theta_int, phi_int = spherical_grids(
        r_max=R_MAX,
        nr=NR,
        ntheta=NTHETA_INT,
        nphi=NPHI_INT,
    )

    print("Computing dipole vector for 1s <-> 2p_z ...", flush=True)
    d_1s_2pz = dipole_vector(psi_100, psi_210, r, theta_int, phi_int)

    print("Computing dipole vector for 1s <-> 2s ...", flush=True)
    d_1s_2s = dipole_vector(psi_100, psi_200, r, theta_int, phi_int)

    print("\nDipole vector results:")
    print(
        f"1s <-> 2p_z : dx={d_1s_2pz[0]:.6e}, dy={d_1s_2pz[1]:.6e}, dz={d_1s_2pz[2]:.6e}"
    )
    print(
        f"1s <-> 2s   : dx={d_1s_2s[0]:.6e}, dy={d_1s_2s[1]:.6e}, dz={d_1s_2s[2]:.6e}"
    )

    dx_rel = np.abs(d_1s_2pz[0]) / max(np.abs(d_1s_2pz[2]), 1e-30)
    dy_rel = np.abs(d_1s_2pz[1]) / max(np.abs(d_1s_2pz[2]), 1e-30)
    forb_rel = np.linalg.norm(d_1s_2s) / max(np.linalg.norm(d_1s_2pz), 1e-30)

    print("\nDerived checks:")
    print(f"|dx|/|dz| for 1s <-> 2p_z = {dx_rel:.6e}")
    print(f"|dy|/|dz| for 1s <-> 2p_z = {dy_rel:.6e}")
    print(f"|d_1s,2s| / |d_1s,2p_z|   = {forb_rel:.6e}")

    save_bar(d_1s_2pz, ["dx", "dy", "dz"], r"$1s \leftrightarrow 2p_z$ dipole components", "task01_dipole_components_1s_2pz.png")
    save_bar(d_1s_2s, ["dx", "dy", "dz"], r"$1s \leftrightarrow 2s$ forbidden-transition check", "task01_forbidden_transition_components_1s_2s.png")

    theta_ff = np.linspace(0.0, np.pi, NTHETA_FF)
    phi_ff = np.linspace(0.0, 2.0 * np.pi, NPHI_FF, endpoint=False)

    i_map = dipole_intensity_map(d_1s_2pz, theta_ff, phi_ff)
    i_map = normalize_by_peak(i_map)

    phi_mid = NPHI_FF // 2
    i_profile = i_map[:, phi_mid]
    i_analytic = np.sin(theta_ff) ** 2
    i_analytic = normalize_by_peak(i_analytic)

    max_err = float(np.max(np.abs(i_profile - i_analytic)))
    print(f"\nMax profile error against sin^2(theta): {max_err:.6e}")

    save_profile(theta_ff, i_profile, i_analytic, "task01_angular_profile.png")
    save_map(theta_ff, phi_ff, i_map, "task01_sphere_map.png")

    print("\nTask 1 complete.", flush=True)


if __name__ == "__main__":
    main()