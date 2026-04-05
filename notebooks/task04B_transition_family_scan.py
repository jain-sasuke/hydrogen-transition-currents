#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrogen.general_states import hydrogen_state_xyz
from hydrogen.general_integrals import dipole_matrix_element
from hydrogen.general_transition_current import (
    transition_current_amplitude_equal_superposition,
)
from hydrogen.spherical_benchmarks import spherical_dipole_vector
from hydrogen.radiation import (
    wave_number_au,
    integrated_current_vector,
    far_field_intensity_from_current,
    dipole_pattern_z,
)

FIG_ROOT = PROJECT_ROOT / "figures"
FIG_DIR = FIG_ROOT / "task04B"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NOTES_ROOT = PROJECT_ROOT / "notes"
NOTES_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Radiation grid (coarse enough to run, good enough for pattern)
# -----------------------------
XMAX = 40.0
NGRID = 71
GRAD_H = 1.0e-4

# -----------------------------
# Spherical benchmark grid (accurate dipole benchmark)
# -----------------------------
RMAX_SPH = 100.0
NR_SPH = 4001
NTH_SPH = 401
NPH_SPH = 401

# -----------------------------
# Angular scan
# -----------------------------
NTHETA = 121
NPHI = 13

TRANSITIONS = [
    ("1s", "2pz"),
    ("1s", "3pz"),
    ("1s", "4pz"),
    ("2s", "3pz"),
    ("2s", "4pz"),
    ("3s", "4pz"),
]


def hydrogen_energy_au(n: int) -> float:
    return -0.5 / (n * n)


def parse_n_from_state(state: str) -> int:
    m = re.match(r"^\s*(\d+)", state)
    if m is None:
        raise ValueError(f"Could not parse principal quantum number from state: {state}")
    return int(m.group(1))


def transition_omega_au_from_states(state_a: str, state_b: str) -> float:
    n_a = parse_n_from_state(state_a)
    n_b = parse_n_from_state(state_b)
    return abs(hydrogen_energy_au(n_b) - hydrogen_energy_au(n_a))


def make_cartesian_grid(xmax: float, n: int):
    x = np.linspace(-xmax, xmax, n)
    y = np.linspace(-xmax, xmax, n)
    z = np.linspace(-xmax, xmax, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


def scan_one_transition(
    state_a: str,
    state_b: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    r_sph: np.ndarray,
    theta_sph: np.ndarray,
    phi_sph: np.ndarray,
):
    omega = transition_omega_au_from_states(state_a, state_b)
    k = wave_number_au(omega)

    psi_a = hydrogen_state_xyz(state_a, X, Y, Z)
    psi_b = hydrogen_state_xyz(state_b, X, Y, Z)

    dvec_cart = dipole_matrix_element(psi_a, psi_b, X, Y, Z, x, y, z)
    dvec_sph = spherical_dipole_vector(state_a, state_b, r_sph, theta_sph, phi_sph)

    dz_cart = float(np.real_if_close(dvec_cart[2]))
    dz_sph = float(np.real_if_close(dvec_sph[2]))
    rel_dipole_error = abs(dz_cart - dz_sph) / max(abs(dz_sph), 1e-30)

    _, _, Jx_amp, Jy_amp, Jz_amp = transition_current_amplitude_equal_superposition(
        state_a, state_b, X, Y, Z, h=GRAD_H
    )

    # Keep old current-integral diagnostic only as auxiliary info
    Jint = integrated_current_vector(Jx_amp, Jy_amp, Jz_amp, x, y, z)
    Jint_z = float(np.real_if_close(Jint[2]))
    expected_Jz_from_sph = abs(omega * dz_sph)
    rel_current_aux_error = abs(abs(Jint_z) - expected_Jz_from_sph) / max(expected_Jz_from_sph, 1e-30)

    theta_grid = np.linspace(0.0, np.pi, NTHETA)
    phi0 = 0.0

    I_exact = []
    for theta in theta_grid:
        I_val = far_field_intensity_from_current(
            Jx_amp,
            Jy_amp,
            Jz_amp,
            X,
            Y,
            Z,
            x,
            y,
            z,
            k=k,
            theta=float(theta),
            phi=phi0,
        )
        I_exact.append(I_val)

    I_exact = np.asarray(I_exact, dtype=float)
    I_exact_norm = I_exact / np.max(I_exact)

    I_dip = dipole_pattern_z(theta_grid)
    I_dip_norm = I_dip / np.max(I_dip)

    max_profile_deviation = float(np.max(np.abs(I_exact_norm - I_dip_norm)))

    theta_eq = np.pi / 2.0
    phi_grid = np.linspace(0.0, 2.0 * np.pi, NPHI)

    I_phi = []
    for phi in phi_grid:
        I_val = far_field_intensity_from_current(
            Jx_amp,
            Jy_amp,
            Jz_amp,
            X,
            Y,
            Z,
            x,
            y,
            z,
            k=k,
            theta=theta_eq,
            phi=float(phi),
        )
        I_phi.append(I_val)

    I_phi = np.asarray(I_phi, dtype=float)
    I_phi_norm = I_phi / np.max(I_phi)
    phi_spread = float((np.max(I_phi_norm) - np.min(I_phi_norm)) / np.mean(I_phi_norm))

    return {
        "state_a": state_a,
        "state_b": state_b,
        "omega": omega,
        "k": k,
        "dvec_cart": dvec_cart,
        "dvec_sph": dvec_sph,
        "rel_dipole_error": rel_dipole_error,
        "Jint_z": Jint_z,
        "expected_Jz_from_sph": expected_Jz_from_sph,
        "rel_current_aux_error": rel_current_aux_error,
        "theta_grid": theta_grid,
        "I_exact_norm": I_exact_norm,
        "I_dip_norm": I_dip_norm,
        "max_profile_deviation": max_profile_deviation,
        "phi_grid": phi_grid,
        "I_phi_norm": I_phi_norm,
        "phi_spread": phi_spread,
    }


def save_bar_log(labels, values, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_worst_profile_plot(result, filename):
    label = f"{result['state_a']}→{result['state_b']}"

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(
        result["theta_grid"],
        result["I_exact_norm"],
        lw=2.5,
        label=f"exact far field ({label})",
    )
    ax.plot(
        result["theta_grid"],
        result["I_dip_norm"],
        "--",
        lw=2.5,
        label=r"dipole limit $\sin^2\theta$",
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"Task 4B: worst-case profile comparison ({label})")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def write_summary(results, filename):
    lines = []
    lines.append("Task 4B — hydrogen transition-family scan (fixed benchmark)")
    lines.append("=" * 64)
    lines.append("")
    lines.append("Transitions scanned:")
    for r in results:
        lines.append(f"  {r['state_a']} -> {r['state_b']}")
    lines.append("")

    header = (
        f"{'transition':<12s} "
        f"{'omega[a.u.]':>14s} "
        f"{'rel_dip_err':>14s} "
        f"{'max_prof_dev':>16s} "
        f"{'phi_spread':>14s} "
        f"{'aux_J_err':>12s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        label = f"{r['state_a']}->{r['state_b']}"
        lines.append(
            f"{label:<12s} "
            f"{r['omega']:14.6e} "
            f"{r['rel_dipole_error']:14.6e} "
            f"{r['max_profile_deviation']:16.6e} "
            f"{r['phi_spread']:14.6e} "
            f"{r['rel_current_aux_error']:12.6e}"
        )

    lines.append("")
    worst_by_dev = max(results, key=lambda rr: rr["max_profile_deviation"])
    worst_by_dip = max(results, key=lambda rr: rr["rel_dipole_error"])
    lines.append(
        f"Worst profile deviation: {worst_by_dev['state_a']}->{worst_by_dev['state_b']} "
        f"with {worst_by_dev['max_profile_deviation']:.6e}"
    )
    lines.append(
        f"Worst Cartesian-vs-spherical dipole error: {worst_by_dip['state_a']}->{worst_by_dip['state_b']} "
        f"with {worst_by_dip['rel_dipole_error']:.6e}"
    )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- The fixed benchmark is the Cartesian dipole compared against high-accuracy spherical quadrature.")
    lines.append("- This is stronger than the old coarse-grid current-moment diagnostic.")
    lines.append("- The old current-moment quantity is retained only as an auxiliary numerical indicator.")
    lines.append("- Radiation-pattern validity is judged primarily by max profile deviation and phi-spread.")
    lines.append("- Source-amplitude validity is judged primarily by Cartesian-vs-spherical dipole agreement.")

    path = FIG_DIR / filename
    path.write_text("\n".join(lines))
    print(f"[Saved] {path}")


def main():
    print("Starting Task 4B hydrogen transition-family scan (fixed benchmark)...", flush=True)
    print(f"Saving figures to: {FIG_DIR}")

    x, y, z, X, Y, Z = make_cartesian_grid(XMAX, NGRID)

    r_sph = np.linspace(0.0, RMAX_SPH, NR_SPH)
    theta_sph = np.linspace(0.0, np.pi, NTH_SPH)
    phi_sph = np.linspace(0.0, 2.0 * np.pi, NPH_SPH)

    results = []
    for state_a, state_b in TRANSITIONS:
        print(f"\nScanning {state_a} -> {state_b} ...", flush=True)
        res = scan_one_transition(
            state_a,
            state_b,
            x,
            y,
            z,
            X,
            Y,
            Z,
            r_sph,
            theta_sph,
            phi_sph,
        )
        results.append(res)

        print(f"  omega                 = {res['omega']:.6e}")
        print(f"  rel dipole error      = {res['rel_dipole_error']:.6e}")
        print(f"  max profile deviation = {res['max_profile_deviation']:.6e}")
        print(f"  phi spread            = {res['phi_spread']:.6e}")
        print(f"  aux current error     = {res['rel_current_aux_error']:.6e}")

    labels = [f"{r['state_a']}→{r['state_b']}" for r in results]
    profile_devs = [r["max_profile_deviation"] for r in results]
    phi_spreads = [r["phi_spread"] for r in results]
    dipole_errors = [r["rel_dipole_error"] for r in results]
    aux_current_errors = [r["rel_current_aux_error"] for r in results]

    save_bar_log(
        labels,
        profile_devs,
        ylabel=r"$\max_\theta |I_{\rm exact}^{\rm norm} - I_{\rm dip}^{\rm norm}|$",
        title="Task 4B: deviation from dipole-limit profile across transitions",
        filename="task04B_profile_deviation_scan.png",
    )

    save_bar_log(
        labels,
        phi_spreads,
        ylabel=r"equatorial $\phi$-spread fraction",
        title="Task 4B: azimuthal-invariance check across transitions",
        filename="task04B_phi_spread_scan.png",
    )

    save_bar_log(
        labels,
        dipole_errors,
        ylabel=r"relative error in $d_z$ (Cartesian vs spherical benchmark)",
        title="Task 4B: dipole benchmark across transitions",
        filename="task04B_dipole_benchmark_error.png",
    )

    save_bar_log(
        labels,
        aux_current_errors,
        ylabel=r"auxiliary coarse-grid current-moment error",
        title="Task 4B: old current benchmark retained as auxiliary only",
        filename="task04B_aux_current_error.png",
    )

    worst_result = max(results, key=lambda rr: rr["max_profile_deviation"])
    save_worst_profile_plot(
        worst_result,
        "task04B_worst_profile_comparison.png",
    )

    write_summary(results, "task04B_summary.txt")

    print("\nTask 4B complete.", flush=True)


if __name__ == "__main__":
    main()