#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrogen.general_states import (
    radial_hydrogen,
    complex_spherical_harmonic,
    hydrogen_state_xyz,
)
from hydrogen.general_integrals import dipole_matrix_element

FIG_ROOT = PROJECT_ROOT / "figures"
FIG_DIR = FIG_ROOT / "task04A"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NOTES_ROOT = PROJECT_ROOT / "notes"
NOTES_ROOT.mkdir(parents=True, exist_ok=True)


# --- Cartesian grid for dipole checks only ---
XMAX = 30.0
NGRID = 101

# --- Spherical grids for normalization/overlap checks ---
RMAX = 80.0
NR = 4001
NTH = 401
NPH = 401


def make_cartesian_grid(xmax: float, n: int):
    x = np.linspace(-xmax, xmax, n)
    y = np.linspace(-xmax, xmax, n)
    z = np.linspace(-xmax, xmax, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


def make_angular_mesh():
    theta = np.linspace(0.0, np.pi, NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, NPH)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return theta, phi, TH, PH


def factorized_state(spec, r: np.ndarray, TH: np.ndarray, PH: np.ndarray):
    """
    Return factorized hydrogen state:
        psi(r,theta,phi) = R(r) * Y(theta,phi)

    Supports:
      - tuple (n,l,m)
      - strings '1s', '2s', '2px', '2py', '2pz'
    """
    if isinstance(spec, tuple):
        if len(spec) != 3:
            raise ValueError("Tuple state must be (n,l,m)")
        n, l, m = spec
        R = radial_hydrogen(n, l, r)
        Y = complex_spherical_harmonic(l, m, TH, PH)
        return R, Y

    if not isinstance(spec, str):
        raise TypeError("State spec must be string or tuple (n,l,m)")

    state = spec.lower().strip()

    if state.endswith("s"):
        n = int(state[:-1])
        R = radial_hydrogen(n, 0, r)
        Y = complex_spherical_harmonic(0, 0, TH, PH)
        return R, Y

    if state.endswith("pz"):
        n = int(state[:-2])
        R = radial_hydrogen(n, 1, r)
        Y = complex_spherical_harmonic(1, 0, TH, PH)
        return R, Y

    if state.endswith("px"):
        n = int(state[:-2])
        R = radial_hydrogen(n, 1, r)
        Yp = complex_spherical_harmonic(1, +1, TH, PH)
        Ym = complex_spherical_harmonic(1, -1, TH, PH)
        Y = np.real_if_close((Ym - Yp) / np.sqrt(2.0))
        return R, Y

    if state.endswith("py"):
        n = int(state[:-2])
        R = radial_hydrogen(n, 1, r)
        Yp = complex_spherical_harmonic(1, +1, TH, PH)
        Ym = complex_spherical_harmonic(1, -1, TH, PH)
        Y = np.real_if_close(1j * (Ym + Yp) / np.sqrt(2.0))
        return R, Y

    raise ValueError("Supported string states: '1s', '2s', '2px', '2py', '2pz'")


def radial_inner(Ra: np.ndarray, Rb: np.ndarray, r: np.ndarray) -> complex:
    integrand = np.conjugate(Ra) * Rb * (r**2)
    return simpson(integrand, x=r)


def angular_inner(Ya: np.ndarray, Yb: np.ndarray, theta: np.ndarray, phi: np.ndarray, TH: np.ndarray) -> complex:
    integrand = np.conjugate(Ya) * Yb * np.sin(TH)
    out_phi = simpson(integrand, x=phi, axis=1)
    out_theta = simpson(out_phi, x=theta, axis=0)
    return out_theta


def spherical_norm(spec, r, theta, phi, TH, PH) -> float:
    R, Y = factorized_state(spec, r, TH, PH)
    val = radial_inner(R, R, r) * angular_inner(Y, Y, theta, phi, TH)
    return float(np.real_if_close(val))


def spherical_overlap(spec_a, spec_b, r, theta, phi, TH, PH) -> complex:
    Ra, Ya = factorized_state(spec_a, r, TH, PH)
    Rb, Yb = factorized_state(spec_b, r, TH, PH)
    return radial_inner(Ra, Rb, r) * angular_inner(Ya, Yb, theta, phi, TH)


def save_normalization_plot(labels, norm_results, filename):
    errors = [abs(norm_results[k] - 1.0) for k in labels]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.bar(labels, errors)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\left|\int |\psi|^2 d^3r - 1\right|$")
    ax.set_title("Task 4A: normalization error for general hydrogen states")
    ax.axhline(1e-6, color="gray", ls="--", lw=1.3, label=r"$10^{-6}$ reference")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_overlap_plot(labels, values, filename):
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel(r"$|\langle a|b\rangle|$")
    ax.set_title("Task 4A: orthogonality checks")
    ax.axhline(1e-6, color="gray", ls="--", lw=1.3, label=r"$10^{-6}$ reference")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_dipole_plot(case_labels, dipoles_abs, filename):
    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    idx = np.arange(len(case_labels))
    width = 0.24

    ax.bar(idx - width, dipoles_abs[:, 0], width=width, label=r"$|d_x|$")
    ax.bar(idx,         dipoles_abs[:, 1], width=width, label=r"$|d_y|$")
    ax.bar(idx + width, dipoles_abs[:, 2], width=width, label=r"$|d_z|$")

    ax.set_xticks(idx)
    ax.set_xticklabels(case_labels)
    ax.set_ylabel(r"$|\langle a|r_i|b\rangle|$")
    ax.set_title("Task 4A: dipole selection-rule checks")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def write_summary(
    path: Path,
    norm_results: dict[str, float],
    overlap_results: dict[str, complex],
    dipole_results: dict[str, np.ndarray],
):
    max_norm_error = max(abs(v - 1.0) for v in norm_results.values())
    max_overlap_mag = max(abs(v) for v in overlap_results.values())

    d_1s_2s = dipole_results["1s->2s"]
    d_1s_2pz = dipole_results["1s->2pz"]
    d_1s_2px = dipole_results["1s->2px"]

    forbidden_mag = float(np.max(np.abs(d_1s_2s)))
    pz_main = float(abs(d_1s_2pz[2]))
    pz_leak = float(max(abs(d_1s_2pz[0]), abs(d_1s_2pz[1])))
    px_main = float(abs(d_1s_2px[0]))
    px_leak = float(max(abs(d_1s_2px[1]), abs(d_1s_2px[2])))

    lines = []
    lines.append("Task 4A — general hydrogen state engine")
    lines.append("=" * 44)
    lines.append("")
    lines.append("Normalization results:")
    for k, v in norm_results.items():
        lines.append(f"  {k:>6s} : {v:.12e}")
    lines.append(f"Max normalization error: {max_norm_error:.12e}")
    lines.append("")

    lines.append("Orthogonality results:")
    for k, v in overlap_results.items():
        lines.append(f"  {k:>14s} : {abs(v):.12e}")
    lines.append(f"Max overlap magnitude: {max_overlap_mag:.12e}")
    lines.append("")

    lines.append("Dipole selection-rule checks:")
    lines.append(f"  1s->2s  : |dx|,|dy|,|dz| = {[float(abs(c)) for c in d_1s_2s]}")
    lines.append(f"  1s->2pz : |dx|,|dy|,|dz| = {[float(abs(c)) for c in d_1s_2pz]}")
    lines.append(f"  1s->2px : |dx|,|dy|,|dz| = {[float(abs(c)) for c in d_1s_2px]}")
    lines.append("")
    lines.append(f"Forbidden-transition max dipole magnitude (1s->2s): {forbidden_mag:.12e}")
    lines.append(f"Allowed 1s->2pz main component |dz|: {pz_main:.12e}")
    lines.append(f"Allowed 1s->2pz leakage max(|dx|,|dy|): {pz_leak:.12e}")
    lines.append(f"Allowed 1s->2px main component |dx|: {px_main:.12e}")
    lines.append(f"Allowed 1s->2px leakage max(|dy|,|dz|): {px_leak:.12e}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Scalar validation is now done in spherical coordinates, not on a Cartesian cube.")
    lines.append("- Norms should be very close to 1.")
    lines.append("- Overlaps between distinct states should be very small.")
    lines.append("- 1s->2s should be dipole-forbidden.")
    lines.append("- 1s->2pz should be z-polarized.")
    lines.append("- 1s->2px should be x-polarized.")

    path.write_text("\n".join(lines))
    print(f"[Saved] {path}")


def main():
    print("Starting Task 4A general hydrogen state engine...", flush=True)
    print(f"Saving figures to: {FIG_DIR}")

    # --- spherical grids for scalar checks ---
    r = np.linspace(0.0, RMAX, NR)
    theta = np.linspace(0.0, np.pi, NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, NPH)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    # --- normalization / overlap test states ---
    states = {
        "1s": "1s",
        "2s": "2s",
        "2pz": "2pz",
        "2px": "2px",
        "2py": "2py",
        "2p+1": (2, 1, +1),
    }

    norm_results = {}
    for label, spec in states.items():
        norm_results[label] = spherical_norm(spec, r, theta, phi, TH, PH)

    print("\nNormalization checks:")
    for k, v in norm_results.items():
        print(f"  {k:>6s} : {v:.12e}")

    save_normalization_plot(
        list(norm_results.keys()),
        norm_results,
        "task04A_normalization.png",
    )

    overlap_results = {
        "<1s|2s>": spherical_overlap("1s", "2s", r, theta, phi, TH, PH),
        "<1s|2pz>": spherical_overlap("1s", "2pz", r, theta, phi, TH, PH),
        "<2s|2pz>": spherical_overlap("2s", "2pz", r, theta, phi, TH, PH),
        "<2px|2py>": spherical_overlap("2px", "2py", r, theta, phi, TH, PH),
        "<2pz|2p+1>": spherical_overlap("2pz", (2, 1, +1), r, theta, phi, TH, PH),
    }

    print("\nOrthogonality checks:")
    for k, v in overlap_results.items():
        print(f"  {k:>14s} : {abs(v):.12e}")

    save_overlap_plot(
        list(overlap_results.keys()),
        [abs(v) for v in overlap_results.values()],
        "task04A_orthogonality.png",
    )

    # --- Cartesian grid only for dipole checks ---
    x, y, z, X, Y, Z = make_cartesian_grid(XMAX, NGRID)

    psi_cart = {
        "1s": hydrogen_state_xyz("1s", X, Y, Z),
        "2s": hydrogen_state_xyz("2s", X, Y, Z),
        "2pz": hydrogen_state_xyz("2pz", X, Y, Z),
        "2px": hydrogen_state_xyz("2px", X, Y, Z),
    }

    dipole_results = {
        "1s->2s": dipole_matrix_element(psi_cart["1s"], psi_cart["2s"], X, Y, Z, x, y, z),
        "1s->2pz": dipole_matrix_element(psi_cart["1s"], psi_cart["2pz"], X, Y, Z, x, y, z),
        "1s->2px": dipole_matrix_element(psi_cart["1s"], psi_cart["2px"], X, Y, Z, x, y, z),
    }

    print("\nDipole checks:")
    for k, vec in dipole_results.items():
        mags = [float(abs(c)) for c in vec]
        print(f"  {k:>8s} : |dx|,|dy|,|dz| = {mags}")

    dipole_abs = np.array(
        [[abs(c) for c in dipole_results[key]] for key in ["1s->2s", "1s->2pz", "1s->2px"]],
        dtype=float,
    )

    save_dipole_plot(
        ["1s->2s", "1s->2pz", "1s->2px"],
        dipole_abs,
        "task04A_dipole_selection_rules.png",
    )

    write_summary(
        FIG_DIR / "task04A_summary.txt",
        norm_results=norm_results,
        overlap_results=overlap_results,
        dipole_results=dipole_results,
    )

    print("\nTask 4A complete.", flush=True)


if __name__ == "__main__":
    main()