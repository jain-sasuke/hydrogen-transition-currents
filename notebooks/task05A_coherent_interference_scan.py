#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydrogen.coherent_superpositions import (
    coherent_state_three_component,
    coherent_current_density,
    hydrogen_energy_au,
    normalize_coefficients,
)
from hydrogen.general_integrals import dipole_matrix_element
from hydrogen.general_states import hydrogen_state_xyz
from hydrogen.radiation import wave_number_au, far_field_intensity_from_current
from hydrogen.radiation_jax import dipole_limit_map_from_vector_numpy

FIG_DIR = PROJECT_ROOT / "figures" / "task05A"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Controls
# -----------------------------
# Keep this moderate on laptop. The old 61^3 + 121x181 scan is too heavy.
XMAX = 28.0
NGRID = 41
GRAD_H = 1.0e-4

# Angular map resolution
NTHETA = 81
NPHI = 91

# The full batch JAX kernel can materialize an enormous
# (theta, phi, x, y, z) phase tensor and get killed on laptop memory.
# Use a slower but memory-safe angular loop by default.
SAFE_ANGLE_LOOP = True

GROUND = "1s"
PX = "2px"
PZ = "2pz"

A0 = 1.0
BX0 = 0.5
BZ0 = 0.5

# Start with 3 physically distinct phase cases only.
DELTA_LIST = [
    0.0,
    0.5 * np.pi,
    np.pi,
]

TIME_FRACTION = 0.25  # evaluate at T/4


def make_cartesian_grid(xmax: float, n: int):
    x = np.linspace(-xmax, xmax, n)
    y = np.linspace(-xmax, xmax, n)
    z = np.linspace(-xmax, xmax, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


def transition_omega_1s_2p() -> float:
    E1 = hydrogen_energy_au(1)
    E2 = hydrogen_energy_au(2)
    return abs(E2 - E1)


def compute_dipole_vector_from_density(
    psi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    rho = np.abs(psi) ** 2

    x3 = x[:, None, None]
    y3 = y[None, :, None]
    z3 = z[None, None, :]

    dx = np.trapezoid(np.trapezoid(np.trapezoid(rho * x3, z, axis=2), y, axis=1), x, axis=0)
    dy = np.trapezoid(np.trapezoid(np.trapezoid(rho * y3, z, axis=2), y, axis=1), x, axis=0)
    dz = np.trapezoid(np.trapezoid(np.trapezoid(rho * z3, z, axis=2), y, axis=1), x, axis=0)

    return np.array([dx, dy, dz], dtype=float)


def compute_transition_dipole_basis(X, Y, Z, x, y, z):
    psi_1s = hydrogen_state_xyz(GROUND, X, Y, Z)
    psi_2px = hydrogen_state_xyz(PX, X, Y, Z)
    psi_2pz = hydrogen_state_xyz(PZ, X, Y, Z)

    d_1s_2px = dipole_matrix_element(psi_1s, psi_2px, X, Y, Z, x, y, z)
    d_1s_2pz = dipole_matrix_element(psi_1s, psi_2pz, X, Y, Z, x, y, z)

    return np.real_if_close(d_1s_2px), np.real_if_close(d_1s_2pz)


def exact_far_field_map_memory_safe(
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
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
) -> np.ndarray:
    """
    Build the exact far-field map one angle at a time.

    This is intentionally slower than the fully batched JAX path, but it avoids
    the huge temporary arrays that were causing the process to be killed.
    """
    out = np.zeros((len(theta_grid), len(phi_grid)), dtype=float)
    total = len(theta_grid) * len(phi_grid)
    counter = 0

    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            out[i, j] = far_field_intensity_from_current(
                Jx,
                Jy,
                Jz,
                X,
                Y,
                Z,
                x,
                y,
                z,
                k=k,
                theta=float(theta),
                phi=float(phi),
            )
            counter += 1

        if (i + 1) % max(1, len(theta_grid) // 8) == 0 or (i + 1) == len(theta_grid):
            print(f"    exact map progress: {counter}/{total} angles done", flush=True)

    max_val = np.max(out)
    if max_val > 0.0:
        out = out / max_val
    return out


def save_map(map_data, theta_grid, phi_grid, title, filename):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    im = ax.pcolormesh(phi_grid, theta_grid, map_data, shading="auto", cmap="viridis")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized intensity")
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_polar_cut(theta_grid, exact_cut, dip_cut, delta_label, filename):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(theta_grid, exact_cut, lw=2.5, label="exact current-source")
    ax.plot(theta_grid, dip_cut, "--", lw=2.5, label="dipole vector prediction")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"Task 5A: phi=0 cut, delta={delta_label}")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_dipole_rotation_plot(px_vals, pz_vals, delta_vals, filename):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot(px_vals, pz_vals, "-o")
    for px, pz, d in zip(px_vals, pz_vals, delta_vals):
        ax.text(px, pz, f"{d / np.pi:.2f}π")
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_z$")
    ax.set_title("Task 5A: dipole-vector trajectory vs phase")
    ax.axhline(0.0, color="gray", lw=1.0)
    ax.axvline(0.0, color="gray", lw=1.0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def main():
    print("Starting Task 5A coherent interference scan (memory-safe exact radiation kernel)...", flush=True)

    a, bx, bz = normalize_coefficients(A0, BX0, BZ0)

    x, y, z, X, Y, Z = make_cartesian_grid(XMAX, NGRID)

    omega = transition_omega_1s_2p()
    T = 2.0 * np.pi / omega
    t_eval = TIME_FRACTION * T
    k = wave_number_au(omega)

    print(f"omega_1s-2p = {omega:.6e} a.u.")
    print(f"T           = {T:.6e} a.u.")
    print(f"t_eval      = {t_eval:.6e} a.u.")
    print(f"grid        = {NGRID}^3, XMAX={XMAX}")
    print(f"angles      = {NTHETA} x {NPHI}")
    print(f"angle loop  = {SAFE_ANGLE_LOOP}")
    print("Using memory-safe exact far-field evaluation...", flush=True)

    d_1s_2px, d_1s_2pz = compute_transition_dipole_basis(X, Y, Z, x, y, z)
    d_1s_2px = np.real_if_close(d_1s_2px)
    d_1s_2pz = np.real_if_close(d_1s_2pz)

    theta_grid = np.linspace(0.0, np.pi, NTHETA)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, NPHI)

    px_vals = []
    pz_vals = []

    summary_lines = []
    summary_lines.append("Task 5A coherent interference scan")
    summary_lines.append("=" * 40)
    summary_lines.append(f"a = {a:.6f}, b_x = {bx:.6f}, b_z = {bz:.6f}")
    summary_lines.append(f"t_eval = {t_eval:.6e} a.u.")
    summary_lines.append(f"grid = {NGRID}^3, XMAX = {XMAX}")
    summary_lines.append(f"angles = {NTHETA} x {NPHI}")
    summary_lines.append("")

    for idx, delta in enumerate(DELTA_LIST):
        print(f"\nScanning delta = {delta:.6f} rad", flush=True)

        psi = coherent_state_three_component(
            X, Y, Z,
            ground_state=GROUND,
            excited_x=PX,
            excited_z=PZ,
            a=a, b_x=bx, b_z=bz,
            delta=delta,
            t=t_eval,
        )

        pvec_density = compute_dipole_vector_from_density(psi, x, y, z)

        phase_time = np.exp(-1j * omega * t_eval)
        pvec_expected = np.real_if_close(
            a * bx * phase_time * d_1s_2px
            + a * bz * np.exp(1j * delta) * phase_time * d_1s_2pz
            + a * bx * np.conjugate(phase_time) * np.conjugate(d_1s_2px)
            + a * bz * np.exp(-1j * delta) * np.conjugate(phase_time) * np.conjugate(d_1s_2pz)
        )

        px_vals.append(float(np.real(pvec_density[0])))
        pz_vals.append(float(np.real(pvec_density[2])))

        _, Jx, Jy, Jz = coherent_current_density(
            X, Y, Z,
            ground_state=GROUND,
            excited_x=PX,
            excited_z=PZ,
            a=a, b_x=bx, b_z=bz,
            delta=delta,
            t=t_eval,
            h=GRAD_H,
        )

        if SAFE_ANGLE_LOOP:
            I_exact = exact_far_field_map_memory_safe(
                Jx,
                Jy,
                Jz,
                X,
                Y,
                Z,
                x,
                y,
                z,
                k=k,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )
        else:
            raise RuntimeError(
                "SAFE_ANGLE_LOOP=False is disabled in this notebook because the old batched path "
                "was causing the process to be killed on this machine."
            )

        I_dip = dipole_limit_map_from_vector_numpy(
            np.real_if_close(pvec_density),
            theta_grid,
            phi_grid,
        )

        max_map_dev = float(np.max(np.abs(I_exact - I_dip)))

        phi0_index = 0
        exact_cut = I_exact[:, phi0_index]
        dip_cut = I_dip[:, phi0_index]

        print(f"  p_density = ({pvec_density[0]:.6e}, {pvec_density[1]:.6e}, {pvec_density[2]:.6e})")
        print(f"  p_expected= ({pvec_expected[0]:.6e}, {pvec_expected[1]:.6e}, {pvec_expected[2]:.6e})")
        print(f"  max exact-vs-dipole map deviation = {max_map_dev:.6e}")

        summary_lines.append(f"delta = {delta:.6f} rad")
        summary_lines.append(
            f"  p_density = ({pvec_density[0]:.6e}, {pvec_density[1]:.6e}, {pvec_density[2]:.6e})"
        )
        summary_lines.append(
            f"  p_expected = ({pvec_expected[0]:.6e}, {pvec_expected[1]:.6e}, {pvec_expected[2]:.6e})"
        )
        summary_lines.append(f"  max exact-vs-dipole map deviation = {max_map_dev:.6e}")
        summary_lines.append("")

        frac = delta / np.pi
        delta_label = f"{frac:.2f}π"

        save_map(
            I_exact,
            theta_grid,
            phi_grid,
            title=f"Task 5A exact far-field map (delta={delta_label})",
            filename=f"task05A_exact_map_delta_{idx}.png",
        )

        save_map(
            I_dip,
            theta_grid,
            phi_grid,
            title=f"Task 5A dipole-limit map from p(t) (delta={delta_label})",
            filename=f"task05A_dipole_map_delta_{idx}.png",
        )

        save_polar_cut(
            theta_grid,
            exact_cut,
            dip_cut,
            delta_label=delta_label,
            filename=f"task05A_phi0_cut_delta_{idx}.png",
        )

    save_dipole_rotation_plot(
        px_vals,
        pz_vals,
        DELTA_LIST,
        filename="task05A_dipole_rotation.png",
    )

    summary_path = FIG_DIR / "task05A_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[Saved] {summary_path}")

    print("\nTask 5A complete.", flush=True)


if __name__ == "__main__":
    main()