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

from hydrogen.energies import hydrogen_energy_au, transition_omega_au
from hydrogen.transition_current import (
    cartesian_grid,
    psi_100_xyz,
    psi_210_xyz,
    grad_psi_100_xyz,
    grad_psi_210_xyz,
    equal_superposition_phase,
    charge_density_equal_superposition,
    oscillating_charge_density_equal_superposition,
    time_derivative_charge_density_equal_superposition,
    electric_current_density_equal_superposition,
    dipole_moment_from_charge_density,
)
from hydrogen.continuity import divergence_cartesian, continuity_residual


FIG_ROOT = PROJECT_ROOT / "figures"
FIG_DIR = FIG_ROOT / "task02"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Grid
XMAX = 14.0
NGRID = 101

# Time samples
N_TIME = 200


def save_scalar_slice(x, z, field_xz, title, filename, cmap="RdBu_r", vcenter=True):
    field_xz = np.real_if_close(np.asarray(field_xz, dtype=np.complex128)).astype(float)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    if vcenter:
        vmax = np.max(np.abs(field_xz))
        im = ax.pcolormesh(
            x, z, field_xz.T,
            shading="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )
    else:
        im = ax.pcolormesh(
            x, z, field_xz.T,
            shading="auto",
            cmap=cmap,
        )

    ax.set_xlabel("x [a.u.]")
    ax.set_ylabel("z [a.u.]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_quiver_slice(x, z, Jx_xz, Jz_xz, title, filename, stride=4):
    Jx_xz = np.real_if_close(np.asarray(Jx_xz, dtype=np.complex128)).astype(float)
    Jz_xz = np.real_if_close(np.asarray(Jz_xz, dtype=np.complex128)).astype(float)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    speed_full = np.sqrt(Jx_xz**2 + Jz_xz**2)
    im = ax.imshow(
        speed_full.T,
        origin="lower",
        extent=[x[0], x[-1], z[0], z[-1]],
        aspect="auto",
        cmap="viridis",
    )

    xx = x[::stride]
    zz = z[::stride]
    Xq, Zq = np.meshgrid(xx, zz, indexing="ij")

    U = Jx_xz[::stride, ::stride]
    V = Jz_xz[::stride, ::stride]

    ax.quiver(Xq, Zq, U, V, color="white")

    ax.set_xlabel("x [a.u.]")
    ax.set_ylabel("z [a.u.]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=r"$\sqrt{J_x^2 + J_z^2}$")
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_time_trace(t_arr, rho_trace, Jz_trace, omega, filename):
    rho_trace = np.real_if_close(np.asarray(rho_trace, dtype=np.complex128)).astype(float)
    Jz_trace = np.real_if_close(np.asarray(Jz_trace, dtype=np.complex128)).astype(float)

    rho_norm = rho_trace / max(np.max(np.abs(rho_trace)), 1e-30)
    Jz_norm = Jz_trace / max(np.max(np.abs(Jz_trace)), 1e-30)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(t_arr, rho_norm, label=r"$\rho_{\rm osc}$ (normalized)")
    ax.plot(t_arr, Jz_norm, label=r"$J_z$ (normalized)")

    ax.plot(t_arr, np.cos(omega * t_arr), "--", alpha=0.7, label=r"$\cos(\omega t)$")
    ax.plot(t_arr, np.sin(omega * t_arr), "--", alpha=0.7, label=r"$\sin(\omega t)$")

    ax.set_xlabel("t [a.u.]")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Time trace at a representative spatial point")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_dipole_trace(t_arr, pz_trace, exact_abs_d, filename):
    pz_trace = np.real_if_close(np.asarray(pz_trace, dtype=np.complex128)).astype(float)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(t_arr, pz_trace, label=r"$p_z(t)$ from charge density")
    ax.axhline(+exact_abs_d, color="gray", ls="--", label=r"$\pm |\langle 1s|z|2p_z\rangle|$")
    ax.axhline(-exact_abs_d, color="gray", ls="--")

    ax.set_xlabel("t [a.u.]")
    ax.set_ylabel(r"$p_z(t)$ [a.u.]")
    ax.set_title("Dipole moment extracted from charge density")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def write_summary(path, omega, t_quarter, pz0, exact_abs_d, continuity_metric):
    lines = []
    lines.append("Task 2 transition-current benchmark summary")
    lines.append("=" * 48)
    lines.append("")
    lines.append(f"Hydrogen energies (a.u.): E1 = {hydrogen_energy_au(1):.12e}, E2 = {hydrogen_energy_au(2):.12e}")
    lines.append(f"Transition frequency omega_21 = {omega:.12e} a.u.")
    lines.append(f"T/4 = {t_quarter:.12e} a.u.")
    lines.append("")
    lines.append(f"|p_z(t=0)| from charge density = {abs(pz0):.12e}")
    lines.append(f"|<1s|z|2p_z>| benchmark      = {exact_abs_d:.12e}")
    lines.append(f"Relative dipole-amplitude error = {abs(abs(pz0) - exact_abs_d) / exact_abs_d:.12e}")
    lines.append("")
    lines.append(f"Masked continuity residual metric (p99) = {continuity_metric:.12e}")
    lines.append("")
    lines.append("Expected phase relation:")
    lines.append("rho_osc ~ cos(omega t)")
    lines.append("J_osc   ~ sin(omega t)")
    path.write_text("\n".join(lines))
    print(f"[Saved] {path}")


def main():
    print("Starting Task 2 transition-current benchmark...", flush=True)

    x, y, z, X, Y, Z = cartesian_grid(xmax=XMAX, n=NGRID)
    y0_idx = len(y) // 2

    # Hydrogen states
    psi_1s = psi_100_xyz(X, Y, Z)
    psi_2pz = psi_210_xyz(X, Y, Z)

    grad_1s = grad_psi_100_xyz(X, Y, Z)
    grad_2pz = grad_psi_210_xyz(X, Y, Z)

    # Energies / phase
    omega_21 = transition_omega_au(1, 2)
    period = 2.0 * np.pi / omega_21
    t0 = 0.0
    t_quarter = period / 4.0
    t_half = period / 2.0

    print(f"omega_21 = {omega_21:.6e} a.u.")
    print(f"T/4      = {t_quarter:.6e} a.u.")

    # Oscillating charge density maps
    rho_osc_t0 = oscillating_charge_density_equal_superposition(
        psi_1s, psi_2pz, equal_superposition_phase(omega_21, t0)
    )
    rho_osc_thalf = oscillating_charge_density_equal_superposition(
        psi_1s, psi_2pz, equal_superposition_phase(omega_21, t_half)
    )

    rho_osc_t0_xz = rho_osc_t0[:, y0_idx, :]
    rho_osc_thalf_xz = rho_osc_thalf[:, y0_idx, :]

    save_scalar_slice(
        x, z, rho_osc_t0_xz,
        title=r"Oscillating charge density $\rho_{\rm osc}(x,z)$ at $t=0$",
        filename="task02_rho_osc_t0.png",
        cmap="RdBu_r",
        vcenter=True,
    )

    save_scalar_slice(
        x, z, rho_osc_thalf_xz,
        title=r"Oscillating charge density $\rho_{\rm osc}(x,z)$ at $t=T/2$",
        filename="task02_rho_osc_thalf.png",
        cmap="RdBu_r",
        vcenter=True,
    )

    # Current density at quarter period
    phase_quarter = equal_superposition_phase(omega_21, t_quarter)
    Jx, Jy, Jz = electric_current_density_equal_superposition(
        psi_1s, psi_2pz, grad_1s, grad_2pz, phase_quarter
    )

    Jx_xz = Jx[:, y0_idx, :]
    Jz_xz = Jz[:, y0_idx, :]

    save_quiver_slice(
        x, z, Jx_xz, Jz_xz,
        title=r"Transition current in the $x$-$z$ plane at $t=T/4$",
        filename="task02_current_quiver_tquarter.png",
        stride=4,
    )

    # Continuity residual at quarter period
    drho_dt = time_derivative_charge_density_equal_superposition(
        psi_1s, psi_2pz, omega_21, phase_quarter
    )
    divJ = divergence_cartesian(Jx, Jy, Jz, x, y, z)
    resid = continuity_residual(drho_dt, divJ)

    interior = resid[2:-2, 2:-2, 2:-2]
    signal = (np.abs(drho_dt) + np.abs(divJ))[2:-2, 2:-2, 2:-2]

    threshold = 1e-3 * np.max(signal)
    mask = signal > threshold
    masked_resid = interior[mask]

    max_resid_masked = float(np.max(masked_resid))
    p99_resid_masked = float(np.percentile(masked_resid, 99))
    mean_resid_masked = float(np.mean(masked_resid))

    print(f"Masked continuity residual max  : {max_resid_masked:.6e}")
    print(f"Masked continuity residual p99  : {p99_resid_masked:.6e}")
    print(f"Masked continuity residual mean : {mean_resid_masked:.6e}")

    resid_xz = resid[:, y0_idx, :]
    save_scalar_slice(
        x, z, resid_xz,
        title=r"Continuity residual in the $x$-$z$ plane at $t=T/4$",
        filename="task02_continuity_residual_tquarter.png",
        cmap="magma",
        vcenter=False,
    )

    # Time trace at representative point
    ix = int(np.argmin(np.abs(x - 0.0)))
    iy = int(np.argmin(np.abs(y - 0.0)))
    iz = int(np.argmin(np.abs(z - 2.0)))

    t_arr = np.linspace(0.0, period, N_TIME)
    rho_trace = []
    Jz_trace = []
    pz_trace = []

    for t in t_arr:
        phase = equal_superposition_phase(omega_21, t)

        rho_osc = oscillating_charge_density_equal_superposition(psi_1s, psi_2pz, phase)
        rho_full = charge_density_equal_superposition(psi_1s, psi_2pz, phase)
        Jx_t, Jy_t, Jz_t = electric_current_density_equal_superposition(
            psi_1s, psi_2pz, grad_1s, grad_2pz, phase
        )

        rho_trace.append(float(np.real_if_close(rho_osc[ix, iy, iz])))
        Jz_trace.append(float(np.real_if_close(Jz_t[ix, iy, iz])))

        p_vec = dipole_moment_from_charge_density(rho_full, X, Y, Z, x, y, z)
        pz_trace.append(float(np.real_if_close(p_vec[2])))

    rho_trace = np.array(rho_trace)
    Jz_trace = np.array(Jz_trace)
    pz_trace = np.array(pz_trace)

    save_time_trace(
        t_arr, rho_trace, Jz_trace, omega_21,
        filename="task02_time_trace_point.png"
    )

    # Dipole amplitude benchmark from Task 1 exact result
    exact_abs_d = 128.0 * np.sqrt(2.0) / 243.0
    save_dipole_trace(
        t_arr, pz_trace, exact_abs_d,
        filename="task02_dipole_time_trace.png"
    )

    p0 = pz_trace[0]
    print(f"|p_z(t=0)| from charge density = {abs(p0):.6e}")
    print(f"|<1s|z|2p_z>| benchmark      = {exact_abs_d:.6e}")
    print(f"Relative dipole-amplitude error = {abs(abs(p0) - exact_abs_d) / exact_abs_d:.6e}")

    write_summary(
        FIG_DIR / "task02_summary.txt",
        omega_21, t_quarter, p0, exact_abs_d, p99_resid_masked
    )

    print("\nTask 2 complete.", flush=True)


if __name__ == "__main__":
    main()