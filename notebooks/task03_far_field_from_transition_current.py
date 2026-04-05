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

from hydrogen.energies import transition_omega_au
from hydrogen.transition_current import (
    cartesian_grid,
    psi_100_xyz,
    psi_210_xyz,
    grad_psi_100_xyz,
    grad_psi_210_xyz,
    electric_current_density_equal_superposition,
)
from hydrogen.radiation import (
    wave_number_au,
    integrated_current_vector,
    far_field_intensity_from_current,
    dipole_pattern_z,
)

FIG_ROOT = PROJECT_ROOT / "figures"
FIG_DIR = FIG_ROOT / "task03"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Grid for source integration
XMAX = 14.0
NGRID = 81

# Angular resolution
NTHETA = 181
NPHI_CHECK = 13


def save_profile(theta, profile_exact, profile_dip, filename):
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(theta, profile_exact, lw=2.5, label="exact current-source far field")
    ax.plot(theta, profile_dip, "--", lw=2.5, label=r"dipole limit $\sin^2\theta$")

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(r"Task 3: exact far-field profile vs dipole limit")
    ax.legend()
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_deviation(theta, rel_dev, filename):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(theta, rel_dev, lw=2.2)
    ax.axhline(0.0, color="gray", ls="--", lw=1.3)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$I_{\rm exact}/I_{\rm dip} - 1$")
    ax.set_title("Task 3: relative deviation from dipole-limit profile")
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_phi_check(phi_vals, intensities, filename):
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(phi_vals, intensities, marker="o", lw=2.0)

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel("Normalized intensity at $\\theta=\\pi/2$")
    ax.set_title("Task 3: azimuthal invariance check at the equator")
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def save_integrated_current_check(Jint_z, expected_z, filename):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    labels = [r"$\int J_z\,d^3r$", r"$\omega\,|\langle 1s|z|2p_z\rangle|$"]
    values = [Jint_z, expected_z]

    ax.bar(labels, values)
    ax.set_ylabel("Value [a.u.]")
    ax.set_title("Task 3: integrated-current benchmark")
    fig.tight_layout()

    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {path}")


def write_summary(
    path: Path,
    omega: float,
    k: float,
    Jint_z: float,
    expected_z: float,
    rel_current_error: float,
    max_profile_deviation: float,
    equatorial_phi_spread: float,
):
    lines = []
    lines.append("Task 3 far-field from actual transition current")
    lines.append("=" * 52)
    lines.append("")
    lines.append(f"omega_21 [a.u.] = {omega:.12e}")
    lines.append(f"k = omega/c [a.u.] = {k:.12e}")
    lines.append("")
    lines.append("Integrated-current benchmark:")
    lines.append(f"Integral J_z d^3r                 = {Jint_z:.12e}")
    lines.append(f"Expected omega * |<1s|z|2p_z>|    = {expected_z:.12e}")
    lines.append(f"Relative error                    = {rel_current_error:.12e}")
    lines.append("")
    lines.append("Angular-pattern checks:")
    lines.append(f"Max |I_exact_norm - I_dip_norm|   = {max_profile_deviation:.12e}")
    lines.append(f"Equatorial phi spread fraction    = {equatorial_phi_spread:.12e}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- If the profile matches sin^2(theta), hydrogen is in the dipole regime.")
    lines.append("- If phi variation is negligible, the z-polarized source is axisymmetric.")
    path.write_text("\n".join(lines))
    print(f"[Saved] {path}")


def main():
    print("Starting Task 3 far-field from actual transition current...", flush=True)
    print(f"Saving figures to: {FIG_DIR}")

    # Hydrogen transition frequency
    omega = transition_omega_au(1, 2)
    k = wave_number_au(omega)

    print(f"omega_21 = {omega:.6e} a.u.")
    print(f"k        = {k:.6e} a.u.")

    # Exact dipole benchmark magnitude from Task 1
    d_exact = 128.0 * np.sqrt(2.0) / 243.0

    # Spatial grid and orbitals
    x, y, z, X, Y, Z = cartesian_grid(xmax=XMAX, n=NGRID)

    psi_1s = psi_100_xyz(X, Y, Z)
    psi_2pz = psi_210_xyz(X, Y, Z)

    grad_1s = grad_psi_100_xyz(X, Y, Z)
    grad_2pz = grad_psi_210_xyz(X, Y, Z)

    # Current amplitude:
    # using phase = pi/2 gives sin(phase)=1, i.e. the full current amplitude.
    Jx_amp, Jy_amp, Jz_amp = electric_current_density_equal_superposition(
        psi_1s, psi_2pz, grad_1s, grad_2pz, phase=np.pi / 2.0
    )

    # Benchmark from continuity / dipole relation:
    # dp_z/dt = ∫ J_z d^3r, and for p_z(t) = -d cos(omega t),
    # the amplitude at quarter phase is +omega d.
    Jint = integrated_current_vector(Jx_amp, Jy_amp, Jz_amp, x, y, z)
    Jint_z = float(np.real_if_close(Jint[2]))
    expected_z = float(omega * d_exact)
    rel_current_error = abs(Jint_z - expected_z) / abs(expected_z)

    print("\nIntegrated-current benchmark:")
    print(f"Integral J_z d^3r              = {Jint_z:.6e}")
    print(f"Expected omega * |<1s|z|2p_z>| = {expected_z:.6e}")
    print(f"Relative current error         = {rel_current_error:.6e}")

    save_integrated_current_check(
        Jint_z, expected_z, "task03_integrated_current_check.png"
    )

    # Angular profile in the x-z plane (phi = 0)
    theta_grid = np.linspace(0.0, np.pi, NTHETA)
    phi0 = 0.0

    I_exact = []
    for theta in theta_grid:
        I_val = far_field_intensity_from_current(
            Jx_amp, Jy_amp, Jz_amp,
            X, Y, Z, x, y, z,
            k=k,
            theta=float(theta),
            phi=phi0,
        )
        I_exact.append(I_val)

    I_exact = np.array(I_exact, dtype=float)
    I_exact_norm = I_exact / np.max(I_exact)

    I_dip = dipole_pattern_z(theta_grid)
    I_dip_norm = I_dip / np.max(I_dip)

    max_profile_deviation = float(np.max(np.abs(I_exact_norm - I_dip_norm)))

    print("\nAngular-profile benchmark:")
    print(f"Max normalized-profile deviation from sin^2(theta) = {max_profile_deviation:.6e}")

    save_profile(
        theta_grid,
        I_exact_norm,
        I_dip_norm,
        "task03_profile_vs_dipole.png",
    )

    # Relative deviation away from the poles
    pole_mask = I_dip_norm > 1e-6
    rel_dev = np.zeros_like(I_exact_norm)
    rel_dev[pole_mask] = I_exact_norm[pole_mask] / I_dip_norm[pole_mask] - 1.0

    save_deviation(
        theta_grid[pole_mask],
        rel_dev[pole_mask],
        "task03_relative_deviation.png",
    )

    # Azimuthal invariance check at equator
    theta_eq = np.pi / 2.0
    phi_grid = np.linspace(0.0, 2.0 * np.pi, NPHI_CHECK)

    I_phi = []
    for phi in phi_grid:
        I_val = far_field_intensity_from_current(
            Jx_amp, Jy_amp, Jz_amp,
            X, Y, Z, x, y, z,
            k=k,
            theta=theta_eq,
            phi=float(phi),
        )
        I_phi.append(I_val)

    I_phi = np.array(I_phi, dtype=float)
    I_phi_norm = I_phi / np.max(I_phi)
    equatorial_phi_spread = float((np.max(I_phi_norm) - np.min(I_phi_norm)) / np.mean(I_phi_norm))

    print("\nAzimuthal-invariance benchmark:")
    print(f"Equatorial phi spread fraction = {equatorial_phi_spread:.6e}")

    save_phi_check(
        phi_grid,
        I_phi_norm,
        "task03_phi_invariance_check.png",
    )

    # Summary
    write_summary(
        FIG_DIR / "task03_summary.txt",
        omega=omega,
        k=k,
        Jint_z=Jint_z,
        expected_z=expected_z,
        rel_current_error=rel_current_error,
        max_profile_deviation=max_profile_deviation,
        equatorial_phi_spread=equatorial_phi_spread,
    )

    print("\nTask 3 complete.", flush=True)


if __name__ == "__main__":
    main()