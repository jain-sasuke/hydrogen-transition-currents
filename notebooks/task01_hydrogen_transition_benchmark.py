from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Task 1 benchmark: hydrogen 1s <-> 2p_z
# Self-contained, publication-grade benchmark script
# Atomic units are used: a0 = 1, e = 1
# ============================================================

PI = np.pi
SQRT2 = np.sqrt(2.0)

# -------- exact hydrogen orbitals in atomic units --------
# Convention: real orbitals only, for benchmark states
# psi_100 = 1s
# psi_200 = 2s
# psi_210 = 2p_z

def psi_100(r, theta, phi):
    return np.exp(-r) / np.sqrt(PI)

def psi_200(r, theta, phi):
    return (2.0 - r) * np.exp(-r / 2.0) / (4.0 * np.sqrt(2.0 * PI))

def psi_210(r, theta, phi):
    return r * np.exp(-r / 2.0) * np.cos(theta) / (4.0 * np.sqrt(2.0 * PI))

# -------- exact analytic benchmark --------
def exact_abs_d_1s_2pz():
    # |<1s|z|2p_z>| = 128 sqrt(2) / 243 in atomic units
    return 128.0 * np.sqrt(2.0) / 243.0

# -------- numerical integration helper --------
def integrate_3d(integrand, r, theta, phi):
    """
    Integrate array of shape (nr, ntheta, nphi) over spherical coordinates.
    Uses:
      - periodic rectangle sum in phi
      - trapezoid in theta and r
    """
    dphi = phi[1] - phi[0]
    out_phi = np.sum(integrand, axis=2) * dphi
    out_theta = np.trapezoid(out_phi, theta, axis=1)
    out_r = np.trapezoid(out_theta, r, axis=0)
    return out_r

def dipole_vector(psi_a, psi_b, r, theta, phi):
    """
    d = - ∫ psi_a^* r psi_b d^3r
    Atomic units, with electron charge included as -1.
    """
    R, TH, PH = np.meshgrid(r, theta, phi, indexing="ij")

    x = R * np.sin(TH) * np.cos(PH)
    y = R * np.sin(TH) * np.sin(PH)
    z = R * np.cos(TH)

    jac = R**2 * np.sin(TH)
    overlap = np.conjugate(psi_a(R, TH, PH)) * psi_b(R, TH, PH)

    dx = -integrate_3d(overlap * x * jac, r, theta, phi)
    dy = -integrate_3d(overlap * y * jac, r, theta, phi)
    dz = -integrate_3d(overlap * z * jac, r, theta, phi)

    return np.array([complex(dx), complex(dy), complex(dz)])

# -------- radiation pattern from dipole --------
def far_field_intensity_from_dipole(dvec, theta, phi):
    """
    I(theta,phi) ∝ |n x (n x d)|^2 = |d|^2 - |n·d|^2
    """
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    nx = np.sin(TH) * np.cos(PH)
    ny = np.sin(TH) * np.sin(PH)
    nz = np.cos(TH)

    n_dot_d = nx * dvec[0] + ny * dvec[1] + nz * dvec[2]
    d2 = np.sum(np.abs(dvec) ** 2)

    intensity = d2 - np.abs(n_dot_d) ** 2
    intensity = np.real_if_close(intensity)
    intensity = np.clip(intensity, 0.0, None)
    return intensity

def normalized_profile_at_phi0(dvec, theta):
    phi0 = np.array([0.0])
    I = far_field_intensity_from_dipole(dvec, theta, phi0)[:, 0]
    I /= np.max(I)
    return I

# -------- plotting helpers --------
def save_dipole_component_plot(dvec, outpath, title, ylog=False):
    labels = ["dx", "dy", "dz"]
    vals = np.abs(dvec)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel("Absolute value")

    if ylog:
        floor = 1e-22
        vals_safe = np.maximum(vals, floor)
        ax.cla()
        ax.bar(labels, vals_safe)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("Absolute value (log scale)")

    for i, v in enumerate(vals):
        ax.text(i, max(v, 1e-22) * (1.08 if ylog else 1.01), f"{v:.2e}",
                ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_angular_profile_plot(theta, profile_num, profile_exact, exact_d, num_d, forbidden_ratio, outpath):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(theta, profile_num, label="numerical", linewidth=2.5)
    ax.plot(theta, profile_exact, "--", label=r"analytic $\sin^2\theta$", linewidth=2.5)

    text = (
        rf"$|\langle 1s|z|2p_z\rangle|_{{\rm exact}} = {exact_d:.7f}$" "\n"
        rf"$|\langle 1s|z|2p_z\rangle|_{{\rm num}} = {num_d:.7f}$" "\n"
        rf"relative error $= {abs(num_d - exact_d)/exact_d:.3e}$" "\n"
        rf"$|d_{{1s,2s}}|/|d_{{1s,2p_z}}| = {forbidden_ratio:.3e}$"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.set_title(r"Hydrogen benchmark: $1s \leftrightarrow 2p_z$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_far_field_map(theta, phi, intensity, outpath):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(phi, theta, intensity, shading="auto")
    ax.set_title(r"Far-field map for $1s \leftrightarrow 2p_z$")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized intensity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def write_summary(summary_path, d_1s_2pz, d_1s_2s, exact_d, max_profile_error):
    abs_num = abs(d_1s_2pz[2])
    abs_forbidden = np.linalg.norm(d_1s_2s)
    forbidden_ratio = abs_forbidden / abs_num

    lines = []
    lines.append("Task 1 hydrogen benchmark summary")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Allowed transition: 1s <-> 2p_z")
    lines.append(f"dx = {d_1s_2pz[0]:.12e}")
    lines.append(f"dy = {d_1s_2pz[1]:.12e}")
    lines.append(f"dz = {d_1s_2pz[2]:.12e}")
    lines.append("")
    lines.append("Forbidden transition: 1s <-> 2s")
    lines.append(f"dx = {d_1s_2s[0]:.12e}")
    lines.append(f"dy = {d_1s_2s[1]:.12e}")
    lines.append(f"dz = {d_1s_2s[2]:.12e}")
    lines.append("")
    lines.append(f"|<1s|z|2p_z>| exact    = {exact_d:.12e}")
    lines.append(f"|<1s|z|2p_z>| numerical = {abs_num:.12e}")
    lines.append(f"relative error          = {abs(abs_num - exact_d)/exact_d:.12e}")
    lines.append(f"|d_1s,2s| / |d_1s,2p_z|  = {forbidden_ratio:.12e}")
    lines.append(f"max angular-profile error vs sin^2(theta) = {max_profile_error:.12e}")
    lines.append("")
    lines.append("Pass/fail gates")
    lines.append(f"Gate A (dx,d y << dz for 1s<->2p_z): "
                 f"{'PASS' if (abs(d_1s_2pz[0])/abs_num < 1e-10 and abs(d_1s_2pz[1])/abs_num < 1e-10) else 'FAIL'}")
    lines.append(f"Gate B (forbidden ratio < 1e-10): "
                 f"{'PASS' if forbidden_ratio < 1e-10 else 'FAIL'}")
    lines.append(f"Gate C (profile error < 1e-10): "
                 f"{'PASS' if max_profile_error < 1e-10 else 'FAIL'}")

    summary_path.write_text("\n".join(lines))

def main():
    project_root = Path(__file__).resolve().parents[1]
    figures_root = project_root / "figures"
    figures_dir = figures_root / "task01"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Task 1 hydrogen transition benchmark...")

    # -----------------------------
    # Integration grid
    # -----------------------------
    # Chosen to be accurate while still running comfortably on laptop CPU.
    r_max = 40.0
    nr = 260
    ntheta = 181
    nphi = 256

    r = np.linspace(0.0, r_max, nr)
    theta_int = np.linspace(0.0, PI, ntheta)
    phi_int = np.linspace(0.0, 2.0 * PI, nphi, endpoint=False)

    print("Computing dipole vector for 1s <-> 2p_z ...")
    d_1s_2pz = dipole_vector(psi_100, psi_210, r, theta_int, phi_int)

    print("Computing dipole vector for 1s <-> 2s ...")
    d_1s_2s = dipole_vector(psi_100, psi_200, r, theta_int, phi_int)

    exact_d = exact_abs_d_1s_2pz()
    num_d = abs(d_1s_2pz[2])
    forbidden_ratio = np.linalg.norm(d_1s_2s) / num_d

    print("\nDipole vector results:")
    print(f"1s <-> 2p_z : dx={d_1s_2pz[0]:.6e}, dy={d_1s_2pz[1]:.6e}, dz={d_1s_2pz[2]:.6e}")
    print(f"1s <-> 2s   : dx={d_1s_2s[0]:.6e}, dy={d_1s_2s[1]:.6e}, dz={d_1s_2s[2]:.6e}")

    print("\nDerived checks:")
    print(f"|dx|/|dz| for 1s <-> 2p_z = {abs(d_1s_2pz[0])/num_d:.6e}")
    print(f"|dy|/|dz| for 1s <-> 2p_z = {abs(d_1s_2pz[1])/num_d:.6e}")
    print(f"|d_1s,2s| / |d_1s,2p_z|   = {forbidden_ratio:.6e}")
    print(f"|<1s|z|2p_z>| exact        = {exact_d:.6e}")
    print(f"|<1s|z|2p_z>| numerical    = {num_d:.6e}")
    print(f"relative error             = {abs(num_d - exact_d)/exact_d:.6e}")

    # -----------------------------
    # Angular profile
    # -----------------------------
    theta_plot = np.linspace(0.0, PI, 400)
    profile_num = normalized_profile_at_phi0(d_1s_2pz, theta_plot)
    profile_exact = np.sin(theta_plot) ** 2
    profile_exact /= np.max(profile_exact)

    max_profile_error = np.max(np.abs(profile_num - profile_exact))
    print(f"\nMax profile error against sin^2(theta): {max_profile_error:.6e}")

    # -----------------------------
    # Far-field map
    # -----------------------------
    theta_map = np.linspace(0.0, PI, 240)
    phi_map = np.linspace(0.0, 2.0 * PI, 360)
    intensity_map = far_field_intensity_from_dipole(d_1s_2pz, theta_map, phi_map)
    intensity_map /= np.max(intensity_map)

    # -----------------------------
    # Save figures
    # -----------------------------
    f1 = figures_dir / "task01_dipole_components_1s_2pz.png"
    f2 = figures_dir / "task01_forbidden_transition_components_1s_2s.png"
    f3 = figures_dir / "task01_angular_profile.png"
    f4 = figures_dir / "task01_sphere_map.png"
    f5 = figures_dir / "task01_summary.txt"

    save_dipole_component_plot(
        d_1s_2pz,
        f1,
        r"$1s \leftrightarrow 2p_z$ dipole components"
    )

    save_dipole_component_plot(
        d_1s_2s,
        f2,
        r"$1s \leftrightarrow 2s$ forbidden-transition check",
        ylog=True
    )

    save_angular_profile_plot(
        theta_plot,
        profile_num,
        profile_exact,
        exact_d,
        num_d,
        forbidden_ratio,
        f3
    )

    save_far_field_map(
        theta_map,
        phi_map,
        intensity_map,
        f4
    )

    write_summary(
        f5,
        d_1s_2pz,
        d_1s_2s,
        exact_d,
        max_profile_error
    )

    print(f"[Saved] {f1}")
    print(f"[Saved] {f2}")
    print(f"[Saved] {f3}")
    print(f"[Saved] {f4}")
    print(f"[Saved] {f5}")
    print("\nTask 1 complete.")

if __name__ == "__main__":
    main()