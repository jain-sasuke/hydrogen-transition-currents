from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .general_states import radial_hydrogen, complex_spherical_harmonic


def make_angular_mesh(theta: np.ndarray, phi: np.ndarray):
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return TH, PH


def factorized_state(spec, r: np.ndarray, TH: np.ndarray, PH: np.ndarray):
    """
    Return hydrogen state in factorized form:
        psi(r,theta,phi) = R(r) * Y(theta,phi)

    Supported specs:
      - tuple (n,l,m)
      - strings like '1s', '2s', '2pz', '2px', '2py'
    """
    if isinstance(spec, tuple):
        if len(spec) != 3:
            raise ValueError("Tuple state must be (n,l,m)")
        n, l, m = spec
        R = radial_hydrogen(n, l, r)
        Y = complex_spherical_harmonic(l, m, TH, PH)
        return R, Y

    if not isinstance(spec, str):
        raise TypeError("State spec must be a string or tuple (n,l,m)")

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

    raise ValueError(f"Unsupported state spec: {spec}")


def radial_integral(f: np.ndarray, r: np.ndarray) -> complex:
    return simpson(f, x=r)


def angular_integral(f: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> complex:
    out_phi = simpson(f, x=phi, axis=1)
    out_theta = simpson(out_phi, x=theta, axis=0)
    return out_theta


def spherical_overlap(spec_a, spec_b, r, theta, phi):
    TH, PH = make_angular_mesh(theta, phi)
    Ra, Ya = factorized_state(spec_a, r, TH, PH)
    Rb, Yb = factorized_state(spec_b, r, TH, PH)

    radial_part = radial_integral(np.conjugate(Ra) * Rb * r**2, r)
    angular_part = angular_integral(np.conjugate(Ya) * Yb * np.sin(TH), theta, phi)
    return radial_part * angular_part


def _angular_operator(axis: str, TH: np.ndarray, PH: np.ndarray):
    axis = axis.lower()
    if axis == "x":
        return np.sin(TH) * np.cos(PH)
    if axis == "y":
        return np.sin(TH) * np.sin(PH)
    if axis == "z":
        return np.cos(TH)
    raise ValueError("axis must be one of 'x', 'y', 'z'")


def spherical_dipole_matrix_element(spec_a, spec_b, axis: str, r, theta, phi):
    """
    Compute <a|r_axis|b> using spherical quadrature:

        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cos(theta)
    """
    TH, PH = make_angular_mesh(theta, phi)
    Ra, Ya = factorized_state(spec_a, r, TH, PH)
    Rb, Yb = factorized_state(spec_b, r, TH, PH)

    radial_part = radial_integral(np.conjugate(Ra) * Rb * r**3, r)

    op_ang = _angular_operator(axis, TH, PH)
    angular_part = angular_integral(
        np.conjugate(Ya) * op_ang * Yb * np.sin(TH),
        theta,
        phi,
    )

    return radial_part * angular_part


def spherical_dipole_vector(spec_a, spec_b, r, theta, phi):
    dx = spherical_dipole_matrix_element(spec_a, spec_b, "x", r, theta, phi)
    dy = spherical_dipole_matrix_element(spec_a, spec_b, "y", r, theta, phi)
    dz = spherical_dipole_matrix_element(spec_a, spec_b, "z", r, theta, phi)
    return np.array([dx, dy, dz], dtype=complex)