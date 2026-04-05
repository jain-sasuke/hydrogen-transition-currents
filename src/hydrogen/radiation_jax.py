from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def _make_nhat(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """
    Return observation directions n_hat with shape (..., 3)
    """
    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    cp = jnp.cos(phi)
    sp = jnp.sin(phi)
    return jnp.stack([st * cp, st * sp, ct], axis=-1)


@jax.jit
def far_field_intensity_batch_from_current(
    Jx: jnp.ndarray,
    Jy: jnp.ndarray,
    Jz: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    dV: float,
    k: float,
    theta_grid: jnp.ndarray,
    phi_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute normalized far-field intensity map I(theta, phi) from a source current.

    Inputs:
      Jx,Jy,Jz : shape (Nx,Ny,Nz)
      X,Y,Z    : shape (Nx,Ny,Nz)
      theta_grid : shape (Nt,)
      phi_grid   : shape (Np,)

    Output:
      intensity : shape (Nt,Np)
    """
    TH, PH = jnp.meshgrid(theta_grid, phi_grid, indexing="ij")   # (Nt,Np)
    nhat = _make_nhat(TH, PH)                                    # (Nt,Np,3)

    # phase = exp(-i k n·r)
    ndotr = (
        nhat[..., 0, None, None, None] * X[None, None, :, :, :]
        + nhat[..., 1, None, None, None] * Y[None, None, :, :, :]
        + nhat[..., 2, None, None, None] * Z[None, None, :, :, :]
    )
    phase = jnp.exp(-1j * k * ndotr)                             # (Nt,Np,Nx,Ny,Nz)

    Jvec = jnp.stack([Jx, Jy, Jz], axis=0)                       # (3,Nx,Ny,Nz)

    # Fourier transform of current:
    Jkx = jnp.sum(Jx[None, None, :, :, :] * phase, axis=(2, 3, 4)) * dV
    Jky = jnp.sum(Jy[None, None, :, :, :] * phase, axis=(2, 3, 4)) * dV
    Jkz = jnp.sum(Jz[None, None, :, :, :] * phase, axis=(2, 3, 4)) * dV
    Jk = jnp.stack([Jkx, Jky, Jkz], axis=-1)                     # (Nt,Np,3)

    # E ~ n x (n x Jk) = n (n·Jk) - Jk
    ndotJ = jnp.sum(nhat * Jk, axis=-1, keepdims=True)           # (Nt,Np,1)
    E = nhat * ndotJ - Jk                                         # (Nt,Np,3)

    intensity = jnp.real(jnp.sum(E * jnp.conjugate(E), axis=-1)) # (Nt,Np)

    maxval = jnp.max(intensity)
    intensity = jnp.where(maxval > 0.0, intensity / maxval, intensity)
    return intensity


def far_field_intensity_batch_from_current_numpy(
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    dV: float,
    k: float,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
) -> np.ndarray:
    """
    NumPy wrapper around the jitted JAX kernel.
    """
    out = far_field_intensity_batch_from_current(
        jnp.asarray(Jx),
        jnp.asarray(Jy),
        jnp.asarray(Jz),
        jnp.asarray(X),
        jnp.asarray(Y),
        jnp.asarray(Z),
        float(dV),
        float(k),
        jnp.asarray(theta_grid),
        jnp.asarray(phi_grid),
    )
    return np.asarray(out)


def dipole_limit_map_from_vector_numpy(
    pvec: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
) -> np.ndarray:
    """
    Fast vectorized dipole-limit map from instantaneous dipole vector p.
    """
    theta_grid = np.asarray(theta_grid)
    phi_grid = np.asarray(phi_grid)
    px, py, pz = np.asarray(pvec, dtype=float)

    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    st = np.sin(TH)
    ct = np.cos(TH)
    cp = np.cos(PH)
    sp = np.sin(PH)

    nx = st * cp
    ny = st * sp
    nz = ct

    ndotp = nx * px + ny * py + nz * pz
    p2 = px * px + py * py + pz * pz
    out = p2 - ndotp * ndotp

    m = np.max(out)
    if m > 0.0:
        out = out / m
    return out