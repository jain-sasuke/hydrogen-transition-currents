# Task 4A — General hydrogen state engine

## Goal

Replace the one-transition-only setup with a reusable hydrogen state engine for arbitrary bound states.

We want access to

$$
\psi_{n\ell m}(r,\theta,\phi)=R_{n\ell}(r)Y_{\ell m}(\theta,\phi)
$$

and also convenient real orbitals such as

- $1s$
- $2s$
- $2p_x$
- $2p_y$
- $2p_z$

---

## Why this matters

Without a general state engine, the project is trapped at the benchmark stage.

With this engine, we can later study:

- different hydrogen transitions
- scaling with $n,\ell,m$
- dipole selection rules numerically
- which transitions remain deep in the dipole regime
- where finite-source or multipole corrections first become visible

---

## Physics implemented

### Radial hydrogen function

In atomic units,

$$
R_{n\ell}(r)
=
\sqrt{
\left(\frac{2}{n}\right)^3
\frac{(n-\ell-1)!}{2n\,(n+\ell)!}
}
\,
e^{-r/n}
\left(\frac{2r}{n}\right)^\ell
L_{n-\ell-1}^{2\ell+1}\!\left(\frac{2r}{n}\right)
$$

### Angular part

$$
Y_{\ell m}(\theta,\phi)
$$

is implemented directly from associated Legendre functions.

### Real orbitals

For the $p$ states, we build real orbitals from the complex spherical basis:

$$
p_z \sim Y_{10},
\qquad
p_x \sim \frac{Y_{1,-1}-Y_{11}}{\sqrt2},
\qquad
p_y \sim \frac{i(Y_{1,-1}+Y_{11})}{\sqrt2}.
$$

---

## Numerical checks

### 1. Normalization
For each tested state,

$$
\int |\psi|^2 d^3r \approx 1.
$$

### 2. Orthogonality
For distinct states,

$$
\int \psi_a^*\psi_b\,d^3r \approx 0.
$$

### 3. Dipole selection rules
We numerically test the coordinate matrix element

$$
\langle a|\mathbf r|b\rangle.
$$

Expected:

- $1s\to 2s$: forbidden
- $1s\to 2p_z$: only $z$ component survives
- $1s\to 2p_x$: only $x$ component survives

---

## Deliverables

- `src/hydrogen/general_states.py`
- `src/hydrogen/general_integrals.py`
- `notebooks/task04A_general_hydrogen_state_engine.py`

---

## What comes after this

If Task 4A passes, the next serious task is:

### Task 4B
Scan families of transitions and compare their exact radiation pattern to the dipole-limit prediction.

That is where the benchmark phase ends and actual hydrogen-structure investigation begins.