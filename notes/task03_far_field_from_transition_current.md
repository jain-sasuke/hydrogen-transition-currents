# Task 3 — Far field from the actual hydrogen transition current

## Goal

Use the **actual real-space transition current** for the hydrogen superposition

$$
\Psi(\mathbf r,t)=\frac{1}{\sqrt2}
\left[
\psi_{1s}(\mathbf r)e^{-iE_1 t}
+
\psi_{2p_z}(\mathbf r)e^{-iE_2 t}
\right]
$$

to compute the radiated far-field angular pattern, and compare it against the dipole-limit result

$$
I_{\rm dip}(\theta)\propto \sin^2\theta.
$$

---

## Physics

From Task 2, the current density for real orbitals is

$$
\mathbf J(\mathbf r,t)
=
\frac12 \sin(\omega t)
\left[
\psi_{1s}\nabla\psi_{2p_z}-\psi_{2p_z}\nabla\psi_{1s}
\right].
$$

Define the current amplitude field

$$
\mathbf J_{\rm amp}(\mathbf r)
=
\frac12
\left[
\psi_{1s}\nabla\psi_{2p_z}-\psi_{2p_z}\nabla\psi_{1s}
\right].
$$

Then the far-field electric amplitude in direction $\hat{\mathbf n}$ is, up to an overall prefactor,

$$
\mathbf E(\hat{\mathbf n})
\propto
\hat{\mathbf n}\times
\left[
\hat{\mathbf n}\times
\widetilde{\mathbf J}(k\hat{\mathbf n})
\right],
$$

where

$$
\widetilde{\mathbf J}(k\hat{\mathbf n})
=
\int d^3r\,
\mathbf J_{\rm amp}(\mathbf r)\,
e^{-ik\hat{\mathbf n}\cdot\mathbf r}.
$$

The intensity is

$$
I(\hat{\mathbf n})\propto |\mathbf E(\hat{\mathbf n})|^2.
$$

---

## Dipole limit

For a localized source with

$$
ka \ll 1,
$$

we may replace

$$
e^{-ik\hat{\mathbf n}\cdot\mathbf r}\approx 1,
$$

so the radiation pattern reduces to the standard dipole form.

For a dipole oriented along $z$,

$$
I_{\rm dip}(\theta)\propto \sin^2\theta.
$$

For hydrogen $1s\leftrightarrow 2p_z$, this long-wavelength approximation should be excellent because

$$
k = \frac{\omega}{c}
$$

is very small in atomic units.

---

## Extra benchmark: integrated current

Continuity implies

$$
\frac{d\mathbf p}{dt}=\int d^3r\,\mathbf J(\mathbf r,t).
$$

For the transition dipole moment

$$
p_z(t)=-d_{1s,2p_z}\cos(\omega t),
$$

the current amplitude must satisfy

$$
\int d^3r\,J_{z,\rm amp}(\mathbf r)=\omega\,|d_{1s,2p_z}|.
$$

This is an important self-consistency check.

---

## Deliverables

1. Exact far-field profile vs $\sin^2\theta$
2. Relative deviation from dipole-limit profile
3. Azimuthal invariance check at $\theta=\pi/2$
4. Integrated-current benchmark

---

## Pass-fail gates

### Gate A — current normalization
The integrated current should match

$$
\omega\,|\langle 1s|z|2p_z\rangle|
$$

well.

### Gate B — angular profile
The normalized exact profile should closely follow

$$
\sin^2\theta.
$$

### Gate C — azimuthal symmetry
At fixed $\theta=\pi/2$, the intensity should be essentially independent of $\phi$.

---

## Interpretation

If Task 3 passes, then you have shown that:

- the actual hydrogen transition current reproduces the dipole pattern,
- the atom is deeply in the long-wavelength regime,
- the source-current formulation is consistent with the standard radiation result.

This does **not** yet produce new physics.
It establishes the exact benchmark that later structured or extended-source models must be compared against.