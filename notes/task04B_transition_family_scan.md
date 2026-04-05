# Task 4B — Hydrogen transition-family scan

## Goal

Use the validated general hydrogen state engine to scan a family of allowed hydrogen transitions and compare

$$
I_{\rm exact}(\theta)
$$

computed from the actual transition current against the dipole-limit prediction

$$
I_{\rm dip}(\theta)\propto \sin^2\theta.
$$

---

## Transition family

This first scan is restricted to real, axisymmetric $s \to p_z$ transitions:

- $1s \to 2p_z$
- $1s \to 3p_z$
- $1s \to 4p_z$
- $2s \to 3p_z$
- $2s \to 4p_z$
- $3s \to 4p_z$

This keeps the symmetry clean and makes interpretation straightforward.

---

## Physics

For a real equal-weight superposition of two bound states,

$$
\Psi(\mathbf r,t)=\frac{1}{\sqrt2}
\left[
\psi_a(\mathbf r)e^{-iE_a t}
+
\psi_b(\mathbf r)e^{-iE_b t}
\right],
$$

the current density has the form

$$
\mathbf J(\mathbf r,t)=\sin(\omega t)\,\mathbf J_{\rm amp}(\mathbf r),
$$

with

$$
\mathbf J_{\rm amp}(\mathbf r)
=
\frac12\left[\psi_a\nabla\psi_b-\psi_b\nabla\psi_a\right].
$$

The exact far-field electric amplitude is computed from the spatial Fourier transform of this current:

$$
\mathbf E(\hat{\mathbf n})
\propto
\hat{\mathbf n}\times
\left[
\hat{\mathbf n}\times
\widetilde{\mathbf J}(k\hat{\mathbf n})
\right].
$$

The radiated intensity is then

$$
I(\hat{\mathbf n})\propto |\mathbf E(\hat{\mathbf n})|^2.
$$

---

## Why this task matters

Task 3 proved that the benchmark transition $1s \leftrightarrow 2p_z$ is deep in the dipole regime.

Task 4B asks a harder question:

> Is that just one lucky case, or is the dipole regime effectively universal across a family of ordinary hydrogen transitions?

That is a real physics question, even if the answer turns out to be “yes, still dipolar.”

---

## Deliverables

1. Maximum profile deviation from $\sin^2\theta$ for each transition
2. Equatorial $\phi$-spread for each transition
3. Integrated-current benchmark error for each transition
4. Exact-vs-dipole profile for the worst-case transition in the scan

---

## Pass-fail logic

### Gate A — integrated-current consistency
For each transition, the amplitude should satisfy

$$
\int J_z\,d^3r \approx \omega\,|d_z|.
$$

### Gate B — dipole-profile consistency
If hydrogen remains deep in the dipole regime, then the normalized exact profile should stay close to

$$
\sin^2\theta.
$$

### Gate C — azimuthal symmetry
For $p_z$ transitions, the radiation should remain axisymmetric, so the $\phi$-spread at $\theta=\pi/2$ should remain tiny.

---

## Interpretation

If all scanned transitions remain very close to the dipole limit, then this task shows that low-lying hydrogen transitions are robustly in the long-wavelength regime.

That is not exotic new physics.
It is a family-level structural result that tells us where the ordinary regime ends and where more ambitious ideas would need to begin.