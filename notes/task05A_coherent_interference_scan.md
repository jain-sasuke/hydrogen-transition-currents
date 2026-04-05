# Task 5A — coherent two-channel interference scan

## Goal

Move beyond single-channel hydrogen transitions and study a coherent mixed state

$$
\Psi(\mathbf r,t)
=
a\,\psi_{1s}e^{-iE_1 t}
+
b_x\,\psi_{2p_x}e^{-iE_2 t}
+
b_z e^{i\delta}\psi_{2p_z}e^{-iE_2 t}.
$$

This introduces two allowed transition channels with a controllable relative phase:
- $1s \leftrightarrow 2p_x$
- $1s \leftrightarrow 2p_z$

The question is whether coherent interference rotates or reshapes the radiation pattern.

## Why this matters

All previous scans showed that ordinary single-channel $s\to p_z$ transitions remain deep in the dipole regime and stay axisymmetric.

So the next serious question is:

> Can structured superpositions generate nontrivial source topology and directional radiation structure?

## Main observables

1. Instantaneous dipole vector from the charge density:
   $$
   \mathbf p(t)=\int \mathbf r\,|\Psi(\mathbf r,t)|^2\,d^3r
   $$

2. Exact far-field intensity from the current density:
   $$
   I_{\rm exact}(\theta,\phi)
   $$

3. Dipole-limit prediction from the instantaneous dipole vector:
   $$
   I_{\rm dip}(\theta,\phi)\propto |\hat{\mathbf n}\times(\hat{\mathbf n}\times\mathbf p)|^2
   $$

4. Maximum map deviation:
   $$
   \max_{\theta,\phi}\big|I_{\rm exact}^{\rm norm}-I_{\rm dip}^{\rm norm}\big|
   $$

## What to look for

- rotation of the dipole vector in the $x$-$z$ plane as $\delta$ changes,
- phase-controlled changes in radiation maps,
- whether exact and dipole-limit maps still agree,
- whether any unexpected non-dipolar structure appears.

## Expected outcome

At low $n$, hydrogen will probably still stay close to dipole behavior.
But this is the first task where non-axisymmetric interference can produce genuinely interesting structure.