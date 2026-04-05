# Task 2 — Transition charge/current density benchmark

## Goal

Construct the real-space charge and current source for the hydrogen superposition

$$
\Psi(\mathbf r,t)=\frac{1}{\sqrt2}
\left[
\psi_{1s}(\mathbf r)e^{-iE_1 t}
+
\psi_{2p_z}(\mathbf r)e^{-iE_2 t}
\right]
$$

and verify:

1. the oscillating charge density has dipolar parity,
2. the current density is phase-shifted relative to the charge density,
3. the continuity equation holds numerically,
4. the dipole moment extracted from the charge density matches the Task 1 benchmark.

---

## Physics

For equal-amplitude superposition of real orbitals,

### Charge density
$$
\rho(\mathbf r,t)=-\frac12\left[
\psi_{1s}^2+\psi_{2p_z}^2+2\psi_{1s}\psi_{2p_z}\cos(\omega_{21}t)
\right].
$$

The oscillating interference piece is
$$
\rho_{\rm osc}(\mathbf r,t)=-\psi_{1s}\psi_{2p_z}\cos(\omega_{21}t).
$$

### Electric current density
For real orbitals,
$$
\mathbf J(\mathbf r,t)=
\frac12\sin(\omega_{21}t)\,
\left[
\psi_{1s}\nabla\psi_{2p_z}
-
\psi_{2p_z}\nabla\psi_{1s}
\right].
$$

### Continuity equation
$$
\frac{\partial \rho}{\partial t}+\nabla\cdot\mathbf J=0.
$$

---

## Expected signatures

- \(\rho_{\rm osc}\) is dipolar in \(z\),
- \(\rho_{\rm osc}\propto \cos(\omega_{21} t)\),
- \(\mathbf J_{\rm osc}\propto \sin(\omega_{21} t)\),
- the current transports charge between the positive and negative lobes,
- the dipole moment from \(\rho\) has amplitude
$$
|\langle 1s|z|2p_z\rangle| = \frac{128\sqrt2}{243}.
$$

---

## Deliverables

1. \(\rho_{\rm osc}(x,z)\) at \(t=0\)
2. \(\rho_{\rm osc}(x,z)\) at \(t=T/2\)
3. current quiver plot in \(x\)-\(z\) plane at \(t=T/4\)
4. continuity residual map at \(t=T/4\)
5. time trace of \(\rho_{\rm osc}\) and \(J_z\) at a representative point
6. dipole-moment time trace \(p_z(t)\)

---

## Pass-fail gates

### Gate A — dipolar symmetry
The charge-density map shows the correct odd \(z\)-parity.

### Gate B — phase relation
At a representative point:
- \(\rho_{\rm osc}\sim \cos(\omega_{21}t)\)
- \(J_z\sim \sin(\omega_{21}t)\)

### Gate C — dipole consistency
The amplitude of \(p_z(t)\) agrees with
$$
\frac{128\sqrt2}{243}
$$
to good numerical accuracy.

### Gate D — continuity
The continuity residual remains small on the well-resolved interior grid.