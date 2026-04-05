# Task 1 — Hydrogen transition benchmark

## Goal

Reproduce the standard dipole-radiation benchmark for the hydrogen transition

$$
1s \leftrightarrow 2p_z
$$

and verify that the far-field angular pattern is

$$
I(\theta)\propto \sin^2\theta.
$$

Also verify the forbidden dipole benchmark

$$
\langle 1s|\mathbf r|2s\rangle = 0.
$$

---

## Physics

Use the hydrogen orbitals

$$
\psi_{100}(r,\theta,\phi)=\frac{1}{\sqrt{\pi}\,a_0^{3/2}}e^{-r/a_0}
$$

$$
\psi_{200}(r,\theta,\phi)=\frac{1}{4\sqrt{2\pi}\,a_0^{3/2}}
\left(2-\frac{r}{a_0}\right)e^{-r/(2a_0)}
$$

$$
\psi_{210}(r,\theta,\phi)=\frac{1}{4\sqrt{2\pi}\,a_0^{3/2}}
\left(\frac{r}{a_0}\right)e^{-r/(2a_0)}\cos\theta
$$

The transition dipole is

$$
\mathbf d_{ab}=-e\int \psi_a^*(\mathbf r)\,\mathbf r\,\psi_b(\mathbf r)\,d^3r.
$$

The far-field dipole radiation pattern is

$$
I(\theta,\phi)\propto
\left|\hat{\mathbf n}\times(\hat{\mathbf n}\times\mathbf d_{ab})\right|^2.
$$

For $1s \leftrightarrow 2p_z$, the dipole should point along $z$, so the pattern must be

$$
I(\theta)\propto \sin^2\theta.
$$

---

## Deliverables

1. Compute $(d_x,d_y,d_z)$ for $1s \leftrightarrow 2p_z$
2. Compute $(d_x,d_y,d_z)$ for $1s \leftrightarrow 2s$
3. Plot numerical $I(\theta)$ against analytic $\sin^2\theta$
4. Plot a full $(\theta,\phi)$ far-field map

---

## Pass-fail gates

### Gate A
For $1s \leftrightarrow 2p_z$,

$$
\frac{|d_x|}{|d_z|}<10^{-2},\qquad
\frac{|d_y|}{|d_z|}<10^{-2}
$$

### Gate B
For $1s \leftrightarrow 2s$,

$$
\frac{|\mathbf d_{1s,2s}|}{|\mathbf d_{1s,2p_z}|}<10^{-3}
$$

### Gate C
For normalized angular profiles,

$$
\max_\theta\left|I_{\mathrm{num}}(\theta)-\sin^2\theta\right|<0.03
$$
