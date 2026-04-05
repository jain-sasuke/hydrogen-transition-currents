# hydrogen-transition-currents

Hydrogenic transition-current benchmarks for radiation physics.

## First goal

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

## Repo structure

```text
src/hydrogen/
    constants.py
    wavefunctions.py
    dipole.py
    far_field.py

notebooks/
    task01_hydrogen_transition_benchmark.py

notes/
    task01_hydrogen_transition_benchmark.md

figures/