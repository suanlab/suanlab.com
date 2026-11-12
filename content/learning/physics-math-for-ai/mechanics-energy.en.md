## Motion is not the same as optimization

A mass attached to an ideal spring oscillates because inertia carries it past equilibrium. Plain gradient descent instead uses a first-order update to reduce an objective. Both may contain a gradient, but their state variables, units and energy balance differ. We will derive one physical model before discussing analogies to learning.

Assume a one-dimensional spring, fixed mass $m=1$ kg, stiffness $k>0$ in N/m, no friction, no external forcing and displacement $q$ in meters from equilibrium. Initial velocity is zero. These assumptions define the system boundary and determine what conservation claim is appropriate.

## From force to the exact trajectory

Hooke’s law and Newton’s second law give

$$
F=-kq,\qquad m\ddot q=-kq.
$$

With $q(0)=q_0,v(0)=0$, the exact solution is $q(t)=q_0\cos(\omega t)$ and $v(t)=-q_0\omega\sin(\omega t)$, where $\omega=\sqrt{k/m}$. At $m=k=q_0=1$ in the stated units, the period is $2\pi$ seconds.

The negative force restores the mass toward equilibrium. It does not make position move monotonically toward zero: velocity is an independent state variable.

## Derive the conserved energy

Potential energy is $V=kq^2/2$, whose negative position derivative is the spring force. Kinetic energy is $mv^2/2$. Thus

$$
E=\frac12mv^2+\frac12kq^2,\qquad
\frac{dE}{dt}=v(m\dot v+kq)=0.
$$

The cancellation uses the equation of motion and the absence of friction or external work. Adding damping changes the energy balance; conservation cannot simply be retained as an unquestioned label.

In the reference case $q_0=1$, initial energy is 0.5 J. At a quarter period, position is zero and all energy is kinetic. At half a period, position is $-1$ m and the energy is again potential.

## A discrete method with a chosen order

The lab uses velocity-first symplectic Euler:

$$
v_{n+1}=v_n-h(k/m)q_n,\qquad
q_{n+1}=q_n+h v_{n+1}.
$$

The position uses the **new** velocity. This differs from forward Euler using the old velocity in both updates. For $h=0.1,m=k=q_0=1$, the first new velocity is $-0.1$ and position is 0.99. Its energy is 0.49505 J, not exactly 0.5.

Symplectic structure preservation does not mean exact preservation of the original energy at every step. This distinction matters when evaluating a “physics-preserving” learned model: specify which structure is preserved and at what numerical tolerance.

## Derive the stability range

In state coordinates $(q,v)$, the update matrix is

$$
M=\begin{pmatrix}1-h^2\omega^2&h\\-h\omega^2&1\end{pmatrix},\qquad\det M=1.
$$

Its trace is $2-h^2\omega^2$. Within $0<h\omega<2$, eigenvalues lie on the unit circle and the oscillator remains bounded for this linear problem. At the boundary $h\omega=2$, repeated eigenvalues need not yield bounded powers: a defective update can produce linear growth. Volume preservation alone does not guarantee stability.

The lab computes 100 steps and reports actual maximum energy deviation over those states. Changing $h$ changes final time. To measure convergence order, hold the physical final time fixed and adjust the step count.

## Read and extend the experiment

Blue is exact position and orange is numerical position at matching displayed times. The table includes terminal position and velocity, initial energy, final energy and maximum energy error. A visually close position curve does not prove energy conservation or long-time accuracy.

For AI research, compare learned forces against known force derivatives and inspect trajectories, units, invariants and resolution sensitivity. A network that matches a short trajectory may fail at longer times or different initial states. A Hamiltonian parameterization supplies structure only under its stated state variables and solver assumptions.

## Research extension: a conserved quadratic form can fail to control motion

The energy wiggle in the existing plot is a property of the specified solver, not friction. We can explain its boundedness without fitting a trend to 100 samples. Keep the same unforced linear spring and mass of 1 kg, and choose a **fixed** positive step h. In exact arithmetic, velocity-first symplectic Euler preserves the following modified quadratic form:

$$
H_h(q,v)=\frac12\big(mv^2+kq^2-hkqv\big),\qquad m=1\ \mathrm{kg}.
$$

The correction has energy units: h times stiffness times displacement times velocity is a joule. This expression belongs to this particular linear oscillator and update order. It is not a claim that every symplectic solver exactly preserves a nearby energy, or that this modified quantity is the apparatus's physical energy.

### Prove the invariant before inspecting a plot

Write v'=v−h(k/m)q and q'=q+hv'. Expanding the new quadratic form gives:

$$
\begin{aligned}
2H_h(q',v')
&=mv'^2+k(q+hv')^2-hk(q+hv')v'\\
&=mv'^2+kq^2+hkqv'\\
&=m(v-h(k/m)q)^2+kq^2+hkq(v-h(k/m)q)\\
&=mv^2+kq^2-hkqv=2H_h(q,v).
\end{aligned}
$$

Thus H_h is constant for every step count in exact arithmetic. Physical E generally oscillates. Floating-point arithmetic adds roundoff; a computed tiny invariant residual measures numerical evaluation, rather than an exact physical conservation law. The formula also depends on h: changing h between updates changes the quadratic form, so this proof does not authorize adaptive steps.

### Why positive definiteness supplies the missing bound

Use energy-scaled coordinates; with the fixed mass, Q and P have the same units. The normalized step z is dimensionless:

$$
Q=\sqrt{k/m}\,q,\quad P=v,\quad z=h\sqrt{k/m},\qquad
H_h=\frac{m}{2}\begin{pmatrix}Q&P\end{pmatrix}
\begin{pmatrix}1&-z/2\\-z/2&1\end{pmatrix}
\begin{pmatrix}Q\\P\end{pmatrix}.
$$

The symmetric matrix has eigenvectors (1,1) and (1,−1), with eigenvalues 1−z/2 and 1+z/2. When 0<z<2, both are positive. Diagonalize in those two directions, or use 2|QP|≤Q²+P², to obtain the first inequality below. Because v0=0, H_h(q0,0)=E0; division by the positive constants gives the second:

$$
(1-z/2)E_n\le H_h(q_n,v_n)\le(1+z/2)E_n,
\qquad
\frac{1}{1+z/2}\le\frac{E_n}{E_0}\le\frac{1}{1-z/2}.
$$

The ratio requires q0≠0. If q0=0 and v0=0, the trajectory is identically zero and no energy ratio is needed. For arbitrary initial velocity, replace E0 in the middle of the bound by H_h(q0,v0); its equality with E0 is specific to the zero-velocity start.

Intuitively, constant H_h traces a tilted ellipse in the (Q,P) plane. An ellipse has finite extent, which bounds q and v and hence physical energy. As z approaches 2, one eigenvalue tends to zero and the ellipse elongates. A bound can become large even while the invariant stays perfectly constant. This is a statement about all discrete times, not just the 100 plotted states, and it does not guarantee small phase error over arbitrarily long physical times.

### Work the boundary example exactly

Use the existing unstable preset h=1 s, k=4 N/m and q0=1 m. Then z=2. The repeated eigenvalue −1 has a nonzero nilpotent part:

$$
M=\begin{pmatrix}-3&1\\-4&1\end{pmatrix}=-I+N,
\quad N=\begin{pmatrix}-2&1\\-4&2\end{pmatrix},\quad N^2=0,
\quad M^n=(-1)^n(I-nN).
$$

$$
q_n=(-1)^n(1+2n),\quad v_n=(-1)^n4n,\qquad
E_n=16n^2+8n+2,\qquad H_h=\frac12(v-2q)^2=2.
$$

The last line uses these initial data. After 100 updates, q=201 m, v=400 m/s and E=160802 J, whereas E0=2 J. The modified form is still exactly 2 J. Its level set is now a pair of parallel lines, not a bounded ellipse: the missing positive eigenvalue lets motion escape. Determinant one and eigenvalue modulus one therefore cannot replace a bounded-powers argument.

![For a stable step the physical energy oscillates around an exactly constant modified quadratic form; at the defective boundary physical energy grows quadratically despite the same invariant.](/assets/learning/classical-energy-invariant.en.svg)

The upper panel shows the first 12 updates with h=0.5, k=4; the lower panel shows 100 updates with h=1, k=4. Both start at q=1, v=0. Points are discrete solver states, not interpolated exact physical trajectories. The independent vertical scales are explicitly labeled. The table below includes the initial state and all 100 updates; each row has a different final physical time T=100h, so it is **not** a fixed-time convergence study.

| h (s) | z | min E/E0 | max E/E0 |
|---|---|---|---|
| 1/10 | 0.2 | 0.909100 | 1.111059 |
| 1/2 | 1.0 | 1.000000 | 2.000000 |
| 9/10 | 1.8 | 0.526986 | 9.999591 |
| 1 | 2.0 | 1.000000 | 80401.000000 |

### Reproduce with rational arithmetic

The standalone code extends the explanation; it does not add a modified-energy readout to the current interactive lab. Fractions make every invariant and boundary assertion exact. The stable-case test uses k=4, hence z/2=h in the declared numerical units.

```python
from fractions import Fraction as F

def energy(q, v, k):
    return (v*v + k*q*q) / 2

def run(h, k, steps=100):
    q, v = F(1), F(0)  # m = 1 kg, q0 = 1 m
    E0 = energy(q, v, k)
    H0 = E0 - h*k*q*v/2
    energies = [E0]
    for n in range(1, steps+1):
        v -= h*k*q
        q += h*v
        E = energy(q, v, k)
        assert E - h*k*q*v/2 == H0
        if h < 1:  # k=4, so z=2h and z/2=h
            assert E0/(1+h) <= E <= E0/(1-h)
        else:  # h=1, k=4: exact defective-boundary solution
            assert q == (-1)**n * (1+2*n)
            assert v == (-1)**n * 4*n
            assert E == 16*n*n + 8*n + 2
        energies.append(E)
    return q, v, min(energies)/E0, max(energies)/E0

for h in [F(1,10), F(1,2), F(9,10), F(1)]:
    q, v, lo, hi = run(h, F(4))
    print(f"h={str(h):>4}, min E/E0={float(lo):.6f}, "
          f"max E/E0={float(hi):.6f}")
q, v, _, _ = run(F(1), F(4))
print("boundary q100, v100, E100:", q, v, energy(q, v, F(4)))
```

### Research checks and worked questions

1. **What does h=0.5, k=4 guarantee for all n?** Here z=1, so 2/3≤E_n/E0≤2. At n=1, v=−2, q=0 and E=2 J; at n=2, v=−2, q=−1 and E=4 J. The upper bound is attained. All these states have H_h=2 J.
2. **Can a zero invariant residual certify trajectory accuracy?** No. The boundary example has zero residual and unbounded position. In the stable regime, phase can drift while the state remains bounded. Compare against the analytic oscillator at a fixed physical final time and refine h with the number of steps adjusted.
3. **What should a learned integrator report?** State which energy or quadratic form it preserves, distinguish its residual from physical E−E0, and test the coercivity margin 1−z/2. Short-run energy agreement cannot establish long-run phase accuracy or validity beyond the assumed linear spring.

This extension is a direct algebraic derivation for the existing update; no new empirical force law or external theorem is assumed.

## Exercises and reference

1. Recompute the first numerical step and its energy. **Answer:** $q=0.99,v=-0.1,E=0.49505$.
2. Double stiffness and explain how frequency and stable step size change.
3. Add a damping force $-cv$ on paper and derive $dE/dt=-cv^2$.
4. Compare methods at the same physical final time, not merely the same number of updates.

See [MIT Classical Mechanics](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/) for forces, energy and oscillation. The numerical method and checks above are specified directly for this teaching oscillator, not a validated model of contact or rigid-body motion.
