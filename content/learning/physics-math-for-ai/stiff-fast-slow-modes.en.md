## Slow observations can require fast computations

Many physical systems combine a rapidly disappearing transient with a persistent slow evolution. An observer may care only about the slow component, yet an explicit integrator can require tiny steps because of the fast decay rate. This is a useful starting point for understanding stiffness. A steep-looking curve alone is not a sufficient definition.

Consider independent modes s′=-s and q′=-kq. Their exact solutions are s(t)=exp(-t) and q(t)=(A+ε)exp(-kt). The perturbation ε is injected only into the initial fast component; no new random noise is added at later steps. For large k, the fast scale 1/k is much shorter than the slow observation scale.

## Compute the fast-mode multiplier

$$
q_{n+1}=(1-kh)q_n\quad\text{for Euler},\qquad
q_{n+1}=\frac{q_n}{1+kh}\quad\text{for backward Euler}.
$$

For positive real k, Euler is non-amplifying only when 0≤kh≤2. At kh=2, the multiplier is -1: it oscillates without damping, which is different from faithfully reproducing physical decay. With k=40 and h=0.1, the multiplier is -3, so a tiny perturbation triples at every step while alternating sign. Backward Euler instead has multiplier 1/5. That is stable, but still quite different from the exact one-step multiplier exp(-4).

## An absent mode can hide instability

If A=ε=0, exact-arithmetic Euler keeps q=0 forever. A quiet observed trajectory does not establish stability to all perturbations. Changing ε to a small positive value reveals the same underlying amplification factor. The experiment reports both the initialized fast amplitude and the step multiplier so that these two questions remain separate.

Blue shows the numerical slow coordinate. Orange shows the numerical fast coordinate and the exact fast reference. They are separate coordinates, not terms that must be added to obtain a single measured quantity. If a huge fast value dominates the vertical scale, inspect the slow-mode error in the numeric table.

## A-stability does not imply decay of unresolved fast modes

For the complex test equation y′=μy, put z=hμ with Re z≤0. Backward Euler has R=1/(1−z); since |1−z|²≥1 it is A-stable. The trapezoidal rule has R=(1+z/2)/(1−z/2). The difference between squared denominator and numerator magnitudes is −2 Re z≥0, so it is also A-stable. Nevertheless their large negative-real limits differ:

$$
\lim_{x\to\infty}R_{BE}(-x)=0,\qquad
\lim_{x\to\infty}R_{trap}(-x)=-1.
$$

The first property supplies L-stable stiff damping; the second can leave nearly undamped alternating fast components. Stability prevents amplification but does not promise a faithful elimination of an unresolved transient. No nonconstant explicit Runge–Kutta stability polynomial can be bounded on the entire negative real axis, since its highest-degree term grows there.

| z | Backward Euler | Trapezoidal |
|---|---:|---:|
| −1 | 1/2 | 1/3 |
| −10 | 1/11 | −2/3 |
| −100 | 1/101 | −49/51 |

```python
from fractions import Fraction as F
for z in (F(-1),F(-10),F(-100)):
    backward=1/(1-z)
    trapezoid=(1+z/2)/(1-z/2)
    assert abs(backward)<=1 and abs(trapezoid)<=1
    print(z,backward,trapezoid)
for z in (-1+2j,-.01+10j,-10+3j):
    assert abs(1/(1-z))<=1
    assert abs((1+z/2)/(1-z/2))<=1
```

The trapezoidal comparison is an auxiliary analysis, not an added live method. For research solvers, a step can damp fast modes well while still making the slow mode inaccurate. Nonlinear solve convergence, nonnormal amplification and the chosen observation functional also matter; a scalar A-stability plot is not a general nonlinear well-posedness proof.

## Worked example and exercise

For k=40, T=1, N=10 and A=1, explicit Euler gives fast terminal value (-3)^{10}=59049. The true value exp(-40) is nearly zero. Meanwhile slow Euler gives 0.9^{10}≈0.348678, to be compared with exp(-1). The catastrophic fast error and moderate slow error are different effects.

Exercise: for k=80, how small must h be for non-amplifying Euler? Solution: h≤2/80=0.025. At the endpoint there is no damping, so reproducing the actual decay requires a stricter accuracy choice. Satisfying a stability inequality is not the same as meeting an error target.

## Connection to AI and multiple time scales

Reaction systems, discretized diffusion and continuous-depth models can produce widely separated Jacobian scales. This diagonal example does not cover nonnormal transient growth or all forms of nonlinear stiffness. In research, record Jacobian structure, nonlinear-solve cost, observation horizon, accuracy targets and computational budget. Selecting an implicit method by name alone guarantees neither accuracy nor efficiency.

Reference: [official SciPy discussion of IVP methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
