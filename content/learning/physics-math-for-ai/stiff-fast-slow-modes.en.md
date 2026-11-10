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

## Worked example and exercise

For k=40, T=1, N=10 and A=1, explicit Euler gives fast terminal value (-3)^{10}=59049. The true value exp(-40) is nearly zero. Meanwhile slow Euler gives 0.9^{10}≈0.348678, to be compared with exp(-1). The catastrophic fast error and moderate slow error are different effects.

Exercise: for k=80, how small must h be for non-amplifying Euler? Solution: h≤2/80=0.025. At the endpoint there is no damping, so reproducing the actual decay requires a stricter accuracy choice. Satisfying a stability inequality is not the same as meeting an error target.

## Connection to AI and multiple time scales

Reaction systems, discretized diffusion and continuous-depth models can produce widely separated Jacobian scales. This diagonal example does not cover nonnormal transient growth or all forms of nonlinear stiffness. In research, record Jacobian structure, nonlinear-solve cost, observation horizon, accuracy targets and computational budget. Selecting an implicit method by name alone guarantees neither accuracy nor efficiency.

Reference: [official SciPy discussion of IVP methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
