## Separate equilibria, bifurcations and chaos diagnostics

These concepts describe different properties. Stability concerns responses near a state. A bifurcation changes the qualitative structure as a parameter varies. Chaos concerns a much stronger long-time dynamical behavior, and an irregular-looking finite trace is insufficient evidence. The lab uses a continuous scalar ODE for a pitchfork bifurcation and a separate discrete logistic map for sensitivity diagnostics.

The plot selector explicitly switches systems. An autonomous smooth scalar ODE cannot repeatedly fold state order in the way the logistic map does. Its trajectories should not be relabeled as an example of discrete-map chaos simply because both models share a parameter-controlled interface.

## Derive the pitchfork equilibria

For y'=mu y-y³, equilibria solve y(mu-y²)=0. When mu is negative, only zero exists and is attracting. Positive mu creates the additional states plus/minus sqrt(mu). The derivative of the field is mu-3y², so the new branches have derivative -2mu and attract, while zero becomes repelling.

$$
y(t)=\frac{y_0e^{\mu t}}{\sqrt{1+y_0^2(e^{2\mu t}-1)/\mu}},
\qquad
 y(t)\big|_{\mu=0}=\frac{y_0}{\sqrt{1+2y_0^2t}}.
$$

The formula follows by solving the logistic-type equation for y² and preserving the initial sign. A small-argument series supplies a stable zero-parameter limit in the implementation.

## Zero linearization does not finish the argument

At mu zero the linearized derivative at the origin is zero, so the strict-sign linearization test is inconclusive. Nevertheless, the exact cubic solution tends to zero for every initial state, proving attraction with algebraic rather than exponential decay. At positive mu, starting exactly at the unstable equilibrium zero still leaves the trajectory there forever; its lack of motion does not establish stability.

## A finite Lyapunov mean is a diagnostic

The separate map is x_(n+1)=r x_n(1-x_n). Its derivative is r(1-2x_n). After a selected burn-in, the lab computes the finite mean of log absolute derivatives and compares two nearby initial orbits.

$$
\Lambda_{m,N}=\frac1{N-m}\sum_{n=m}^{N-1}\log|r(1-2x_n)|.
$$

This average depends on initial state, observation window and finite arithmetic. A positive value is not an asymptotic chaos certificate. An empty window has no average. If a derivative factor is exactly zero, the logarithmic sum is minus infinity, so no finite number is displayed; the zero-factor count distinguishes this from missing samples.

## A potential proves what linearization misses at criticality

Write the pitchfork ODE as y′=−V′(y), with V(y)=y⁴/4−μy²/2. Along a solution, dV/dt=−(μy−y³)²≤0. For μ≤0, V has its unique minimum at zero; its compact sublevel sets and the fact that V′ vanishes only at zero prove convergence to zero. At μ=0 the linearized derivative is zero, yet the nonlinear restoring term gives asymptotic stability. Its decay is algebraic:

$$
y(t)=\frac{y_0}{\sqrt{1+2y_0^2t}},\qquad
\lim_{t\to\infty}\sqrt t\,|y(t)|=1/\sqrt2\quad(y_0\ne0).
$$

For μ>0, zero is a local maximum and ±√μ are minima. Uniqueness prevents crossing equilibria, so each nonzero initial sign selects its corresponding attracting branch. This one-dimensional smooth autonomous flow has no nonconstant periodic orbit: any such orbit would have an extremum where its vector field vanishes and uniqueness would force a constant solution.

| μ | Origin linearization | Nonlinear conclusion |
|---|---|---|
| −1 | −1 | asymptotically stable |
| 0 | 0 | asymptotically stable, algebraic decay |
| 1 | 1 | unstable, two attracting branches |

```python
from math import sqrt, sin, pi
for mu in (-1., 0., 1.):
    for y in (-1., -.25, 0., .25, 1.):
        field = mu*y-y**3
        potential_derivative = y**3-mu*y
        assert potential_derivative*field == -field**2
    print(mu, 'origin derivative:', mu)
for theta in (.1, .2, .3):
    x = sin(pi*theta)**2
    assert abs(4*x*(1-x)-sin(2*pi*theta)**2) < 1e-14
for t in (1., 100., 10000.):
    y = .5/sqrt(1+2*.5**2*t)
    print(t, sqrt(t)*abs(y))
x = .5
for _ in range(2): x = 4*x*(1-x)
assert x == 0
```

The separate logistic map at r=4 obeys x=sin²(πθ), with θ↦2θ mod1. This identity connects it to an expanding map, but exceptional orbits and finite precision still matter. For example x₀=1/2 reaches 1 then 0 exactly. A positive finite-window logarithmic derivative average is a diagnostic, not a proof about every initial condition or an infinite-time computer orbit.

## Exercise: compare critical cases

For mu zero and y0=1/2, evaluate the ODE at t=2 to obtain 1/(2sqrt(2)). For positive mu, verify both nonzero equilibrium derivatives are negative. Then choose map r=4, seed 1/2 and no burn-in: the first derivative factor is zero, even though r=4 is often associated with chaotic behavior for other initial conditions. Explain why a parameter label cannot replace a statement about the actual orbit and diagnostic window.

Further reading: [MIT nonlinear systems and dynamics](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). The two controlled models and finite-window distinctions are derived directly.
