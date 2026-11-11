## A jump requires an integral interpretation

A shock has no ordinary spatial derivative at its discontinuity. Conservation laws therefore need a weak, distributional interpretation that remains meaningful for piecewise smooth fields. For Burgers' equation the conserved flux is f(u)=u²/2. The Riemann initial data are constant uL to the left of zero and uR to the right.

A candidate discontinuity moving at speed s satisfies the weak conservation law precisely when its delta-function coefficient vanishes:

$$
s[u]=[f(u)],\qquad
s=\frac{u_L+u_R}{2},\qquad [u]=u_R-u_L.
$$

The speed formula extends continuously to equal states, where there is no actual jump. A zero Rankine–Hugoniot defect checks conservation across the interface; it does not by itself select the physically admissible evolution.

## Compression and expansion have different entropy behavior

If uL>uR, faster left characteristics meet slower right characteristics, forming a compressive shock. Its speed lies strictly between the two states. If uL<uR, an expansion jump with the same algebraic speed still satisfies the weak law, but characteristics move away from it rather than into it.

For the convex quadratic entropy eta=u²/2 and entropy flux q=u³/3, the jump coefficient is

$$
[q]-s[\eta]=\frac{(u_R-u_L)^3}{12}.
$$

The entropy inequality requires a nonpositive coefficient. Compression satisfies it; expansion violates it. The lab's candidate-jump mode exposes this counterexample rather than silently treating every conservative discontinuity as an entropy solution.

## A rarefaction fills the expanding gap

For uL<uR the entropy solution is a continuous fan at positive time:

$$
u(x,t)=\begin{cases}
u_L&x\le u_Lt,\\
x/t&u_Lt<x<u_Rt,\\
u_R&x\ge u_Rt.
\end{cases}
$$

With states zero and one at t=1, the value at x=1/4 is one quarter. The candidate expansion jump instead returns zero at that point. Both conserve weak flux, but only the fan satisfies the entropy selection. At time zero the plot returns the stated step initial data.

## Open observation intervals need boundary flux

The readout integrates mass over [-2,2] and separately integrates flux through both endpoints in time. Their balance is M(t)-M(0)+Fout-Fin=0. A wave leaving this window can change its mass without violating conservation. Boundary events are integrated piecewise exactly for these shock and fan formulas, rather than approximated by an unrelated generic decay curve.

Jumps are drawn as separate horizontal segments with endpoint conventions. A connecting diagonal would depict a false smooth transition and imply nonexistent intermediate states at fixed time.

## Vanishing viscosity selects the compressive jump

Distributionally, a moving step has uₜ=−s[u]δ and f(u)ₓ=[f]δ, yielding Rankine–Hugoniot. The same calculation for an entropy pair gives ηₜ+qₓ=([q]−s[η])δ. This explains why conservation alone is insufficient: for η=u²/2, q=u³/3, an expansion jump has a positive coefficient.

Add viscosity uₜ+(u²/2)ₓ=εuₓₓ, ε>0, and seek U(z), z=x−st, joining L>R. One integration and the endpoint limits give

$$
s=(L+R)/2,\quad \varepsilon U'=\tfrac12(U-L)(U-R),
\quad U(z)=s-\tfrac{L-R}{2}\tanh\!\left(\frac{(L-R)z}{4\varepsilon}\right).
$$

U decreases from L to R, and its width is proportional to ε/(L−R). For an increasing join L<R, the same ODE has U′<0 between the endpoints, so no increasing connecting profile exists. Positive viscosity rejects the expansion shock. For a general convex entropy, ηₜ+qₓ=ε(η′uₓ)ₓ−εη″uₓ², making the dissipative sign explicit under a justified limit. A complete vanishing-viscosity convergence theorem needs compactness and initial-data hypotheses beyond this traveling-wave calculation.

| ε for L=1,R=0 | Profile at z=0 | Center slope |
|---|---:|---:|
| 1/2 | 1/2 | −1/4 |
| 1/4 | 1/2 | −1/2 |
| 1/8 | 1/2 | −1 |

```python
from math import tanh
L, R = 1., 0.
s, jump = (L+R)/2, L-R
for eps in (.5, .25, .125):
    k = jump/(4*eps)
    for z in (-1., 0., 1.):
        h = tanh(k*z)
        U = s-jump*h/2
        first = -jump*k*(1-h*h)/2
        second = jump*k*k*h*(1-h*h)
        assert abs(eps*first-(U-L)*(U-R)/2) < 1e-14
        assert abs((U-s)*first-eps*second) < 1e-14
    print(eps, s, -jump*jump/(8*eps))
```

The smooth viscous profiles below are auxiliary analytic examples; the live inviscid jump remains drawn as two separate segments. The calculation checks the integrated profile ODE and its smooth PDE residual without replacing a discontinuity by a fictitious plotting ramp.

## Exercise: a conservative wrong answer

Compare uL=0, uR=1 in entropy and candidate modes. Both Rankine–Hugoniot defects vanish, while the candidate's entropy coefficient is positive 1/12. Next use states two and one at t=2: the shock has passed the observation window, whose mass changes from six to eight. Verify that accumulated inflow minus outflow equals two and explain why unchanged window mass is not the correct conservation test.

Further reading: [Clawpack Riemann-book Burgers chapter](https://www.clawpack.org/riemann_book/html/Burgers.html). The entropy and exact window-flux experiments are independently derived.
