## Choose time and space steps together

For many PDE discretizations, time-step selection depends on spatial spacing. Explicit central heat diffusion requires dt proportional to h², while positive-speed upwind transport requires dt proportional to h. The parameter r therefore has different definitions in the two models. Equal values of r do not mean equal physical time.

The periodic initial samples are cos(2πx)+ε cos(πNx) on N nodes in [0,1). The small second component is the highest representable grid frequency. It is a deliberately seeded initial mode, not fresh noise added at every step. All nodes update simultaneously from the previous array.

## Two stencils and two stability bounds

$$
u_i^{n+1}=u_i^n+r(u_{i-1}^n-2u_i^n+u_{i+1}^n),\qquad r=\nu\Delta t/h^2
$$

is explicit central heat diffusion for u_t=νu_xx. Its Fourier multiplier is 1-4r sin²(θ/2), so all modes are non-amplifying when 0≤r≤1/2. Distinguish the field u from the diffusivity ν in this notation.

$$
u_i^{n+1}=u_i^n-r(u_i^n-u_{i-1}^n),\qquad r=c\Delta t/h
$$

is upwind for u_t+cu_x=0 with c>0. Its multiplier is 1-r+r exp(-iθ), with modulus at most one for 0≤r≤1. A negative speed requires the other neighbor; this finite experiment allows only positive c.

## Physical time, conservation and observation

The reported time is T=n dt. Changing r or N also changes dt and T, so do not compare separate final pictures as if they automatically had a common physical horizon. Within each selected experiment, orange is the exact continuous reference evaluated at that same current T. Heat damps both continuous modes exponentially; transport translates both by cT.

Periodic neighbor exchanges cancel in the global sum, preserving discrete mass h∑u_i in exact arithmetic. This conservation does not imply stability: very large alternating values can still sum to zero. Inspect the Nyquist multiplier, maximum nodal error and mass change separately.

## Energy estimates explain the Fourier bounds in physical space

Use the unweighted periodic grid norm; multiplying it by h changes none of the stability conclusions. Write dᵢ=uᵢ₊₁−uᵢ and Δ₂uᵢ=uᵢ₋₁−2uᵢ+uᵢ₊₁. Summation by parts gives the heat identity

$$
\|u+r\Delta_2u\|^2-\|u\|^2
=-2r\sum_i d_i^2+r^2\sum_i(\Delta_2u_i)^2
\le-2r(1-2r)\sum_i d_i^2.
$$

The inequality uses Σ(Δ₂u)²≤4Σd². Upwind is a convex combination of u and its periodic shift S, giving exactly ‖(1−r)u+rSu‖²−‖u‖²=−r(1−r)‖u−Su‖². Hence the same CFL intervals follow without diagonalizing the stencil. Nonnegative combination weights also prove maximum bounds and positivity when the initial data are nonnegative; the lab's cosine data need not be.

| Scheme and r | Coefficient multiplying squared neighbor differences |
|---|---:|
| Heat, r=1/4, upper bound | −1/4 |
| Upwind, r=1/2, identity | −1/4 |
| Upwind, r=1, identity | 0 |

```python
from fractions import Fraction as F
u=list(map(F,(1,2,-1,0)))
sq=lambda v:sum(x*x for x in v)
diff=[u[(i+1)%4]-u[i] for i in range(4)]
lap=[u[(i-1)%4]-2*u[i]+u[(i+1)%4] for i in range(4)]
r=F(1,4);heat=[x+r*d for x,d in zip(u,lap)]
change=sq(heat)-sq(u)
assert change==-2*r*sq(diff)+r*r*sq(lap)
assert change<=-2*r*(1-2*r)*sq(diff)
for r in (F(1,2),F(1)):
    upwind=[(1-r)*u[i]+r*u[(i-1)%4] for i in range(4)]
    assert sq(upwind)-sq(u)==-r*(1-r)*sq(diff)
    print(r,sq(upwind)-sq(u))
```

The live upwind update now uses the combination form. At r=1 this is a bit-exact array permutation, including very small samples that subtraction could damage. Above the stability interval, telescoping still preserves mass in exact arithmetic, but these energy estimates no longer give contraction. In a huge unstable floating rollout even mass cancellation can lose accuracy.

## Worked example and exercise

For heat with r=0.75, the Nyquist multiplier is 1-4r=-2. An initial amplitude ε=0.001 becomes 0.001·(-2)²⁰=1048.576 after twenty steps. Physical diffusion instead damps high frequencies most strongly, so this growth is a numerical artifact.

Exercise: simplify an upwind step at r=1. Solution: u_i_next=u_i-(u_i-u_left)=u_left, an exact one-cell shift. Since dt=h/c, the physical translation distance is also h. This special integer-cell translation does not imply that upwind is nondiffusive or exact for arbitrary r.

## Diagnose learned-simulator failures

An exploding learned PDE rollout can reflect model error, unstable time stepping, unresolved inputs or boundary mistakes. Correct mass alone cannot certify the solution. These bounds concern the stated linear one-dimensional stencils; their numerical constants cannot be transferred unconditionally to nonlinear PDE solvers. Multiple dimensions, nonuniform meshes and higher-order reconstruction require their own amplification or energy analysis. Report both the spatial discretization and the physical horizon when comparing learned and conventional solvers.

References: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/) and [the Clawpack authors on finite-volume methods](https://www.clawpack.org/riemann_book/html/Approximate_solvers.html).
