## Cell averages represent conserved content

For u_t+c u_x=0 with positive speed c on [0,1], a finite-volume state is a cell average, not a point sample and not the cell's mass itself. Multiplying by cell width dx converts an average to content. The lab initializes a rectangular pulse on [1/4,1/2] using exact overlap lengths, so its mass is A/4 even when cell boundaries do not align with the pulse.

The blue and orange stems compare computed and exactly advected cell averages at the same physical time. They are not probability masses. Integrating the exact translated pulse over each cell provides a fair reference that avoids confusing unresolved point values with numerical error.

## Derive the shared-flux update

With positive c, first-order upwind flux uses the state on the left of an interface. Adjacent cells share that same flux:

$$
u_i^{n+1}=u_i^n-\frac{\Delta t}{\Delta x}
(F_{i+1/2}-F_{i-1/2}),\qquad
F_{i+1/2}=c u_i^n.
$$

Writing nu=CFL=c dt/dx gives u_new=(1-nu)u_i+nu u_left. Summing cell contents cancels all internal interface contributions. This telescoping property is the discrete conservation mechanism; it does not require every individual cell to retain its value.

## Boundary data determine global mass change

Periodic mode supplies the last cell as the first cell's upwind neighbor, so net boundary flux vanishes. Inflow mode supplies a prescribed constant g at the left and uses the last interior state at outflow. Therefore,

$$
M^{n+1}-M^n=\Delta t(cg-cu_{N-1}^n).
$$

The lab accumulates the actual per-step boundary input and checks M-M0 against it. Reporting only final mass would incorrectly label a valid open-boundary calculation nonconservative. Initial data, inflow data and outflow conventions are separate pieces of the model.

## Conservation is distinct from stability

For CFL between zero and one, the update is a convex combination of nonnegative neighboring values, preserving positivity and the corresponding maximum bound. Above one, the scheme still telescopes but can create oscillations, negative values and growing amplitudes. A tiny global mass residual is therefore not evidence that the solution is accurate or stable.

At CFL one, a periodic update shifts the array by exactly one cell. Because both reference and numerical state are cell averages, the translated pulse is reproduced up to ordinary arithmetic error. At smaller CFL the first-order scheme spreads sharp edges, producing numerical diffusion even though total mass stays correct.

## L1 contraction and the random-walk origin of numerical diffusion

For two periodic upwind solutions with 0≤ν≤1, their difference obeys eᵢⁿ⁺¹=(1−ν)eᵢⁿ+νeᵢ₋₁ⁿ. The triangle inequality and an index shift give ΔxΣ|eᵢⁿ⁺¹|≤ΔxΣ|eᵢⁿ|. The same proof works with equal prescribed inflow data, with an outflow loss. Thus this scheme is L1 contractive, not just mass conservative. Above CFL one, the nonnegative-weight proof fails while flux cancellation still holds.

On the infinite grid, iterating a unit cell pulse gives binomial weights: each step stays in place with probability 1−ν or moves right by one cell with probability ν. This is a representation of normalized nonnegative weights, not a reinterpretation of arbitrary cell averages as probabilities. After n steps, the mean physical shift is nνΔx=ct and the variance is nν(1−ν)Δx². Comparing with variance 2Dt gives

$$
D_{\mathrm{num}}=\frac{c\Delta x}{2}(1-\nu).
$$

The same coefficient arises by Taylor expansion of the fully discrete method, including the uₜₜ term. It vanishes at ν=1, when the scheme is an exact cell shift. At fixed physical time and fixed CFL, grid refinement reduces this leading diffusion; near discontinuities it is not a pointwise second-order error theorem.

![Binomial upwind pulse weights on an infinite grid at CFL one half for three step counts.](/assets/learning/differential-binomial-en.svg)

| n, CFL=1/2 | Mean cell shift | Variance in cell units |
|---|---:|---:|
| 4 | 2 | 1 |
| 8 | 4 | 2 |
| 16 | 8 | 4 |

```python
from fractions import Fraction as F
from math import comb
nu = F(1, 2)
for n in (4, 8, 16):
    weights = [F(comb(n, j))*nu**j*(1-nu)**(n-j) for j in range(n+1)]
    mass = sum(weights)
    mean = sum(j*w for j, w in enumerate(weights))
    variance = sum((j-mean)**2*w for j, w in enumerate(weights))
    assert mass == 1 and mean == n*nu and variance == n*nu*(1-nu)
    print(n, mass, mean, variance)
```

Exact fractions verify mass, mean and variance independently of the live finite-domain solver. The picture uses cell offsets from the mean, and a wider spread at more steps; periodic wrapping and inflow boundaries require their own interpretation. Shared fluxes certify the balance, while this separate stability argument explains why the conservative CFL>1 experiment can still fail.

## Exercise: match physical time before comparing grids

The time step is CFL dx/c, so after n updates the physical time is n CFL/(Nc). Changing N at fixed update count changes the time being observed. For N=12, c=1, CFL=0.8 and eight updates, calculate t=8/15. Verify periodic mass remains one quarter, then switch to inflow and reconcile its changed mass with accumulated flux. Finally set CFL=1.2: identify an overshoot or negative cell while checking that the mass balance can still hold.

Further reading: [Clawpack conservation-law and Riemann framework](https://www.clawpack.org/riemann_book/html/Burgers.html). The linear upwind scheme, exact cell averages and boundary accounting are independently implemented.
