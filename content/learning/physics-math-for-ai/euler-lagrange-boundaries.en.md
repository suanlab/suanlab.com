## Interior equations come with boundary assumptions

Consider minimizing J[y]=∫₀¹[(y′)²/2−qy]dt for a constant load q≥0. The derivative term penalizes large slopes, while the load rewards larger y. The admissible endpoint conditions determine which curves and which variations are allowed.

The lab compares zero values at both endpoints with a fixed left endpoint and a free right endpoint. These are different optimization problems, even though their interior integrand is identical.

## Derive the Euler equation

For a smooth candidate and direction h, integrate by parts:

$$
\delta J[y;h]=\int_0^1(y'h'-qh)\,dt
=[y'h]_0^1+\int_0^1(-y''-q)h\,dt.
$$

Directions supported strictly inside the interval force −y″−q=0 at a stationary curve. For weak Sobolev solutions the same statement begins with an integral identity; pointwise differentiation requires sufficient regularity. This polynomial example is smooth, so both descriptions agree.

If both endpoint values are fixed, h(0)=h(1)=0 and the boundary term vanishes automatically. If only the left endpoint is fixed, h(1) is arbitrary. Stationarity then also requires the natural condition y′(1)=0. A free endpoint does not mean that its variation vanishes.

## Solve the two boundary-value problems

With fixed endpoints the unique minimizing curve is y*=(q/2)t(1−t), with J*=−q²/24. With a free right endpoint it is y*=q(t−t²/2), with J*=−q²/6. The latter has nonzero endpoint value but zero endpoint slope.

For q=2 these become t(1−t) and 2t−t². Their minima are −1/6 and −2/3. Removing a constraint enlarges the feasible set, so the lower free-end minimum is consistent with optimization geometry.

## Stationarity versus minimality

The Euler equation is generally a stationarity condition, not a universal global-minimum theorem. Here convexity supplies the stronger conclusion. For an admissible h=y−y*, stationarity removes the linear term and

$$
J[y]-J[y^*]=\frac12\int_0^1(h')^2\,dt\ge0.
$$

Because the left endpoint is fixed, a nonzero constant difference is not admissible; equality therefore identifies the unique minimizer. Without appropriate boundary or function-space assumptions, uniqueness and even existence require further work.

## Turn the weak residual into an energy-error certificate

An interior differential residual is only part of the free-end problem. Let V={v∈H¹(0,1):v(0)=0}, with norm ‖v′‖₂, and let y be a smooth candidate with the correct left endpoint. Write e=y−y*, r=−y″−q and b=y′(1). Integration by parts and the exact solution's weak equation give

$$
R(v):=\int_0^1rv\,dt+bv(1)=\int_0^1e'v'\,dt,
\qquad
\|R\|_{V^*}=\|e'\|_2,\qquad
J[y]-J[y^*]=\tfrac12\|R\|_{V^*}^2.
$$

The dual norm is a supremum over unit-norm test functions. Cauchy–Schwarz proves the upper inequality; choosing v=e/‖e′‖₂ proves equality when e is nonzero. The zero case follows directly. This converts a weak residual into the actual energy error for this coercive quadratic problem.

To obtain a simpler computable bound, use ‖v‖₂≤‖v′‖₂ and |v(1)|≤‖v′‖₂. Therefore

$$
\|e'\|_2\le\|r\|_2+|b|,\qquad
0\le J[y]-J[y^*]\le\tfrac12(\|r\|_2+|b|)^2.
$$

These constants are sufficient, rather than optimal for every term. For fixed endpoints the test functions vanish at both ends, so there is no b term; candidates must then satisfy both essential endpoint values. If a candidate violates the essential values, first account for a boundary lifting instead of applying this certificate unchanged.

## An exactly zero interior residual can still give the wrong solution

Take the free-end solution and add e(t)=a t+c t². The interior residual is −2c and the natural boundary residual is a+2c. The excess is (a²+2ac+4c²/3)/2. In particular e=t/2 has zero interior residual but excess 1/8, because its right-end slope is wrong.

| (a,c) | r | b | Excess | Bound |
|---|---:|---:|---:|---:|
| (1/2,0) | 0 | 1/2 | 1/8 | 1/8 |
| (0,1/2) | −1 | 1 | 1/6 | 2 |
| (1,−1/2) | 1 | 0 | 1/6 | 1/2 |

```python
from fractions import Fraction as F

# Free-right-end problem. e(t)=a*t+c*t*t is the candidate error.
for a, c in ((F(1, 2), F(0)), (F(0), F(1, 2)), (F(1), F(-1, 2))):
    interior, boundary = -2*c, a+2*c
    derivative_squared_integral = a*a+2*a*c+F(4, 3)*c*c
    gap = derivative_squared_integral/2
    certificate = (abs(interior)+abs(boundary))**2/2
    assert 0 <= gap <= certificate
    print("a,c:", a, c, "r,b:", interior, boundary,
          "gap,bound:", gap, certificate)
# Check the bound on an independent rational polynomial grid.
for i in range(-8, 9):
    for j in range(-8, 9):
        a, c = F(i, 8), F(j, 8)
        gap = (a*a+2*a*c+F(4, 3)*c*c)/2
        assert gap <= (abs(2*c)+abs(a+2*c))**2/2
print("289 polynomial residual certificates verified")
```

This independent polynomial example and its 289-case rational grid do not add controls to the lab's sine-based family. They explain why physics-informed training should monitor the specified boundary conditions as well as interior equations. For nonconvex energies, noncoercive operators or approximate quadrature, the equality and certificate require new arguments; a small sampled training loss alone is insufficient.

## Exercise and explained answer

The candidate is a polynomial basis curve plus b sin πt. In free-end mode use q=a=2 and b=0.2. Its right-end slope is −0.2π≈−0.628319. Its interior residual is π²·0.2 sin πt, so the candidate violates both interior stationarity and the natural condition. Its objective excess is 0.01π²≈0.098696.

Set b=0 and verify both residuals vanish. Then switch boundary mode and update a to q/2. Explain why reusing the same interior differential equation without changing the boundary conditions would solve the wrong problem. This distinction carries directly into variational PDE models and physics-informed learning.

Further reading: [MIT, Calculus of Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
