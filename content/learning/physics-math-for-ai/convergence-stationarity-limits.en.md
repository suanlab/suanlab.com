## State exactly what converges

Optimization papers can discuss convergence of objective values, gradient norms, iterates, distributions or distance to a solution set. These are different claims. A finite stopping test is further evidence about the current calculation, not an automatic asymptotic theorem.

The lab compares f=x²/2, f=(x²−1)²/4 and f=exp(−x), all with infimum zero. Blue plots gradient magnitude and orange plots objective minus infimum. The examples are dimensionless, but their two vertical quantities remain different measures; a visually smaller gradient is not a universal bound on objective error.

## A convex quadratic links several errors

For f=x²/2, the gradient is x, the unique optimizer is zero, and GD gives xₖ=(1−η)ᵏx₀. Under the displayed positive step range this contracts. The relations

$$
|\nabla f(x)|=|x|,\qquad f(x)-f^*=x^2/2
$$

link parameter distance, gradient and objective error because this particular function has strong curvature. They are not universal identities for arbitrary objectives. Bounds between such quantities need assumptions such as strong convexity, an error-bound condition or a Polyak–Łojasiewicz inequality.

## Zero gradient can mean a local maximum

For the double well,

$$
f(x)=\frac{(x^2-1)^2}{4},\qquad
f'(x)=x(x^2-1),\qquad f''(x)=3x^2-1.
$$

The global minima are ±1. Starting exactly at zero leaves every deterministic GD iterate zero. The gradient and step magnitude vanish, yet objective error stays 0.25 and Hessian is −1. A first-order stationary point is therefore not a global optimum certificate in nonconvex optimization.

Starting nearby can lead toward a minimum, but that observation does not change the exact-zero counterexample. The initialization and perturbation mechanism belong in the algorithm description.

## Vanishing objective error need not yield a finite minimizer

The exponential has infimum zero only as x→∞. Its gradient magnitude equals its objective value, both positive at every finite point. With a positive constant step, GD increases x by ηexp(−x). If x had a finite limit, those increments would tend to a positive constant, a contradiction. Thus x grows without a finite limit while the objective and gradient approach zero.

This is a case where bounded-below descent and vanishing gradients do not imply bounded iterates or an attained minimum. Compactness and existence assumptions cannot be omitted from stronger convergence claims.

## A gradient tolerance depends on coordinates and units

Change coordinates to u=cx in the same physical quadratic f=x²/2. Then f=u²/(2c²), its u-gradient is x/c and its strong-convexity constant is 1/c². Making c large produces a small numerical gradient without changing the physical point or objective error. A raw threshold on gradient components therefore depends on feature and parameter units.

Under a known μ-strong-convexity bound, the coordinate-scaled objective certificate remains consistent:

$$
f(x)-f^*\le\frac{\|\nabla f(x)\|^2}{2\mu},\qquad
\|x-x^*\|\le\frac{\|\nabla f(x)\|}{\mu}.
$$

The quadratic attains equality in both. In u coordinates the distance bound refers to u-distance; divide by c to recover physical x-distance. Omitting the curvature transformation mixes units and can turn a tolerance into a misleading success flag.

| c at physical x=1 | u-gradient | Objective error |
|---:|---:|---:|
| 1 | 1 | 1/2 |
| 1,000 | 1/1,000 | 1/2 |
| 1,000,000 | 1/1,000,000 | 1/2 |

For nonconvex objectives without an error bound, the original maximum and escaping exponential remain counterexamples to stronger conclusions. A finite floating-point update may also stagnate because its displacement rounds away. Report step size, scale-aware residuals, objective behavior, curvature assumptions and the actual stopping target separately. Numerical stagnation, first-order stationarity, second-order stationarity and global optimality are different outcomes.

## Reproduce the calculation in Python

```python
from fractions import Fraction as Q
for c in [Q(1),Q(1000),Q(1000000)]:
    physical=Q(1); u=c*physical
    gradient=physical/c; mu=1/(c*c)
    objective_bound=gradient*gradient/(2*mu)
    coordinate_distance=abs(gradient)/mu
    assert objective_bound==Q(1,2) and coordinate_distance/c==physical
    print('scale:',c,'gradient:',gradient,'gap bound:',objective_bound,'physical distance:',coordinate_distance/c)
```

## Exercise and stopping criteria

Use the quadratic with x₀=1,η=0.2 and ten steps. Gradient magnitude is 0.8¹⁰≈0.107374 and objective error approximately 0.005765. Compare the double-well zero start, whose gradient is smaller but objective error larger.

Set η=0 at a nonstationary point. Step magnitude becomes zero even though the gradient remains nonzero, demonstrating why a step-only tolerance can be misleading. For η>0, a small step can still reflect a tiny learning rate rather than a small gradient.

When reporting a solver result, name the objective, feasible domain, stationarity norm, tolerances and any certificate of global optimality. Distinguish empirical finite-run behavior from a theorem about the entire sequence.

Further reading: [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
