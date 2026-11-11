## A lower bound and an equality theorem are different claims

For feasible inequality multipliers, the Lagrangian infimum gives a lower bound on every feasible primal objective. Maximizing those bounds gives d*≤p*: weak duality. Strong duality means equality of optimal values. Whether a supremum is attained by a finite multiplier is a separate question.

Slater's condition is a sufficient qualification for convex problems with finite optimum: a feasible point in the appropriate relative interior strictly satisfies the nonlinear inequalities, with the usual affine-equality qualifications. Failure of this sufficient condition does not by itself determine whether the values coincide.

## Strict feasibility with an attained multiplier

The first mode uses min (x−2)²/2 subject to x≤1. Choosing x=0 is strictly feasible. Primal optimum is 0.5 at x=1, and

$$
d(\lambda)=\lambda-\lambda^2/2,\quad\lambda\ge0,
\qquad\lambda^*=1,\quad d^*=0.5.
$$

The plotted dual curve touches the primal reference level. The witness-size control is unused in this regular example.

## Equal values without a finite dual maximizer

Now minimize x subject to x²≤0. Only x=0 is feasible, so p*=0, and no strict feasible point exists. For λ>0, minimizing x+λx² gives

$$
x_\lambda=-1/(2\lambda),\qquad d(\lambda)=-1/(4\lambda).
$$

At λ=0 the infimum is negative infinity, witnessed by x=−M as M grows. For positive λ, d approaches zero only as λ→∞. Thus d*=p*=0, but no finite optimal multiplier exists. KKT stationarity at the primal optimum would require 1+2λ·0=0, which is impossible. Convexity alone did not guarantee multipliers.

## A genuine positive gap

The third example is adapted explicitly from Boyd and Vandenberghe's exercise 5.21:

$$
\min_{x\in\mathbb R,\ y>0} e^{-x}
\quad\text{subject to}\quad x^2/y\le0.
$$

The perspective quadratic x²/y is convex on y>0 and nonnegative. Feasibility forces x=0, giving attained primal value one. For every finite λ≥0 the Lagrangian is nonnegative, while the sequence x=M,y=M⁴ gives e⁻ᴹ+λ/M²→0. Therefore d(λ)=0 for every feasible λ and the optimal-value gap is one.

Those sequence points lie in the function domain but are not primal feasible. They witness an unconstrained Lagrangian infimum, not better feasible solutions. The dual supremum is attained here, although its inner Lagrangian infimum is not. Distinguish all three attainment statements.

## Perturbation geometry explains both failures

![Two convex perturbation value functions distinguish a continuous value with no finite supporting slope from an upward endpoint jump producing a positive duality gap.](/assets/learning/convex-perturbation-en.svg)

After eliminating affine equalities, define p(u) as the infimum value with inequality bounds relaxed from zero to u. Mixing feasible witnesses proves convexity of p. For this proof, assume the strictly feasible version of Slater: a finite point in the relative interior of the common convex domain satisfies every inequality strictly, and p(0) is finite. Then zero lies in the interior of the effective perturbation domain and p has a supporting subgradient s there. Monotonicity under relaxation gives s≤0. Put λ=−s≥0. For every x in the objective domain,

$$
p(u)\ge p(0)-\lambda^Tu,\qquad
f(x)+\lambda^Tg(x)\ge p(0).
$$

The second inequality follows by choosing u=g(x). Taking the Lagrangian infimum and combining with weak duality proves equality of values and an attained multiplier. The support argument supplies the reason behind the qualification; a numerical dual plot cannot supply it.

For min x with x²≤u, the perturbation value is −√u on u≥0. It is continuous at zero but has no finite supporting slope there, explaining multiplier nonattainment. For the perspective example, p(0)=1 while p(u)=0 for every u>0 and p(u)=+∞ for u<0. Its upward endpoint jump makes the epigraph nonclosed; the lower semicontinuous hull has value zero at the origin, matching the dual bound in this example.

| Perturbation model | Value at zero | Right-hand limit | Finite supporting multiplier |
|---|---:|---:|---|
| x²≤u, minimize x | 0 | 0 | No |
| x²/y≤u, minimize exp(−x) | 1 | 0 | No support certifying primal value |

For u>0 in the second model, choosing y=M²/u permits any x=M and sends the objective to zero. These are feasible for the perturbed problem; at u=0 they cease to be feasible. This is different from claiming a better feasible solution to the original problem.

## Reproduce the calculation in Python

```python
from math import exp,sqrt
for u in [1,.1,.01,.0001]:
    print('u:',u,'first value:',-sqrt(u))
    for M in [2,4,8,16]:
        y=M*M/u
        assert abs(M*M/y-u) < 1e-14
    print('second feasible objective at M=16:',exp(-16))
# The analytic sequence M -> infinity proves the infimum, not this finite list.
```

## Exercise and interpretation

In mode two increase M from four to sixteen. The witness Lagrangian value decreases toward zero, while the primal optimum remains one. Explain why this supports the calculated dual infimum without refuting weak duality. Compare with the second mode, where the value gap is zero but multiplier attainment fails.

When using a dual certificate in AI optimization, state the primal/dual domains, convexity and qualification assumptions. A plotted finite grid cannot establish an unbounded-domain infimum; the explicit sequences and lower-bound arguments provide the missing reasoning.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
