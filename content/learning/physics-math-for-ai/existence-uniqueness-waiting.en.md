## Existence and uniqueness ask different questions

Existence asks whether the equation and data admit a solution. Uniqueness asks whether two solutions with those same data must coincide on their common interval. A solver returning one trajectory answers neither question by itself. In particular, visually separating curves are not evidence of nonuniqueness unless their initial conditions and governing equation match.

For an ODE y'=f(t,y), continuity of the field can provide local existence under the usual local-domain assumptions. A local Lipschitz bound in the state is a familiar sufficient hypothesis for uniqueness. These are local results: they do not automatically guarantee a solution remains bounded or exists for all future time.

## A linear field gives a controlled comparison

For y'=lambda y, the field is globally Lipschitz with constant |lambda|. The solution starting at zero is identically zero. The solution starting at epsilon is epsilon exp(lambda t). Positive lambda separates them, yet uniqueness is intact because their initial values differ.

$$
|y_\epsilon(t)-y_0(t)|=|\epsilon|e^{\lambda t}.
$$

The lab reports that initial-data difference explicitly. Sensitivity to initial data and nonuniqueness are distinct properties. A unique flow can amplify uncertainty, making prediction difficult without making the mathematical initial-value problem ambiguous.

## Construct several solutions with identical data

Now use the continuous field f(y)=2 sqrt(max(y,0)) and initial value zero. For every nonnegative waiting time c, define

$$
y_c(t)=(t-c)_+^2,\qquad t\ge0.
$$

Before c, both the state and derivative vanish. After c, the derivative is 2(t-c), equal to twice the square root of the state. At c the one-sided derivatives both equal zero, so the piecewise curve is continuously differentiable and genuinely satisfies the ODE, rather than merely fitting a plot.

All members start at zero. Taking c=0 and c=1/2 gives different solutions at t=1: one and one quarter. This explicit construction proves nonuniqueness for this problem. A solution that stays zero forever is another member of the broader family.

## Why the sufficient condition fails

Near zero, |f(h)-f(0)|/|h|=2/sqrt(h) for positive h. It grows without bound, so no finite local Lipschitz constant works. Failure of a sufficient theorem alone would not prove nonuniqueness for an arbitrary field; the constructed distinct solutions are the decisive evidence here.

## A local theorem, a quantitative bound, and finite-time escape

On a closed rectangle around (0,y₀), suppose f is continuous, bounded by M, and Lipschitz in y with constant L. Choose h so Mh fits inside the rectangle and Lh<1. The Picard map $T[y](t)=y_0+\int_0^t f(s,y(s))\,ds$ maps the closed continuous-function ball into itself and contracts its supremum distance by at most Lh. The complete-space fixed-point theorem gives a local solution and uniqueness in that ball. Continuing the argument gives uniqueness on common intervals, not automatic existence for all time.

For a differentiable approximation v with residual ρ=v′−f(t,v), Gronwall gives, on a common domain with the same L,

$$
|v(t)-y(t)|\le e^{Lt}|v(0)-y_0|+
\int_0^t e^{L(t-s)}|\rho(s)|\,ds.
$$

A uniform residual ε contributes ε(expm1(Lt))/L, with limit εt at L=0. This also bounds model perturbations in neural ODEs; a residual at sampled times alone is insufficient.

| L=1, interval h | Picard distance factor Lh | Strict contraction argument |
|---|---:|---|
| 1/4 | 1/4 | applies |
| 1/2 | 1/2 | applies |
| 1 | 1 | inconclusive by this bound |

```python
from fractions import Fraction as F
from math import exp, factorial
for T in (F(1, 4), F(1, 2), F(1)):
    coefficients = [F(1)]
    for iteration in range(8):
        coefficients = [F(1)] + [a/F(i+1) for i, a in enumerate(coefficients)]
    value = sum(a*T**i for i, a in enumerate(coefficients))
    error = abs(float(value)-exp(float(T)))
    bound = exp(float(T))*float(T)**9/factorial(9)
    assert error <= bound + 1e-15
    print(T, error, bound)
assert F(1)/(1-F(3, 4)) == 4  # y'=y^2 before its pole.
```

The code constructs Picard polynomials for y′=y and checks a Taylor remainder. Failure of this contraction bound at h=1 does not imply nonuniqueness. Conversely, y′=y², y(0)=1 is locally Lipschitz but y=1/(1−t) escapes at t=1: local uniqueness is compatible with finite maximal lifetime. The waiting solutions above prove nonuniqueness by construction, not merely by failure of a theorem's hypotheses.

## Exercise: diagnose the right phenomenon

Set the waiting time to one and query t=1/2. The waiting solution is zero while the immediate solution is one quarter, and both initial values are zero. Then switch to the linear model with epsilon 0.1. At t=1 the curves again differ, but now their initial values differ by 0.1. State which comparison disproves uniqueness and verify the ODE residual for the waiting solution exactly at its departure time.

Further reading: [MIT first-order differential equations](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). The waiting family and matching-data comparison are constructed here.
