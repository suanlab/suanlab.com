## Sets and functions have separate convexity conditions

A convex set contains the entire segment between any two of its points. With t between zero and one, that segment is z=(1−t)x+ty. An interval is convex; the union [−2,−1]∪[1,2] is not. Its endpoints −1 and one belong to the set, but their midpoint zero falls in the missing gap.

A convex function must have a convex domain and satisfy the chord inequality:

$$
f((1-t)x+ty)\le(1-t)f(x)+tf(y),\qquad x,y\in\operatorname{dom}f,\quad0\le t\le1.
$$

The lab treats the selected set and the function domain separately. In an optimization problem the effective feasible domain is their intersection; a nonconvex selected set can sometimes have a convex intersection with a restricted function domain. Do not classify a whole problem from one label alone.

## Read the Jensen gap

Blue shows the function and orange connects the endpoint function values. Their vertical difference at the mixture point is the Jensen gap, defined as chord height minus function value. For f(x)=ax², direct expansion gives

$$
(1-t)ax^2+tay^2-a((1-t)x+ty)^2
=a\,t(1-t)(x-y)^2.
$$

With a=1,x=−1,y=1,t=1/2, the gap is one. With a=−1 it becomes −1 and gives a concrete counterexample to convexity. When a=0 the function is affine and every gap vanishes. For a>0 the second derivative is 2a, so the function is also strongly convex with parameter 2a under the standard Euclidean convention.

## Nonsmooth and restricted-domain examples

The absolute-value function is convex despite its cusp: differentiability is not part of the definition. Its chord can touch or lie strictly above the graph depending on whether the chosen segment crosses zero.

The negative logarithm is convex on positive inputs because its second derivative is 1/x²>0. Zero and negative inputs are outside its domain. If either endpoint is invalid, no ordinary finite Jensen gap is assigned, even if the mixture happens to be positive. The plot shows only a finite positive window; it does not draw a fictitious finite value at zero.

## A finite experiment is not the universal statement

One negative gap disproves convexity, but a positive gap at one pair cannot prove it. The definition quantifies over every pair in the domain and every t in [0,1]. Analytic expansion, a valid second-derivative criterion, or a composition theorem can supply a proof under their respective assumptions. Sampling more points is useful for debugging, not a replacement for those hypotheses.

## Epigraphs and separation turn pictures into certificates

The epigraph consists of points (x,r) with r≥f(x). It is convex exactly when the extended-valued function f is convex: combine two epigraph points for one direction of the proof, and choose their heights f(x),f(y) for the other. Closedness is an additional condition, equivalent to lower semicontinuity for a proper function. Convexity alone does not guarantee attainment of an infimum.

A nonempty closed convex set C in finite-dimensional Euclidean space has a unique closest point q to any p. Existence follows from restricting the continuous squared distance to a sufficiently large compact ball; strict convexity gives uniqueness. Differentiating toward every x in C yields a separating hyperplane:

$$
\langle p-q,x-q\rangle\le0\quad(x\in C),\qquad
\langle p-q,p-q\rangle>0\quad(p\notin C).
$$

The normal is the residual p−q. For the triangle with vertices (0,0),(1,0),(0,1), take p=(2,1/2),q=(1,0). Checking the three vertices suffices for the inequality on every convex combination. This supplies an actual certificate instead of guessing a separating line from a drawing.

| Triangle vertex x | (p−q)·(x−q) |
|---|---:|
| (0,0) | -1 |
| (1,0) | 0 |
| (0,1) | -1/2 |

Numerically, subtracting nearly identical chord heights can invent a negative log Jensen gap. The lab now uses the exact quadratic identity, sign-aware absolute-value geometry and a local logarithm series. Displayed heights remain separately rounded. For x=1,y=1+2⁻⁵²,t=1/2, the quadratic gap is 2⁻¹⁰⁶ and the negative-log gap is approximately 2⁻¹⁰⁷, both positive. This does not turn a finite example into a global convexity proof.

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
p,q=(F(2),F(1,2)),(F(1),F(0))
normal=[p[i]-q[i] for i in range(2)]
for x in [(0,0),(1,0),(0,1)]:
    product=sum(normal[i]*(x[i]-q[i]) for i in range(2))
    assert product <= 0
    print(x,product)
delta=F(1,2**52)
print('adjacent-input quadratic gap:',delta*delta/4)
```

## Exercise and explained answer

Keep the convex quadratic and select the two-interval set. At x=−1,y=1,t=1/2, the function passes the chord test while the set fails segment membership. Next choose −log with x=0,y=1. The endpoint-domain indicator is false and the gap is undefined. Explain why interpolating a plotted line cannot create the missing logarithm value.

## Connection to AI

Convex losses, regularizers and feasible sets enable global optimality certificates when combined correctly. A convex-looking loss slice does not certify a neural network objective in all parameters, and a convex function paired with incompatible constraints needs a separate feasible-domain analysis.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
