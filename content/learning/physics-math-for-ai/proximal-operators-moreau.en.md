## An optimization step that includes the penalty

A gradient step linearizes a differentiable objective. A proximal step instead solves a small optimization problem containing the penalty exactly. For a proper closed convex function f and η>0,

$$
P(y)=\operatorname{prox}_{\eta f}(y)
=\arg\min_x\left[f(x)+\frac{(x-y)^2}{2\eta}\right].
$$

The quadratic term makes the objective strongly convex, so the minimizer is unique even when f has corners or an extended-valued indicator. Think of a spring pulling toward y while the penalty resists. Increasing η weakens the spring relative to f.

## Optimality through a subgradient

The necessary and sufficient condition is (y−P(y))/η∈∂f(P(y)). The lab displays this required subgradient, the actual subdifferential interval and their distance. A small objective value alone does not establish stationarity.

For f(x)=λ|x|, the zero-point subdifferential is [−λ,λ]. If |y|≤ηλ the spring force can be balanced at zero. Outside that region the penalty has constant slope of the appropriate sign. Consequently

$$
P(y)=\operatorname{sign}(y)\max(|y|-\eta\lambda,0).
$$

With y=2, λ=0.5 and η=1, the answer is 1.5 and the proximal objective is 0.875. The threshold includes η; using λ alone changes the method when the step size changes.

## Projection and ridge shrinkage

For an indicator of [−λ,λ], the penalty is zero inside and infinity outside. The proximal point is the Euclidean projection clip(y,−λ,λ), independent of η. Here λ means box radius. Infinite values are omitted from the finite-height plot, rather than drawn as zero.

For f(x)=λx²/2, stationarity gives P(y)=y/(1+ηλ). At y=2, λ=1 and η=0.5 the result is 4/3. An explicit gradient step would give one, so these operations are different. Ridge shrinkage is an implicit update using the derivative at the new point.

## The Moreau envelope smooths the value function

Define Eηf(y) as the minimized proximal objective. Despite the absolute-value corner, this value function is differentiable for proper closed convex f in finite dimensions:

$$
\nabla E_\eta f(y)=\frac{y-P(y)}{\eta}.
$$

This derivative is with respect to the input y, not the optimization variable x. The displayed proximal objective value is the envelope value; the required subgradient is also its derivative. For the box indicator the envelope is squared distance to the box divided by 2η.

Monotonicity of subgradients at two proximal points gives ‖P(y)−P(z)‖²≤(P(y)−P(z))(y−z). This firm nonexpansiveness implies a Lipschitz constant of one, but not a strict contraction in every case. An identity map is an immediate counterexample to strictness.

## Moreau decomposition connects a primal step and a dual projection

![Soft thresholding and a scaled dual clipping projection sum to the input, while the Moreau envelope smooths the corner of the absolute-value penalty.](/assets/learning/convex-moreau-en.svg)

Let p=proxηf(y) and s=(y−p)/η. Optimality says s∈∂f(p), which is equivalent to p∈∂f*(s) by Fenchel equality. Rearranging the latter is exactly the optimality condition for the dual proximal problem, proving

$$
\operatorname{prox}_{\eta f}(y)+\eta\operatorname{prox}_{f^*/\eta}(y/\eta)=y.
$$

For f=λ|x|, the conjugate is the indicator of [−λ,λ]. Soft thresholding and a scaled clipping projection therefore add up to the input. This gives an independent check of both scaling factors, not just another way to memorize the threshold.

Strong convexity improves nonexpansiveness. If f is μ-strongly convex, strong monotonicity of ∂f at p,q gives (p−q)·(y−z)≥(1+ημ)‖p−q‖². Cauchy–Schwarz then gives contraction factor 1/(1+ημ). Absolute value is not strongly convex; its proximal map preserves distances between two inputs on the same outer branch.

| y, with λ=4/5 and η=3/4 | Primal prox p | Scaled dual term ηs |
|---:|---:|---:|
| -2 | -7/5 | -3/5 |
| -3/5 | 0 | -3/5 |
| 0 | 0 | 0 |
| 3/5 | 0 | 3/5 |
| 2 | 7/5 | 3/5 |

The envelope derivative is now evaluated from each analytic proximal formula. For ridge with y=1,η=1,λ=10⁻²⁰, the displayed point rounds to one but the derivative remains approximately 10⁻²⁰. Subtracting that rounded point from y would incorrectly give zero.

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
lam,eta=F(4,5),F(3,4)
def clip(y): return max(-lam,min(lam,y))
for y in [F(-2),-F(3,5),F(0),F(3,5),F(2)]:
    p=(1 if y>=0 else -1)*max(abs(y)-eta*lam,0)
    s=clip(y/eta)
    assert p+eta*s==y and lam*abs(p)==s*p
    print(y,p,eta*s)
tiny=F(1,10**20); p=1/(1+tiny); derivative=tiny*p
assert 1-p==derivative and derivative>0
print('rounded ridge point:',float(p),'analytic derivative:',float(derivative))
```

## Exercise and boundary cases

Set λ=0 in each mode. Absolute-value and ridge penalties disappear, so their proximal maps become identity. The indicator mode instead becomes projection onto the singleton {0}; its normal cone is the whole real line. Explain why the same numerical parameter has different meanings across modes.

Check the envelope derivative by a centered finite difference in y away from a threshold, then across a threshold with a smaller step. The envelope derivative stays continuous even when the proximal map changes its formula. This is one reason proximal methods connect nonsmooth optimization to smooth value functions.

Further reading: [Parikh and Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).
