## A differential becomes a gradient through a metric

A loss differential tells how the objective changes along a tangent vector. A gradient is the vector representing that differential under a chosen inner product. Natural gradient uses the statistical model's Fisher metric rather than the Euclidean metric of a particular parameter chart.

The lab minimizes Bernoulli cross entropy L(p)=−r log p−(1−r)log(1−p), with current probability p and target r both interior. Its minimum is at p=r. Objective excess is KL(Ber(r)||Ber(p)), which makes the direction of KL explicit.

## Derive the invariant tangent direction

The probability-coordinate derivative is (p−r)/[p(1−p)]. Multiplying its negative by inverse Fisher gives

$$
\dot p=-I_p^{-1}\frac{dL}{dp}=r-p.
$$

In logit coordinates the derivative is p−r and information is p(1−p), giving θ̇=(r−p)/[p(1−p)]. Mapping this vector back with dp/dθ=p(1−p) recovers exactly ṗ=r−p. This is a statement about tangent vectors and the corresponding continuous flow.

## Finite Euler stepping depends on coordinates

Probability Euler gives p⁺=p+η(r−p). Logit Euler instead gives

$$
p^+=\operatorname{sigmoid}\left(\operatorname{logit}(p)+
\eta\frac{r-p}{p(1-p)}\right).
$$

The two agree to first order in η, not at arbitrary finite η. Nonlinear coordinate changes do not commute with finite additive stepping. The exact natural flow solves p(t)=r+(p₀−r)e⁻ᵗ and is coordinate invariant when the same flow is represented consistently.

Blue plots that exact flow against elapsed step parameter. Orange plots the selected coordinate's single Euler update as a function of its step size; it is not a multi-iteration training trajectory.

## Worked coordinate comparison

At p=0.2,r=0.8,η=0.1, probability Euler gives 0.26, logit Euler approximately 0.266727 and exact flow approximately 0.257098. The mapped tangent velocities still agree at 0.6. Finite disagreement therefore does not refute tangent invariance.

Ordinary probability GD gives 0.575 in the same example. At p=0.1,r=0.9,η=0.5 it instead gives approximately 4.544444, outside the probability domain. Natural probability Euler stays at 0.5 because its displayed step range forms a convex combination of p and r. Ordinary logit GD remains interior but follows a different metric-dependent direction.

## A loss guarantee for this flow and probability step

Cross entropy is convex in its probability argument. The exact flow is a convex combination p(t)=e⁻ᵗp₀+(1−e⁻ᵗ)r. Convexity therefore gives an explicit objective guarantee. Probability Euler has the same argument with mixing weight η when 0≤η≤1:

$$
L(p(t))-L(r)\le e^{-t}[L(p_0)-L(r)],\qquad
L(p^+)-L(r)\le(1-\eta)[L(p)-L(r)].
$$

This proof uses the specified Bernoulli loss, interior endpoints and the probability update. It is not a theorem for every natural-gradient discretization or the logit Euler step at every step size. Along the continuous flow, direct differentiation also gives dL/dt=−(p−r)²/[p(1−p)].

## Why adding the same identity damping breaks tangent invariance

Replace Fisher I by I+d in each scalar chart. Set a=dp/dθ=p(1−p). A probability-coordinate damping d must transform to a²d in the logit chart to represent the same metric. Adding the same numeric d to both charts instead yields different mapped velocities.

For p=1/5,r=4/5, the exact rational calculation is:

| d | Probability velocity | Mapped logit velocity |
|---|---:|---:|
| 0 | 3/5 | 3/5 |
| 1/10 | 75/127 | 24/65 |
| 1 | 15/29 | 12/145 |

```python
from fractions import Fraction as F
p, target = F(1, 5), F(4, 5)
a = p*(1-p)
gp, gt = (p-target)/a, p-target
for damping in (F(0), F(1, 10), F(1)):
    vp = -gp/(1/a+damping)
    mapped = a*(-gt/(a+damping))
    covariant = a*(-gt/(a+a*a*damping))
    assert covariant == vp
    if damping: assert mapped != vp
    print(damping, vp, mapped)
```

The last assertion uses the transformed damping and recovers tangent agreement. This is a metric-coordinate calculation; it does not implement a large-model approximate Fisher solver. Both damping and finite stepping must be specified before extending an invariance claim.

## Exercise and practical limits

Halve η in the first example and compare the two natural Euler probabilities. Their discrepancy should decrease quadratically to leading order here. Derive the natural directional loss derivative −(p−r)²/[p(1−p)] and verify that it is nonpositive.

The experiment uses an exact nonsingular scalar Fisher. Large neural models require structured or approximate solves, and coordinate-specific damping can alter invariance. Neither an approximate Fisher nor a fixed learning rate inherits every property of the exact flow automatically. State the metric, parameterization, solver and discretization before claiming invariance or convergence.

Further reading: [Martens, Natural Gradient Method](https://www.jmlr.org/papers/v21/17-678.html).
