## Curvature limits a useful step

An L-smooth differentiable function has a gradient that is L-Lipschitz in the specified norm. In Euclidean space this yields a global quadratic upper model. When a twice-differentiable function has Hessian operator norm at most L everywhere, that bound is sufficient for smoothness. Merely measuring curvature at one point does not establish a global constant.

The lab chooses f(x)=(x₁²+κx₂²)/2, κ≥1. Its constant Hessian is diag(1,κ), so L=κ and strong-convexity parameter μ=1. The initial vector is always (1,1), allowing the learning rate to be compared against both curvature directions.

## Derive the descent lemma update

Substitute the GD step s=−η∇f(x) into the smooth upper model:

$$
f(x+s)\le f(x)+\nabla f(x)^Ts+\frac L2\|s\|^2,
\qquad
f(x^+)\le f(x)-\eta(1-\eta L/2)\|\nabla f(x)\|^2.
$$

For 0<η<2/L the coefficient of the gradient norm is positive. This guarantees descent whenever the gradient is nonzero. At the boundary the bound only guarantees nonincrease; beyond it the inequality still holds but no longer promises decrease.

The orange value at iteration k is the upper bound computed at iteration k−1. It is not another optimizer trajectory or an estimate of the unknown optimum. Blue is the actual objective. At k=0 both start with the initial value.

## Read every eigendirection

For this quadratic the error recurrence is exact:

$$
x_{i,k+1}=(1-\eta\lambda_i)x_{i,k},\qquad
\rho=\max_i|1-\eta\lambda_i|.
$$

Every mode contracts exactly when ρ<1. With κ=8, η=0.2 produces factors 0.8 and −0.6: the steep coordinate alternates sign while shrinking. Sign oscillation therefore does not itself mean divergence.

At η=0.25 the steep factor is −1. Its coordinate magnitude remains one and its objective contribution remains four. At η=0.3 the factor is −1.4, so that direction grows despite the gentle coordinate still shrinking by factor 0.7. A single coordinate or averaged-looking plot can conceal the unstable mode.

## Conditioning and stronger claims

Taking η=1/L is a convenient sufficient choice, not the only stable choice. For a positive definite quadratic with eigenvalues between μ and L, minimizing the worst-case error factor gives η=2/(L+μ), with factor (L−μ)/(L+μ). This conclusion uses the quadratic spectral model and should not be applied blindly to arbitrary nonconvex objectives.

An L-smooth nonconvex function can have descent without global-optimum convergence. Strong convexity is additional structure. These assumptions support different conclusions and must remain separate in a training argument.

## Telescope descent before claiming an optimum

For a lower-bounded L-smooth objective, constant 0<η≤1/L and exact gradients, summing the descent lemma yields

$$
\min_{0\le k<T}\|\nabla f(x_k)\|^2
\le\frac1T\sum_{k=0}^{T-1}\|\nabla f(x_k)\|^2
\le\frac{2(f(x_0)-f_{\inf})}{\eta T}.
$$

For an infinite run, the same sum is finite, so the gradient norms tend to zero. Neither this conclusion nor the finite-time minimum bound proves that iterates converge to a global optimizer. The boundedness of iterates and the meaning of stationarity need separate analysis.

A Polyak–Łojasiewicz condition, ‖∇f‖²/2≥μ(f−f*), strengthens the objective conclusion to a linear rate with factor 1−ημ under this step range. It does not require a unique optimizer. For f(x,y)=(x+y)²/2, the Hessian has eigenvalues zero and two. The function satisfies PL with μ=2 but is not strongly convex; every point with x+y=0 is optimal.

| Assumption | Consequence |
|---|---|
| Lower bound, L-smoothness, valid constant step | Vanishing gradient norms |
| Additional PL inequality | Linear objective-error bound |
| Strong convexity | Unique optimizer and parameter-error control |

The exact code preserves x−y while reducing x+y. A null direction does not obstruct objective convergence. Numerically, the lab tests strict quadratic stability through 0<ηL<2, so a factor rounded to one at tiny η is not mistaken for the mathematical boundary case.

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
x,y=F(1),F(0); eta=F(1,4); initial=F(1,2); norms=[]
for k in range(8):
    value=(x+y)**2/2; grad=x+y
    assert grad*grad == 2*value
    norms.append(2*grad*grad)
    nxtx,nxty=x-eta*grad,y-eta*grad
    assert (nxtx+nxty)**2/2 <= (1-eta*2)*value
    assert nxtx-nxty==1
    print(k,float(value),tuple(map(float,[x,y])))
    x,y=nxtx,nxty
assert sum(norms)<=2*initial/eta
```

## Exercise and explained answer

Set κ=8 and compare η=0.2,0.25,0.3 for eight iterations. Their contraction factors are 0.8, one and 1.4. Compute x₂=(1−8η)⁸ and verify the readout. Then set η=0: both variables remain one, ρ=1 and no optimization occurs. Explain why a bounded sequence and a decreasing objective are different observations from strict error contraction.

Further reading: [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
