## Differentiate the gradient

For a smooth scalar f:Rⁿ→R, the gradient is a vector and its derivative is an n×n Hessian. Its directional action is

$$
H_{ij}=\partial_i\partial_j f,\qquad H v=D(\nabla f)[v].
$$

Continuous mixed-partial assumptions give symmetry, not positive definiteness. Computing derivatives does not establish convexity.

## Work a nonlinear example

For f(x,y)=½x²y+sin y,

$$
\nabla f=(xy,\tfrac12x^2+\cos y),\qquad
H=\begin{pmatrix}y&x\\x&-\sin y\end{pmatrix}.
$$

At (1,0), gradient=(0,1.5) and H=[[0,1],[1,0]], with eigenvalues ±1. For v=(1,1), Hv=(1,1) and vᵀHv=2. Since v is not unit length, two is not a claim about the maximum eigenvalue.

## Directional curvature and a quadratic model

Along g(t)=f(z+tv),

$$
g'(0)=\nabla f(z)^Tv,\qquad g''(0)=v^TH(z)v.
$$

The quadratic approximation is f(z)+t∇f·v+½t²vᵀHv. For the reference this is 1.5t+t². The actual path is ½(1+t)²t+sin t and retains higher-order terms. The plot compares them.

Along v=(1,0), the reference path is constant zero. A zero direction makes any path constant; it does not remove the input space.

## Products without materializing a Hessian

A full Hessian has n² entries. Differentiating the gradient in direction v can compute Hv without storing all entries. Differentiating the scalar ∇f·v while treating v as constant gives Hᵀv, equal to Hv under symmetry.

The interactive engine evaluates analytic expressions; it does not claim to run an autodiff library. An implementation with JVPs, VJPs or nested differentiation must state shapes, graph retention and whether v is treated as constant.

## Independent checks and AI use

$$
Hv\approx\frac{\nabla f(z+hv)-\nabla f(z-hv)}{2h}.
$$

The regression checks use central differences of an independently evaluated gradient. Zero steps are invalid; very small steps can suffer cancellation.

Newton-CG, influence calculations and curvature diagnostics use HVPs. Nonsmooth operations, stochastic samples or changing batches can invalidate a naive derivative comparison; hold the relevant computation fixed.

## Estimate a trace using only products

Hold a real symmetric Hessian H fixed, including parameters, data and model randomness. Draw a vector v whose coordinates are independent, equally likely −1 and 1 (Rademacher probes). Since E[vᵢvⱼ]=δᵢⱼ,

$$
\mathbb E[v^THv]=\operatorname{tr}H,\qquad
v^THv=\operatorname{tr}H+2\sum_{i<j}H_{ij}v_i v_j.
$$

The diagonal contribution is exact because vᵢ²=1. Different unordered pairs have zero cross expectation: disjoint pairs each contain independent mean-zero factors, and pairs sharing one index leave two distinct mean-zero factors after cancellation. Thus

$$
\operatorname{Var}(v^THv)=4\sum_{i<j}H_{ij}^2,\qquad
\operatorname{Var}\left(\frac1M\sum_{m=1}^M v_m^THv_m\right)
=\frac4M\sum_{i<j}H_{ij}^2
$$

for M independent probes. Repeating the same random vector M times leaves the single-probe variance unchanged. These are Rademacher formulas; a different probe distribution needs its own fourth moments.

## Enumerate the reference and interpret its curvature

At the existing lab’s reference (1,0), Hv=(v₂,v₁). All four equally likely vectors give:

| v | Hv | vᵀHv |
|---|---|---|
| (1,1) | (1,1) | 2 |
| (1,−1) | (−1,1) | −2 |
| (−1,1) | (1,−1) | −2 |
| (−1,−1) | (−1,−1) | 2 |

The mean is zero and the single-probe variance is four. For M=1,2,8, averaging independent probes gives variances 4,2,1/2. If K of M values are positive, K is Binomial(M,1/2) and the estimate is 2(2K−M)/M.

Trace measures total signed curvature. For H=diag(2,−1), every Rademacher probe gives trace one exactly, yet a negative eigenvalue remains. Convexity requires appropriate curvature conditions beyond the trace. Trace alone also loses the opposing curvatures in the lab’s zero-trace reference.

## Check every probe and batch moment in Python

The code checks analytic HVPs against central differences of the gradient, then enumerates the exact probe and binomial moments using rational arithmetic. Finite differences provide a numerical check; exact enumeration establishes the stated finite-distribution moments. The interactive engine remains analytic.

```python
from fractions import Fraction as Q
from itertools import product
from math import comb, cos

def gradient(x, y):
    return (x * y, 0.5 * x * x + cos(y))

def hvp(v):
    return (v[1], v[0])  # at (x, y) = (1, 0)

values = []
for v in product([-1, 1], repeat=2):
    h = 1e-4
    plus = gradient(1 + h * v[0], h * v[1])
    minus = gradient(1 - h * v[0], -h * v[1])
    difference = tuple((a - b) / (2 * h) for a, b in zip(plus, minus))
    assert max(abs(a - b) for a, b in zip(difference, hvp(v))) < 1e-8
    values.append(sum(a * b for a, b in zip(v, hvp(v))))
assert sum(values) == 0
assert Q(sum(q * q for q in values), 4) == 4

for m in [1, 2, 8]:
    mean, second = Q(0), Q(0)
    for k in range(m + 1):
        probability = Q(comb(m, k), 2**m)
        estimate = Q(2 * (2 * k - m), m)
        mean += probability * estimate
        second += probability * estimate**2
    assert mean == 0
    assert second - mean**2 == Q(4, m)
    print(m, mean, second - mean**2)
```

For a nonsymmetric dynamics Jacobian J, vᵀJv=vᵀSv with S=(J+Jᵀ)/2, and tr S=tr J. The expectation identity still holds, while the variance formula uses S. A vector field’s divergence is tr J; a scalar loss’s curvature trace is tr H. State which target your experiment estimates.

Source connection: [FFJORD v3](https://arxiv.org/abs/1810.01367v3), §3.1, Eq. (7), applies a stochastic trace identity to dynamics Jacobians. The Hessian enumeration and variance derivation here are independently constructed. Reproducing the paper’s flow training and numerical integration is a separate project. For this fixed-matrix experiment, compare fresh independent probes with one reused probe at equal HVP budgets.

## Exercises and answers

1. At the reference with v=(1,−1): **Hv=(−1,1), vᵀHv=−2**.
2. Symmetry **does not imply PSD**; the reference is indefinite.
3. Hv is **a vector**; vᵀHv is scalar.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
