## Velocities live in a tangent space

At a point p on a smooth manifold, tangent vectors describe velocities of smooth curves through p. In a coordinate chart they can also act as directional derivatives on functions. The tangent space is linear even though the manifold itself need not be a vector space. Adding two points on a circle usually leaves the circle; adding two tangent vectors at the same point stays in its tangent space.

For p(θ)=(cos θ,sin θ), differentiation gives a unit tangent basis. A vector with coefficient a is

$$
\partial_\theta p=(-\sin\theta,\cos\theta),\qquad
v=a\partial_\theta.
$$

The orange segment begins at p and ends at p+a∂θp in the ambient plane. It is a tangent vector, not a path constrained to the circle. Its squared intrinsic norm in the unit-circle metric is a².

## A covector consumes a vector

The cotangent space is the dual of the tangent space. A covector is a linear functional that returns a scalar when given a tangent vector. For α=b dθ and v=a∂θ, α(v)=ab because dθ(∂θ)=1. A function differential df is a covector: df(v) is the directional rate of change of f.

Drawing a covector as an ordinary arrow silently uses an identification with vectors. A metric can provide that identification, but it is additional structure. This diagram therefore draws only the tangent vector and reports the covector components numerically.

## Change the local coordinate scale

Let u=sθ with positive s. The chain rule gives ∂u=(1/s)∂θ and du=s dθ. The geometric vector and covector must stay fixed, so their components change reciprocally:

$$
v^u=sa,\qquad \alpha_u=b/s,\qquad
\alpha_uv^u=ab,\qquad g_{uu}=1/s^2.
$$

Thus the metric squared norm is gᵤᵤ(vᵘ)²=a². At a=1,b=2,s=2, the new components are two and one, pairing remains two, and squared norm remains one. Simply squaring the vector's new component would incorrectly give four.

## The chain rule explains reciprocal transformation

For a smooth local coordinate change with invertible Jacobian J, a curve's velocity transforms by J. A covector must transform by the inverse transpose so that its value on every velocity stays the same. The metric follows from the same requirement on lengths:

$$
v'=Jv,\qquad \alpha'=J^{-T}\alpha,\qquad
g'=J^{-T}gJ^{-1},\qquad
\operatorname{grad}'f=J\operatorname{grad}f.
$$

These statements are local; vectors at different points still require a transport rule before addition. In one dimension take u=θ+(1/2)sinθ. Then J=1+(1/2)cosθ lies between 1/2 and 3/2, so this is a valid increasing coordinate change on an angle neighborhood. For a=b=1 at θ=0, J=3/2, vector component is 3/2, covector component 2/3 and metric 4/9. Pairing and squared length both remain one.

The code uses three local Jacobians and exact fractions. Multiplying a covector component directly into an optimization step fails the gradient transformation law unless the metric is accounted for. This is the geometric reason that ordinary Euclidean gradient descent depends on parameterization.

| J | Vector component | Covector component | Metric |
|---:|---:|---:|---:|
| 3/2 | 3/2 | 2/3 | 4/9 |
| 1 | 1 | 1 | 1 |
| 1/2 | 1/2 | 2 | 4 |

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
for j in [F(3, 2), F(1), F(1, 2)]:
    v, alpha, g = j, 1/j, 1/j**2
    grad = alpha/g
    assert alpha*v == 1 and g*v*v == 1 and grad == j
    print(j, v, alpha, g)
```

## Exercise and explained answer

Set a=−2,b=0.5,s=2. Predict vᵘ=−4, αᵤ=0.25, pairing −1 and squared norm four. Check each readout. Move θ around the circle: the physical arrow rotates with its tangent space, but these scalar quantities stay fixed. Then set a=0: every covector evaluates the zero vector to zero, regardless of b.

## Connection to learning algorithms

Automatic differentiation commonly returns partial derivatives, which are covector components. Treating them directly as a gradient vector assumes a particular metric, typically Euclidean coordinates. On manifolds or in nonorthonormal parameters, one must raise the covector index with the inverse metric. Keep the derivative, the metric, the gradient vector and their coordinate components separate when deriving natural or Riemannian optimization methods.

Further reading: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
