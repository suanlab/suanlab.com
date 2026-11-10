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

## Exercise and explained answer

Set a=−2,b=0.5,s=2. Predict vᵘ=−4, αᵤ=0.25, pairing −1 and squared norm four. Check each readout. Move θ around the circle: the physical arrow rotates with its tangent space, but these scalar quantities stay fixed. Then set a=0: every covector evaluates the zero vector to zero, regardless of b.

## Connection to learning algorithms

Automatic differentiation commonly returns partial derivatives, which are covector components. Treating them directly as a gradient vector assumes a particular metric, typically Euclidean coordinates. On manifolds or in nonorthonormal parameters, one must raise the covector index with the inverse metric. Keep the derivative, the metric, the gradient vector and their coordinate components separate when deriving natural or Riemannian optimization methods.

Further reading: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
