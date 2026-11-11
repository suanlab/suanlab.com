## Local rank and global shape answer different questions

A smooth map is an immersion when its derivative is injective at every source point. For a one-dimensional source this means the derivative vector never vanishes. An embedding additionally identifies the source homeomorphically with its image. These requirements involve both local derivatives and global topology.

The first two examples use a circle source: θ=π(u+1) for u in [−1,1], with the parameter endpoints identified. The last example uses the real line and is merely displayed on [−1,1]. Stating the source prevents the repeated endpoint of a circle parameterization from being mistaken for a failure of injectivity on S¹.

## A circle is an embedding

For f(θ)=(cosθ,sinθ), the derivative has unit speed with respect to θ and speed π with respect to u. It has rank one everywhere. On the circle source the map is one-to-one and continuous; compactness of S¹ and the Hausdorff target ensure it is a homeomorphism onto its image.

$$
f'(\theta)=(-\sin\theta,\cos\theta),\qquad
\|df/du\|=\pi.
$$

Blue shows the image, orange the query point and a normalized nonzero tangent. The tangent's fixed drawing length does not encode derivative speed; read speed numerically.

## An immersion can cross itself

The figure-eight map is f(θ)=(sinθ,sin2θ). At θ=0 and θ=π it reaches the same origin, so it is not injective. Yet

$$
f'(\theta)=(\cos\theta,2\cos2\theta)
$$

never vanishes: if cosθ=0, then cos2θ=−1. It is an immersion everywhere but not an embedding. The two crossing branches correspond to different source points and different tangent directions; a single tangent space for the image at the crossing would not describe that source information correctly.

## An injective map can fail immersion

For the cusp c(u)=(u²,u³), the second coordinate determines u uniquely by a continuous cube root. Thus it is a homeomorphism onto its image, including for the full real-line source. But

$$
c'(u)=(2u,3u^2),\qquad c'(0)=(0,0).
$$

Its derivative loses rank at the origin. It is a topological embedding with this parameterization, but not a smooth embedding. The lab leaves the tangent segment absent at zero rather than normalizing a zero derivative into an invented direction.

## Why global hypotheses cannot be replaced by rank

An injective immersion from a compact manifold into a Hausdorff manifold is an embedding. The missing step is topology: every closed subset of the compact source is compact, its image is compact and therefore closed in the target. Thus the continuous bijection to its image has a continuous inverse. More generally a proper injective immersion is an embedding for the usual manifolds; properness prevents parts of the source escaping to infinity while their images accumulate locally.

A counterexample without properness is the irrational winding of the real line into a product of two circles:

$$
f(t)=(\cos t,\sin t,\cos(\sqrt2t),\sin(\sqrt2t)).
$$

It is injective because equality would require both a nonzero integer multiple of 2π and a compatible irrational multiple. Its speed is √3, hence it is an immersion. Rational approximations p/q to √2 give t=2πq tending to infinity while f(t) tends to f(0). An inverse on the image cannot be continuous at f(0). Thus even injectivity plus full rank is insufficient globally.

Below, successive Pell approximations show the recurrence quantitatively. The first circle returns exactly in the mathematical formula, so its discrepancy is assigned zero; the second is evaluated using the small phase q√2−p. This extension is a separate example, not an extra selectable curve in the lab.

| p/q | q | Distance to f(0), approximately |
|---|---:|---:|
| 3/2 | 2 | 1.027 |
| 17/12 | 12 | 0.185 |
| 99/70 | 70 | 0.032 |

## Reproduce the calculation in Python

```python
from decimal import Decimal, localcontext
from math import pi, sin
with localcontext() as ctx:
    ctx.prec = 60
    root2 = Decimal(2).sqrt()
    p, q = 3, 2
    for _ in range(6):
        error = Decimal(q)*root2-Decimal(p)
        distance = 2*abs(sin(pi*float(error)))
        assert p*p-2*q*q == 1
        print(p, q, round(distance, 9))
        p, q = 3*p+4*q, 2*p+3*q
```

## Exercise and explained answer

Select the cusp at u=0 and then u=0.5. Local rank changes from zero to one, but the global smooth-embedding indicator stays false: an everywhere condition is disproved by the origin. For the figure-eight, compare θ=0 and π, corresponding to u=−1 and zero. Their positions agree while derivatives differ.

## Connection to representations

A learned latent map can lose rank, identify distinct inputs, or distort neighborhood topology. These are separate failure modes. Reconstruction on sampled points alone does not certify a smooth embedding of the full population manifold. Check derivative conditions, source identifications and global injectivity assumptions before interpreting a low-dimensional plot as preserved geometry.

Further reading: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
