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

## Exercise and explained answer

Select the cusp at u=0 and then u=0.5. Local rank changes from zero to one, but the global smooth-embedding indicator stays false: an everywhere condition is disproved by the origin. For the figure-eight, compare θ=0 and π, corresponding to u=−1 and zero. Their positions agree while derivatives differ.

## Connection to representations

A learned latent map can lose rank, identify distinct inputs, or distort neighborhood topology. These are separate failure modes. Reconstruction on sampled points alone does not certify a smooth embedding of the full population manifold. Check derivative conditions, source identifications and global injectivity assumptions before interpreting a low-dimensional plot as preserved geometry.

Further reading: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
