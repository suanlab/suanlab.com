## A metric measures tangent displacements

A Riemannian metric is a smoothly varying positive-definite inner product on tangent spaces. It tells us how to measure local lengths and angles. On a radius-R circle with angle θ, differentiating the physical position gives speed R, so

$$
ds^2=R^2d\theta^2,\qquad
L[\theta]=\int_0^1R|\dot\theta(t)|\,dt.
$$

This is intrinsic geometry: allowed paths stay on the circle. The ambient straight chord is a different path through the surrounding plane. Neither distance is wrong; they solve different constrained problems.

## Choose a displacement and a winding

The endpoints determine angle only modulo 2π. Let δ be the representative in (−π,π], with zero used when endpoints coincide. Adding 2πm selects a winding path. A constant-speed lift is θ(t)=θ₀+t(δ+2πm), with

$$
L_m=R|\delta+2\pi m|,\qquad
E_m=\frac12\int_0^1R^2\dot\theta^2dt
=\frac12R^2(\delta+2\pi m)^2.
$$

The lab permits m=−1,0,1. Its shortest intrinsic distance uses the principal displacement R|δ|. At antipodes δ=π, the m=0 and m=−1 paths tie. There is no unique globally shortest direction there.

## A geodesic need not be globally shortest

In this constant metric coordinate the geodesic equation is θ̈=0. Every constant-speed winding arc satisfies it locally, including paths that go around extra times. A local differential equation does not ensure global minimality over arbitrarily long intervals.

For a quarter circle with R=1, shortest distance is π/2≈1.571, while the chord is √2≈1.414. With one extra positive winding the selected length becomes 5π/2≈7.854, although the endpoints and shortest distance are unchanged. Blue draws the entire circle, orange the selected arc; overlapping windings can trace the same visible curve multiple times, so inspect length and energy as well.

## Exercise and explained answer

At R=2 and opposite endpoints, predict intrinsic distance 2π, chord four, and constant-speed energy 2π² on [0,1]. Both shortest directions have the same energy. Set separation zero and winding one: the path is a full loop of length 4π, but the endpoint distance is zero. This illustrates why distance is an infimum over paths rather than the length of whichever path was selected.

## Connection to geometric learning

Interpolation, nearest-neighbor methods and path planning on constrained representations depend on the metric and admissible path class. Euclidean closeness in an embedding need not equal intrinsic closeness. Metric coefficients also change under coordinates: with arc-length coordinate s=Rθ, the coefficient becomes one, while geometric distances stay fixed. State the metric, parameter interval and global path restrictions when comparing geodesic-based algorithms.

Further reading: [John M. Lee, Introduction to Riemannian Manifolds](https://sites.math.washington.edu/~lee/Books/RM/).
