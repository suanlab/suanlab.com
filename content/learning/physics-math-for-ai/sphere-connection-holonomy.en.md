## Comparing tangent vectors requires a rule

Vectors at different manifold points belong to different tangent spaces. A connection specifies how to differentiate a vector field along a path, and parallel transport requires that covariant derivative to vanish. On a Riemannian manifold the Levi-Civita connection is metric compatible and torsion free. Its transport preserves inner products.

This example uses a sphere of radius R, colatitude θ measured from the north pole, and longitude φ. The metric and an orthonormal tangent frame satisfy

$$
ds^2=R^2d\theta^2+R^2\sin^2\theta\,d\phi^2,\qquad
v=\cos\alpha\,e_\theta+\sin\alpha\,e_\phi.
$$

We avoid the polar coordinate singularities by restricting θ away from zero and π. The plot is a plane of components in the local orthonormal frame. It is not a global flat chart of the sphere or the physical path viewed from above.

## Derive transport along a latitude

Along fixed θ, the tangential derivatives of the frame are ∇φeθ=cosθ eφ and ∇φeφ=−cosθ eθ. Substitute the vector representation and set its covariant derivative to zero:

$$
\frac{d\alpha}{d\phi}=-\cos\theta,\qquad
\alpha(\phi)=\alpha_0-\phi\cos\theta.
$$

The blue arrow is the initial direction and orange the transported components after the selected longitude travel. Their unit length is preserved. A fixed component angle would generally rotate physically relative to parallel transport because the local frame itself turns.

## A loop reveals curvature

After one full loop the frame returns to its starting point. The component-angle change is −2πcosθ, interpreted modulo 2π. At θ=π/3 it is −π, so the vector returns reversed. At the equator cosθ=0 and the component direction stays unchanged. Among these latitude paths only the equator is a geodesic; parallel transport is defined along nongeodesic paths as well.

The sphere has Gaussian curvature K=1/R². Its north cap has area 2πR²(1−cosθ), so

$$
\int_{\mathrm{cap}}K\,dA=2\pi(1-\cos\theta)
\equiv-2\pi\cos\theta\pmod{2\pi}.
$$

The sign and frame convention are explicit. A partial open path has a transported angle, but no closed-loop holonomy is reported. The full-cap area readout always refers to the complete cap, not an open-path swept region.

## Exercise and research connection

Double R at fixed θ=π/3. Curvature becomes one quarter and cap area quadruples; their product and the holonomy stay fixed. Verify each quantity. Then use half a longitude loop and explain why comparing endpoint component angles now depends on the chosen frame trivialization.

Transport and connections help compare tangent features on meshes and curved latent spaces. A visualization of component rotation is useful only after specifying frames, path orientation and connection. Do not infer that a surface is flat from one loop with rotation equal to zero modulo 2π; many distinct curvature distributions can share that result.

Further reading: [John M. Lee, Introduction to Riemannian Manifolds](https://sites.math.washington.edu/~lee/Books/RM/).
