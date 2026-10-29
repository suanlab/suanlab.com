## How much lies along a direction?

After learning vector addition, compare two vectors. How much of `(3,2,0)` lies along x? The same question makes sense for a tilted reference direction. **An inner product expresses a comparison as a number; projection expresses the retained component as a vector.**

This unit separates dot products, angles and projections, then checks why the residual is perpendicular. Coordinates are dimensionless Euclidean orthogonal coordinates. Actual features require a separate examination of units and scale.

## Symbols and prerequisites

| Symbol | Meaning | Reference example |
| --- | --- | --- |
| u | Vector to decompose | `(3,2,0)` |
| v | Vector specifying the reference direction | `(2,0,0)` |
| u·v | Sum of products of matching components | 6 |
| α | Coefficient multiplying v | 1.5 |
| p | Projection onto the line spanned by v | `(3,0,0)` |
| r=u−p | Remaining residual | `(0,2,0)` |

We draw p as a point but read it as the vector from the origin to that point. Although v and p lie on the same line, their lengths need not agree.

## Calculate the inner product

Define the Euclidean dot product by

$$
\mathbf u\cdot\mathbf v=u_1v_1+u_2v_2+u_3v_3.
$$

The example gives `3×2 + 2×0 + 0×0=6`. Two vectors enter and one scalar comes out. Vector addition, by contrast, returns a vector.

Components pointing together contribute positively; opposing components contribute negatively. A negative dot product is not a calculation error: it can indicate an obtuse angle. Both vectors must be nonzero for their angle to be defined.

## Separate length from angle

The coordinate definition connects to geometry through

$$
\mathbf u\cdot\mathbf v=\|\mathbf u\|\|\mathbf v\|\cos\theta.
$$

For two nonzero vectors,

$$
\cos\theta=\frac{\mathbf u\cdot\mathbf v}{\|\mathbf u\|\|\mathbf v\|}.
$$

The reference lengths are √13 and 2. Cosine is `6/(2√13)=3/√13≈0.832`, giving approximately 33.69°. Dot product 6 and cosine 0.832 are different quantities. Cosine removes the length factors and lies between −1 and 1.

| Comparison | Interpretation |
| --- | --- |
| Same direction | Cosine 1 |
| Orthogonal nonzero vectors | Dot product 0, cosine 0, angle 90° |
| Opposite directions | Cosine −1 |
| A zero vector is present | Dot product exists, but angle is undefined |

## Projection finds a coefficient along v

Every point on the line spanned by v is αv. We want its closest point p=αv to u. At that point, the residual must be perpendicular to the line:

$$
\mathbf v\cdot(\mathbf u-\alpha\mathbf v)=0.
$$

Expanding and solving for α gives

$$
\alpha=\frac{\mathbf u\cdot\mathbf v}{\mathbf v\cdot\mathbf v},\qquad
\mathbf p=\operatorname{proj}_{\mathbf v}\mathbf u=\alpha\mathbf v.
$$

Here α=6/4=1.5, so p=1.5(2,0,0)=(3,0,0). **Distinguish coefficient 1.5, projected vector `(3,0,0)`, and projected length 3.**

If v is unit length, the denominator is one. Omitting it for an arbitrary vector is incorrect. Doubling v preserves its line and therefore the projected vector, while halving the coefficient.

## Why is the perpendicular foot closest?

For any real t, the squared distance from u to tv is

$$
\|\mathbf u-t\mathbf v\|^2=\|\mathbf u\|^2-2t(\mathbf u\cdot\mathbf v)+t^2\|\mathbf v\|^2.
$$

Using the coefficient α above, complete the square:

$$
\|\mathbf u-t\mathbf v\|^2
=\|\mathbf v\|^2(t-\alpha)^2+\|\mathbf u-\alpha\mathbf v\|^2.
$$

For nonzero v, the first term is nonnegative and vanishes only at t=α. Thus p is the unique closest point. No differentiation is needed for this proof.

Geometrically, any extra displacement along the line adds a perpendicular leg to the residual, making the hypotenuse longer. The same principle will explain least squares.

## Check the residual

The reference residual is `(3,2,0)−(3,0,0)=(0,2,0)`:

$$
\mathbf v\cdot\mathbf r=(2,0,0)\cdot(0,2,0)=0.
$$

Also, `‖u‖²=13`, `‖p‖²=9`, and `‖r‖²=4`, so `13=9+4`. This is a squared-length identity for orthogonal components, not a rule for adding lengths of arbitrary vectors.

Changing u's z component in the experiment uses the same formula. Rotate the 3D view to inspect the vectors and residual. The concept diagram above remains a fixed z=0 reference example.

## Zero vectors and negative projections

Reference v=0 has no direction and makes denominator v·v zero. This **formula specifying projection by a direction vector** is undefined there. Projection onto the separate subspace `{0}` can be defined as zero, but doing so does not create a direction or angle for a zero vector. The experiment states its boundary explicitly.

If u=0 and v is nonzero, the projection is zero, even though the angle remains undefined.

For u=(-3,0,0), v=(2,0,0), the dot product is −6, coefficient −1.5, and projection `(-3,0,0)`. A negative coefficient uses the direction opposite v. Projection is onto a full line, not necessarily its positive half.

## AI examples: similarity and component removal

Representations u=(1,1), v=(1,−1) have dot product zero in the shared Euclidean coordinates. Multiply the second feature by 10 in both representations, producing `(1,10)` and `(1,−10)`: their dot product becomes −99 and cosine becomes −99/101. Changing feature scales changes the comparison geometry.

Cosine similarity can be −1, so it is not itself a probability. High similarity also does not guarantee accuracy on a task. Record feature meanings, whether matching coordinates mean the same thing, and preprocessing and normalization.

The residual `u−projᵥu` removes one directional component and is orthogonal to that direction. This does not establish that the direction fully represents a social attribute or causal factor. Geometric removal and semantic guarantees are different claims.

## Extend to several directions

For mutually orthogonal unit basis vectors q₁,…,qₖ, add their projections:

$$
\mathbf p=\sum_{j=1}^{k}(\mathbf q_j^T\mathbf u)\mathbf q_j.
$$

Do not use this sum unchanged for nonorthogonal basis vectors. Their coefficients must be solved together, connecting to least squares. Orthogonalization and QR decomposition will later provide more general computational tools.

## Challenge and research checks

Keep u=(3,2,0), and change v to (1,1,0). Then α=5/2, p=(2.5,2.5,0), and r=(0.5,−0.5,0). Check v·r=0 and the squared-length identity.

1. What changes when v is multiplied by three, and what stays unchanged?
2. Why do u=0 and v=0 have different implications for projection and angle?
3. Can you reproduce the feature scales and normalization in a research comparison of cosine similarity?

Further study: [MIT — Projections and Least Squares](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projection-matrices-and-least-squares/).
