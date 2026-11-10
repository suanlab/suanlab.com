## 1. A field with continuous local values

The preceding spatial blocking chapter retains a discrete magnetization at each block. Here each site carries a real coordinate. A lattice fixes the positions and adjacency while the configuration space remains continuous. The two-site example integrates over a plane whose axes are the two field values. Changing a field value moves within that plane.

## 2. Energy and the reference measure

Use dimensionless field coordinates and let the action A denote E/T. For an undirected lattice graph with each edge counted once, consider

$$
A(x)=\sum_i (a x_i^4+b x_i^2-h_i x_i)+\frac{c}{2}\sum_{\langle i,j\rangle}(x_i-x_j)^2,
\qquad
Z=\int_{\mathbb R^N} e^{-A(x)}\prod_i dx_i .
$$

Positive c penalizes differences across an edge. The source hᵢ favors positive or negative values at the specified site. The normalized Gibbs density relative to the stated measure is exp(−A)/Z, and F/T=−log Z. The quartic field and gradient terms are familiar from Landau–Ginzburg theory. [Tong, Statistical Field Theory §1.3](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S1)

## 3. Why the quartic term stabilizes the integral

Set dᵢ=max(0,−b)+|hᵢ|/2. The inequality |x|≤(x²+1)/2 bounds the source by a quadratic and a constant. Completing a square in x² gives

$$
a x_i^4+b x_i^2-h_i x_i\ge
\tfrac a2 x_i^4-C_i,\qquad C_i=\frac{d_i^2}{2a}+\frac{|h_i|}{2}.
$$

When a>0 and c≥0, the product of quartic envelopes is integrable with any polynomial weight. This proves finite normalization and moments even for negative b. Subtracting the offset from the log of the weight keeps the shifted action nonnegative. Recover the original normalization using log Z=log Zs+ΣCᵢ.

## 4. Symmetry and fluctuations

With all sources zero, global sign reversal preserves the action and measure. The mean of each site is zero and every monomial of odd total degree integrates to zero. Even total degree products can have nonzero expectation. For instance x₀x₁ is unchanged by a simultaneous flip. At c=0 the sites factorize, so the covariance between distinct sites is zero. Positive c weights similar field values more strongly.

## 5. A local Hessian describes a neighborhood

Take a=1/4, b=−1/2, c=1 and h=0 on two sites. The uniform configurations (1,1) and (−1,−1) minimize the action. At the origin the Hessian has eigenvalues −1 and 1. At either minimum it is

$$
H=\begin{pmatrix}3&-1\\-1&3\end{pmatrix},
\qquad H^{-1}=\frac18\begin{pmatrix}3&1\\1&3\end{pmatrix}.
$$

The inverse Hessian gives the covariance of a quadratic approximation centered at one minimum. The full integral includes both minima and the connecting region. At R=4 with 64 cells along each integration axis, the calculated enclosure for the full off-diagonal covariance lies within [0.50,0.54]. The one-minimum approximation gives 0.125. The full mean is zero by symmetry. Expansions around a minimum retain their local interpretation even when an integral samples both basins. [Tong, Statistical Field Theory §2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2)

## 6. Bound a finite integration box

For a polynomial P, define f=P exp(−A−ΣCᵢ). The second derivative used by the midpoint rule is

$$
\partial_i^2 f=e^{-A-\sum_j C_j}
\left(\partial_i^2P-2\partial_i P\,\partial_i A+
P[(\partial_i A)^2-\partial_i^2 A]\right).
$$

On each cell, enclose this expression with interval arithmetic. The cell average differs from the midpoint value by at most Σᵢ Δᵢ² sup|∂ᵢ²f|/24. Apply the one-dimensional midpoint bound along each coordinate and add the errors. The function range on the cell supplies a second enclosure. Intersect the two enclosures and multiply by cell volume.

## 7. Bound what lies outside the box

Write a coordinate beyond radius R as R+u with u≥0. The tangent bound (R+u)⁴≥R⁴+4R³u yields

$$
\int_{|x|>R}|x|^p e^{-\alpha x^4}dx
\le 2e^{-\alpha R^4}\sum_{k=0}^p
\binom pk\frac{R^{p-k}k!}{(4\alpha R^3)^{k+1}},\qquad \alpha=a/2 .
$$

Outside a product box, at least one coordinate exceeds R in magnitude. Add one product bound per coordinate using the tail factor for that coordinate and full-envelope factors for the others. Use this upper bound on the magnitude of the omitted integral. For nonnegative P the omitted contribution is nonnegative.

## 8. Normalize moments with interval division

Divide each full moment integral by the interval for Zs, which has a positive lower endpoint. For the covariance subtract the product of the mean intervals from the second product moment. Division and subtraction can widen the enclosure, especially if the mass interval is broad. Keep the mass, box integral and omitted-tail bound available when interpreting the final result.

## 9. Separate three refinements

Increasing the number of integration cells at fixed R refines the midpoint calculation. Increasing R changes both the omitted tail and the cell width when the cell count is fixed. Refining the physical lattice changes the graph and the action itself. Hold the lattice fixed to study integration error. When studying physical lattice refinement, specify how the coefficients and measure scale with spacing.

## 10. A solvable case and an identity

For A(x)=a x⁴, substitute u=a x⁴ in each half-line integral. For p even, the unnormalized moment is Γ((p+1)/4)/(2a^((p+1)/4)), so E[x⁴]=1/(4a). For the coupled model, integrate ∂ᵢ(xᵢ exp(−A)) over the entire space. The boundary term vanishes by quartic decay. The result is E[xᵢ∂ᵢA]=1, which gives a check involving the second and fourth moments, the source response and the neighbor product.

## 11. Run a two-site calculation

This NumPy exercise approximates the two-dimensional integral with Gauss–Legendre nodes. Increase the node count and radius independently and compare with the directed enclosures.

```python
import numpy as np

def moments(a=.25, b=-.5, c=1., h=(0., 0.), radius=5., nodes=96):
    t, w = np.polynomial.legendre.leggauss(nodes)
    x, y = radius * t[:, None], radius * t[None, :]
    action = a*(x**4+y**4) + b*(x*x+y*y) - h[0]*x - h[1]*y + c*(x-y)**2/2
    weights = radius**2 * w[:, None] * w[None, :] * np.exp(-action)
    z = weights.sum()
    ex, ey = (weights*x).sum()/z, (weights*y).sum()/z
    ex2, ex4, exy = [(weights*p).sum()/z for p in (x*x, x**4, x*y)]
    identity = 4*a*ex4 + 2*b*ex2 - h[0]*ex + c*(ex2-exy)
    return np.log(z), ex, exy-ex*ey, identity

uncoupled = moments(c=0.)
coupled = moments()
assert abs(uncoupled[2]) < 1e-10
assert 0.50 < coupled[2] < 0.54
assert abs(coupled[3]-1) < 1e-9
print("uncoupled", uncoupled)
print("coupled", coupled)
```

## 12. Check your reasoning

1. Set b negative. The quartic bound still proves a finite partition function because the residual quartic decay is integrable.
2. Flip both coordinates at zero source. Their product keeps its sign, so symmetry permits nonzero covariance.
3. Increase the cell count at fixed R. Only the box approximation changes. The tail formula depends on the radius and envelope.
4. Compare the local covariance 1/8 with the full covariance near 0.52. The full ensemble includes the variation between its two aligned basins.
5. Add a constant K to A. The normalized moments stay the same and log Z decreases by K.
6. Increase c from zero. The term c(x₀−x₁)²/2 suppresses large differences, and the two-site example lets you measure the resulting covariance.
