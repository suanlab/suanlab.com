## From one direction to all directions

A symmetric quadratic form q(v)=vᵀAv depends on length as well as direction: q(αv)=α²q(v). Normalize by squared length to compare directions:

$$
R_A(v)=\frac{v^TAv}{v^Tv},\qquad v\ne0.
$$

This Rayleigh quotient is undefined at zero. The lab uses v=(cosθ,sinθ), so every probe is a unit vector and q equals the quotient.

## Extreme eigenvalues bound the quotient

With A=QΛQᵀ and z=Qᵀv,

$$
R_A(v)=\sum_i\lambda_i\frac{z_i^2}{\sum_j z_j^2},\qquad
\lambda_{\min}\le R_A(v)\le\lambda_{\max}.
$$

The nonnegative weights sum to one. Extremal eigenvectors attain the bounds, but repeated eigenvalues can make extremal directions nonunique.

For [[2,1],[1,2]], eigenvalues are one and three. At zero degrees q=2; at 45 degrees q=3. One small directional value does not establish a universal upper bound.

## Loewner order is not entrywise order

For real symmetric matrices of the same size,

$$
A\preceq B\iff B-A\succeq0\iff
v^TAv\le v^TBv\quad\text{for every }v.
$$

The difference [[1,−1],[−1,1]] contains negative entries but is PSD, with eigenvalues two and zero. Conversely, [[1,2],[2,1]] has positive entries but is indefinite. Entrywise positivity and Loewner positivity are distinct. Loewner order is partial: not every pair is comparable.

## Certify a scalar matrix bound

Taking B=CI gives

$$
A\preceq CI\iff\lambda_{\max}(A)\le C.
$$

For the reference, C=3 works because 3I−A=[[1,−1],[−1,1]] is PSD. With C=2 the zero-degree probe passes, yet the 45-degree value three violates the bound. The lab reports the single-probe result separately from the universal matrix certificate.

The certificate uses exact small-integer two-dimensional PSD predicates. The trigonometric probe comparison allows 10⁻¹² roundoff. That computational tolerance is not an assumption of the mathematical theorem.

## Congruence preserves the order

For every compatible real M,

$$
A\preceq B\implies M^TAM\preceq M^TBM.
$$

Indeed, wᵀMᵀ(B−A)Mw=(Mw)ᵀ(B−A)(Mw)≥0. If M is invertible the converse follows too. A singular M can discard a violating direction, so the converse may fail.

This is a two-sided congruence, not arbitrary one-sided multiplication. A one-sided product may not even remain symmetric.

## Optimization connections and exercises

For a quadratic loss f(x)=½xᵀHx−bᵀx, bounds 0≺mI⪯H⪯LI control curvature, linking to strong convexity, smoothness and gradient-step analysis. Nonlinear losses need Hessian control throughout the relevant region, not only at one point.

The graph samples q over angles and compares it with C. The selected probe appears in the numeric readout. Finite angular samples illustrate a universal bound; eigendecomposition or PSD conditions certify it.

1. The smallest valid C for the reference is **three**.
2. Eigenvalues of 2I−A are **one and minus one**, so it is not PSD.
3. Does A⪯B imply aᵢᵢ≤bᵢᵢ? **Yes, by testing basis vectors. The converse generally fails.**

Further reading: [MIT Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/). The symmetric-matrix arguments above are directly derived.
