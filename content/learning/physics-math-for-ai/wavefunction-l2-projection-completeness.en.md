## 1. A density curve is not a list of point probabilities

A continuous position description assigns probabilities to intervals. A high density at one coordinate is not the probability of the singleton containing that coordinate. Likewise, a narrow region can carry nearly all the probability while a coarse display samples mostly zeros.

We will use exactly integrable functions to separate the state, its plotted samples, its interval probabilities and a finite-basis approximation. This lets us study approximation without mistaking a denser plot for a more accurate integral.

## 2. Specify the space and its equivalence relation

For a finite interval of physical length L, the state space in this chapter is the complex Hilbert space L²([0,L],dx). Its elements are equivalence classes of measurable functions that agree almost everywhere, not functions distinguished by an arbitrarily assigned value at one point.

$$
\mathcal H=L^2([0,L],dx),\qquad \langle\psi,\phi\rangle=\int_0^L\overline{\psi(x)}\phi(x)\,dx,\qquad \|\psi\|_2^2=\int_0^L|\psi(x)|^2\,dx.
$$

The first argument is conjugate-linear, as in the finite-dimensional chapter. A normalized nonzero element can represent a pure state. L² includes functions with jumps; membership alone does not specify a Hamiltonian, boundary condition or finite kinetic-energy expectation. The [MIT functional-analysis notes](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/8fb8d5c170f1613151aca71de21027bc_MIT18_102s21_full_lec.pdf) supply the mathematical background for this space. Those mathematics notes use a first-slot-linear convention; here we retain the physics convention with conjugation in the first slot. Norms and completeness agree under the corresponding change of notation.

## 3. Normalize with the physical integration measure

Let u=x/L and choose a nonzero dimensionless, complex, piecewise-affine preparation g(u). It can have different constant and slope coefficients on either side of a specified split. Its dimensionless norm is Z.

$$
Z=\int_0^1|g(u)|^2\,du>0,\qquad \psi(x)=\frac{g(x/L)}{\sqrt{LZ}},\qquad \int_0^L|\psi(x)|^2dx=1.
$$

The factor L comes from dx=Ldu. If x has length units, ψ has inverse-square-root-length units and |ψ|² has inverse-length units. Removing L from the normalization would give the wrong physical probability. Zero-norm inputs are rejected rather than normalized by a small artificial denominator.

## 4. Integrate bins before drawing samples

On each segment, |g|² is a real quadratic polynomial. Its exact integral gives the probability in a bin, even when the split lies inside that bin. The production calculation integrates both relevant subintervals and then adds their masses.

For g=1 on[0,1/2) and0 elsewhere, Z=1/2. The normalized density is2/L on[0,L/2) and0 on the rest. The left-half probability is1. If support instead ends at u=10⁻⁶, the normalized density there is10⁶/L. Sampling only u=0,1/8,…,1 misses almost its entire width; exact bin integration still assigns mass1 to the containing first bin. Sample count changes the display, not the normalization or integral.

## 5. Completeness concerns limits in a norm

A Cauchy condition controls differences between sufficiently late terms of a sequence:

$$
\forall\varepsilon>0\;\exists N\;\forall m,n\ge N:\quad\|\psi_m-\psi_n\|_2<\varepsilon.
$$

Completeness says such a sequence has an L² limit inside the space. It does not require convergence at every point or uniform convergence of function values. This is why the norm and its equivalence relation must be stated before discussing convergence.

An infinite orthonormal Hilbert basis represents vectors by norm-convergent series. This differs from a Hamel basis, whose linear combinations are finite. A finite coefficient array is a subspace approximation; it is not a replacement definition of the infinite-dimensional Hilbert space.

## 6. Build a concrete orthogonal polynomial family

Use shifted Legendre polynomials P̃n(u)=Pn(2u−1). Starting from P̃₀=1 and P̃₁=2u−1, generate the rest by recurrence.

$$
(n+1)\widetilde P_{n+1}=(2n+1)(2u-1)\widetilde P_n-n\widetilde P_{n-1},\qquad \int_0^1\widetilde P_m\widetilde P_n\,du=\frac{\delta_{mn}}{2n+1}.
$$

The norm on the right is not1. Forgetting this factor gives incorrect coefficients and captured probability. The [NIST orthogonal-polynomial definitions](https://dlmf.nist.gov/18.3) fix the normalization. Our production recurrence and independent closed-binomial expansion are two different ways to generate this same polynomial family.

## 7. Distinguish raw expansion coefficients from orthonormal coordinates

Let aₙ be the raw coefficient multiplying P̃n in g. The physical orthonormal basis includes both a polynomial norm factor and the physical length.

$$
a_n=(2n+1)\int_0^1\widetilde P_n(u)g(u)\,du,\qquad e_n(x)=\sqrt{\frac{2n+1}{L}}\widetilde P_n(x/L),\qquad c_n=\langle e_n,\psi\rangle=\frac{a_n}{\sqrt{(2n+1)Z}}.
$$

Each aₙ can be complex. The probability weight associated with the orthonormal coefficient is |cₙ|², not |aₙ|². A change of L leaves these dimensionless coefficient weights unchanged for the same g, while it rescales the physical wavefunction and its spatial density.

## 8. Orthogonal projection retains its captured norm

Project ψ onto degrees0 throughN. Orthogonality makes the residual perpendicular to that subspace, yielding the Pythagorean identity:

$$
\Pi_N\psi=\sum_{n=0}^Nc_ne_n,\qquad C_N=\|\Pi_N\psi\|_2^2=\sum_{n=0}^N|c_n|^2,\qquad \|\psi-\Pi_N\psi\|_2^2=1-C_N.
$$

Captured mass is nonnegative, at most1, and nondecreasing with degree. The squared L² error is therefore nonincreasing. The program also integrates |g−gN|² directly, rather than defining the error only as a subtraction, and compares the two expressions exactly. These facts do not say the pointwise error decreases at every displayed coordinate.

## 9. Renormalization is a separate approximation

A truncated projection generally has norm below1. Its integral is the captured mass, not an invalid computation to repair silently. If a unit-norm approximate state is desired, normalize it explicitly and label the result separately.

$$
\widehat\psi_N=\frac{\Pi_N\psi}{\sqrt{C_N}}\quad(C_N>0),\qquad |\langle\psi,\widehat\psi_N\rangle|^2=C_N.
$$

The probability bins of this normalized approximation sum to1, whereas the unnormalized projection bins sum to Cₙ. When Cₙ=0, the normalized approximation and its fidelity are unavailable. Returning an all-zero probability vector or choosing an arbitrary state would obscure the fact that the selected subspace captured no component.

## 10. Work through a discontinuous but normalizable state

For the half-interval step g, the first coefficients are particularly simple:

$$
a_0=\frac12,\qquad a_1=-\frac34,\qquad a_2=0,\qquad a_3=\frac7{16},\qquad Z=\frac12.
$$

The odd coefficients carry the discontinuity; the even coefficients above degree0 vanish by symmetry. The following values are exact, not quadrature estimates.

| Maximum degree N | Captured mass Cₙ | Squared L² projection error | Projection renormalizable? |
| --- | --- | --- | --- |
| 0 | 1/2 | 1/2 | Yes |
| 1 | 7/8 | 1/8 | Yes |
| 2 | 7/8 | 1/8 | Yes |
| 3 | 119/128 | 9/128 | Yes |

At N=0, the projected physical wavefunction is1/√(2L), with norm squared1/2. Renormalizing gives a uniform wavefunction1/√L. Those are different approximations with different interval masses; neither equals the original step state.

## 11. Pointwise appearance and probability measures differ

For the chosen representative, g(1/2)=0. Yet every finite Legendre projection of this step takes value1/2 at u=1/2: odd Legendre polynomials vanish there, and the higher even coefficients are zero. The persistent error at that single point is compatible with convergence in L². Changing g at that point does not change any integral or the L² state.

Length rescaling offers another useful comparison:

| Physical length | Target density on its left half | Probability of the left half | Same dimensionless coefficient weights? |
| --- | --- | --- | --- |
| 1 length unit | 2 per length unit | 1 | Yes |
| 2 length units | 1 per length unit | 1 | Yes |
| 3 length units | 2/3 per length unit | 1 | Yes |

Density changes inversely with length; probability remains dimensionless. A density can exceed1 without violating a probability bound. Only its integral over a specified measurable set must lie between0 and1 for a normalized state.

## 12. Do not infer an operator domain from normalization

For an unbounded operator with a proper domain, belonging to the Hilbert space does not guarantee belonging to that domain:

$$
D(H)\subsetneq\mathcal H,\qquad \psi\in\mathcal H\;\not\Rightarrow\;\psi\in D(H).
$$

For example, a constant normalized function on a finite interval is an L² state, but it does not satisfy the endpoint conditions of the Dirichlet Laplacian operator domain. This does not turn it into the zero state or an invalid Hilbert vector. Whether an operator can act on it, a strong derivative exists, or an energy expectation is finite requires further assumptions.

No Hamiltonian or wall condition is imposed in this experiment. Delta-normalized position labels and plane waves on the whole real line are also not ordinary normalized L² vectors; their generalized-state formalism belongs to a later treatment of spectra and domains.

## 13. Six worked checks

**1. Normalize a step.** With g=1 on[0,1/2), Z=1/2. Therefore ψ=√(2/L) on[0,L/2), and integrating its density gives1. The value at the single split point does not alter this result.

**2. Change the length to3.** The density becomes2/3 on a region of length3/2. Their product is1. Coefficient probabilities and projection errors are unchanged because they are computed in u=x/L with the correct measure.

**3. Keep degrees0 and1.** Their raw coefficient masses are1/4 and3/16, totaling7/16. Dividing by Z=1/2 gives C₁=7/8 and squared error1/8. Direct integration of the residual gives the same answer.

**4. Find a zero projection.** Let g=1 on the left half and−1 on the right half. Its integral is0 and Z=1. Projection onto constants is exactly zero, so its squared error is1. A normalized constant approximation is not defined by this projection.

**5. Change one endpoint value.** Take a constant g=1 almost everywhere, but assign999 at u=1. Every coefficient and interval probability is unchanged. A displayed endpoint marker changes; the L² state does not.

**6. Expand g=1+iu.** Z=4/3, a₀=1+i/2 and a₁=i/2. Degree0 captures(5/4)/(4/3)=15/16, leaving squared error1/16. Degree1 reconstructs the affine function exactly. The coefficient weights require the polynomial norm factor1/(2n+1).

## 14. Why the full basis is complete, and what remains to study

The completeness argument is mathematical rather than a finite numerical observation. Continuous functions are dense in L² on a compact interval. A continuous function can be uniformly approximated by polynomials, and the Legendre polynomials span every finite-degree polynomial space. Combining these approximations gives

$$
\|\psi-p\|_2\le\|\psi-f\|_2+\sqrt L\,\|f-p\|_\infty<\varepsilon,
$$

with f continuous and p polynomial chosen for the desired tolerance. Thus the closed span is L². The norm-limit statement does not promise convergence at every point, differentiability or finite energy.

For research and AI approximation, report the measure, normalization, subspace, captured mass and chosen error norm before comparing discretizations. Finite mode truncation, neural representation and sample-grid display make different approximation choices. General function spaces, unbounded operators, spectral measures, rigged Hilbert spaces and quantitative approximation rates remain further study. This finite experiment does not complete that wider research scope.
