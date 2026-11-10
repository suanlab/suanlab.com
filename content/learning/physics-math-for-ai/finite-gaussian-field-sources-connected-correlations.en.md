## 1. A field is a collection of fluctuating values

Picture several neighboring sensors measuring a flexible surface at the same instant. Nearby readings can rise together, but part of that agreement may simply come from a nonzero average surface height. A correlation function separates questions about the average profile from questions about fluctuations around it.

Our field is a finite vector φ=(φ₀,…,φₙ₋₁). Its indices label positions on a chain or ring, not consecutive time steps. We average over an equilibrium probability distribution of entire configurations. We do not follow one trajectory or claim that observations at different sites establish causation.

This chapter makes the source, measure, boundary conditions and constraints explicit. That care pays off: means, connected correlations, responses and fourth moments can all be calculated exactly, and an apparent exception caused by a zero mode can be understood rather than hidden by a numerical regularizer.

## 2. Specify the energy and the boundary convention

Choose n sites and dimensionless energy

$$
U_J(\phi)=\frac12\phi^TK\phi-J^T\phi,
\qquad K=m^2I+\kappa L,\qquad T>0.
$$

Here m² is the supplied nonnegative coefficient, not a mass that the calculator squares again. The nonnegative κ penalizes differences between neighbors, T is temperature with Boltzmann's constant set to one, and J is a physical source coupled linearly to the field. The probability weight is exp(−U_J/T).

For open boundaries, the gradient energy is κ/2 times ∑(φᵢ₊₁−φᵢ)² over the n−1 chain edges. A periodic ring adds the edge from n−1 to 0. Fixed-zero exterior boundaries add κ(φ₀²+φₙ₋₁²)/2 to the open energy: the exterior values φ₋₁=φₙ=0 are fixed, not integrated variables.

Thus an open Laplacian has endpoint diagonal entries 1, a ring has diagonal 2, and fixed-zero exterior boundaries give diagonal 2 everywhere. Off-diagonal neighbor entries are −1. We use n≥3 so a ring is not confused with a two-site multigraph. The lattice spacing is one; no continuum limit is performed.

## 3. Ask whether the probability distribution exists

On the full space, a Gaussian weight is normalizable when K is positive definite. With m²>0, the mass term confines every direction. At m²=0 and κ>0, fixed-zero exterior boundaries still confine the field. Open and periodic boundaries instead leave a constant configuration with zero gradient energy.

$$
L\mathbf1=0\quad\text{(open or periodic)},\qquad
\phi=c\mathbf1\ \Longrightarrow\ \phi^TK\phi=0\quad(m^2=0).
$$

Integrating this unconstrained constant direction has infinite volume. A linear source does not fix it: it either leaves a flat direction or makes the weight grow in one direction. A failed matrix inverse here reflects a missing probability distribution, not a reason to label a huge floating number a physical variance.

The calculation rejects the massless open/ring model without a constraint, and rejects m²=κ=0 even with a mean-zero constraint. The latter would leave n−1 flat directions. A declared mean-zero ensemble is a different, legitimate model that we derive below.

## 4. Precision is local; covariance can be nonlocal

The matrix K is a precision in the energy, while K/T is the precision of the full Gaussian density. A nonzero neighbor coupling in K is local. Its inverse generally has nonzero entries between sites that are not immediate neighbors.

One way to see positivity and compute the inverse is an LDLᵀ factorization. A symmetric positive matrix has a unit lower triangular L₀ and positive diagonal D such that K=L₀DL₀ᵀ. Solve triangular systems to obtain the inverse, rather than using rounded eigenvectors to certify a nearly soft mode.

The implementation performs this factorization with exact rational arithmetic, on K or on the declared reduced matrix. Every diagonal pivot must be positive. This is a finite calculation on at most eight sites; exactness does not remove the computational cost of large rational numerators or justify extrapolation to infinitely many sites.

## 5. Complete the square to obtain the mean and fluctuations

For the full positive-definite model, let G=K⁻¹ and μ=GJ. Completing the square gives

$$
U_J(\phi)=\frac12(\phi-\mu)^TK(\phi-\mu)
-\frac12J^TGJ,\qquad
\langle\phi\rangle_J=\mu,\qquad C=T G.
$$

The source shifts the center. It does not change the covariance in this quadratic model. Temperature changes the width, while the mean at fixed K,J stays unchanged. This independence is a property of the declared Gaussian family, not a general statement about interacting systems.

The normalization is Z(0)=(2πT)ⁿᐟ²/√det K. The calculation returns the exact determinant and the analytic formula; it does not pretend that π, square roots or an exponential are rational numbers. The source-dependent logarithmic ratio does happen to be rational for rational inputs:

$$
\log\frac{Z(J)}{Z(0)}=\frac{J^TGJ}{2T}.
$$

## 6. Separate the raw and connected two-point functions

Define the fluctuation η=φ−μ. The connected two-point function is its covariance:

$$
C_{ij}=\langle\eta_i\eta_j\rangle
=\langle\phi_i\phi_j\rangle-\mu_i\mu_j,
\qquad \langle\phi_i\phi_j\rangle=C_{ij}+\mu_i\mu_j.
$$

A large raw two-point value can result from two large means without large cofluctuations. For independent sites K=I, off-diagonal Cᵢⱼ vanishes, yet a source can make the raw off-diagonal moment nonzero. Subtract the product of means before interpreting connected structure.

A covariance is not a Pearson correlation coefficient: the latter divides by √(CᵢᵢCⱼⱼ). Nor is a nonzero covariance a proof of causal influence in an unknown system. Here response has a meaning because we explicitly define a family of source-dependent distributions.

## 7. Sources and generating functions carry different temperature factors

Keep two conventions separate. Differentiation with respect to the physical energy source J inserts φ/T. Consequently,

$$
\partial_{J_i}\log Z(J)=\frac{\mu_i}{T},\qquad
\partial_{J_i}\partial_{J_j}\log Z(J)=\frac{C_{ij}}{T^2}.
$$

For a dimensionless auxiliary source h inserted as exp(hᵀφ) in a normalized expectation, define W(h)=log⟨exp(hᵀφ)⟩. Completing the square gives

$$
W(h)=h^T\mu+\frac12h^TCh.
$$

Derivatives of this W directly give cumulants without extra temperature factors. Z generates raw moments; its logarithm generates connected cumulants. Mixing h with the energy source J would produce missing factors of T and an incorrect response law.

## 8. Response and the Green identity

Within the full Gaussian family, differentiating μ=GJ yields the static susceptibility:

$$
\chi_{ij}=\frac{\partial\mu_i}{\partial J_j}
=G_{ij}=\frac{C_{ij}}{T},\qquad KG=GK=I.
$$

The j-th column of G is the profile induced by a unit source at site j. This explains why an inverse local operator appears as a spatial response kernel. Multiplying that profile by K recovers the localized source.

Equivalently, Gaussian integration by parts gives ⟨ηᵢ(Kη)ⱼ⟩=Tδᵢⱼ. Boundary conditions matter: changing K changes the Green function. This is a static equilibrium identity, not a retarded response in time and not a claim about relativistic propagation.

## 9. Work out a four-site ring

Set m²=κ=T=1 on a four-site ring. The precision has diagonal 3 and entries −1 between ring neighbors. Its inverse and covariance are

$$
G=C=\frac1{15}
\begin{pmatrix}7&3&2&3\\3&7&3&2\\2&3&7&3\\3&2&3&7\end{pmatrix}.
$$

Each row sums to one, as K1=1. A unit source J=(1,0,0,0) gives μ=(7/15,1/5,2/15,1/5). The largest response is at the driven site, but the opposite site also responds.

For sites 0 and 1 the connected moment is 1/5. The raw moment is 1/5+(7/15)(1/5)=22/75. Set the source to zero and the raw moment returns to 1/5, while the connected moment never changed. Double T instead: μ stays fixed, C doubles, and χ=C/T stays fixed.

## 10. Derive Wick factorization for centered fields

Because W(h) is quadratic, every connected cumulant of order three or higher vanishes. Differentiating exp(W) still produces nonzero higher raw or centered moments. At fourth order,

$$
\langle\eta_i\eta_j\eta_k\eta_l\rangle
=C_{ij}C_{kl}+C_{ik}C_{jl}+C_{il}C_{jk}.
$$

The three terms pair the four labeled insertions in every possible way. Repeated indices retain multiplicity: ⟨ηᵢ⁴⟩=3Cᵢᵢ². Odd centered Gaussian moments vanish. The same generating-function argument produces all pairings at higher even order.

For four different sites of the ring, the three pair products are 1/25,4/225,1/25, so the centered fourth moment is 22/225. A zero connected fourth cumulant therefore does not mean that the fourth moment or all correlations vanish.

## 11. Nonzero means expose a common fourth-order mistake

At nonzero μ, the raw fourth moment contains the product of four means, six terms with one covariance and two means, and the three covariance pairings. Correct fourth cumulants can be formed from centered variables, followed by subtraction of the three centered pair products.

Subtracting products of raw two-point functions from the raw fourth moment is generally wrong. In this Gaussian family that incorrect operation returns

$$
\langle\phi_i\phi_j\phi_k\phi_l\rangle
-\!\sum_{\mathrm{three\ pairings}}\langle\phi_a\phi_b\rangle
\langle\phi_c\phi_d\rangle=-2\mu_i\mu_j\mu_k\mu_l.
$$

For the localized-source ring, the raw fourth moment is 2732/16875, the correct fourth cumulant is zero, but the incorrect raw-pair subtraction is −28/5625. This is an algebraic error in defining “connected,” not evidence for a non-Gaussian interaction.

Gaussian Wick factorization is not a rule for every distribution with the same covariance. A zero-mean variable taking ±1 equally has variance one but fourth cumulant 1−3=−2. Means and covariances alone cannot identify general higher-order fluctuations.

## 12. Declare a mean-zero ensemble instead of hiding a zero mode

Now impose 1ᵀφ=0. Parameterize the allowed configurations by φ=Bz, where B has columns eₐ−eₙ₋₁ for a=0,…,n−2. The reduced precision is A=BᵀKB. Its positivity is the relevant condition on this (n−1)-dimensional space.

$$
G=B A^{-1}B^T,\qquad \mu=GJ,\qquad C=TG,
\qquad P=I-\frac{\mathbf1\mathbf1^T}{n}.
$$

This G has a zero constant direction. It is not an inverse of K on the full ambient space. The mean and covariance obey 1ᵀμ=0 and C1=0. The physical field values are constrained together even if the original energy had no neighbor terms.

The choice is a different ensemble, not merely replacing a zero eigenvalue by a tiny positive number. Every draw has zero sum. The calculation keeps this choice in the input and result.

## 13. Track the integration measure and projected Green equations

The coordinates z are not orthonormal: BᵀB=I+11ᵀ and det(BᵀB)=n. We use induced Euclidean measure on the allowed hyperplane, so dσ=√n dz. Therefore the restricted precision determinant is det(A)/n and

$$
Z(0)=\frac{(2\pi T)^{(n-1)/2}}{\sqrt{\det(A)/n}}.
$$

A delta-function integral with δ(1ᵀφ) would carry a different constant measure factor. Normalized expectations agree after normalization, but absolute partition functions require the convention. The source-dependent log ratio remains JᵀGJ/(2T).

The correct residual equations on the hyperplane are

$$
P(K\mu-J)=0,\qquad PKG=P,\qquad GKP=P.
$$

The ambient residual Kμ−J can be a nonzero constant vector. A Lagrange multiplier α enforces Kμ−J+α1=0; the returned multiplier makes that constraint force visible. It is not a failure of the projected equation.

In particular, imposing KG=I would be impossible for a rank n−1 covariance. The projected integration-by-parts identity is CKP=TP. Exact residual matrices test these declared equations; they are not a statistical estimate from sampled configurations.

## 14. Conditioning is not always the same as centering each sample

For a full positive Gaussian covariance C₀, conditioning on a zero sum gives

$$
C_{\mathrm{cond}}=C_0-
\frac{C_0\mathbf1\mathbf1^TC_0}{\mathbf1^TC_0\mathbf1}.
$$

By contrast, subtracting the arithmetic mean of every sample gives covariance PC₀P. These agree when the constant vector is an eigenvector of C₀, as for a uniform massive open chain or ring. They need not agree with fixed-zero exterior boundaries.

For three fixed-zero sites at m²=0, κ=T=1, C₀ has first row (3/4,1/2,1/4) and row sums (3/2,2,3/2). Conditioning gives C_cond,00=3/10, while post-centering gives (PC₀P)₀₀=11/36. The difference follows from the stated distributions; neither operation should be silently substituted for the other.

Even independent sites reveal the constraint's effect. At K=I,T=1, the constrained covariance is P, with diagonal 1−1/n and off-diagonal −1/n. Negative connected entries here express a fixed total, not a negative neighbor stiffness.

## 15. Compare legitimate massless and nearly massless limits

For the massless four-site ring with κ=T=1 and zero sum, G has first row (5/16,−1/16,−3/16,−1/16). Its nonzero restricted eigenvalues multiply to 16. The constant direction has been removed explicitly, so this is a proper Gaussian on three dimensions.

The unconstrained massive ring behaves differently: G1=1/m²·1, so its response to a uniform source diverges as m²→0. The covariance contribution of the constant mode is T/(nm²) at every pair of sites. Dropping that mode before the limit changes the question.

Fixed-zero exterior boundaries provide another mechanism. For three sites with m²=0 and κ=1, det K=4 and the full inverse exists. Pinning exterior values is not the same as imposing a zero spatial average. Finite-volume, zero-mode and boundary effects must be separated before discussing spatial decay or critical limits.

## 16. Use exact transformations as experiments

At fixed K,J, multiplying T by two doubles connected two-point functions and multiplies centered fourth moments by four. Means and static susceptibility stay unchanged. At fixed K,T, changing J moves the mean and raw moments while leaving the connected covariance unchanged.

On the mean-zero hyperplane, J and J+c1 give the same distribution because their energy difference vanishes for every allowed configuration. Hence G1=0 makes the mean and source log ratio invariant under that shift. This invariance also holds with fixed-zero exterior boundaries when the additional mean-zero constraint is explicitly chosen.

Simultaneously multiplying K,J,T by a positive number leaves the whole normalized distribution unchanged. Susceptibility with respect to the rescaled energy source still scales inversely with that number; the source coordinate has changed. The source spelling can differ while fractions represent the same number. These are useful tests of conventions and exact arithmetic, not replacements for checking the matrix identities.

## 17. Connections to AI: precision graphs, kernels and scores

A full Gaussian Markov random field uses local entries of K/T to describe conditional distributions. Holding all other coordinates fixed, φᵢ has mean (Jᵢ−∑ⱼ≠ᵢKᵢⱼφⱼ)/Kᵢᵢ and variance T/Kᵢᵢ. A zero off-diagonal precision entry describes conditional independence in the full nonsingular Gaussian model; marginal covariance can still be nonzero.

The mean-zero ensemble has a singular ambient covariance and deterministic dependence. Do not apply the same full-space inverse-covariance interpretation without accounting for its constraint. Similarly, a covariance kernel can define Gaussian-process fluctuations, while source response is C/T under our energy convention. “Kernel” does not specify temperature factors automatically.

The full-density score is −(Kφ−J)/T. On the constrained hyperplane the intrinsic score is its tangent projection −P(Kφ−J)/T. A learned score or neural field solver should be checked on its declared integration space. A component normal to the hyperplane cannot be tested by unconstrained formulas as if it were a fluctuating coordinate.

## 18. Read outputs and numerical claims carefully

The bounded preparations use 3–8 sites, nonnegative rational m²,κ≤16, positive rational T≤16 and source entries of magnitude at most16. Each numerator and denominator has at most12 digits. Four probe indices may repeat. Singular unconstrained preparations and completely flat energies are rejected.

The calculation returns exact fractions for precision, reduced factors, mean, response, connected/raw two-point functions, selected fourth moments and source log ratios. Positive factor pivots and projected residuals are exact algebraic checks. The absolute normalization remains an analytic expression with π and a square root.

The independent reference constructs the precision from incidence outer products and uses a bordered inverse for constrained cases, rather than the production basis solve. It also uses a separate determinant algorithm and recursive Gaussian moments. A future heat map or spatial drawing is a floating projection of these rational results, not another proof. A finite ring's matrix is not by itself a fitted infinite-system correlation length.

## 19. Worked questions with explanations

1. **Raw versus connected.** In the localized-source four-site ring, C₀₁=1/5 but ⟨φ₀φ₁⟩=22/75. The extra7/75 is the product of means, not an extra fluctuation correlation.
2. **Temperature factors.** Double T at fixed K,J. Covariance doubles and log Z(J)/Z(0) halves; μ and χ do not change. Differentiating the physical source inserts1/T.
3. **A repeated fourth index.** For a centered field with Cᵢᵢ=7/15, ⟨ηᵢ⁴⟩=3(7/15)²=49/75, while its fourth cumulant is zero.
4. **A false interaction signal.** Why does raw fourth minus raw pair products give −28/5625 in the default source example? The missing mean terms leave −2μ₀μ₁μ₂μ₃. Center first to compute the actual cumulant.
5. **A missing probability distribution.** Why reject m²=0 on an unconstrained ring? The constant direction is unconfined. A numerical pseudo-inverse alone does not define the omitted measure or ensemble.
6. **An anticorrelation without repulsion.** At independent K=I on n=4 sites constrained to zero sum, C₀₁=−1/4. A positive fluctuation must be balanced elsewhere even though κ=0.
7. **The measure factor.** For n=4, the basis Gram determinant is4 and dσ=2 dz. Omitting that factor changes Z(0) but cancels from normalized expectations. The restricted determinant is det A/4.
8. **Conditioning versus demeaning.** In the three-site pinned example the two variances are3/10 and11/36. Their difference shows that projecting samples need not produce the conditional ensemble. Specify the operation before comparing covariance matrices.

## 20. Reading and remaining scope

The source-dependent Gaussian construction, connected two-point functions and Green-function interpretation are discussed in [David Tong's Statistical Field Theory, §2.2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2). Our finite matrices, constraint basis, induced measure and rational verification are derived explicitly for the declared lattice model.

This chapter supplies the Gaussian starting point for correlation functions. Interacting non-Gaussian connected correlations, spatial decay, momentum spectra, finite-size limits and the definition of correlation length require subsequent treatments. It does not define a continuum path-integral measure, prove critical scaling, or replace quantum time-ordered correlators with a classical covariance.
