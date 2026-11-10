## 1. Random preparation and coherent addition are different operations

Suppose a source tosses a fair coin and prepares either e₀ or e₁. If we keep only the quantum system and discard the coin record, a Z measurement gives the two results equally often. The coherent state (e₀+e₁)/√2 gives the same Z histogram. That agreement does not identify the preparations: an X measurement gives equal probabilities for the random source but a certain positive result for the coherent state.

The distinction is between choosing one preparation on each run and adding amplitudes before measurement. A relative phase affects the latter. Averaging vectors is not a rule for random preparation. As an extreme check, choosing ψ or −ψ describes the same pure state on every run, but their average vector is zero. Normalizing that average cannot repair the mistake.

A useful conceptual picture is **preparation labels → weighted pure-state projectors → density operator → measurement statistics**. The labels can contain more information than the density operator. A density operator describes the system available for measurement; it is not a stochastic transition matrix or a list of which label occurred on each run.

## 2. Derive the density operator from measurement probabilities

Let normalized states ψᵢ be selected with classical probabilities pᵢ. A rank-one test onto a normalized direction v has probability Σᵢpᵢ|v†ψᵢ|². Moving the scalar products into a trace isolates one operator that works for every test direction.

$$
\rho=\sum_i p_i|\psi_i\rangle\langle\psi_i|
=\sum_{i:w_i>0}\frac{w_i}{\sum_jw_j}\frac{z_i z_i^\dagger}{z_i^\dagger z_i},\qquad
p(v)=v^\dagger\rho v=\operatorname{Tr}(\rho vv^\dagger).
$$

The raw vectors zᵢ need not have unit length. Each occupied vector is normalized through its own squared norm. Multiplying one raw vector by a nonzero complex number leaves its projector unchanged. A zero weight contributes nothing; a zero raw vector has no normalized state and must not be used with positive weight.

For a Hermitian observable A, the same linearity gives moments. The second moment uses A² before taking the trace. Squaring the mean would answer a different question.

$$
\langle A\rangle_\rho=\operatorname{Tr}(\rho A),\qquad
\langle A^2\rangle_\rho=\operatorname{Tr}(\rho A^2),\qquad
\operatorname{Var}_\rho(A)=\operatorname{Tr}(\rho A^2)-[\operatorname{Tr}(\rho A)]^2.
$$

These equations explain why averaging projectors is appropriate: it averages all outcome probabilities and moments consistently. It does not retain a particular classical preparation record.

## 3. Hermiticity, positivity and trace one are all needed

A finite density operator satisfies three conditions. Positivity means a nonnegative quadratic form for every vector, not nonnegative entries in a displayed matrix.

$$
\rho=\rho^\dagger,\qquad x^\dagger\rho x\ge0\quad\text{for every }x,\qquad
\operatorname{Tr}\rho=1
\quad\Longleftrightarrow\quad
\rho=\sum_{k=1}^d\lambda_k|k\rangle\langle k|,\quad
\lambda_k\ge0,\quad\sum_k\lambda_k=1.
$$

For the ensemble construction, x†ρx is a sum of pᵢ|ψᵢ†x|², so it is nonnegative. Each pure projector has trace one. Conversely, the finite-dimensional spectral theorem supplies orthonormal eigenvectors; nonnegative eigenvalues summing to one give an ensemble representation. The existence of that spectral representation does not mean an apparatus originally prepared those eigenvectors.

Positive diagonal entries and trace one do not suffice. The Hermitian matrix with rows (1/2,3/4) and (3/4,1/2) has a valid-looking Z histogram but eigenvalues 5/4 and −1/4. The negative X test probability is −1/4. This is a theoretical invalid-state example, not a preparation accepted by the convex-ensemble calculator.

## 4. Purity, rank and the geometry of the state set

The eigenvalues form a probability distribution. Cauchy–Schwarz gives a lower bound on their sum of squares, while λ²≤λ gives the upper bound.

$$
\frac1d\le\operatorname{Tr}(\rho^2)=\sum_k\lambda_k^2\le1,\qquad
L(\rho)=1-\operatorname{Tr}(\rho^2),\qquad
\rho\text{ is pure}\Longleftrightarrow\operatorname{rank}\rho=1\Longleftrightarrow\rho^2=\rho.
$$

The lower equality requires ρ=I/d. The upper equality requires one eigenvalue one and all others zero. L is the unscaled linear entropy, whose maximum is 1−1/d; it is not von Neumann entropy. In dimension one, the unique state is both pure and maximally mixed. That harmless boundary case helps check formulas.

The density operators form a convex set. A rank-one state is an extreme point: every vector perpendicular to its support has zero expectation, forcing every positive state in a purported convex decomposition to have the same support. A mixed state is not extreme: transfer a sufficiently small amount between two positive eigenvalues to construct two distinct states whose average is ρ.

For a Hermitian positive semidefinite matrix, rank equals the largest size of a positive principal minor. All principal minors are nonnegative. Checking only the leading minors is insufficient in the semidefinite case: diag(0,−1,2) has every leading determinant zero but a negative singleton principal minor. These exact criteria differ from a floating-point decision that an eigenvalue “looks small.”

## 5. Different ensembles can encode one state

Equal mixtures of the two Z eigenstates, the two X eigenstates or the two Y eigenstates all give I/2. Their component states differ, but every measurement performed only on the retained qubit agrees. The table lists state-only laws, not a measurement of the discarded preparation label.

| Preparation pair, equal weights | Resulting density | p(Z positive) | p(X positive) |
| --- | --- | --- | --- |
| e₀ and e₁ | I/2 | 1/2 | 1/2 |
| (e₀+e₁)/√2 and (e₀−e₁)/√2 | I/2 | 1/2 | 1/2 |
| (e₀+ie₁)/√2 and (e₀−ie₁)/√2 | I/2 | 1/2 | 1/2 |

If an external record tells us which source setting was used, the combined description contains additional information. Discarding that record is part of the preparation scenario. The density operator does not assert that the labels never existed; it says which predictions are available without them.

This also separates a random preparation from a coherent superposition. The pure positive X state has the same Z law as I/2 but a different X law and purity. Off-diagonal coherence is meaningful relative to a basis, whereas being pure or mixed is invariant under a unitary basis change.

## 6. A weighted-column matrix connects density and Gram geometry

Place √pᵢψᵢ in column i of B. Multiplication on the state space produces the density operator; multiplication on the label space produces a Gram matrix. The spaces can have different dimensions.

$$
B_i=\sqrt{p_i}\,\psi_i,\qquad \rho=BB^\dagger,\qquad G=B^\dagger B,\qquad
\rho v=\lambda v,\ \lambda>0\ \Longrightarrow\ G(B^\dagger v)=\lambda(B^\dagger v).
$$

The transferred vector is nonzero because its squared norm is λ||v||². Applying the reverse construction proves equality of the nonzero spectra, including multiplicity. Hence density and Gram have the same rank and purity, with extra zero eigenvalues in the larger space. With four label slots and d≤4, χG(λ)=λ^(4−d)χρ(λ).

Purity has another useful form, obtained by expanding the trace of the mixture squared.

$$
\operatorname{Tr}(\rho^2)=\sum_{i,j}p_i p_j|\langle\psi_i|\psi_j\rangle|^2.
$$

Orthogonal components leave only Σpᵢ²; repeated copies of the same ray give purity one regardless of the labels. The raw overlap matrix Sᵢⱼ=zᵢ†zⱼ and factors tᵢ=pᵢ/Nᵢ give a Gram principal determinant det(SJ)∏ᵢ∈Jtᵢ. Thus the finite experiment can certify Gram characteristic coefficients exactly even when the displayed columns contain square roots. It does not claim to numerically diagonalize every general matrix.

## 7. Work an unequal, nonorthogonal mixture by hand

Choose e₀ with probability 2/5 and the positive X state with probability 3/5. Add their projectors, rather than their amplitudes.

$$
\rho=\frac25\begin{pmatrix}1&0\\0&0\end{pmatrix}
+\frac35\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}
=\begin{pmatrix}7/10&3/10\\3/10&3/10\end{pmatrix},\quad
\det\rho=\frac3{25},\quad
\operatorname{Tr}(\rho^2)=\frac{19}{25},\quad
\lambda_\pm=\frac{5\pm\sqrt{13}}{10}.
$$

The Z probabilities are 7/10 and 3/10. For the dimensionless Pauli Z observable, the mean is 2/5, the second moment is one and the variance is 21/25. Purity is 0.76, obtained from both diagonal squares and two off-diagonal squared magnitudes.

Keeping only the diagonal gives purity 29/50=0.58. Their difference is 9/50=0.18, the squared Hilbert–Schmidt norm of the removed off-diagonal part. Computing only the diagonal squares would miss this contribution. Notice also that a negative measurement result is compatible with a positive density operator: positivity concerns probabilities and quadratic forms, not the signs of all observable eigenvalues.

## 8. Total variance separates two ensemble-dependent contributions

Let μᵢ and Varᵢ(A) refer to component i, and let μ=Σpᵢμᵢ. Expand each second moment as its variance plus squared mean. Adding and subtracting μ² gives an exact accounting identity.

$$
\operatorname{Var}_\rho(A)=\sum_i p_i\operatorname{Var}_i(A)
+\sum_i p_i(\mu_i-\mu)^2.
$$

In the preceding unequal mixture, the two Z means are 1 and 0. The within-component contribution is 3/5 and the between-mean contribution is 6/25, summing to 21/25. Neither term is generally recoverable from ρ without specifying the ensemble.

| Ensemble representing I/2 | Mean Z | Within-component variance | Between-mean variance |
| --- | --- | --- | --- |
| Equal Z eigenstates | 0 | 0 | 1 |
| Equal X eigenstates | 0 | 1 | 0 |
| Equal Y eigenstates | 0 | 1 | 0 |

All three rows describe the same state-only Z variance, namely one. Calling the first term “quantum” and the second “classical” without mentioning the chosen preparation can therefore mislead. Their decomposition changes with the external preparation description; the total predicted distribution does not.

## 9. Recover the missing matrix entries with additional measurements

A diagonal histogram reveals only d−1 independent numbers. A general trace-one Hermitian matrix has d²−1 real parameters. To recover its coherences, test each pair i<j in real and imaginary superposition directions. Let dij=ρii+ρjj.

$$
p_{X\pm}=\frac{d_{ij}}2\pm\operatorname{Re}\rho_{ij},\qquad
p_{Y\pm}=\frac{d_{ij}}2\mp\operatorname{Im}\rho_{ij},\qquad
\operatorname{Re}\rho_{ij}=\frac{p_{X+}-p_{X-}}2,\quad
\operatorname{Im}\rho_{ij}=\frac{p_{Y-}-p_{Y+}}2,\quad
p_{\mathrm{outside}}=1-d_{ij}.
$$

Here Y positive means the vector (eᵢ+ieⱼ)/√2. The minus sign in its probability follows from conjugating the bra. For the pure positive Y qubit, ρ01=−i/2, so pY+=1. Reversing that imaginary sign would exchange the two Y outcomes.

For d>2, the two pair directions span only a subspace. Include its orthogonal complement as the outside result. For I/3 and any pair, X positive, X negative and outside each have probability 1/3; the same is true for the Y test. The displayed experiment uses exact probabilities. Finite-shot estimation, inconsistent frequencies and statistical uncertainty require a separate inference model.

## 10. Conditioning on a measurement does not always produce a pure state

For an ideal spectral measurement with orthogonal projectors Pa, the Lüders instrument specifies both the outcome law and the state update. Stating projectors alone as probabilities should not be confused with specifying every possible apparatus that could realize those probabilities.

$$
p_a=\operatorname{Tr}(\rho P_a),\qquad
\rho_{\mid a}=\frac{P_a\rho P_a}{p_a}\quad(p_a>0),\qquad
\rho_{\mathrm{nonselective}}=\sum_aP_a\rho P_a.
$$

Take ρ=I/3 and A=2e₀e₀†−I. The positive result has probability 1/3 and a pure conditional state. The negative result has probability 2/3 and conditional density diag(0,1/2,1/2), whose purity is 1/2. A degenerate eigenspace can retain mixing. The nonselective state remains I/3.

If pa=0 there is no conditional density; the calculation returns null instead of dividing by an epsilon or inventing a state. For A=aI+b(2P−I), b=0 merges the whole eigenspace before conditioning. In d=1 the only eigenvalue is a+b, with rank one; a rank-zero complementary outcome must not be presented as another physical result.

## 11. Nonselective measurement removes specific coherences

Spectral pinching Δ(ρ)=ΣPaρPa retains blocks inside each eigenspace and removes cross-block entries. In the Hilbert–Schmidt inner product, Δ is an orthogonal projection. Its retained and removed components are orthogonal, giving a Pythagorean identity.

$$
\operatorname{Tr}(\rho^2)-\operatorname{Tr}[(\Delta\rho)^2]
=\|\rho-\Delta\rho\|_{\mathrm{HS}}^2\ge0,\qquad
\|M\|_{\mathrm{HS}}^2=\operatorname{Tr}(M^\dagger M).
$$

For a complete rank-one basis measurement, Δ simply keeps the diagonal. For a degenerate measurement, coherence inside a block can survive. The 0.18 difference in the worked mixture is a concrete instance of the complete-basis identity.

The squared Hilbert–Schmidt difference is not a trace distance or a probability that a detector identifies the state. Nor does this pinching identity say that every quantum operation increases entropy or decreases purity. The next chapter supplies other metrics and distinguishes unital dephasing from general state transformations.

## 12. Exact input and near-rank boundaries

The finite family accepts forty-nine owned bounded integers for d=1…4, four nonnegative weights, complex raw representatives, a probe and rational observable center/gap. Occupied representatives and the probe must be nonzero. Unused coordinates are zero explicitly; reducing dimension does not silently truncate them. A zero-weight nonzero vector may be described, but it contributes no weighted column to ρ.

Density entries, principal minors, rank, Gram invariants, tomography and measurement accounts are exact complex rationals. Normalized amplitudes and weighted columns use directed square-root intervals, with 768-bit work arithmetic and outward 512-bit observations. An unresolved observation can remain null; neither output normalization nor a rounded near-zero classification replaces its certificate.

For example, mix raw vectors (M,1) and (M,−1) equally with M=1000000.

$$
\rho=\operatorname{diag}\!\left(\frac{M^2}{M^2+1},\frac1{M^2+1}\right),\qquad
\operatorname{rank}\rho=2\quad\text{for every finite nonzero }M.
$$

A plot may look pure, but the small eigenvalue is positive. Independent symbolic rank and matrix calculations check this case. Limits such as M→∞ are different statements from equality at a bounded input.

## 13. Exercises with explanations

**1. Why does an equal choice of ψ and −ψ remain pure?**

**Explanation.** Their projectors are identical, so their mixture is ψψ† and has purity one. Averaging their vectors would produce zero, demonstrating why that operation is inappropriate.

**2. Does the matrix with diagonal 1/2 and off-diagonal 3/4 describe a state?**

**Explanation.** No. Its eigenvalues are 5/4 and −1/4. The negative X probability exposes the failure of positivity despite trace one and a valid Z histogram.

**3. What is missing if the unequal mixture's purity is reported as 0.58?**

**Explanation.** The two off-diagonal squared magnitudes, totaling 0.18. Full purity is 0.76. The smaller number is the purity after complete Z-basis dephasing.

**4. Do the two variance contributions belong to ρ alone?**

**Explanation.** No. Z and X eigenstate ensembles for I/2 give within/between contributions 0/1 and 1/0 for a Z measurement. The total variance is one in both cases.

**5. Why can the Gram matrix have more eigenvalues than ρ?**

**Explanation.** Its dimension counts label slots rather than state-space coordinates. The nonzero spectra agree; a larger matrix carries extra zeros. Trace and purity therefore agree too.

**6. What is the Y-positive probability when ρ01=−i/2?**

**Explanation.** One for this pure qubit, because pY+=1/2−Imρ01. For higher dimension also retain the pair's outside result before interpreting a full probability law.

**7. Does every nonzero measurement result yield purity one?**

**Explanation.** No. The rank-two negative eigenspace in the I/3 example yields purity 1/2. A zero-probability result has no conditional state at all.

**8. Can leading principal determinants alone certify semidefinite positivity?**

**Explanation.** No. The leading determinants of diag(0,−1,2) are all zero. Its negative singleton minor rules out positivity, so checking all relevant principal minors matters.

## 14. Connections to learning and remaining scope

The weighted-column construction connects density operators with Gram matrices, covariance geometry, positive kernels and low-rank factorization. In an AI model, a trace-one matrix is not automatically positive; a positive matrix is not automatically a calibrated probabilistic description of data. A parameterization, an estimation model and the meaning of measurements remain separate choices.

The following chapter studies general spectral entropy and analytic reduced-state examples alongside a mixed-qubit experiment. Composite-system dynamics, finite-shot tomography, general channel learning and research-level operator questions continue later. Open the textbook through **Book → Interactive Book**.

Background ensemble and state conventions are linked in [Preskill Chapter2](https://www.preskill.caltech.edu/ph219/chap2_15.pdf) and [IBM's convex-combination lesson](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/convex-combinations). The derivations, weighted-column certificates, worked preparations and numerical comparisons here are developed for this textbook.
