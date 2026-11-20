## 1. Specify the cut before discussing entanglement

An MPS bond has a capacity, but that number does not say how much of the capacity the physical state uses. We now examine the state across a chosen physical cut. The same vector can have different ranks and spectra at different cuts, so an entanglement claim must name its bipartition.

Take a nonzero pure n-qubit vector and group its first k bits into the left index i, with the remaining bits forming the right index j. Reshape the raw amplitudes into C without changing their order. Site0 is the most significant bit. The global squared norm N is independent of the cut.

$$
|\psi\rangle=\frac1{\sqrt N}\sum_{i,j}C_{ij}|i\rangle_L|j\rangle_R,\quad C\in\mathbb C^{2^k\times2^{n-k}},\quad N=\sum_{i,j}|C_{ij}|^2>0.
$$

| Four-qubit cut k | Left dimension | Right dimension | Side used for the small spectrum |
| --- | --- | --- | --- |
| 1 | 2 | 8 | Left |
| 2 | 4 | 4 | Left, by tie convention |
| 3 | 8 | 2 | Right |

An array reshape is not a physical measurement, a normalization step or a change of the state. It changes which physical labels index rows and columns. The calculator evaluates every contiguous cut, while the selected cut chooses which result to inspect. Arbitrary noncontiguous bipartitions require their own explicit regrouping and are not a hidden input mode.

## 2. Derive both physical density matrices from ket coefficients

Trace over the other subsystem by matching its basis label. Because C contains ket amplitudes, the complex conjugate belongs to the bra coefficient. Deriving the entries first prevents a common mistake on the right subsystem.

$$
(\rho_L)_{ii'}=\frac1N\sum_jC_{ij}\overline{C_{i'j}}=\left(\frac{CC^\dagger}{N}\right)_{ii'},\qquad (\rho_R)_{jj'}=\frac1N\sum_iC_{ij}\overline{C_{ij'}}=\left(\frac{C^{\mathsf T}\overline C}{N}\right)_{jj'}.
$$

Both matrices are Hermitian, positive semidefinite and have trace1. The right density is Cᵀ conjugate(C)/N. The matrix C†C/N has the same eigenvalues but is the complex conjugate of the physical right density; it need not have the same entries or predict the same phase-sensitive observables.

For the product row C=(1,i), the right density has upper-right entry−i/2. With the usual Pauli convention, Y=−2Imρ₀₁=+1. Copying C†C as the physical density would reverse that sign. Spectral tests alone cannot find this error, so we will check both the density entries and the truncated complex amplitudes later in the chapter.

## 3. Singular values become Schmidt coefficients

Take a singular value decomposition of the normalized coefficient matrix. Its left and right singular vectors are orthonormal in their respective coordinate spaces. Expanding the product entry by entry identifies the two ket factors.

$$
\frac C{\sqrt N}=U\Sigma V^\dagger,\qquad |\psi\rangle=\sum_{a=1}^{q}\sigma_a|u_a\rangle_L|\overline{v_a}\rangle_R,\qquad\lambda_a=\sigma_a^2,\quad\sum_a\lambda_a=1.
$$

The right ket is the complex conjugate of a column of V in this fixed coefficient convention. The λ values are normalized density weights, while σ values are their nonnegative square roots. Neither is the singular value of the unnormalized C unless N=1. A global rescaling of raw amplitudes leaves λ unchanged.

The nonzero spectra on both physical sides agree because each paired Schmidt component contributes the same λ to either partial trace. The larger side can have additional zeros. A spectrum represented as distinct groups must retain each group's multiplicity: a single group with weight1/2 and multiplicity2 carries total probability1, not1/2.

## 4. Rank constrains bonds and depends on the physical partition

Schmidt rank counts the strictly positive weights, including multiplicity. For a nonzero pure state it equals the coefficient matrix rank. A factorization through a virtual bond limits that rank, as established in the previous chapter.

$$
q=\operatorname{rank}C\le\min(2^k,2^{n-k}),\qquad C=LB\implies q\le D_k.
$$

GHZ₃ has rank2 at either contiguous cut. A product state has rank1 even if stored with a much larger virtual dimension. Conversely, a state with rank4 across a cut cannot be represented there by an exact bond2 MPS. Reducing the bond changes the state unless some positive components are discarded with an explicitly accepted error.

| Four-qubit pure state | Rank at cut1 | Rank at cut2 | Rank at cut3 |
| --- | --- | --- | --- |
| Bell pair01 times Bell pair23 | 2 | 1 | 2 |
| Bell pair02 times Bell pair13 | 2 | 4 | 2 |

At the middle cut the first state has both pairs entirely inside one side or the other, while the second has two pairs crossing. Each crossing pair doubles the rank in this example. Reordering sites changes which physical subsystem a contiguous cut names; it is not a contradiction of invariance under local basis changes within a fixed bipartition.

## 5. Purity and entropy describe the distribution of weights

Rank records support, not how probability is distributed over that support. Purity and entropy add information about the shape of the spectrum. Throughout this chapter the input is globally pure, so the reduced-state entropy is a bipartite entanglement entropy.

$$
\mathcal P=\operatorname{Tr}\rho_L^2=\sum_a\lambda_a^2,\qquad S=-\sum_a\lambda_a\ln\lambda_a,\qquad S_{\mathrm{bits}}=\frac S{\ln2},\qquad0\le S\le\ln q.
$$

For a product state the spectrum is(1,0,…) and the entropy is0. A Bell pair has two weights1/2, purity1/2 and entropy ln2, or one bit. The convention0 ln0=0 is the continuous limit at zero. If a weight has multiplicity m, multiply its entropy contribution by m.

The upper bound follows because a normalized distribution on q support points has entropy at most that of the uniform distribution. A small entropy does not prove rank1: a tiny strictly positive second weight still makes rank2. For mixed global states, reduced entropy can include classical mixing and cannot simply be reused as this pure-state entanglement measure.

## 6. Why keeping the largest weights is optimal at one cut

We approximate the normalized coefficient matrix by a matrix B of rank at most r, using Frobenius distance. This is exactly Hilbert-space distance between the corresponding states because the product basis is orthonormal. The approximation B is not required to have unit norm in this problem.

Let P project onto the column space of B. The error outside that space cannot be cancelled by B, so it supplies a lower bound. In the eigenbasis of ρL, the diagonal elements of P lie between0 and1 and sum to at most r. The weighted sum is therefore largest when it occupies the largest r weights.

$$
\|C/\sqrt N-B\|_F^2\ge\|(I-P)C/\sqrt N\|_F^2=1-\operatorname{Tr}(P\rho_L)\ge1-\sum_{a=1}^{r}\lambda_a,\qquad\operatorname{rank}B\le r.
$$

The two error components inside and outside the range of P are orthogonal. This proves the first inequality. Allocating the limited diagonal mass to the largest λ proves the second; keeping the corresponding Schmidt terms attains both bounds. Thus the lower bound is achievable, rather than merely a heuristic for choosing entries.

This optimization is over a specified rank budget at one cut in the Hilbert/Frobenius norm. It does not minimize a chosen energy, every observable error or a global cost subject to several simultaneous MPS bond limits. Those are different optimization problems with additional constraints.

## 7. Keep the three states separate

Write the normalized original state, its retained Schmidt sum and the renormalized retained state as separate objects. We order the positive λ values from largest to smallest. A budget larger than the support keeps all positive components without inventing extra zero-weight Schmidt vectors.

$$
|\widetilde\psi_r\rangle=\sum_{a=1}^{\min(r,q)}\sqrt{\lambda_a}|u_a\rangle|\overline{v_a}\rangle,\quad S_r=\sum_{a=1}^{\min(r,q)}\lambda_a,\quad |\widehat\psi_r\rangle=\frac{|\widetilde\psi_r\rangle}{\sqrt{S_r}}\quad(S_r>0).
$$

The tilde state is the orthogonal retained component of the original state. Its squared norm is S_r. The discarded component is perpendicular to it, so their squared norms add to1. Renormalization changes the surviving amplitudes by1/sqrt(S_r); it does not restore discarded components.

The raw matrix C_ret, the retained vector C_ret/sqrt(N), and its renormalization C_ret/sqrt(N S_r) are also distinct numerical outputs. Comparing an unnormalized raw matrix directly with a normalized original vector would produce a scale-dependent error unrelated to the theorem. Name the denominator whenever showing an amplitude.

## 8. Error, fidelity and normalized distance have different formulas

Orthogonality gives the unnormalized error. The overlap between the original and the renormalized retained state is positive sqrt(S_r) with our phase choice. Expanding the squared distance between two unit vectors then gives the normalized distance.

$$
\epsilon_r=\|\psi-\widetilde\psi_r\|^2=1-S_r,\qquad|\langle\psi|\widehat\psi_r\rangle|^2=S_r,\qquad\|\psi-\widehat\psi_r\|^2=2(1-\sqrt{S_r}).
$$

| Quantity relative to normalized original | Formula | Requirement |
| --- | --- | --- |
| Retained squared norm | S_r | Any budget |
| Optimal unnormalized squared error | 1−S_r | Any budget |
| Normalized fidelity squared | S_r | S_r>0 |
| Normalized squared distance | 2(1−sqrt(S_r)) | S_r>0, aligned phase |

Fidelity here always means the squared overlap when labeled fidelity squared. Hilbert distance depends on global phase, so the distance uses the natural positive-overlap phase, equivalently the minimum over a global phase. Quoting fidelity without this distinction can turn a correct number into an incorrect geometric interpretation.

The renormalized retained state also maximizes overlap among normalized states of rank at most r. For a fixed allowed column space, Cauchy–Schwarz bounds overlap by the norm of the original state's projection, and the largest possible projected norm is sqrt(S_r). This connects the two optimization problems while keeping their different error values explicit.

## 9. Work through unequal weights without a solver

Take a raw two-qubit vector with amplitudes2 at00 and1 at11. Its raw squared norm is5. The normalized reduced densities are diagonal, so the weights can be read directly rather than estimated from an iterative algorithm.

$$
|\psi\rangle=\frac{2|00\rangle+|11\rangle}{\sqrt5},\quad(\lambda_1,\lambda_2)=(4/5,1/5),\quad\mathcal P=17/25,\quad\epsilon_1=1/5,\quad d_1^2=2-4/\sqrt5.
$$

At budget1 the retained raw matrix has only entry2 at00. On the normalized input it gives(2/sqrt(5))|00〉, with squared norm4/5. Renormalization gives |00〉. The unnormalized squared error is1/5, but normalized distance squared is2−4/sqrt(5), approximately0.2111. Replacing one formula by the other loses the effect of rescaling the retained component.

Purity17/25 and entropy−(4/5)ln(4/5)−(1/5)ln(1/5) describe the original spectrum. They are not properties of the normalized retained product state, whose reduced purity is1 and entropy0. Truncation changes entanglement as well as amplitudes; label which state a displayed entropy belongs to.

## 10. Degeneracy can make the best state nonunique

The Bell state has equal positive weights. At budget1 there is no distinguished direction in the two-dimensional Schmidt eigenspace. Two visibly different retained vectors attain the same minimum error.

$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},\qquad |\widetilde\psi_1^{(0)}\rangle=\frac{|00\rangle}{\sqrt2},\qquad |\widetilde\psi_1^{(1)}\rangle=\frac{|11\rangle}{\sqrt2},\qquad\epsilon_1=1/2.
$$

After renormalization the two choices are |00〉 and |11〉. Both have fidelity squared1/2 with the original, and both have aligned distance squared2−sqrt(2). Equal optimal values do not imply a unique optimal vector. Arbitrarily selecting an eigenvector would hide a choice that the mathematics does not specify.

The same ambiguity persists after rotating the Schmidt basis jointly on both sides. The calculator returns the whole eigenspace projector for the weight group, but if a budget cuts through that positive group it returns no chosen state. It still reports the optimal retained weight and error. Neither silently raising the budget nor choosing a convenient basis is the requested rank-constrained result.

## 11. Degeneracy inside a retained block is harmless

Use a four-qubit middle-cut matrix with diagonal entries3,2,2,1. The squared amplitudes give one largest weight, a repeated middle group and one smallest weight. This separates a degenerate spectrum from a degenerate truncation boundary.

$$
C=\operatorname{diag}(3,2,2,1),\quad N=18,\quad\lambda=(1/2,2/9,2/9,1/18),\quad\mathcal P=19/54.
$$

| Rank budget | Positive rank kept | Retained S_r | Discarded error | Unique unnormalized optimum | Normalized optimum exists |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1 | Yes | No |
| 1 | 1 | 1/2 | 1/2 | Yes | Yes |
| 2 | 2 | 13/18 | 5/18 | No | Yes |
| 3 | 3 | 17/18 | 1/18 | Yes | Yes |
| 4 | 4 | 1 | 0 | Yes | Yes |

Budget2 must choose one direction out of the two weight2/9 directions, so the state is nonunique. Budget3 keeps the whole repeated group; its summed projector is unique even though a basis inside the group is not. At budget1 the entire repeated group is discarded, which is also unambiguous.

An implementation should decide multiplicity from the actual spectrum, not merge nearby values using a display tolerance. A gap that is small but positive still distinguishes the leading subspace mathematically. Numerical conditioning and mathematical uniqueness are separate questions, both relevant to interpreting a visualization.

## 12. Zero budget and budgets beyond support are defined cases

The allowed rank is an upper bound. Budget0 has exactly one unnormalized feasible vector, the zero vector, with error1 relative to the normalized input. No normalized zero vector exists. A request for its normalized fidelity or normalized distance therefore has no defined answer.

$$
r=0:\ S_0=0,\quad\widetilde\psi_0=0,\quad\epsilon_0=1;\qquad r\ge q:\ S_r=1,\quad\epsilon_r=0,\quad\widehat\psi_r=\psi.
$$

The calculator represents these normalized outputs as null rather than0. For a product input of rank1, budget4 still retains only rank1 positive support. It reports zero discarded weight and the original state. It does not append arbitrary nullspace vectors to claim four physically occupied Schmidt modes.

A full positive-support projector can have rank smaller than the ambient density dimension. Its action on the given coefficient matrix is nevertheless the identity on that matrix's support. Zero eigenvalues on the larger physical side are structural padding, not additional positive probabilities or new ambiguity in the retained physical vector.

## 13. A right physical projector acts through a transpose

On the right subsystem, the new coefficient of |i〉|j〉 is the sum over the old right index l of C_il P_jl. This is the matrix product C Pᵀ. On the left, the corresponding action is ordinary left multiplication.

$$
(I\otimes P)|\psi\rangle\quad\longleftrightarrow\quad C_{\mathrm{ret}}=CP^{\mathsf T},\qquad (P\otimes I)|\psi\rangle\quad\longleftrightarrow\quad C_{\mathrm{ret}}=PC.
$$

For a Hermitian complex projector, P†=P but Pᵀ can differ from P. Thus replacing the transpose with an adjoint is not harmless. Consider a three-qubit cut after two sites, where the right side is smaller and is used for the spectrum.

$$
C=\begin{pmatrix}2&2i\\1&-i\\0&0\\0&0\end{pmatrix},\quad\rho_R=\begin{pmatrix}1/2&-3i/10\\3i/10&1/2\end{pmatrix},\quad P_+=\frac12\begin{pmatrix}1&-i\\i&1\end{pmatrix},\quad CP_+^{\mathsf T}=\begin{pmatrix}2&2i\\0&0\\0&0\\0&0\end{pmatrix}.
$$

The raw norm is10 and the two positive weights are4/5 and1/5. The leading right ket is(1,i)/sqrt(2). Correct multiplication C P₊ᵀ keeps row(2,2i) and removes row(1,−i), giving discarded weight1/5. Multiplying C P₊ instead retains the smaller component, with discarded weight4/5. Matching eigenvalues alone would not catch this reversal.

This example ties together ket convention, physical density entries, projectors and actual amplitudes. It is stronger evidence than checking only the trace, purity or entropy of the right matrix. A phase-sensitive visual arrow must follow the correct ket, not merely display the correct weight beside a conjugated direction.

## 14. Tiny weights, observable error and multiple cuts

An almost rank-deficient matrix may still have exact full rank. Let M=10⁶ in the following integer matrix, whose determinant remains−1. Its smallest normalized weight is positive even though it is about6.25×10⁻²⁶.

$$
C_M=\begin{pmatrix}M&M-1\\M-1&M-2\end{pmatrix},\quad\det C_M=-1,\quad N=4M^2-8M+6,\quad\lambda_{\min}=\frac{1-\sqrt{1-4/N^2}}2>0.
$$

Subtracting nearly equal floating-point numbers in that last formula can erase the weight. The finite model uses exact rank and certified root intervals rather than treating the displayed Number value as the definition. Small weights may justify a controlled approximation, but they do not make the exact state a product state.

For the normalized optimum with positive retained norm, a bounded Hermitian observable has an expectation error controlled by the pure-state trace distance. The same operator must be compared on the two states.

$$
|\langle A\rangle_\psi-\langle A\rangle_{\widehat\psi_r}|\le2\|A\|\sqrt{\epsilon_r}\qquad(A=A^\dagger\text{ bounded},\ S_r>0).
$$

The bound does not guarantee a small error for an unbounded operator without further moment control. It also does not turn independent best truncations at different cuts into a jointly optimal MPS. A sequential truncation changes the state seen by later cuts. Global error bounds, conditioning and variational refinement require a separate analysis of that actual procedure.

## Research extension: accumulate errors in an actual sequential compression

A best truncation at one cut is not yet a compression algorithm. We now specify a left-to-right tensor-train SVD (TT-SVD) on a finite normalized pure tensor. Imagine passing a folder of amplitudes down a line: each step keeps an orthonormal summary of the prefix and passes its coefficients onwards. Budgets are positive integers; a budget above the available dimension simply keeps that dimension. No intermediate residual is renormalized.

### Each step uses the current residual

At step k, combine the incoming virtual index with physical site k, take its SVD, retain the leading columns of U, and pass Sigma times V-adjoint to the next step. The singular values here belong to the current residual, not necessarily to the original state at that cut. Q0 is the scalar identity, and k runs from 1 to n−1. When fewer columns exist, the retained dimension is their minimum with the requested budget. The next unfolding uses the actual column count of Q, so a large requested budget does not create nonexistent virtual coordinates.

$$
Q_0=1,\quad R_0=\psi,\quad\|\psi\|=1,\quad
 M_k=\operatorname{reshape}(R_{k-1})\in\mathbb C^{(\operatorname{cols}Q_{k-1})d_k\times(d_{k+1}\cdots d_n)},\quad
 M_k=U_k\Sigma_kV_k^\dagger,\quad
 R_k=\Sigma_{k,\le r_k}V_{k,\le r_k}^\dagger,\quad
 \delta_k^2=\sum_{a>r_k}\sigma_a(M_k)^2.
$$

### Build the retained physical spaces

Qk embeds the retained prefix coordinates into the full first-k-site Hilbert space. Its columns stay orthonormal because each retained U has orthonormal columns. Extend its projector by the identity on later sites. Each new retained physical space lies inside the previous one, so the projectors are nested. Their realized values depend on the input; once the SVD choices are fixed these are ordinary orthogonal projectors.

$$
Q_k=(Q_{k-1}\otimes I_{d_k})U_{k,\le r_k},\quad Q_k^\dagger Q_k=I,\quad
 P_k=Q_kQ_k^\dagger\otimes I_{k+1:n},\quad
 P_kP_{k-1}=P_k,\quad X_k=P_k\psi,\quad X_0=\psi.
$$

### Why squared errors add in this procedure

The discarded piece at step j is orthogonal to the whole retained space after that step, including every later discarded piece and the final tensor. Pythagoras therefore gives an equality for the sum of the current discarded squared singular values. The overlap with the final unnormalized tensor is its squared norm. This argument uses the stated nested, isometric, unnormalized sweep; it is not a rule for arbitrary truncation routines.

$$
e_k=X_{k-1}-X_k,\quad\|e_k\|^2=\delta_k^2,\quad
 \langle e_j,e_k\rangle=0\ (j\ne k),\quad B=X_{n-1},\quad
 E=\|\psi-B\|^2=\sum_{k=1}^{n-1}\delta_k^2,\quad
 \|B\|^2=\langle\psi,B\rangle=1-E.
$$

### Normalize only after finishing

All positive budgets retain a nonzero component of a nonzero input, so E<1 in exact arithmetic. Renormalize the final tensor once. Squared overlap, trace distance and Hilbert distance have distinct formulas. The observable bound compares the same bounded Hermitian operator on both states; it does not certify an unbounded observable or a particular energy optimum.

$$
\widehat B=B/\sqrt{1-E},\quad F_2(\psi,\widehat B)=1-E,\quad
 D_{\rm tr}(\psi,\widehat B)=\sqrt E,\quad
 \|\psi-\widehat B\|^2=2(1-\sqrt{1-E}),\quad
 |\langle O\rangle_\psi-\langle O\rangle_{\widehat B}|\le2\|O\|\sqrt E
 \quad(O=O^\dagger\text{ bounded},\ E<1).
$$

### Relate current losses to the original cut tails

Let Ck be the original normalized unfolding, and epsilon-k its best rank-rk squared error. The current unfolding is a left contraction of Ck. Apply that contraction to an original best rank-rk approximation: rank cannot increase and Frobenius error cannot increase. Since the current SVD is optimal for its own unfolding, its loss is at most epsilon-k. This proves an upper bound using original spectra, not equality with their sum.

$$
\varepsilon_k=\sum_{a>r_k}\lambda_a^{(k)}(\psi),\quad
 M_k=(Q_{k-1}^\dagger\otimes I_{d_k})C_k(\psi),\quad
 \delta_k^2\le\varepsilon_k,\quad E\le\sum_{k=1}^{n-1}\varepsilon_k.
$$

### What quasi-optimal means

Any tensor satisfying every bond budget is also a feasible rank-rk approximation at each individual cut. Thus each original tail is a lower bound on the best global unnormalized error squared. Combining them yields the familiar square-root-of-the-number-of-cuts factor. This is an approximation guarantee, not a proof that the sweep found the global minimizer. It does not compare directly with a separately normalized optimization problem.

$$
e_*^2=\inf_{\operatorname{rank}C_k(Y)\le r_k\ \forall k}\|\psi-Y\|^2,\quad
 \varepsilon_k\le e_*^2,\quad
 \|\psi-B\|\le\sqrt{\sum_k\varepsilon_k}\le\sqrt{n-1}\,e_* .
$$

### A three-qubit calculation you can follow

Take nonnegative real amplitudes at four even-parity strings. The two rows at the first cut have disjoint right support. Under the stated strict inequalities, the first step keeps site0=0; the second keeps the larger of a and b. One obtains a product tensor and two separately identifiable losses. These inequalities avoid a split degeneracy in this example.

$$
\psi=\frac{a|000\rangle+b|011\rangle+c|101\rangle+d|110\rangle}{\sqrt N},\quad
 N=a^2+b^2+c^2+d^2,\quad a>b\ge0,\quad a^2+b^2>c^2+d^2,\quad(r_1,r_2)=(1,1):\qquad
 B=\frac a{\sqrt N}|000\rangle,\quad
 (\delta_1^2,\delta_2^2)=\frac1N(c^2+d^2,b^2),\quad
 E=\frac{b^2+c^2+d^2}{N}.
$$

### Original tails give a different number

At the second original cut the two columns also have disjoint support. The original tail includes both surviving and previously discarded amplitudes; it need not equal the second loss in the actual sweep. For 3,2,1,1 the true total is 2/5 while the original-tail upper bound is 7/15. A repeated singular value allows multiple valid SVD choices; any chosen orthonormal leading subspace obeys the bound, but the final state need not be unique.

$$
\varepsilon_1=\frac{c^2+d^2}{N},\quad
 \varepsilon_2=\frac{\min(a^2+d^2,b^2+c^2)}N,\qquad
 (a,b,c,d)=(3,2,1,1):\quad
 E=\frac25<\varepsilon_1+\varepsilon_2=\frac7{15},\quad
 F_2=\frac35,\quad D_{\rm tr}=\sqrt{\frac25}.
$$

```python
from fractions import Fraction as F
for a,b,c,d in [(3,2,1,1),(4,2,1,1),(3,2,2,1),(2,1,1,0)]:
    N=a*a+b*b+c*c+d*d
    first=F(c*c+d*d,N)
    second=F(b*b,N)
    original_second=F(min(a*a+d*d,b*b+c*c),N)
    print(f'{a},{b},{c},{d}',first,second,first+second,first+original_second)
```

| a,b,c,d | current delta1² | current delta2² | actual E | original-tail sum |
| --- | --- | --- | --- | --- |
| 3,2,1,1 | 2/15 | 4/15 | 2/5 | 7/15 |
| 4,2,1,1 | 1/11 | 2/11 | 3/11 | 7/22 |
| 3,2,2,1 | 5/18 | 2/9 | 1/2 | 13/18 |
| 2,1,1,0 | 1/6 | 1/6 | 1/3 | 1/2 |

![Current discarded weights versus original-cut tails, and the actual sequential squared error below its original-tail sum bound.](/assets/learning/quantum-compression-gradient-research-sequential-discarded-weight.en.svg)

The bars refer to 3,2,1,1. Curves vary c from 0 to 2 with a=3,b=2,d=1; all formulas are analytic. This is a standalone compression derivation and independent sweep, not a new multi-cut compression control in the existing one-cut reader.

#### Research check 1: Do not normalize in the middle

If the residual after step1 is divided by its norm, its next discarded weights are conditional weights. Multiply them by the squared norm entering that stage before adding to the original-scale loss. In the 3,2,1,1 example that conditional second loss is 4/13, whereas the correct original-scale loss is (13/15)(4/13)=4/15.

#### Research check 2: A bound is not the global optimum

For 3,2,2,1 the sweep has E=1/2. The normalized product [(2|0>+|1>)/sqrt(5)] cubed has squared overlap 578/1125 with the original state. Its best unnormalized multiple therefore has squared error 547/1125, strictly smaller than 1/2. This explicit competitor proves the sweep is not globally optimal in this case; it does not establish the competitor as the global optimum. The quasi-optimal bound still holds.

#### Research check 3: Degeneracy does not invalidate the certificate

At an equal-weight boundary, choose any orthonormal leading subspace and record that choice. Nested projection and Pythagoras still hold for the realized sweep. The existing one-cut calculator deliberately does not select an arbitrary state at a split degeneracy; this new derivation does not change that policy.

Background: [Schollwöck, §§4.1.3 and 4.5.1](https://arxiv.org/pdf/1008.3477), sequential SVD and compression. The nested-projector proof above states the exact finite procedure and keeps unnormalized and normalized guarantees separate.

## 15. Eight exercises with worked explanations

**1. Name the matrix.** For four qubits at cut3, what are C's dimensions, which physical density is smaller and how many extra zero eigenvalues does the larger side have when the rank is2?

**Explanation.** C is8×2, the right density is2×2 and the left is8×8. Rank2 gives no right zero and six left zeros. The common positive spectrum does not mean the full physical matrices have equal dimensions.

**2. Normalize the unequal example.** For2|00〉+|11〉, compute rank, weights, purity and the budget1 retained state before and after normalization.

**Explanation.** N=5, rank2, weights4/5 and1/5, purity17/25. The retained component of the normalized input is(2/sqrt(5))|00〉, whereas its renormalization is |00〉. Their different norms explain why the squared errors1/5 and2−4/sqrt(5) differ.

**3. Trace a phase.** Why does C=(1,i) give right Y=+1 rather than−1?

**Explanation.** Ket coefficients give ρR=Cᵀ conjugate(C)/2 with upper-right−i/2. The Pauli convention yields Y=−2Imρ₀₁=+1. C†C has the conjugated off-diagonal and the same eigenvalues, so an eigenvalue-only check is insufficient.

**4. Explain the missing state.** A Bell input at budget1 has error1/2. Why can a calculator report that error but return no chosen optimum?

**Explanation.** Either |00〉/sqrt(2) or |11〉/sqrt(2), among other choices, minimizes the unnormalized error. The equal positive weights leave the retained direction unspecified. A correct value does not supply a unique vector; the full weight-group projector does not decide a rank1 subprojector.

**5. Keep a whole group.** For the3,2,2,1 diagonal example, compare budgets2 and3.

**Explanation.** Budget2 retains weight13/18 and discards5/18 but splits the two-dimensional2/9 eigenspace, so no unique state is selected. Budget3 keeps the whole group, retains17/18 and discards1/18. The projector onto a whole eigenspace is basis-independent.

**6. Interpret zero budget.** What are the unnormalized optimum, error and normalized fidelity at budget0? What changes for a rank1 product input at budget4?

**Explanation.** At budget0 the optimum is zero and error1; normalized fidelity is undefined, not0. At budget4 the product input keeps its single positive mode, has error0 and fidelity squared1. An upper rank budget need not be fully used.

**7. Apply the right projector.** In the complex three-qubit example, compare C P₊ᵀ and C P₊ at budget1.

**Explanation.** The transpose keeps raw row(2,2i), with retained norm8/10. Omitting it keeps row(1,−i), with retained norm2/10. The corresponding squared errors are1/5 and4/5. A Hermitian projector is not necessarily a real symmetric matrix.

**8. Connect to MPS capacity.** Compare the middle-cut ranks of adjacent Bell pairs01/23 and crossing pairs02/13. Does a best rank2 result at that cut solve the full MPS compression problem?

**Explanation.** The first state factors across the middle and has rank1. The second has two crossing Bell pairs and rank4. Rank2 must discard positive weight there. A best approximation at one cut does not enforce or optimize every other bond constraint simultaneously.

## 16. Read the certificate before choosing the next algorithm

The model accepts two to four ordered qubits, a nonzero Gaussian-integer raw vector and an at-most-rank budget0…4. Every contiguous cut returns physical densities, exact rank and purity, grouped weights with multiplicities, entropy, optimal error and normalized comparison values. When no positive degeneracy is split, it also returns the retained projector and actual truncated amplitudes; zero budget never produces normalized amplitudes.

Density algebra and rank are exact. Certified spectral and logarithmic intervals bound displayed values, with separate checks of projector identities, reconstruction and actual truncation error. These calculations do not implement automatic MPS compression, DMRG, an optimization sweep or a thermodynamic-limit claim. A small demonstration is evidence for its declared preparation, not every larger system.

For background, consult [Schollwöck's SVD/Schmidt discussion](https://arxiv.org/html/1008.3477v2), §4.1.1. The coefficient convention, projection proof and worked examples above make the assumptions explicit. The next implementation step is to connect the five many-body units to controls, prediction prompts, explained quizzes and accessible diagrams, then verify the real reader. A spectrum picture should distinguish capacity, positive support, kept weight, discarded error and an unavailable degenerate state choice.
