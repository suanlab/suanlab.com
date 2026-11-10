## 1. Local randomness does not identify its origin

Imagine receiving one qubit from a two-qubit source. Your local state is I2/2, so every Pauli measurement has equally likely outcomes. The source might prepare a Bell state, a classical correlated mixture, or two independent maximally mixed qubits. No experiment on your qubit alone can distinguish these possibilities.

$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},\qquad
\rho_{\rm Bell}=|\Phi^+\rangle\langle\Phi^+|,\quad
\rho_{\rm classical}=\frac{|00\rangle\langle00|+|11\rangle\langle11|}{2},\quad
\rho_{\rm independent}=\frac{I_4}{4}.
$$

All three sources have the same two local density matrices. Bell and classical sources also give the same computational-basis joint probabilities: half 00 and half 11. Their other joint measurements differ. The conceptual picture is **valid joint state → local and joint laws → separability test → selected Bell test → optimized Bell test**. Each arrow answers a different question.

| Source | Joint purity | Mutual information in bits | Negativity | Optimized CHSH magnitude |
| --- | --- | --- | --- | --- |
| Bell pair | 1 | 2 | 1/2 | 2√2 |
| Classical correlated mixture | 1/2 | 1 | 0 | 2 |
| Independent I4/4 | 1/4 | 0 | 0 | 0 |

The last column optimizes over traceless qubit observables with eigenvalues +1 and −1. Allowing a constant identity observable changes that measurement class. In particular, the zero for I4/4 is not a claim that every conceivable assignment of outputs has CHSH magnitude zero.

## 2. Verify the whole Hermitian state before normalizing

A density matrix must be Hermitian, positive semidefinite and trace one. The finite family receives a general Hermitian numerator H and normalizes by its positive trace. Its upper triangle contains ten real entries and six independent imaginary entries; the lower triangle is the complex conjugate of the upper one.

$$
H=H^\dagger,\quad \operatorname{Tr}H>0,\quad
\rho=\frac{H}{\operatorname{Tr}H},\qquad
H\succeq0\ \Longleftrightarrow\ \det H[J,J]\ge0
\quad\text{for every nonempty }J\subseteq\{0,1,2,3\}.
$$

There are fifteen nonempty principal submatrices. Inspecting only the four leading principal minors is insufficient for semidefiniteness. For example, diag(1,0,−1,1) has nonnegative leading determinants 1,0,0,0 and positive trace, yet its third diagonal direction is negative. Dividing by the trace cannot repair that defect.

Why do all principal minors suffice? They are necessary because every principal restriction of a PSD matrix is PSD. Conversely, their sums give the coefficients of det(tI+H). If every principal minor is nonnegative, this monic polynomial is strictly positive for every positive t. A negative Hermitian eigenvalue λ would give a positive root t=−λ, a contradiction. Hermiticity is essential because it ensures real eigenvalues.

An invalid input is rejected. Clipping its negative eigenvalues would create a different state and hide the problem. Singular valid states, including pure states and exact PPT boundaries, remain valid; positivity does not require strictly positive eigenvalues.

## 3. A Pauli expansion separates local means from correlations

Use the ordered matrices X,Y,Z with Y's upper-right entry −i. Together with I they are orthogonal under the trace inner product. Expanding an arbitrary two-qubit density in their tensor products gives fifteen real state parameters after trace normalization.

$$
r_i=\operatorname{Tr}(\rho\,\sigma_i\otimes I),\quad
s_j=\operatorname{Tr}(\rho\,I\otimes\sigma_j),\quad
T_{ij}=\operatorname{Tr}(\rho\,\sigma_i\otimes\sigma_j),\qquad
\rho=\frac14\left[I\otimes I+\sum_i r_i\sigma_i\otimes I+\sum_j s_jI\otimes\sigma_j+\sum_{ij}T_{ij}\sigma_i\otimes\sigma_j\right].
$$

The Bloch vectors r and s determine the reduced states. The real three-by-three tensor T records joint Pauli correlations, including those not determined by the marginals. For local directions a,b, the expectation of (a·σ)⊗(b·σ) is aᵀTb. The connected correlation subtracts the product of local means, giving aᵀ(T−rsᵀ)b.

For a product density ρA⊗ρB, T=rsᵀ and every such connected correlation vanishes. A separable mixture need not satisfy this factorization: its shared preparation label can create classical connected correlations. Nonzero connected correlation is therefore not by itself an entanglement witness.

Reconstructing the full density from this expansion checks every component, including Y signs and subsystem order. The convention is A-major, with joint index 2a+b. A plot of only the local Bloch vectors omits nine joint parameters.

## 4. Separability concerns a joint convex decomposition

A state is separable when it can be prepared as a classical mixture of local product densities. The same preparation label k chooses both factors, which allows ordinary classical correlations.

$$
\rho\ \text{separable}\quad\Longleftrightarrow\quad
\rho=\sum_k p_k\rho_A^{(k)}\otimes\rho_B^{(k)},\qquad
p_k\ge0,\quad\sum_kp_k=1.
$$

Entanglement means that no such decomposition exists. It is not equivalent to finding a decomposition that happens to contain entangled vectors. For example, the equally weighted mixture of |Φ+⟩ and |Φ−⟩ equals the classical state of section 1: their off-diagonal coherences cancel. The existence of one entangled preparation ensemble does not exclude a separable ensemble for the same density.

The pure-state rank-one coefficient test from the previous chapter does not apply to an arbitrary matrix of mixed-state entries. A mixed density has two bra indices and two ket indices; treating it as one pure amplitude table changes the mathematical object. Local mixedness likewise does not determine whether its joint source is entangled.

## 5. Partial transpose gives a necessary separability condition

Transpose only Bob's matrix indices in a fixed product basis. The resulting operator remains Hermitian and trace one but need not be positive. Transpose is positive on one isolated density, while identity tensor transpose is not positive on all joint states.

$$
(\rho^{T_B})_{ab,cd}=\rho_{ad,cb},\qquad
\left(\sum_kp_k\rho_A^{(k)}\otimes\rho_B^{(k)}\right)^{T_B}
=\sum_kp_k\rho_A^{(k)}\otimes(\rho_B^{(k)})^T\succeq0.
$$

The final inequality proves that every separable state has positive partial transpose, abbreviated PPT. Consequently, any negative partial-transpose eigenvalue certifies entanglement. The transposition is a mathematical test, not an instruction to implement an unphysical universal transpose channel on Bob's laboratory system.

For the Bell state, partial transpose moves the coherence between 00 and 11 into the block spanned by 01 and 10. Its eigenvalues are 1/2,1/2,1/2,−1/2. The negative direction has no analogue in either local I/2 state.

Although partial transpose uses a chosen basis, changing local bases changes it by unitary conjugation, with Bob's unitary complex-conjugated. Its eigenvalues and positivity therefore do not depend on that choice. Exchanging A and B also leaves the partial-transpose spectrum unchanged, because the two partial transposes are full transposes of each other.

## 6. In two qubits PPT is also sufficient

For dimensions 2×2, and also 2×3, PPT implies separability. This is a dimension-specific theorem. The two-qubit calculator applies it to its 4×4 input; it does not infer the same implication for arbitrary larger mixed states. In higher dimensions there are PPT entangled states.

The necessity proof in section 5 is elementary. Sufficiency requires further structure, commonly expressed through positive-map characterizations of separability and the structure of positive maps in these low dimensions. Checking principal minors does not itself prove that theorem in every dimension. A computational certificate must state the theorem and dimensions on which its interpretation depends.

A PPT result here certifies the existence of a separable decomposition without constructing its product factors. A negative partial-transpose result certifies entanglement without determining whether the selected CHSH settings violate a Bell inequality. Keep those conclusions distinct when reading an output.

## 7. Quantify spectral negativity and joint information

Negativity sums the magnitudes of negative partial-transpose eigenvalues. Our convention includes the factor one-half in its trace-norm definition. Some sources use a different normalization, so check the Bell value before comparing numbers.

$$
\mathcal N(\rho)=\frac{\|\rho^{T_B}\|_1-1}{2}
=-\sum_{\nu_k<0}\nu_k,\qquad
E_{\mathcal N}=\log_2(1+2\mathcal N),\qquad
I(A:B)=S(\rho_A)+S(\rho_B)-S(\rho),\quad
S(A|B)=S(\rho)-S(\rho_B).
$$

Entropy uses natural logarithms unless explicitly converted to bits. In a Bell pair the joint entropy is zero and both local entropies are ln2. Thus the mutual information is 2ln2 and the conditional entropy S(A|B) is −ln2. The classical correlated mixture has joint entropy ln2, mutual information ln2 and conditional entropy zero. Independent I4/4 has joint entropy 2ln2 and zero mutual information.

Negative quantum conditional entropy is possible because these are differences of density-operator entropies, not a classical conditional probability distribution's Shannon entropy. Mutual information includes both classical and quantum correlations; it is not a pure entanglement measure for general mixed states.

For a two-qubit PPT state, negativity is exactly zero. An interval numerically close to zero is not a substitute for that exact positivity certificate. The calculation sums negative magnitudes directly and preserves exact zero eigenvalues rather than using a tolerance to decide spectral sign.

## 8. Derive each setting pair's observable probability law

A nonzero complex spinor z defines a rank-one projector P=zz†/‖z‖² and an observable O=2P−I. Its outcomes are +1 and −1. Four spinors define A0,A1,B0,B1; each pair (x,y) corresponds to a separate repeatable preparation experiment on independent copies of the source.

$$
P_a^x=\frac{I+a A_x}{2},\quad Q_b^y=\frac{I+b B_y}{2},\qquad
p(a,b|x,y)=\operatorname{Tr}[\rho(P_a^x\otimes Q_b^y)],\quad
E_{xy}=\sum_{a,b=\pm1}ab\,p(a,b|x,y)=\operatorname{Tr}[\rho(A_x\otimes B_y)].
$$

Summing over Bob's outcomes inserts I on his subsystem, so Alice's marginal is independent of y. Summing over Alice gives the reciprocal statement. These identities check the full four probability tables, not just their correlations.

To condition Bob's state on Alice's outcome, apply P locally, form the unnormalized joint post-measurement density and trace Alice out. Divide by the outcome probability only when it is positive. A zero-probability conditional density and its purity are null. Summing the unnormalized conditionals returns Bob's original reduced density for each Alice setting.

Conditioning can change the state assigned to a labelled subensemble. Ignoring the outcome leaves the remote law unchanged. Shared outcome information requires classical communication and does not supply superluminal signalling. The finite model calculates ideal population laws, not a finite experimental sample or an inference procedure for noisy tomography.

## 9. Derive the local hidden-variable CHSH bound

In a deterministic local assignment, Alice has outputs a0,a1 in {−1,+1} and Bob has b0,b1. Exactly one of b0+b1 and b0−b1 vanishes, while the other equals ±2. Therefore the absolute value of a0(b0+b1)+a1(b0−b1) is two. A mixture of such assignments cannot exceed two in expectation.

$$
S=E_{00}+E_{01}+E_{10}-E_{11},\qquad
|S|\le2\quad\text{for local hidden-variable assignments},\qquad
\mathcal B=A_0\otimes(B_0+B_1)+A_1\otimes(B_0-B_1).
$$

The stochastic local model is covered by including its local randomness in the hidden variable. The comparison assumes that the shared hidden-variable distribution is independent of the setting choices and that each local response depends only on the local setting and hidden variable. Experimental tests additionally need justified sampling and detection assumptions.

Quantum theory calculates S=Trρℬ using four distinct setting-pair experiments. It does not require simultaneous outcomes for noncommuting A0 and A1 on one copy. Bell violation rules out the stated local model for the observed laws; it is not a failure of probability theory.

The calculator classifies its selected settings by exact rational |S|>2. Failing to violate this one expression at one set of directions cannot establish that no other directions, outcome relabellings or Bell scenarios can show nonlocality.

## 10. Optimize CHSH over all projective directions

Write Ax=ax·σ and By=by·σ with real unit vectors. The vectors b0+b1 and b0−b1 are orthogonal, so write them as 2cosθ v and 2sinθ w with v,w orthonormal. For fixed v,w, align Alice's axes with Tv and Tw, choosing signs as needed. Optimizing θ then gives a square root of a sum of squared lengths.

$$
G=T^T T,\quad 0\le\mu_1\le\mu_2\le\mu_3,\qquad
\max_{A_0,A_1,B_0,B_1}|S|=2\sqrt{\mu_2+\mu_3}=2\sqrt M,
\qquad M>1\ \Longleftrightarrow\ \text{some CHSH directions violate }2.
$$

To see why the two largest eigenvalues appear, maximize vᵀGv+wᵀGw over orthonormal v,w. In an eigenbasis of G, the two vectors assign weights between zero and one to each eigenvalue, with total weight two. The maximum puts weight one on each of the two largest eigenvalues. This also covers degeneracy without selecting unique axes.

For a unit direction b, the length ‖Tb‖ equals the maximum correlation with an Alice unit direction. The corresponding product Pauli observable has eigenvalues ±1, so that correlation cannot exceed one in magnitude. Hence μ2,μ3≤1 and the optimized magnitude is at most 2√2, the Tsirelson bound.

This optimization is over all continuous traceless projective directions, whereas the chosen inputs are bounded integer spinor representatives. The reported maximum need not be attained by the selected directions or by an exactly representable bounded spinor. The engine provides the maximum and an exact violation classification, but no maximizing eigenvectors or axes.

## 11. Work Bell and Werner thresholds explicitly

For |Φ+⟩, T=diag(1,−1,1). Choose A0=Z, A1=X and Bob spinors (5,2) and (5,−2). Their Bloch directions are (20/29,0,21/29) and (−20/29,0,21/29), giving S=82/29, approximately 2.827586. This exceeds two but is below 2√2. Poor directions can miss the violation for the same density.

The Werner family mixes a singlet with independent noise. Its correlation tensor is −pI3, and its partially transposed spectrum can be calculated directly.

$$
\rho_W(p)=p|\Psi^-\rangle\langle\Psi^-|+(1-p)\frac{I_4}{4},\quad 0\le p\le1,\qquad
\operatorname{spec}(\rho_W^{T_B})=\left\{\frac{1+p}{4}\text{ (three times)},\frac{1-3p}{4}\right\},\quad
\mathcal N=\max\left(0,\frac{3p-1}{4}\right),\quad M=2p^2.
$$

Thus entanglement starts above p=1/3, while optimized CHSH violation starts only above p=1/√2. At p=1/2, negativity is 1/8 but every projective CHSH choice stays at or below √2. Entanglement and CHSH violation are inequivalent tests.

| Werner parameter | PPT and entanglement | Optimized CHSH conclusion |
| --- | --- | --- |
| 1/3 | PPT, separable, one zero partial-transpose eigenvalue | No violation |
| 1/2 | NPT, entangled, negativity 1/8 | No violation |
| 353553/500000 versus 353554/500000 | Both entangled | The first is below, the second above the violation threshold |

These rational neighbours avoid pretending that 1/√2 is an exact rational input. Nonviolation of every projective CHSH inequality still does not prove a local hidden-variable description for every possible measurement scenario, filtering operation or multi-copy protocol.

## 12. Certify an exact CHSH boundary without rounding

An eigenvalue interval can straddle a threshold even when the mathematical answer is exact equality. The two-qubit model avoids a midpoint decision. Put q=TrG−1. Since M=TrG−μ1, compare the smallest eigenvalue of G with q by testing the shifted matrix.

$$
q=\operatorname{Tr}G-1,\quad K=G-qI,\qquad
M>1\ \Longleftrightarrow\ K\not\succeq0,\qquad
M=1\ \Longleftrightarrow\ K\succeq0\ \text{and}\ \det K=0,\qquad
M<1\ \Longleftrightarrow\ K\succ0.
$$

All entries of K are rational for the accepted inputs. Its seven principal minors and determinant give exact certificates, including a repeated minimum eigenvalue. The reported interval for M still comes from adding the largest two isolated roots; the exact classification does not depend on rounding that interval.

For a full-rank example, take zero local Bloch vectors and T=diag(3/5,−4/5,1/2). Its Bell-basis weights are 29/40,1/40,3/40,7/40. Its negativity is 9/40, so it is entangled. Meanwhile G=diag(9/25,16/25,1/4), q=1/4 and K=diag(11/100,39/100,0). Therefore M=1 and the optimized CHSH magnitude equals two exactly.

Changing the first correlation by ±1/25000 keeps the density positive and moves M to opposite sides of one. This is a physical perturbation of a full-rank state. Perturbing a rank-deficient boundary without checking positivity can instead produce an invalid “example.” State validity must precede interpretation of the Bell test.

## 13. Exercises with explanations

**1. Explain identical local states.** Why can no Alice-only measurement distinguish the three sources in section 1? Which single joint measurement still fails to distinguish two of them?

**Explanation.** Each Alice marginal is I2/2, which fixes the law of every local measurement. The computational-basis joint measurement also gives the same half-00/half-11 table for Bell and classical sources. Joint X measurements distinguish them: their XX correlations are respectively 1 and 0.

**2. Reject an invalid state.** Normalize diag(1,0,−1,1) by its positive trace. Do its leading principal minors guarantee a density matrix?

**Explanation.** They do not. Its trace is one already and its leading determinants are 1,0,0,0, but the singleton principal minor at the third coordinate is −1. The probability for that basis projector would be negative. All principal minors, rather than just leading ones, are required.

**3. Distinguish a decomposition from a criterion.** Mix |Φ+⟩ and |Φ−⟩ with equal weights. Does using entangled ingredients prove that the result is entangled?

**Explanation.** The two coherences cancel, leaving (|00⟩⟨00|+|11⟩⟨11|)/2. This is an explicit product-state mixture and is separable. Its mutual information is one bit and negativity zero; nonzero classical correlation is compatible with separability.

**4. Derive the selected Bell value.** With the spinors and Alice axes of section 11, compute the four E values and S.

**Explanation.** They are E00=21/29,E01=21/29,E10=20/29,E11=−20/29. Substitution gives S=82/29. The density's optimized maximum is 2√2. Changing to poor settings changes the selected result without changing its entanglement or optimized maximum.

**5. Find entanglement without CHSH violation.** At Werner p=1/2, calculate the smallest partial-transpose eigenvalue, negativity and largest possible CHSH magnitude.

**Explanation.** The smallest eigenvalue is −1/8, the other three are 3/8, and negativity is 1/8. Since M=2p²=1/2, the optimized magnitude is √2. It is entangled but CHSH-nonviolating for every projective direction. That statement alone is not a general local-model certificate.

**6. Recognize a nonzero negativity on the exact boundary.** Use section 12's Bell-diagonal state to explain why a maximum of exactly two does not imply separability.

**Explanation.** Its largest Bell weight is29/40; the partial-transpose eigenvalues are −9/40,19/40,17/40,13/40. Negativity9/40 certifies entanglement. The shifted Gram matrix is PSD with a zero determinant, so M=1 and the CHSH maximum is exactly two. The two certificates answer different questions.

## 14. Read the model as one step toward research

The family accepts 32 owned integer fields of magnitude at most 10⁶: sixteen Hermitian numerator parameters and sixteen components for four nonzero local spinors. Arbitrary complex coherences are allowed subject to exact PSD validation. Scaling a numerator or a local spinor by an allowed nonzero representative factor leaves its normalized physical laws unchanged.

State spectra, partial transpose and G use exact polynomial isolation at 1024 bits. Entropies and nonlinear magnitudes use 768-bit working intervals with outward 512-bit observations. Exact zeros are not rounded versions of small eigenvalues. Null machine-number projections or zero-event conditional states need their stated interpretation; neither is permission to insert a guessed physical value.

Useful next research questions include finite-sample confidence regions for witness measurements, robust tomography under positivity constraints, alternative Bell inequalities, multi-copy activation, LOCC conversion and tensor-network simulation of open systems. In AI, optimizing an entanglement witness or a Bell setting is a constrained modeling problem; a training loss is not automatically a valid physical certificate. Positive-map theory, PPT entanglement and multipartite resources exceed this two-qubit model and remain part of the wider curriculum.

Open this chapter through **Book → Interactive Book**. Fixed concept comparisons use independent preparations; the free lab saves valid settings in this browser. Read the exact certificates alongside the diagrams. The local-mixture and partial-trace conventions follow [Preskill's Chapter 2](https://www.preskill.caltech.edu/ph219/chap2_15.pdf). The dimension restriction on PPT sufficiency is stated in the [Horodecki separability paper](https://arxiv.org/abs/quant-ph/9605038). The projective CHSH optimization and its measurement assumptions are checked against [the authors' review, sections IV.A–B](https://arxiv.org/pdf/quant-ph/0702225). The shifted rational certificate and numerical examples above are derived in this chapter's conventions.
