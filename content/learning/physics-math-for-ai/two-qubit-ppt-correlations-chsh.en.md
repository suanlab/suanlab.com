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

## Research extension: measurable witnesses and finite-shot certificates

### Turn an NPT direction into a testable linear functional

A negative partial-transpose eigenvalue identifies a direction in matrix space. It also defines a witness whose expectation is a linear function of the original density:

$$
\rho^{T_B}v=\nu v,\quad\|v\|=1,\quad\nu<0,\qquad
 W_v=(|v\rangle\langle v|)^{T_B},\quad
 \operatorname{Tr}(\rho W_v)=\langle v|\rho^{T_B}|v\rangle=\nu,\quad
 \operatorname{Tr}(\sigma_{\rm sep}W_v)\ge0.
$$

Partial transpose is self-adjoint under the trace pairing, which proves the equality. Every separable state is PPT, so the same quadratic form is nonnegative there. This direction-dependent witness is useful after a proposed state is modeled, but estimating v from noisy data does not by itself validate a subsequent significance claim on those same data. The separating-functional viewpoint appears in [Horodecki and collaborators' original paper](https://arxiv.org/pdf/quant-ph/9605038).

A particularly simple fixed witness targets the singlet:

$$
W=\tfrac12I_4-|\Psi^-\rangle\langle\Psi^-|
 =\tfrac14(I_4+X\otimes X+Y\otimes Y+Z\otimes Z),\qquad
 \operatorname{Tr}(\rho_W(p)W)=\frac{1-3p}{4}.
$$

Its Pauli decomposition follows by expanding the singlet projector. It detects the Werner family exactly above p=1/3, including entangled states that cannot violate projective CHSH. Its nonnegative result is not a general separability certificate: another entangled state can be poorly aligned with this witness.

### Randomize the trusted measurement and keep every trial

On each independent copy choose one common Pauli axis J uniformly, independently of the source and previous outcomes. Alice and Bob measure that axis locally and report binary outcomes. Assign one score per input copy:

$$
J\sim\operatorname{Uniform}\{X,Y,Z\},\quad a,b\in\{-1,+1\},\quad
 V=\frac{1+3ab}{4}\in\{-\tfrac12,1\},\qquad
 \mathbb EV=\tfrac14\left(1+\sum_{j=X,Y,Z}\langle j\otimes j\rangle\right)=\operatorname{Tr}(\rho W).
$$

Averaging over J produces the witness expectation. There is no simultaneous X,Y,Z outcome on one pair. This protocol assumes trusted calibrated Pauli measurements, iid copies of a fixed density, a witness and trial count fixed before validation, and all trials included in the stated binary-outcome model. Detector loss or an unreported outcome is not permission to drop a trial; it requires a new model and bound.

The score range has width3/2. A one-sided [Hoeffding bound](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf) gives:

$$
\overline V=\frac1n\sum_{i=1}^n V_i,\quad
 r_n(\delta)=\frac32\sqrt{\frac{\ln(1/\delta)}{2n}},\quad
 \Pr\{\operatorname{Tr}(\rho W)>\overline V+r_n(\delta)\}\le\delta,\qquad
 \overline V+r_n(\delta)<0\ \Longrightarrow\ \text{entanglement certificate at level }\delta.
$$

The probability statement concerns repeated data sets for a fixed source, not a posterior probability assigned to one observed density. Because every separable source has nonnegative mean, falsely certifying such a source occurs with probability at most δ under the assumptions. Equality at zero is inconclusive. This fixed-n rule does not justify looking repeatedly and stopping at the first negative bound.

If the reported correlation for each axis differs from its ideal value by at most a justified e_j, add a worst-case systematic allowance:

$$
|E_j^{\rm reported}-E_j^{\rm ideal}|\le e_j\quad(j=X,Y,Z),\qquad
 U=\overline V+r_n(\delta)+\tfrac14(e_X+e_Y+e_Z),\quad
 U<0\ \Longrightarrow\ \text{entanglement certificate under these calibration bounds}.
$$

The plus sign protects against an observed correlation being spuriously too negative. These are assumed deterministic calibration bounds. If they were themselves estimated with error probability, that uncertainty needs its own probability budget. With arbitrary unbounded calibration error, this trusted-witness claim is unavailable; it is not a device-independent Bell test.

### Confidence and detection power are different design questions

A small type-I error level does not guarantee that an entangled source will be detected. In the ideal calibration model, let its true witness margin be γ>0. The opposite Hoeffding tail bounds failure to detect:

$$
\gamma=-\operatorname{Tr}(\rho W)>0,\qquad
 \gamma>r_n(\delta)+r_n(\beta)\ \Longrightarrow\
 \Pr\{\overline V+r_n(\delta)\ge0\}\le\beta,\qquad
 n>\frac9{8\gamma^2}\left(\sqrt{\ln(1/\delta)}+\sqrt{\ln(1/\beta)}\right)^2.
$$

The last sufficient sample-size condition simply solves the preceding strict radius inequality. It can be conservative. Requiring only r_n(δ)<γ compares the radius with the population mean; it is not a stated bound on power. For the Werner state p=1/2, γ=1/8, so finite counts and calibration errors matter even though the exact matrix is clearly NPT.

For comparison, independently choose uniform binary settings x,y for a CHSH game and count wins. For an iid local hidden-variable model with setting independence and binary outcomes on all trials:

$$
C_i=\boldsymbol1\{a_ib_i=(-1)^{x_i y_i}\},\quad
 w=\tfrac12+\tfrac18S,\quad w_{\rm local}\le\tfrac34,\quad
 \Pr_{\rm local}\left\{\overline C>\tfrac34+\sqrt{\frac{\ln(1/\delta)}{2n}}\right\}\le\delta,\qquad
 w_{W,\rm opt}=\tfrac12+\tfrac{\sqrt2}{4}p.
$$

The win formula expands the four correlations with the same signs as this chapter's S. The stated tail bound is for the fixed-n iid experiment; no optional-stopping or general memory-robust protocol is being claimed. For Werner p=1/2 the optimized mean is below3/4, even while the trusted Pauli witness is negative. A failure of this Bell test is therefore compatible with successful witness certification under its different measurement assumptions.

Use δ=exp(−8) to make the witness radius rational at square trial counts. K counts opposite local outcomes in the randomized-axis experiment:

$$
\delta=e^{-8},\quad r_n=\frac3{\sqrt n},\quad
 K=\#\{i:a_ib_i=-1\},\quad\overline V=1-\frac{3K}{2n},\qquad
 K\sim\operatorname{Binomial}\!\left(n,\frac{1+p}{2}\right)\quad\text{for an iid Werner source}.
$$


```python
from fractions import Fraction as Q
from math import isqrt
for n in [100, 400, 1600, 10000]:
    root = isqrt(n)
    assert root*root == n
    k = 3*n//4  # illustrative counts, not measured data
    estimate = 1-Q(3*k, 2*n)
    radius = Q(3, root)  # delta = exp(-8)
    print(n, k, estimate, radius, estimate+radius)
```

| trials n | opposite outcomes K | witness estimate | confidence radius | upper bound |
|---|---|---|---|---|
| 100 | 75 | -1/8 | 3/10 | 7/40 |
| 400 | 300 | -1/8 | 3/20 | 1/40 |
| 1600 | 1200 | -1/8 | 3/40 | -1/20 |
| 10000 | 7500 | -1/8 | 3/100 | -19/200 |

![A fixed singlet witness detects a wider Werner region than projective CHSH; finite sample confidence radii and calibration allowances determine whether an observed negative estimate certifies entanglement.](/assets/learning/quantum-entanglement-research-witness-finite-evidence.en.svg)

These rows are illustrative count records with the same point estimate, not measured data or guaranteed experimental outcomes. The upper bound excludes calibration error until an explicit allowance is added. The present reader computes population laws and exact matrix certificates; no shot sampler, confidence calculator or detector-loss model has been added to its controls.

#### Research check 1: negative estimates can be inconclusive

The 400-trial row estimates −1/8, yet its upper bound is +1/40 and it does not certify entanglement at δ=exp(−8). At1600 trials the ideal-calibration bound is −1/20. If each axis error is bounded by1/10, the systematic allowance is3/40 and that latter bound becomes +1/40, again inconclusive.

#### Research check 2: a witness can miss a Bell state

For Φ+, the singlet overlap is zero, so this W has expectation +1/2. The state remains maximally entangled. A nonnegative value means this fixed witness failed, not that the PPT criterion would classify the density as separable. The exact density and finite measurement record answer related but different questions.

#### Research check 3: train a witness, then validate it

Choosing the most negative-looking witness after inspecting many noisy candidates changes the statistical selection rule. Freeze the candidate using training data and validate on independent iid holdout copies, or derive an appropriate simultaneous bound. Reusing the same one-witness δ bound after selection does not account for that search. Likewise, retaining only successful filter outcomes requires a clearly heralded validation source and a complete sampling model; it cannot silently repair a Bell detection loophole.

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
