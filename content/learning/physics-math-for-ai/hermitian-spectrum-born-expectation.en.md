## 1. What does an observable tell us?

An observable specifies the possible numerical results of an ideal measurement and the subspaces associated with those results. The quantum state then determines their probabilities. A matrix, a measurement outcome and an average over repeated preparations are three different objects. In particular, the average need not be a possible result of a single measurement.

We use a nonzero raw complex vector z and its normalized representative ψ=z/||z|| in C². Each trial starts with the same preparation. There is no simulated physical clock, apparatus trajectory or claim that the two complex coordinates are two spatial positions. The previous chapter on rays explains why a common nonzero complex multiplier does not change the preparation.

## 2. Adjoint means conjugate transpose

Our inner product is conjugate linear in its first argument. The adjoint is the operator that moves across that inner product with the following identity. In finite dimension it is the conjugate transpose, not just the transpose.

$$
\langle u,Av\rangle=\langle A^\dagger u,v\rangle,\qquad A^\dagger=\overline A^{\mathsf T},\qquad A=A^\dagger.
$$

The last condition defines a Hermitian matrix. For example Y has entries Y₀₁=−i and Y₁₀=i, with zero diagonal. Transposition alone gives −Y, while conjugate transposition gives Y. This distinction matters whenever amplitudes or matrix entries are complex.

If Av=λv with v nonzero, then v†Av=λ||v||². Hermiticity also makes that scalar equal to its own conjugate. Since the norm squared is positive, λ is real. This argument concerns finite matrices; an unbounded differential operator requires a separate domain analysis.

## 3. Eigenspaces are the meaningful outcomes

A finite Hermitian matrix has mutually orthogonal eigenspaces. Each distinct eigenvalue labels a projector onto its entire eigenspace. The projector does not depend on which orthonormal basis we choose inside that space.

$$
A=\sum_k a_k P_k,\qquad P_k^\dagger=P_k,\qquad P_kP_j=\delta_{kj}P_k,\qquad \sum_k P_k=I.
$$

The sum is over distinct numerical eigenvalues. If A=3I, there is one outcome, 3, and its spectral projector is I. Listing two arbitrary basis vectors with value 3 would describe a more detailed measurement than the observable alone specifies. Keeping this distinction now prevents an important error in the next chapter on degeneracy.

## 4. An exactly controlled two-dimensional family

Choose another nonzero raw vector v. Its rank-one projector P needs no square root. With rational center a and signed gap b we construct the following family.

$$
P=\frac{vv^\dagger}{v^\dagger v},\qquad Q=I-P,\qquad A=aI+b(2P-I)=(a+b)P+(a-b)Q.
$$

For b nonzero, the two eigenvalues are a+b and a−b. Their projectors remain attached when we sort the displayed values. A negative b reverses their order. When b=0, we merge the projectors to I before calculating a spectral measurement. A tiny nonzero rational gap remains distinct: there is no floating-point tolerance that silently turns it into zero.

The raw vector may be scaled or multiplied by a nonzero phase without changing P. Changing its ray can change the measurement even if the two eigenvalues stay fixed. The family covers the geometry of two-dimensional Hermitian measurements with the selected rational parameters; it does not claim to enumerate arbitrary real numerical inputs.

## 5. Born probabilities are squared projection lengths

The probability of an outcome is the fraction of the initial squared norm retained by its projector. Normalization follows from orthogonality and completeness, not from adjusting the bars after calculation.

$$
p_k=\langle\psi,P_k\psi\rangle=\|P_k\psi\|^2=\frac{\|P_kz\|^2}{z^\dagger z},\qquad \sum_k p_k=1.
$$

Take ψ=(1,i)/√2 and A=Z=diag(1,−1). Both components have modulus 1/√2. Results −1 and +1 therefore each have probability 1/2. The imaginary relative phase is real physical preparation information, but this particular measurement does not detect it. Another observable may distinguish states that share this Z distribution.

A useful diagram will show three linked quantities for each outcome: its eigenvalue, its probability and its contribution to the mean. A coordinate amplitude is not itself a probability, and a signed eigenvalue is not a negative probability.

## 6. The mean and variance answer different questions

The mean describes an ensemble of identically prepared trials. The variance measures the spread of their numerical outcomes around that mean.

$$
\langle A\rangle=\sum_k a_kp_k=\psi^\dagger A\psi,\qquad \operatorname{Var}(A)=\sum_k a_k^2p_k-\langle A\rangle^2=\|(A-\langle A\rangle I)\psi\|^2.
$$

For the preceding Z example the mean is 0 and the variance is 1. Zero is not one of the measurement outcomes. Compare A=0 on the same state: the mean is again 0, but the sole outcome is 0 and the variance is 0. Equal means do not specify equal experiments or equal distributions.

Zero variance says the state belongs to an eigenspace of this observable. It does not determine the state uniquely when that eigenspace has dimension greater than one. If the observable is measured in units U, its mean has units U and its variance has units U²; probabilities are dimensionless.

## 7. Work through a rational shift and gap

Keep equal P and Q probabilities but choose a=−2/3 and b=5/11. The sorted outcomes are −37/33 and −7/33. Their equal-weight average is −22/33=−2/3. Their distances from the mean are ±5/11, so the variance is 25/121.

| Preparation | Outcome law | Mean | Variance |
| --- | --- | --- | --- |
| ψ=(1,i)/√2, A=Z | −1 and +1, each 1/2 | 0 | 1 |
| Same ψ, A=0 | 0 with probability 1 | 0 | 0 |
| Same ψ, shifted/scaled Z | −37/33 and −7/33, each 1/2 | −2/3 | 25/121 |

This table separates three choices: the input state, the measurement eigenspaces and the numerical eigenvalue labels. Changing only a shifts every result and the mean, while leaving probabilities and variance unchanged. Changing b scales deviations and their squared spread.

## 8. Changing coordinates does not change the experiment

For a unitary basis matrix U, transform both the state coordinates and the observable matrix. Transforming just one generally changes the computed answer.

$$
\psi'=U^\dagger\psi,\qquad A'=U^\dagger AU,\qquad (\psi')^\dagger A'\psi'=\psi^\dagger A\psi.
$$

Using U with columns (1,1)/√2 and (−1,1)/√2 gives A′=−X for A=Z. The state above has new coordinates ((1+i)/2,(−1+i)/2). Its mean remains 0. This is a passive coordinate change. Physically selecting a different observable while holding the state fixed is a different operation, even if a similar matrix expression appears.

## 9. A matrix with a complex expectation is a warning

Consider the nilpotent matrix N, whose only nonzero entry is N₀₁=1. Its adjoint has the 1 below the diagonal. On the same state ψ=(1,i)/√2 its expectation is imaginary.

$$
N=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad \psi^\dagger N\psi=\frac{i}{2},\qquad N^\dagger\ne N.
$$

This is a counterexample to treating every square matrix as a real-valued observable. It does not mean every non-Hermitian matrix has a nonreal expectation in every state. A real value in one preparation does not certify Hermiticity. Likewise, the product of two Hermitian matrices is Hermitian only when they commute; reversing factors under the adjoint explains why.

## 10. What is calculated exactly?

State, axis and passive-basis raw coordinates are bounded integers. The center and gap are ratios with positive denominators. The engine forms matrices, projection masses and moments using exact complex rational arithmetic. It reports algebraic residuals for projector identities, total probability, matrix reconstruction and agreement between spectral and direct moments.

Normalized coordinates may involve square roots. They are returned with directed 512-bit intervals; the rounded drawing coordinates are not exact normalization certificates. A probability-zero branch has no conditional normalized vector; the result uses null for its conditional amplitudes. No pseudocount, epsilon denominator or normalized random noise is inserted to make such a vector appear.

The independent reference uses symbolic eigenvectors and orthonormalized eigenspaces rather than trusting the production projector decomposition. A successful comparison supports this finite family. It does not prove an infinite-dimensional spectral theorem.

## Research extension: stable values and a discontinuous ideal measurement

### What does a calibration error actually bound?

An observable reconstructed from a model or calibration is rarely exact. Keep the same normalized preparation ψ and compare two finite Hermitian matrices. The operator norm is the largest change in vector length over unit inputs. If its error is bounded by η, Cauchy–Schwarz immediately controls the mean:

$$
\widetilde A=A+E,\quad A=A^\dagger,\quad E=E^\dagger,\quad \|E\|_{\rm op}\le\eta,
 \qquad |\langle\widetilde A\rangle_\psi-\langle A\rangle_\psi|=|\langle\psi|E|\psi\rangle|\le\eta.
$$

This compares two population means on the same state, not the sampling uncertainty from a finite number of experimental trials. The standard deviation also has a useful bound. The projection Q_ψ removes the component parallel to ψ, leaving (A−⟨A⟩)ψ. The reverse triangle inequality gives:

$$
Q_\psi=I-|\psi\rangle\langle\psi|,\quad \sigma_\psi(A)=\|Q_\psi A\psi\|,
 \qquad |\sigma_\psi(\widetilde A)-\sigma_\psi(A)|\le\|Q_\psi E\psi\|\le\eta.
$$

Use standard deviation here: it has the same units as A and η. Variance has squared units, so asserting the same η bound for variance would be dimensionally wrong. For example, the bound on its difference can instead use η times the sum of the two standard deviations.

Ordered eigenvalues are stable as well. For ascending eigenvalues the finite-dimensional minimum–maximum characterization says:

$$
\lambda_k(A)=\min_{\dim S=k}\ \max_{v\in S,\,\|v\|=1}\langle v|A|v\rangle,
 \qquad |\lambda_k(\widetilde A)-\lambda_k(A)|\le\eta,\quad \lambda_1\le\cdots\le\lambda_d.
$$

To see the last bound, every unit-vector Rayleigh quotient changes by at most η; taking a maximum on each subspace and then a minimum preserves that inequality. Interchange the two matrices to obtain the other direction. This argument bounds numerical values, not individual eigenspace orientations near a closing gap.

### Why merging eigenvalues needs a separate instrument decision

Consider a particularly transparent calibration family in the chapter's C² model:

$$
A_\epsilon=\epsilon Z,\quad \|A_\epsilon-0\|_{\rm op}=|\epsilon|,\quad
 \lambda_\pm=\pm|\epsilon|,\quad P_\pm=\tfrac12(I\pm\operatorname{sgn}(\epsilon)Z)\quad(\epsilon\ne0).
$$

For every nonzero ε there are two distinct outcomes. Negative ε reverses their ordered labels and projectors together. At ε=0 the observable is zero and its sole spectral projector is I: the displayed formula involving sgn ε is no longer the spectral decomposition into distinct outcomes.

An experiment that keeps resolving |0⟩ and |1⟩ at ε=0 is a refinement with an extra record. It is different from the coarse spectral Lüders measurement of the zero observable. A real detector's finite resolution must likewise be specified as part of an instrument model; a numerical tolerance alone is not such a model.

Numerical values can converge while their ideal state update does not. For any real scalar Lipschitz readout f with constant K, the spectral distribution gives:

$$
|\mathbb E_\psi f(A_\epsilon)-f(0)|
 \le\sum_{s\in\{-,+\}}p_s|f(\lambda_s)-f(0)|\le K|\epsilon|,
 \qquad |f(x)-f(y)|\le K|x-y|.
$$

This bound concerns this family approaching the scalar zero observable. It does not claim that every scalar Lipschitz function is operator Lipschitz for arbitrary noncommuting matrix pairs. The example already shows why comparing readout values and comparing subsequent states are different tasks.

### A vanishing spread can hide a fixed downstream disturbance

Prepare the plus state for every ε:

$$
\psi=\tfrac1{\sqrt2}(1,1),\quad \rho=|\psi\rangle\langle\psi|,\quad
 \langle A_\epsilon\rangle=0,\quad \sigma_\psi(A_\epsilon)=|\epsilon|,\quad
 \operatorname{Var}_\psi(A_\epsilon)=\epsilon^2.
$$

Write P_0 and P_1 for the fixed computational projectors, as opposed to projectors sorted by eigenvalue. The ideal nonselective spectral update has two branches for any nonzero ε, but one branch at zero:

$$
\mathcal D_\epsilon(\rho)=\begin{cases}P_0\rho P_0+P_1\rho P_1=I/2,&\epsilon\ne0,\\ \rho,&\epsilon=0,\end{cases}
 \qquad P_0=|0\rangle\langle0|,\quad P_1=|1\rangle\langle1|.
$$

A later yes/no test B asks whether the original plus state survived. Direct multiplication gives:

$$
B=|\psi\rangle\langle\psi|,\quad
 \operatorname{Tr}[B\mathcal D_\epsilon(\rho)]=\begin{cases}1/2,&\epsilon\ne0,\\1,&\epsilon=0,\end{cases}
 \qquad \operatorname{Tr}[\mathcal D_\epsilon(\rho)^2]=\begin{cases}1/2,&\epsilon\ne0,\\1,&\epsilon=0.\end{cases}
$$

Thus both the mean and standard deviation approach the zero-observable values, yet the later probability stays at one half until the exact degeneracy. This is a discontinuity of the stipulated ideal spectral-instrument family, not a claim that a physical apparatus must respond discontinuously. The [projection rule in Preskill §2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf) specifies the update being compared; the limiting example and calibration bounds above are derived directly here.

The following standalone calculation reproduces the table exactly. The existing reader can prepare the εZ family by changing the signed gap; this supplement adds no finite-resolution detector simulation.

```python
from fractions import Fraction as Q
for epsilon in [Q(1), Q(1, 4), Q(1, 16), Q(0)]:
    sigma, variance = abs(epsilon), epsilon*epsilon
    purity = Q(1, 2) if epsilon else Q(1)
    later_probability = Q(1, 2) if epsilon else Q(1)
    assert sigma*sigma == variance
    print(epsilon, sigma, variance, purity, later_probability)
```

| epsilon | sigma | variance | purity | p_B |
|---|---|---|---|---|
| 1 | 1 | 1 | 1/2 | 1/2 |
| 1/4 | 1/4 | 1/16 | 1/2 | 1/2 |
| 1/16 | 1/16 | 1/256 | 1/2 | 1/2 |
| 0 | 0 | 0 | 1 | 1 |

![As a spectral gap closes, the standard deviation decreases continuously, while the later plus-state probability remains one half away from zero and equals one at zero.](/assets/learning/quantum-observable-research-closing-spectral-gap.en.svg)

The open point at ε=0 belongs to the two-branch limiting value. The filled point belongs to the one-branch zero-observable instrument. Do not join them with an interpolated apparatus trajectory.

#### Research check 1: which uncertainty has been bounded?

If η=0.02 in the observable's units, the population mean and standard deviation each change by at most 0.02. This says nothing about finite-shot confidence intervals unless a sampling model and trial count are supplied. A variance bound needs squared units.

#### Research check 2: approach from negative ε

For ε=−1/4 the lower eigenvalue is attached to |0⟩, whereas for ε=1/4 it is attached to |1⟩. On the plus state both probabilities are one half, and ignoring the label gives the same dephased output. Sorting labels cannot restore coherence.

#### Research check 3: keep the internal record at zero

If the detector still records the computational basis branch when ε=0, the later B probability remains one half. That instrument is consistent with reporting the number zero every time, but is not the coarse Lüders instrument with projector I. A value histogram alone does not identify the implementation.

## 11. Exercises with explanations

**1. Why is the bra of (1,i) equal to (1,−i)?**

**Explanation.** The bra is the conjugate transpose. It gives squared norm 1+(−i)i=2. A plain transpose would incorrectly give 1+i²=0.

**2. Which projector gets the lower eigenvalue when a=0 and b=−1?**

**Explanation.** P has value a+b=−1 and Q has value a−b=+1. Sort the pairs together. If z is in P, the lower result has probability 1.

**3. What happens to the distribution when a=7/3 and b=0?**

**Explanation.** There is one value, 7/3, with projector I and probability 1. Every normalized input has variance 0; this does not identify its ray.

**4. Does mean 0 imply that a single reading is 0?**

**Explanation.** No. For the equal-weight Z preparation each result is −1 or +1. The average is 0. The zero operator provides a different experiment with the same mean.

**5. What changes when ψ is replaced by 3iψ?**

**Explanation.** The raw norm squared is multiplied by 9. Every projected squared norm gets the same factor, so probabilities, mean and variance are unchanged after normalization.

**6. Can a real expectation on one state certify an observable?**

**Explanation.** No. N has expectation 0 on the first standard basis vector, yet N† differs from N. The property must hold as an operator statement across states.

## 12. Connections and limits

For AI work, this chapter provides a concrete setting for complex inner products, spectral decompositions, quadratic forms, distributional summaries and coordinate invariance. A real scalar loss at one parameter value is not a substitute for proving the structure of the operator that produced it. An average prediction likewise need not identify the underlying distribution.

The next chapter studies conditional projection and degeneracy. General mixed-state inputs, POVMs, uncertainty relations and operator domains remain separate topics. Our finite matrix derivations do not establish boundary conditions or self-adjointness for differential operators.

Background definitions were checked against [Preskill §2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf) and [TU Delft §8.7–8.8](https://interactivetextbooks.tudelft.nl/mqp/hilbert-spaces-and-operators.html). The controlled preparation family, worked calculations and numerical verification contract are developed for this textbook.
