# Degenerate projectors and sequential observation

Registered and verified in the Interactive Book reader. See the [integration report](quantum-observables-integration-status.md) for evidence.

## 1. The same number can hide different states

Suppose a measurement reports the value 2. Does that tell us which vector the system occupies? Only if the corresponding eigenspace is one-dimensional, and even then only its ray is fixed. In a higher-dimensional eigenspace, many states give that same result with certainty. Resolving the value and resolving an internal basis are different experiments.

We use C³ to make this distinction concrete. The three complex coordinates are basis labels, not three physical position axes. The input is a pure state ψ=z/||z||. All measurements below are specified as ideal projective instruments; this assumption includes how the state changes, rather than deriving an apparatus from the observable alone.

## 2. Build a rank-one and a rank-two eigenspace

A nonzero raw complex vector v determines P. Its orthogonal complement Q has dimension two. Rational labels λ and μ define the observable A.

$$
P=\frac{vv^\dagger}{v^\dagger v},\qquad Q=I-P,\qquad A=\lambda P+\mu Q,\qquad \operatorname{rank}P=1,\quad \operatorname{rank}Q=2.
$$

When λ and μ differ, the two spectral outcomes correspond to P and Q, even though Q contains infinitely many rays. When λ=μ, A is a scalar multiple of I. There is then only one spectral projector, I. Keeping P and Q as distinguishable recorded branches in that case would add information that the value of A does not supply.

Changing the phase or magnitude of v leaves P unchanged. Sorting λ and μ changes only display order when each projector moves together with its label. The matrix and the physical instrument stay the same.

## 3. Selective projection requires a nonzero probability

For this chapter we choose the ideal Lüders update. Conditioned on receiving outcome k, the state is the normalized projection into that entire eigenspace.

$$
p_k=\frac{\|P_kz\|^2}{z^\dagger z},\qquad \psi_k=\frac{P_kz}{\|P_kz\|}\quad(p_k>0).
$$

If the projected vector is zero, the outcome cannot occur in this preparation. There is no conditional state for it. Returning a zero vector as though it were normalized, or silently dividing by an epsilon, would invent a physical preparation. The numerical result therefore uses null for that branch's conditional amplitudes.

If an outcome has rank two, its projection can preserve a superposition inside that subspace. Measurement of its numerical value does not require choosing one internal basis vector. This is exactly where a naive basis-by-basis collapse would give the wrong prediction for the specified instrument.

## 4. Ignoring the outcome still changes the experiment

A selective output includes the recorded outcome. A nonselective output describes the ensemble when the measurement happened but its outcome is ignored. We use a density matrix for this derived output, not as a new arbitrary mixed-state input.

$$
\rho=|\psi\rangle\langle\psi|,\qquad \rho_A=\sum_k P_k\rho P_k=\sum_k p_k|\psi_k\rangle\langle\psi_k|.
$$

The final sum includes only nonzero-probability branches. Its trace is one. The output can be mixed even when the input was pure. Purity Tr(ρ_A²) equals one for a pure output and is lower for a mixed output. It is a diagnostic of the ensemble, not a probability of an additional unspecified outcome.

Ignoring an outcome does not undo the measurement. Coherence between distinct measured eigenspaces is removed by this map. Coherence inside one degenerate eigenspace can remain. Which coherence survives depends on the actual instrument.

## 5. A rank-one refinement is a different instrument

Split Q into orthogonal rank-one projectors S and T. The three projectors P, S and T sum to I. Their associated numerical labels are λ, μ and μ, but the refined experiment also distinguishes the S and T branch records.

$$
Q=S+T,\qquad ST=0,\qquad \rho_{\mathrm{fine}}=P\rho P+S\rho S+T\rho T.
$$

For reproducibility the engine picks the first standard axis whose projection under Q is nonzero, normalizes its direction to build S, and sets T=Q−S. This is one deterministic basis refinement, not a preferred physical basis inherent in A. The independent reference constructs the same direction with Gram–Schmidt.

After discarding the S/T record, the reported numerical distribution of A matches the coarse measurement. The nonselective state can nevertheless differ. This is why specifying only the numerical observable does not specify every possible implementation of its measurement. We explicitly compare the ideal coarse Lüders instrument with the chosen refinement.

## 6. Work through a preserved superposition

Let P project onto e₀ and Q onto the span of e₁ and e₂. Set λ=1, μ=2 and ψ=(e₁+e₂)/√2. The value 2 occurs with certainty. The coarse projection leaves ψ unchanged. Refine Q using S onto e₁ and T onto e₂: each refined branch occurs with probability 1/2.

$$
\rho_A=\frac12\begin{pmatrix}0&0&0\\0&1&1\\0&1&1\end{pmatrix},\qquad \rho_{\mathrm{fine}}=\frac12\begin{pmatrix}0&0&0\\0&1&0\\0&0&1\end{pmatrix}.
$$

The off-diagonal entries between e₁ and e₂ carry the coherence lost by refinement. A later test B=|ψ⟩⟨ψ| returns 1 with probability one after the coarse measurement, but only with probability 1/2 after refinement. The two instruments have the same original A value distribution and different consequences.

| Quantity | Coarse Lüders A | Rank-one refinement |
| --- | --- | --- |
| Reported A value | 2 with probability 1 | 2 with probability 1 |
| Nonselective purity | 1 | 1/2 |
| Later B=1 probability | 1 | 1/2 |
| Squared Frobenius distance between outputs | 1/2, comparing the two outputs | Same comparison |

A and B commute in this example. That fact protects the specified coarse spectral measurement; it does not license replacing it with an arbitrary refinement that fails to commute with B.

## 7. Sequential probabilities include an ordered state update

Construct B with its own spectral projectors R_j. In the experiment A then B, the first projector acts on the raw state before the second one.

$$
p_{A\to B}(i,j)=\frac{\|R_jP_i z\|^2}{z^\dagger z},\qquad p_{B\to A}(j,i)=\frac{\|P_iR_j z\|^2}{z^\dagger z}.
$$

Rows of the first table are A outcomes and columns are B outcomes. Rows of the reverse table are B outcomes and columns are A outcomes. Compare a table with the transpose of the other when asking whether the same labeled pair has the same probability. Otherwise an indexing mismatch can masquerade as a measurement-order effect.

The first marginal is the original Born distribution. The second marginal describes the changed ensemble after the first measurement; it need not equal B's Born distribution on the original state. If the first probability is zero, the conditional second distribution is undefined and returned as null, while its joint-probability row is exactly zero.

## 8. An order effect with simple numbers

Keep A=diag(1,2,2), set B to the rank-one projector onto (e₀+e₁)/√2, and prepare e₀. Measuring A first gives 1 certainly, followed by B=0 or 1 with probability 1/2 each. Measuring B first gives either branch with probability 1/2; both conditional states then give A=1 or 2 with probability 1/2 each.

| Labeled pair (A value, B value) | A then B | B then A |
| --- | --- | --- |
| (1,0) | 1/2 | 1/4 |
| (1,1) | 1/2 | 1/4 |
| (2,0) | 0 | 1/4 |
| (2,1) | 0 | 1/4 |

The second experiment has A=1 probability 1/2, rather than one. This difference comes from the ordered measurement update, not from a physical-time animation or a numerical integration step. All entries are exact rational probabilities.

## 9. One preparation cannot certify commutativity

The operator commutator is independent of the chosen input. Its vanishing is stronger than the accidental agreement of two joint laws on one state.

$$
[A,B]=AB-BA,\qquad \|[A,B]\|_F^2=\operatorname{Tr}([A,B]^\dagger[A,B]).
$$

In the preceding A and B example, use input e₂ instead. Both orders now report (A=2,B=0) with certainty. The laws agree on that state, although the commutator remains nonzero because the operators act differently in the e₀/e₁ plane. Testing a single preparation cannot prove an operator identity.

The exact engine checks the matrix commutator. A tiny nonzero rational entry is not rounded to zero for classification. The displayed drawing can use rounded coordinates, but it does not decide whether the observables commute.

## 10. Independent checks and scope of the model

Every sequence also has a density-matrix expression for its second marginal. It provides a separate way to inspect the meaning of summing over the first outcome.

$$
\sum_i p_{A\to B}(i,j)=\operatorname{Tr}(R_j\rho_A),\qquad \sum_jp_{A\to B}(i,j)=p_A(i).
$$

Production computes exact projected-vector norms. The independent symbolic reference uses trace expressions and separately obtained eigenspaces. Checks cover the projector laws, total mass, both marginal identities, conditional normalization, output purity, negative eigenvalue sorting, exactly coincident rational labels and zero branches. Normalized amplitudes alone require directed 512-bit square-root intervals.

No unitary evolution, general POVM, continuous-spectrum position measurement or environment model is supplied here. Ordinary matrix coordinates do not settle self-adjoint domains in infinite-dimensional spaces. General instruments can have different state updates; the explicit Lüders and refinement choices are essential assumptions.

## 11. Exercises with explanations

**1. If λ=μ=5, how many spectral outcomes does A have?**

**Explanation.** One: value 5 with projector I. Every input is preserved by the chosen coarse Lüders update. Keeping three distinguishable branch records defines a finer instrument.

**2. Does observing value 2 select e₁ or e₂ in the superposition example?**

**Explanation.** No. The coarse projector is Q and Qψ=ψ. Choosing an internal basis vector would add a refinement that was not part of the stated experiment.

**3. What conditional state belongs to a zero-probability branch?**

**Explanation.** None. Its joint row is zero and its conditional vector and conditional second law are undefined. A normalized zero vector does not exist.

**4. How can two measurements report the same A value distribution and differ later?**

**Explanation.** The coarse and fine instruments both report 2 certainly, but their output states have different e₁/e₂ coherence. The later B test yields probabilities 1 and 1/2.

**5. Does agreement of the two orders on e₂ prove [A,B]=0?**

**Explanation.** No. The state lies outside the plane that reveals their noncommuting action. The exact commutator remains nonzero; e₀ reveals an order effect.

**6. Why compare the reverse joint table after transposing?**

**Explanation.** Its row label is B and its column label is A. Transposition aligns the same ordered pair of labels before comparing numerical probabilities.

## 12. Connections to inference and future topics

A coarse outcome does not justify inferring information that the experiment did not record. Two procedures with the same immediate output histogram can produce different downstream behavior. These are useful cautions when designing AI evaluations, latent representations and sequential decision experiments. The quantum update is its own mathematical model; it is not ordinary Bayesian conditioning on a pre-existing hidden basis label.

General density operators, measurement uncertainty, entanglement and quantum channels will extend these ideas later. The present derived output ensembles are a bridge, not a replacement for those topics or a completed theory of measurement devices.

The finite projection and Born rules were checked against [Preskill §2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf); adjoint and finite Hermitian background were checked against [TU Delft §8.7–8.8](https://interactivetextbooks.tudelft.nl/mqp/hilbert-spaces-and-operators.html). The C³ refinement comparison and all worked probability tables are derived for this textbook.
