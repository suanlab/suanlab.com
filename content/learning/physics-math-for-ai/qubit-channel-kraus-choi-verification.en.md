## 1. Why a unitary is not enough

A state vector describes an isolated pure-state computation. A device can also lose energy, interact with an unobserved environment, or forget a measurement result. Its visible output may be mixed even when the joint evolution is unitary. A quantum channel describes the input–output rule for density operators without requiring us to observe the environment.

We assume density matrices, partial traces and tensor products. The key question is stronger than “does this rule return a valid state for the inputs we tried?” The rule must remain valid when the input is correlated with another system. This is essential when composing quantum circuits or training a model whose output is used inside a larger computation.

## 2. Linearity, trace preservation and complete positivity

An unconditioned channel is linear on operators. Operationally, preparing ρ with probability p and σ otherwise must produce the same average output as applying the channel to their mixture. Conditional postselection instead divides by a state-dependent probability; it is generally nonlinear as a normalized state update.

Trace preservation means Tr Φ(A)=Tr A for all operators A. Positivity means positive inputs stay positive. Complete positivity means id_R⊗Φ stays positive for every finite reference system R. In finite dimension it suffices to test a reference as large as the input. A physical channel is completely positive and trace preserving, abbreviated CPTP.

These conditions answer different questions. Trace preservation concerns total weight. Positivity concerns individual systems. Complete positivity concerns compatibility with correlations. Unitality, Φ(I)=I, is a fourth property and is not required of every channel.

## 3. Kraus operators: alternatives whose outputs add

A Kraus representation has the form

$$
\Phi(\rho)=\sum_a K_a\rho K_a^\dagger,\qquad
\sum_a K_a^\dagger K_a=I.
$$

Each term preserves positivity even with a reference attached. The second equation gives trace preservation by cyclicity of trace. Do not add K operators before applying them: (K₀+K₁)ρ(K₀+K₁)† has cross terms that are absent from the channel. An unobserved environment label is not a coherently recombined path.

Use a damping example with survival amplitude3/5 and decay amplitude4/5:

$$
K_0=\begin{pmatrix}1&0\\0&3/5\end{pmatrix},\quad
K_1=\begin{pmatrix}0&4/5\\0&0\end{pmatrix},\quad
\gamma=16/25.
$$

For ρ=[[a,c],[c*,b]], the output has diagonal entries a+16b/25 and 9b/25, with coherence3c/5. The excited-state population can move into the ground state. The trace remains a+b, while Φ(I)=diag(41/25,9/25) differs from I. Thus this channel is TP and CP but not unital. It cannot be a mixture of unitaries, since each unitary fixes I and so does every probabilistic mixture of them.

The adjoint channel acts on observables: Φ*(E)=ΣK†EK. It satisfies Tr[EΦ(ρ)]=Tr[Φ*(E)ρ]. TP is equivalent to Φ*(I)=I, whereas unitality concerns Φ(I). Keeping these two maps separate prevents a common reversal of the completeness condition.

## 4. A Choi matrix tests the whole linear map

Fix the convention before constructing a matrix. We use reference first, output second, and an **unnormalized** maximally entangled vector Ω=|00〉+|11〉:

$$
J(\Phi)=\sum_{i,j=0}^1 |i\rangle\langle j|_R\otimes
\Phi(|i\rangle\langle j|)_B
=(\mathrm{id}_R\otimes\Phi)(|\Omega\rangle\langle\Omega|).
$$

Complete positivity is equivalent to J being positive semidefinite. TP is equivalent to Tr_B J=I_R. A TP qubit channel has Tr J=2; J/2 is a density matrix. A matrix with trace 1 can therefore be the normalized Choi state without matching this unnormalized convention entry by entry.

For the damping example, in order |00〉,|01〉,|10〉,|11〉:

$$
J=\begin{pmatrix}1&0&0&3/5\\0&0&0&0\\0&0&16/25&0\\3/5&0&0&9/25\end{pmatrix}.
$$

Its eigenvalues are0,0,16/25,34/25, all nonnegative. Its output partial trace is I. These checks use off-diagonal matrix units too: testing only |0〉 and |1〉 misses how a map handles coherence.

## 5. Transpose passes a weaker test and fails the necessary one

The transpose map T(ρ)=ρᵀ preserves trace and maps a single density matrix to another density matrix. Its Choi matrix is the two-qubit SWAP operator. The antisymmetric vector (|01〉−|10〉)/√2 has eigenvalue−1, so the Choi matrix is not positive semidefinite.

Applying id⊗T to a normalized Bell state gives J(T)/2, with expectation−1/2 on that vector. A negative measurement probability would result if this were treated as a physical joint state. This explicit reference-system counterexample explains why positivity alone is insufficient; the failure does not depend on rounding or a particular random test set.

## 6. Composition and postselection

If Φ has Kraus operators K_a and Ψ has L_b, then Ψ∘Φ has Kraus operators L_b K_a. Chronology still matters. For the damping example, applying X then damping to |0〉 gives populations16/25 and9/25. Damping then X gives |1〉 with certainty.

A single recorded outcome can use a trace-decreasing map A(ρ)=KρK†. Its trace is the probability of that outcome. If ΣK†K≤I, the operation is CP and trace nonincreasing, but is not an unconditioned TP channel. The normalized state A(ρ)/Tr A(ρ) is unavailable when that probability is0. Reporting an invented conditional state in that case would conceal an impossible event.

## 7. A worked comparison of phase flip, selection and damping

Use the same input |+i〉=(|0〉+i|1〉)/√2 so that a change of input cannot explain a difference between maps. Its initial populations are1/2,1/2 and its upper off-diagonal entry is−i/2.

The random phase-flip map uses Z with probability p and I otherwise. Multiplying Z on both sides changes the signs of the off-diagonal entries while preserving the populations. Thus

$$
\Phi_p(\rho)=(1-p)\rho+pZ\rho Z,\qquad
\rho_{01}\longmapsto(1-2p)\rho_{01}.
$$

At p=1/2 the output is I/2 with purity 1/2. At p=1 it is the pure state |−i〉, with Y expectation−1 and purity 1. The parameter is a probability of applying a particular error, not a monotone measure of lost coherence over the whole interval. A different depolarizing convention can use the same letter p for another rule; compare the map formulas before comparing plots.

Damping with γ=16/25 instead gives populations41/50,9/50 and coherence−3i/10. Its output purity is553/625. Its unital image is diag(41/25,9/25), while the adjoint fixes I. These three entries show population transfer and coherence attenuation in the same example.

For selection, take K=|0〉〈0| and input |1〉 instead. The output operator is0 and its trace, the success probability, is0. This is a CP operation, but it is not TP and has no conditional normalized state on that input. At the opposite extreme, the linear map ρ↦2ρ is CP and has a positive output, but trace 2. Physicality requires the appropriate conditions together; a positive output for one input is not a channel certificate.

In the experiment, first predict the populations, then inspect coherence and the separate CP/TP/unital checks. Finally inspect J and J/2. The matrix called a Choi state in a normalized convention must be multiplied by2 before comparison with this lesson's J.

## 8. Exercises with explanations

**Check:** Why does damping preserve trace? K₀†K₀=diag(1,9/25) and K₁†K₁=diag(0,16/25), whose sum is I. Adding K₀K₀† and K₁K₁† instead tests unitality and yields a different matrix.

**Predict:** Is transpose a valid channel because all sampled one-qubit states passed? No. The antisymmetric witness on its Bell output proves a negative direction with a reference present.

**Normalize:** Your Choi matrix has trace 1 and output marginal I/2. Is TP violated? First check whether you constructed J/2. Under that normalized convention those are precisely the TP marginal and trace values.

## 9. Verification and further study

Reconstruct the map from its Choi blocks and compare its output with the operator sum. Check off-diagonal matrix units as well as basis-state populations. Inspect CP, TP and unitality separately, and distinguish a successful calculation from a physical-channel certificate. Further study includes channel discrimination, tomography and learning physical maps under CP/TP constraints.

Definitions and representation equivalences follow [IBM Quantum Learning, Quantum channels](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels) and [Watrous, The Theory of Quantum Information, chapter2](https://jhwatrous.github.io/TQI.pdf). The rational damping values, matrix convention and explicit hand calculations are stated here so they can be reproduced independently.
