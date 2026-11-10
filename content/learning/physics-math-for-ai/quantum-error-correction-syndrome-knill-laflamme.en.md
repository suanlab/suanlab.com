## 1. Protect a superposition without learning its amplitudes

A logical qubit is $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$. Quantum error correction must preserve arbitrary complex amplitudes and any entanglement with a reference. Recovering the right answer for one computational-basis input is insufficient.

The three-qubit repetition encoding is the isometry

$$
V|0\rangle=|000\rangle,\qquad V|1\rangle=|111\rangle.
$$

It maps the superposition to $\alpha|000\rangle+\beta|111\rangle$, not to three independent copies. This distinction is consistent with no-cloning. The code space has dimension two inside an eight-dimensional physical space.

Review [measurement records and feedforward](/book/online/physics-math-for-ai/quantum-instruments-records-feedforward/) and [channel verification](/book/online/physics-math-for-ai/qubit-channel-kraus-choi-verification/).

## 2. Measure parity, not the logical bit

Use commuting observables

$$
S_0=Z_0Z_1,\qquad S_1=Z_1Z_2.
$$

Both codewords have eigenvalue $+1$ for both observables. Label the eigenvalues as $(-1)^a,(-1)^b$ and define the syndrome projector

$$
P_{ab}=\frac{I+(-1)^aS_0}{2}\frac{I+(-1)^bS_1}{2}.
$$

A syndrome identifies parity disagreements. It does not distinguish $000$ from $111$, so it does not reveal the encoded amplitudes. Measuring all three physical qubits in $Z$ would reveal the logical label and destroy a general logical superposition.

## 3. Single bit flips have different syndromes

Site order is $0,1,2$, with site0 the most significant bit. A recovery uses the minimum-weight X error compatible with the observed syndrome.

| Error | Syndrome $a,b$ | Recovery |
|---|---|---|
| $I$ | 0,0 | $I$ |
| $X_0$ | 1,0 | $X_0$ |
| $X_1$ | 1,1 | $X_1$ |
| $X_2$ | 0,1 | $X_2$ |

For example, $X_1$ produces $\alpha|010\rangle+\beta|101\rangle$. Both terms lie in syndrome11. Applying $X_1$ restores the entire coherent state. The logical decoder $V^\dagger$ then gives the original qubit.

Write the recovered logical branch operator for physical error $E$ as $K_{ab}(E)=V^\dagger C_{ab}P_{ab}EV$. An impossible branch has probability zero and no normalized conditional state. Do not draw a fictitious state for it.

## 4. Check a complex example

Encode $|\psi\rangle=(3|0\rangle+4i|1\rangle)/5$. After $X_0$ the state is $(3|100\rangle+4i|011\rangle)/5$. The syndrome is10 with probability one. Applying $X_0$ and decoding restores the off-diagonal density entries $-12i/25$ and $12i/25$.

A correction that restores only populations would miss this phase information. The numerical model therefore returns the encoded density matrix, the errored matrix, every syndrome probability, the raw recovered vector and its conditional logical density.

This example has a specified single Pauli error. It must be kept separate from a stochastic noise channel that averages over several possible errors.

## 5. Derive the Knill–Laflamme condition

For an error family $\{E_a\}$, exact correction on the code requires and is guaranteed by

$$
V^\dagger E_a^\dagger E_bV=c_{ab}I_2
$$

for every pair. Equivalently, with code projector $P=VV^\dagger$, $PE_a^\dagger E_bP=c_{ab}P$.

The condition says the error-overlap matrix does not distinguish logical basis states and has no unwanted logical off-diagonal terms. Checking only $E_a^\dagger E_a$ misses cross terms. Checking only one encoded input misses dependence on the logical state.

For $\{I,X_0,X_1,X_2\}$ the compressed cross terms vanish and the diagonal terms are $I_2$. The error subspaces are mutually orthogonal. The syndrome measurement distinguishes these errors without distinguishing the logical alternatives.

## 6. What this code cannot correct

Add $Z_0$ to an error family containing $I$. Its cross term is

$$
V^\dagger Z_0V=Z,
$$

which is not a scalar identity. A phase flip changes $\alpha|000\rangle+\beta|111\rangle$ into $\alpha|000\rangle-\beta|111\rangle$ while leaving syndrome00. The X-only recovery does not repair it. The code does not correct an arbitrary single-qubit Pauli error.

Two flips can imitate a different single flip: $X_0X_1$ has the syndrome of $X_2$. Applying $X_2$ yields $X_0X_1X_2$, which acts as logical $X$. A zero syndrome also does not prove absence of error: the triple flip is a logical operation with syndrome00.

A passing condition does not require orthogonal error labels. For the family $\{Z_0,Z_1,Z_2\}$ without $I$, all three operators act as the same logical Z and can be undone by a common phase flip. Their cross terms are scalar identities. This illustrates degeneracy and the importance of specifying the error family; the fixed X recovery is not that alternative recovery.

## 7. Compute the logical noise channel

Assume each physical qubit independently flips with probability $p$, and assume ideal encoding, syndrome measurement and recovery. Zero or one flip is corrected. Two or three flips produce logical X, with probability

$$
q=3p^2(1-p)+p^3=3p^2-2p^3.
$$

Thus the corrected logical channel is

$$
\mathcal L_p(\rho)=(1-q)\rho+qX\rho X.
$$

At $p=1/10$, $q=7/250=0.028$, below the physical error probability. At $p=1/2$, $q=1/2$; at $p=1$, $q=1$. This is a fixed majority recovery. Knowing that a deterministic triple flip always occurs would permit a different recovery, so the displayed rule is not an optimal decoder for every noise parameter.

The model enumerates all eight X-error masks, their exact weights, syndromes and effective logical operators. It derives the channel from these branches rather than using only the final polynomial.

## 8. One preserved input does not certify a preserved channel

Logical $|+\rangle$ is invariant under $X$, so its state fidelity is one for every $q$. That does not make $\mathcal L_p$ the identity. For $|0\rangle$, the squared fidelity with the input is $1-q$.

Attach an untouched reference and prepare $|\Phi^+\rangle_{RL}$. The normalized Choi state after the logical channel is

$$
(1-q)|\Phi^+\rangle\langle\Phi^+|+q|\Psi^+\rangle\langle\Psi^+|.
$$

These Bell states are orthogonal, so the entanglement fidelity for the maximally mixed logical input is $1-q$. At $p=1/10$ it is $243/250=0.972$, even though the input-specific $|+\rangle$ fidelity is one.

The visualization should display these two criteria together. A channel that preserves selected states can still damage correlations needed by a larger computation.

## 9. Exercises with worked answers

1. **Encoding versus cloning.** Is $\alpha|000\rangle+\beta|111\rangle$ three independent copies? **Answer:** no; it is an encoded superposition and generally entangled.
2. **Syndrome11.** Which single X error does it identify? **Answer:** $X_1$ under the specified site/parity order.
3. **Undetected phase error.** Why does $Z_0$ evade these checks? **Answer:** it changes a relative phase but commutes with the Z parities and leaves the code-space syndrome unchanged.
4. **Cross-term test.** Why is checking $E_a^\dagger E_a=I$ insufficient? **Answer:** all Pauli operators pass that test, while cross terms such as $I^\dagger Z_0$ fail the scalar-on-code requirement.
5. **Small error probability.** Evaluate $q$ at $p=1/10$. **Answer:** $3/100-2/1000=7/250$.
6. **Misleading fidelity.** The logical $|+\rangle$ fidelity is one. Is reference entanglement preserved? **Answer:** generally not; the entanglement fidelity is $1-q$, which is below one for $q>0$.

## 10. Beyond the toy code

General Pauli correction needs richer codes. Fault tolerance also accounts for noisy measurements, gates, ancilla preparation and correlated faults. A smaller logical error under ideal recovery is not a fault-tolerance threshold or a scalable quantum computer.

For AI-assisted scientific development, the lesson is to verify the full operation and its invariants, not just one successful example. The foundation currently supplies exact calculations and manuscripts; Worker, diagrams, guided interaction and reader integration remain to be connected.

Primary conditions and the reference-system perspective are discussed in [Preskill, Chapter 7, Sections 7.2–7.3](https://www.preskill.caltech.edu/ph219/chap7_26_6A.pdf). The matrices, examples and finite noise calculations above are specified independently for this course.
