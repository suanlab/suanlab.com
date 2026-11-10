## 1. Capacity depends on the communication task

A quantum channel can carry a classical message, preserve an unknown quantum state or distribute private correlations. These tasks have different performance criteria and resource assumptions. A single “information capacity” number hides those distinctions.

This chapter studies a known memoryless qubit phase-flip channel,

$$
\mathcal N_p(\rho)=(1-p)\rho+pZ\rho Z,\qquad0\le p\le1.
$$

| Quantity | Task | Unit per channel use | Extra resource |
|---|---|---|---|
| $C$ | classical messages | bits | no preshared entanglement |
| $C_E$ | classical messages | bits | preshared entanglement allowed |
| $Q$ | quantum information | qubits | no assistance |

Private communication and two-way-assisted tasks require separate definitions and are not computed here. All three capacities above refer to asymptotic vanishing-error coding, not one-use guessing success. Review [channel and Choi verification](/book/online/physics-math-for-ai/qubit-channel-kraus-choi-verification/) and [system-environment dilation](/book/online/physics-math-for-ai/system-environment-stinespring-reference/).

## 2. A classical bit survives even complete dephasing

Encode a classical message as $|0\rangle$ or $|1\rangle$. Either projector is unchanged by $Z\rho Z$, so Bob can distinguish them perfectly in $Z$ for every $p$. This gives one classical bit per use without entanglement.

An $n$-use output has dimension $2^n$; its Holevo information is at most $\log_2(2^n)=n$. Thus this family has $C=1$ bit per use. The argument combines an explicit achievable encoding and a dimension upper bound. It does not rely on the quantum capacity having the same value.

At $p=1/2$, the channel erases off-diagonal entries but retains the computational-basis populations. Classical labels can survive while arbitrary quantum superpositions and their reference correlations do not.

## 3. Coherent information belongs to a specified input

Purify the chosen input $\rho_A$ as $|\psi\rangle_{RA}$ and send $A$ through the channel. If $B$ is the output and $E$ the complementary environment, define

$$
I_c(\rho,\mathcal N)=S(B)-S(RB)=S(B)-S(E).
$$

The second equality uses a pure joint $RBE$ dilation. For a pure input without a nontrivial reference, $BE$ is pure, so $S(B)=S(E)$ and $I_c=0$. Even the identity channel therefore has coherent information zero for a chosen pure input, while its quantum capacity is one qubit per use.

The input, its purification and the optimization task must not be omitted. Coherent information can be negative for other channels and inputs; it is not a classical mutual information.

## 4. Derive the channel and environment outputs

For Bloch vector $(x,y,z)$, the output vector is $((1-2p)x,(1-2p)y,z)$. A dilation is

$$
W|\psi\rangle=\sqrt{1-p}|\psi\rangle_B|0\rangle_E+
\sqrt p\,Z|\psi\rangle_B|1\rangle_E.
$$

Tracing out Bob gives

$$
\rho_E=\begin{pmatrix}
1-p&\sqrt{p(1-p)}z\\
\sqrt{p(1-p)}z&p
\end{pmatrix}.
$$

Its eigenvalues are $(1\pm r_E)/2$, where

$$
r_E^2=(1-2p)^2+4p(1-p)z^2.
$$

This permits an explicit entropy calculation for any physical rational Bloch input. The environment has coherence for nonzero $z$; its entropy is not generally the Shannon entropy of the classical error flag alone.

For the maximally mixed input, $z=0$, Bob remains maximally mixed and the environment spectrum is $(1-p,p)$. Hence $I_c(I_2/2,\mathcal N_p)=1-h_2(p)$.

## 5. Why this value is the quantum capacity for this family

A channel is degradable if a channel acting on Bob's output can reproduce the complementary output. Here Bob can measure $Z$ and prepare an environment state

$$
|e_\pm\rangle=\sqrt{1-p}|0\rangle\pm\sqrt p|1\rangle
$$

according to the result. Bob's populations are unchanged by the phase-flip channel, so this measure-and-prepare map reproduces $\rho_E$ exactly. It is a valid degrading map for every $p$.

For degradable channels, the quantum coding theorem and additivity reduce the capacity to a single-use coherent-information maximum. That function is concave in the input. For this Pauli-symmetric family, conjugating the input by $I,X,Y,Z$ leaves the coherent-information value unchanged. Averaging those four inputs produces $I_2/2$; concavity shows that this average is an optimizer.

Combining these facts gives

$$
Q(\mathcal N_p)=1-h_2(p).
$$

The degrading map and symmetry are essential premises. Evaluating one input, or scanning a finite grid, would not establish the capacity of a general channel.

## 6. Entanglement assistance changes the rate

The entanglement-assisted classical capacity theorem gives

$$
C_E(\mathcal N)=\max_\rho I(R:B),\qquad
I(R:B)=S(\rho)+S(B)-S(E).
$$

For this family, concavity and the same Pauli averaging select the maximally mixed input. Its input and output entropies are both one, so

$$
C_E(\mathcal N_p)=2-h_2(p).
$$

This allows preshared entanglement and counts classical bits, whereas $Q$ counts unassisted quantum communication. The two rates are not interchangeable. A noisy quantum channel plus entanglement still cannot communicate without physically using the channel. At the noiseless endpoints, the classical rate of two bits per transmitted qubit agrees with dense coding's resource accounting.

## 7. Inspect the endpoints before trusting a plot

| $p$ | Channel behavior | $C$ | $C_E$ | $Q$ |
|---:|---|---:|---:|---:|
| 0 | identity | 1 | 2 | 1 |
| $1/10$ | random phase error | 1 | approximately1.531004 | approximately0.531004 |
| $1/2$ | complete $Z$ dephasing | 1 | 1 | 0 |
| $9/10$ | mostly a known $Z$ | 1 | approximately1.531004 | approximately0.531004 |
| 1 | deterministic $Z$ | 1 | 2 | 1 |

The rates are symmetric under $p\leftrightarrow1-p$, because a known output $Z$ transforms one channel into the other. More frequent application of a known unitary is not monotonically more destructive. At $p=1$, Bob can undo $Z$ exactly.

Near $p=1/2$, subtracting a rounded displayed entropy from one can incorrectly produce zero. For $p=1/2+\delta$, the leading behavior is $Q\approx2\delta^2/\ln2$. The calculation retains directed intervals and exact classification of the midpoint rather than inferring a true zero from display rounding.

## 8. One-use guessing still has another meaning

Encode equal-prior classical labels as $|+\rangle,|-\rangle$ and measure $X$ after the channel. This is a binary symmetric channel with crossover $p$. Its mutual information for uniform labels is $1-h_2(p)$, while the best label-guess probability is $\max\{p,1-p\}$ after relabeling when necessary.

At $p=1/10$, guessing success is $0.9$ and measured information is approximately $0.531004$ bit. The numerical equality between that information and this family's $Q$ does not make their operational tasks the same. A $Z$ encoding instead transmits one classical bit for every $p$.

The proposed experiment keeps separate panels for the chosen input's entropies, the family capacities and the selected classical encodings. Changing the input changes its coherent information; it does not change the channel's capacity.

## 9. General channels may require regularization

For a general finite-dimensional memoryless channel, quantum capacity is characterized by

$$
Q(\mathcal N)=\lim_{n\to\infty}\frac1n
\max_{\rho_{A^n}}I_c(\rho_{A^n},\mathcal N^{\otimes n}).
$$

The input may be entangled across uses. The unassisted classical capacity likewise uses a regularized optimization of ensemble Holevo information, with collective decoding allowed. The entanglement-assisted classical theorem has a single-letter maximum, but its resource model is different.

These formulas do not make every optimization computationally easy. A finite sample of inputs is not an upper bound on all inputs or all block lengths. A hashing-type achievable expression for a different Pauli channel is not automatically its exact quantum capacity. No claim about a current unsolved frontier is needed to establish these distinctions.

## 10. An asymptotic rate is not a finite-block error guarantee

At $p=1/10$, $Q\approx0.531004$. A target rate of $0.5$ qubit per use is below capacity, so a suitable sequence of increasingly long codes can achieve vanishing error under the stated model. This does not specify which code works for a block of100 uses or what its error will be.

For a finite block, the encoded logical dimension is an integer, for example $2^k$, and the rate is $k/n$. Choosing $k=53,n=100$ does not acquire an error guarantee merely because $0.53<Q$. Encoding maps, decoding maps and a fidelity criterion are still needed. Strong-converse and second-order claims require their own hypotheses and proofs.

## 11. Exercises with worked answers

1. **Classical preservation.** Why is $C$ nonzero at complete dephasing? **Answer:** the orthogonal $Z$ labels survive exactly, yielding $C=1$.
2. **Pure input.** What is the coherent information of a selected pure input through this dilation? **Answer:** zero, because the joint output/environment state is pure and the marginal entropies agree.
3. **Capacity example.** Evaluate the rates at $p=1/10$. **Answer:** $C=1$, $Q=1-h_2(1/10)\approx0.531004$, $C_E=1+Q\approx1.531004$.
4. **Known error.** What happens at $p=1$? **Answer:** a deterministic unitary $Z$ can be undone, giving $Q=1$, not zero.
5. **Input search.** Does a grid maximum certify a general channel's capacity? **Answer:** no; untested inputs, entangled block inputs and required converse arguments remain.
6. **Finite block.** Is $53$ logical qubits in100 uses guaranteed accurate by the capacity number? **Answer:** no. The asymptotic threshold alone does not construct that code or bound its finite-block error.

## 12. Research connections and assumptions to preserve

AI-assisted communication design should record the message type, shared resources, block length, input family and error criterion before comparing rates. A model that improves one-use classification may not improve quantum transmission or reference preservation.

Further work includes explicit finite-block codes, noisy assistance, energy or dimension constraints, private communication and recovery guarantees. Those extensions require additional theory and experiments; this chapter supplies a precisely delimited qubit family.

Primary foundations: [Preskill, Quantum Shannon Theory, Sections10.7–10.8](https://www.preskill.caltech.edu/ph219/chap10_6A_2025.pdf), [Devetak–Shor on generalized dephasing and regularization](https://arxiv.org/abs/quant-ph/0311131), and [Bennett–Shor–Smolin–Thapliyal on entanglement-assisted capacity](https://arxiv.org/abs/quant-ph/0106052). The matrices, examples and endpoint checks above are developed explicitly for this course.
