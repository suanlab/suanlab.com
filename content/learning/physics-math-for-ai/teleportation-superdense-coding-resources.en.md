## 1. Write down the resources before drawing the circuit

Quantum teleportation transfers an unknown qubit state using one shared Bell pair and two transmitted classical bits. Superdense coding sends one of four classical messages using one shared Bell pair and one transmitted qubit. The shared entanglement is a resource that must be counted; neither protocol supplies free communication.

These are ideal protocols with perfect gates, complete measurements and reliable transmission. This chapter derives their branches and checks complex amplitudes. Noise models and implementation fidelity require additional work.

Review [circuit order and interference](/book/online/physics-math-for-ai/finite-qubit-circuit-order-interference/) and [classical measurement records and feedforward](/book/online/physics-math-for-ai/quantum-instruments-records-feedforward/).

## 2. Fix the register order and initial state

For teleportation use site order $0,1,2$: Alice's input, Alice's Bell half, Bob's Bell half. Site zero is the most significant computational-basis bit. The input is

$$
\lvert\psi\rangle=\alpha|0\rangle+\beta|1\rangle,\qquad |\alpha|^2+|\beta|^2=1,
$$

and the joint state is

$$
\lvert\psi\rangle_0\otimes|\Phi^+\rangle_{12},\qquad
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2}.
$$

The numerical model can accept a nonzero unnormalized exact input vector. Its raw amplitudes retain that scale; probabilities and density matrices divide by the input norm. An input list such as $(3,4i)$ represents the same normalized state as $(3/5,4i/5)$.

## 3. Derive the four branches

Alice applies CNOT from site0 to site1, then $H$ on site0. Expanding the resulting state gives

$$
\frac12\sum_{a,b\in\{0,1\}}|ab\rangle_{01}\otimes X^bZ^a|\psi\rangle_2.
$$

She measures sites0 and1 in $Z$, obtaining classical bits $a,b$. Bob's unnormalized branch vector is $(1/2)X^bZ^a|\psi\rangle$. Its norm squared is $1/4$, independently of the input. The factor $1/2$ is an amplitude, not a branch probability.

| Record $a,b$ | Bob's raw branch | Gates Bob executes, in time order |
|---|---|---|
| 0,0 | $\lvert\psi\rangle/2$ | none |
| 0,1 | $X\lvert\psi\rangle/2$ | $X$ |
| 1,0 | $Z\lvert\psi\rangle/2$ | $Z$ |
| 1,1 | $XZ\lvert\psi\rangle/2$ | $X$, then $Z$ |

Multiplication acts right to left. Executing $X$ then $Z$ implements $ZX$, so the last correction cancels $XZ$ exactly: $ZX\,XZ=I$.

## 4. A complex input catches missing signs

Take $\alpha=3/5$, $\beta=4i/5$. The four raw vectors are

$$
\tfrac12\begin{pmatrix}\alpha\\\beta\end{pmatrix},\quad
\tfrac12\begin{pmatrix}\beta\\\alpha\end{pmatrix},\quad
\tfrac12\begin{pmatrix}\alpha\\-\beta\end{pmatrix},\quad
\tfrac12\begin{pmatrix}-\beta\\\alpha\end{pmatrix}.
$$

Every corrected raw vector is $(\alpha,\beta)^T/2$. For record11, applying $X$ gives $(\alpha,-\beta)^T/2$, then $Z$ restores $(\alpha,\beta)^T/2$. The normalized input density matrix has off-diagonal entries $-12i/25$ and $12i/25$; those signs must be preserved.

Testing only $|0\rangle$ would not check relative phase. Testing only normalized densities would miss a raw global sign. The exact calculation therefore checks both raw vectors and density matrices.

## 5. Before the message arrives, Bob has no input-dependent local signal

Ignoring Alice's classical record gives

$$
\rho_B=\frac14\sum_{a,b}X^bZ^a\rho\,Z^aX^b=I_2/2.
$$

Bob's local measurement statistics are independent of $\rho$. The four conditional states are different, but their average has no usable signal. Alice must send the two classical bits before Bob can choose the appropriate correction. Entanglement does not permit faster-than-light communication.

The planned visualization will separate the branch-conditioned state from the record-discarded mixture. Selecting a branch is a conditioning operation, not an action Bob can perform without knowing the record.

## 6. Teleportation transfers rather than duplicates

After Alice measures, her two qubits contain the classical record. They do not remain a separate intact copy of $|\psi\rangle$. The Bell-pair resource has also been consumed by the protocol. Repeating the task requires a new shared pair.

Linearity of the corrected operation makes the ideal channel the identity, including when the input is entangled with a reference. A finite list of pure-state examples illustrates that claim but is not its proof. The branch operator identity $K_{ab}=I/2$ after correction gives $\sum_{ab}K_{ab}\rho K_{ab}^\dagger=\rho$ and remains valid with an untouched reference factor.

## 7. Dense coding reverses the resource roles

Now Alice and Bob share $|\Phi^+\rangle_{AB}$. Alice wants to communicate bits $(a,b)$. For a definite convention, she executes $X^b$ followed by $Z^a$ on her half, implementing $Z^aX^b$. She then physically sends that qubit to Bob.

Bob now owns both halves. He applies CNOT from the received qubit to his original half, then $H$ on the received qubit, and measures both in $Z$. The decoded state is $|ab\rangle$ under this convention.

| Message | Operator on Alice's half | Bell state before transmission |
|---|---|---|
| 00 | $I$ | $\Phi^+$ |
| 01 | $X$ | $\Psi^+$ |
| 10 | $Z$ | $\Phi^-$ |
| 11 | $ZX$ | $\Psi^-$ |

Here $|\Psi^\pm\rangle=(|01\rangle\pm|10\rangle)/\sqrt2$. Reversing the two encoding gates changes only the global sign for message11; the decoder then returns $-|11\rangle$, with the same observable message. The calculation retains this sign rather than silently equating raw vectors.

## 8. Why two bits do not violate the Holevo bound

The receiver's joint system has dimension four, and the four Bell states are orthogonal. The task uses a preexisting shared ebit and a newly transmitted qubit. It is not the task of sending two bits in an isolated unassisted qubit.

Before transmission, Bob still has $I_2/2$ for every message. Alice's local Pauli operation does not change his reduced density matrix. The message becomes distinguishable only after Bob receives her qubit and can perform a joint measurement.

| Protocol | Consumed shared entanglement | Transmitted system | Task |
|---|---:|---|---|
| Teleportation | 1 ebit | 2 classical bits | transfer an unknown qubit state |
| Dense coding | 1 ebit | 1 qubit | transmit 2 chosen classical bits |

Preparing and distributing the initial entanglement has a cost outside these transmission-stage counts. Resource accounting must keep that assumption visible.

## 9. Exercises with worked answers

1. **Branch probability.** Why is a raw factor $1/2$ associated with probability $1/4$? **Answer:** the Born rule squares the norm; for unnormalized input divide by its norm squared as well.
2. **Order check.** Bob's record is11 and his branch is $XZ|\psi\rangle/2$. **Answer:** execute $X$, then $Z$; the total correction is $ZX$ and the raw result is $|\psi\rangle/2$.
3. **Complex coherence.** For $(3/5,4i/5)$, find $\rho_{01}$. **Answer:** $(3/5)(-4i/5)=-12i/25$.
4. **No message.** Can Bob distinguish teleporting $|0\rangle$ from $|+\rangle$ before receiving Alice's bits? **Answer:** no; either reduced state is $I_2/2$.
5. **Global phase.** Dense-code11 with $XZ$ instead of $ZX$. **Answer:** decoding gives $-|11\rangle$, which has the same measurement probabilities but a different raw vector.
6. **Resource count.** Why is dense coding not a counterexample to an unassisted qubit information bound? **Answer:** a shared Bell pair and a four-dimensional joint receiver system are additional resources.

## 10. From exact circuits to research questions

Noisy shared entanglement, imperfect classical control, losses, reference preservation and finite-resource distillation lead to different tasks. They require explicit channels and performance criteria. A successful ideal pure-input demonstration does not quantify an implementation's teleportation fidelity or establish a noisy channel capacity.

For AI-assisted scientific software, this example emphasizes explicit register order, chronological operations, raw-versus-normalized representations and resource constraints. The foundation currently supplies exact calculations and manuscripts; Worker, interactive views, guided exercises and reader integration remain to be connected.

Primary publication records: [Bennett et al., teleportation (1993)](https://research.ibm.com/publications/teleporting-an-unknown-quantum-state-via-dual-classical-and-einstein-podolsky-rosen-channels) and [Bennett–Wiesner, dense coding (1992)](https://research.ibm.com/publications/communication-via-one-and-two-particle-operators-on-einstein-podolsky-rosen-states). All branch conventions and worked calculations above are specified for this course.
