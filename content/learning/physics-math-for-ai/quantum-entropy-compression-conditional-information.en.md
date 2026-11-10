## 1. What does the entropy measure?

A pure state can give random measurement outcomes. Prepare every qubit in the known state $|+\rangle$: measuring $Z$ gives a fair coin, but measuring $X$ always gives $+1$. The randomness of one chosen measurement does not define the entropy of the state.

The von Neumann entropy depends on the spectrum of the density matrix:

$$
S(\rho)=-\operatorname{Tr}(\rho\log_2\rho)=-\sum_{j:\lambda_j>0}\lambda_j\log_2\lambda_j.
$$

The eigenvalues are probabilities, not amplitudes. We use bits throughout and define $0\log_2 0=0$ by continuity. A pure qubit has spectrum $(1,0)$ and entropy zero, even when its computational-basis measurement is random. The maximally mixed qubit has spectrum $(1/2,1/2)$ and entropy one.

Before continuing, review [density matrices and spectra](/book/online/physics-math-for-ai/qubit-density-spectrum-entropy-dephasing/). This chapter explains joint information and compression rather than merely recalculating a single-qubit spectrum.

## 2. A numerical spectrum you can inspect

Take $\rho=\operatorname{diag}(4/5,1/5)$. Its entropy is

$$
S(\rho)=h_2(1/5)=-\frac45\log_2\frac45-\frac15\log_2\frac15\approx0.721928.
$$

The label of the less common eigenstate carries more surprisal, $-\log_2(1/5)\approx2.321928$ bits, than the common label, $-\log_2(4/5)\approx0.321928$ bits. Entropy averages these two costs with their probabilities. It is neither the larger cost nor a fractional count of possible outcomes in a single trial.

A unitary rotation changes the eigenvectors but preserves this spectrum. Thus $S(U\rho U^\dagger)=S(\rho)$. Inspecting only the diagonal entries after that rotation can produce a different Shannon entropy; that calculation describes a measurement, not the state spectrum.

## 3. Joint systems require joint states

For a state $\rho_{AB}$, first take partial traces to obtain $\rho_A$ and $\rho_B$. Define

$$
S(A|B)=S(AB)-S(B),\qquad I(A:B)=S(A)+S(B)-S(AB).
$$

The vertical bar is part of a quantum information quantity. It does not instruct you to measure $B$, choose an outcome, and average the entropies of collapsed states. Those are separate procedures with an explicitly specified measurement.

For independent systems, $\rho_{AB}=\rho_A\otimes\rho_B$, the joint spectrum contains all products $\lambda_i\mu_j$. Expanding the logarithm of a product proves entropy additivity and gives $I(A:B)=0$.

## 4. Three preparations with identical local states

Compare these states in the basis $|00\rangle,|01\rangle,|10\rangle,|11\rangle$:

$$
\rho_{\mathrm{Bell}}=|\Phi^+\rangle\langle\Phi^+|,\quad
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},
$$

$$
\rho_{\mathrm{corr}}=\tfrac12|00\rangle\langle00|+\tfrac12|11\rangle\langle11|,
\qquad \rho_{\mathrm{ind}}=I_4/4.
$$

Every local state is $I_2/2$. Local entropy alone cannot distinguish these preparations.

| Preparation | Joint eigenvalues | $S(A)$ | $S(B)$ | $S(AB)$ | $S(A|B)$ | $I(A:B)$ |
|---|---|---:|---:|---:|---:|---:|
| Bell pair | $1,0,0,0$ | 1 | 1 | 0 | −1 | 2 |
| Correlated mixture | $1/2,1/2,0,0$ | 1 | 1 | 1 | 0 | 1 |
| Independent mixture | $1/4,1/4,1/4,1/4$ | 1 | 1 | 2 | 1 | 0 |

A useful visual experiment will keep the two local entropy bars fixed at one while changing the joint spectrum. The Bell state has off-diagonal coherence between $00$ and $11$; deleting it produces the correlated mixture. Filling all four diagonal entries equally produces the independent mixture. Each operation changes the joint information while leaving the local density matrices unchanged.

To interpolate between these examples, use $\rho(c,t)=c\rho_{\mathrm{corr}}(t)+(1-c)I_4/4$, where $\rho_{\mathrm{corr}}(t)$ has populations $1/2$ at $00,11$ and coherence $t/2$ between them. Both parameters lie in $[0,1]$. Its eigenvalues are $(1+c+2ct)/4$, $(1+c-2ct)/4$ and twice $(1-c)/4. The local states stay fixed. Thus $c=1,t=1$ gives the Bell pair, $c=1,t=0$ the correlated mixture, and $c=0$ the independent mixture. Changing $t$ removes coherence; changing $c$ also dilutes the classical correlation.

## 5. Negative conditional entropy has a precise meaning

For the Bell pair, the complete state is pure while either part is mixed. Therefore $S(A|B)=-1$. This does not mean a measurement returns a negative number of uncertain bits. Quantum conditional entropy differs from classical conditional Shannon entropy, which is nonnegative.

Negative conditional entropy certifies entanglement. The converse is false for general mixed states: entanglement may exist without a negative conditional entropy. For example, $c=1/2,t=1$ in the family above has joint spectrum $(5/8,1/8,1/8,1/8)$ and conditional entropy approximately $0.548795$ bit. Nevertheless, its partial transpose has an eigenvalue $-1/8$, proving entanglement. It also does not supply an extra bit that can be read locally. Measuring both halves in $Z$ produces one shared classical bit; the classical mutual information of those two outcomes is one, while the quantum mutual information before measurement is two.

The operational connection to state merging requires an asymptotic protocol, quantum side information and a specified resource convention. The value $-1$ alone is not a complete implementation or finite-copy guarantee.

## 6. Why a compression rate can be fractional

Suppose a memoryless source emits independent copies with average density matrix $\rho$. For $n$ emissions, the average state is $\rho^{\otimes n}$. Its eigenvectors are tensor products, and its eigenvalues are products of the one-copy eigenvalues.

An eigenlabel string with approximately the expected frequencies has probability near $2^{-nS(\rho)}$. Many such strings together carry almost all probability for sufficiently large $n$. Their span is the typical subspace, with dimension roughly $2^{nS(\rho)}$ on the exponential scale.

This leads to the Schumacher compression rate: under the standard independent-source, asymptotically vanishing-error formulation, rates above $S(\rho)$ are achievable. The coding task preserves quantum states, including relevant reference correlations; it is not simply a classical file of measured eigenlabels. A complete theorem must state the source and fidelity criterion. This intuition does not establish a finite-block error bound or a strong converse for every alternative criterion.

## 7. A small block is not the asymptotic limit

Return to eigenvalues $(4/5,1/5)$ and take only two copies. The four eigenlabel strings have the following exact probabilities:

| String | Probability | Probability accumulated in decreasing order |
|---|---:|---:|
| 00 | $16/25$ | $16/25$ |
| 01 | $4/25$ | $20/25$ |
| 10 | $4/25$ | $24/25$ |
| 11 | $1/25$ | 1 |

A rank-two projector onto $00,01$ captures weight $20/25=0.8$. Keeping three vectors captures $24/25=0.96$, but requires a three-dimensional retained subspace. Encoding that subspace in qubits requires $\lceil\log_2 3\rceil=2$ qubits before any failure handling.

The quantity $nS(\rho)\approx1.443856$ is not a physical fractional-qubit register. Projector weight is a useful finite-block statistic, but it must not be silently relabeled as an entanglement fidelity for an unspecified encoder and decoder. Tie choices between $01$ and $10$ do not change the retained weight.

## 8. Dephasing changes the task

Dephasing the Bell state in the local $Z$ basis removes its coherence and increases $S(AB)$ from zero to one. Dephasing $|+\rangle$ changes its entropy from zero to one. A source that always emits a known $|+\rangle$ can be reproduced without transmitting a quantum memory; a source of unknown independent maximally mixed qubits with a purifying reference cannot be treated the same way.

The average density matrix and the preservation task must be specified together. If the sole task were to prepare the average mixed state without preserving source-reference correlations, preparing a new local mixed state would be sufficient. That is a different problem from quantum source compression.

## 9. Exercises with worked answers

1. **Random outcomes, pure source.** A known $|+\rangle$ is measured in $Z$. Find state entropy and outcome entropy. **Answer:** zero and one bit respectively; the measurement basis matters only for the latter.
2. **Product rule.** Both qubits have spectrum $(4/5,1/5)$ and are independent. Find the joint spectrum and entropy. **Answer:** $(16,4,4,1)/25$ and $2h_2(1/5)\approx1.443856$ bits.
3. **Classical side information.** For the correlated mixture, knowing the $Z$ label of $B$ determines that of $A$. Compute $S(A|B)$. **Answer:** $1-1=0$; there is no negative conditional entropy here.
4. **Quantum versus measured correlations.** Compare the Bell pair before and after local $Z$ measurements. **Answer:** quantum mutual information is two bits before measurement; the classical outcome mutual information is one bit.
5. **Finite storage.** Can the rank-three subspace in Section 7 fit exactly into one qubit? **Answer:** no. An isometric encoding requires at least three dimensions; one qubit supplies two. This is a dimension argument, independent of the asymptotic entropy rate.
6. **A plausible but false conclusion.** A mixed entangled state has $S(A|B)\ge0$. Must it be separable? **Answer:** no. Negative conditional entropy is a sufficient entanglement witness, not a necessary characterization of all entanglement.

## 10. Connection to AI and further study

A latent representation can discard information about its input, so it is helpful to ask which correlations a representation must preserve. The quantum version makes the preservation criterion explicit: local statistics can agree while reference correlations differ. This is an analogy for careful task design, not a claim that an ordinary neural representation is a quantum state.

Next topics separate ensemble information, the Holevo bound, relative entropy and channel capacities. Finite-block coding, state merging and research-level converse arguments require additional proofs and experiments. This chapter supplies the worked foundation; its interactive controls and production reader integration are still being prepared.

For primary definitions, entropy inequalities and source coding, see [Watrous, Chapter 5](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf). The examples and explanations above were developed for this course.
