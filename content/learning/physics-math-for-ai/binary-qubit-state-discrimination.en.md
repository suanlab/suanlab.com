## Distinguishing two quantum states

Two laboratories promise to prepare one of two known states. You receive one physical system, not the classical description of which state was chosen. How often can you identify its label correctly?

This question turns a matrix distance into a decision problem. It also separates three ideas that are easy to confuse: identifying a label, extracting classical information, and estimating an unknown state by tomography.

Review [qubit density matrices](/book/online/physics-math-for-ai/qubit-density-spectrum-entropy-dephasing/) and [measurement effects and disturbance](/book/online/physics-math-for-ai/qubit-povm-instrument-backaction/) first. All logarithmic information quantities mentioned here use bits. Distance and success probability are dimensionless.

## 1. State the experiment before choosing a distance

Let the hidden label be $X\in\{0,1\}$. The source chooses $X=0$ with probability $\pi$ and prepares $\rho_0$; otherwise it prepares $\rho_1$. The states and prior are known, and you receive a single copy.

A two-outcome measurement has effects $M_0,M_1$ satisfying

$$
M_0\succeq0,\qquad M_1\succeq0,\qquad M_0+M_1=I.
$$

Interpret outcome $j$ as the guess $X=j$. Its average success probability is

$$
P_{\rm succ}=\pi\operatorname{Tr}(M_0\rho_0)
+(1-\pi)\operatorname{Tr}(M_1\rho_1).
$$

The prior is part of the task. Without touching the system, you can already succeed with probability $\max(\pi,1-\pi)$ by always choosing the more likely label. A proposed measurement should be compared with this baseline.

This is not tomography. Tomography usually uses many independently prepared copies to estimate a state. Here the alternatives are already specified; the unknown is the label of this preparation.

## 2. The weighted difference is the decision object

Write

$$
\Delta=\pi\rho_0-(1-\pi)\rho_1.
$$

Substituting $M_1=I-M_0$ gives

$$
P_{\rm succ}=1-\pi+\operatorname{Tr}(M_0\Delta).
$$

The problem is therefore to choose $0\preceq M_0\preceq I$. Diagonalize the Hermitian matrix $\Delta=\sum_i\lambda_i|i\rangle\langle i|$. Only diagonal entries of $M_0$ in this basis contribute to the objective:

$$
\operatorname{Tr}(M_0\Delta)=\sum_i\lambda_i\langle i|M_0|i\rangle.
$$

Each diagonal entry lies between zero and one. Set it to one for positive eigenvalues and zero for negative eigenvalues. A projector onto the positive eigenspace attains this bound. Zero eigenvalues do not change the objective, so an optimum need not be unique.

Since $\operatorname{Tr}\Delta=2\pi-1$, this argument yields the binary minimum-error formula

$$
P_{\rm opt}=\frac{1+\|\Delta\|_1}{2},\qquad
\|\Delta\|_1=\sum_i|\lambda_i|.
$$

The norm is the trace norm, not the Frobenius norm. Replacing the sum of absolute eigenvalues by the square root of their squared sum solves a different problem. The theorem and measurement construction are discussed in [Watrous, §3.1.1](https://cs.uwaterloo.ca/~watrous/TQI/TQI.3.pdf).

## 3. A qubit makes the geometry explicit

For $\rho_j=(I+r_j\cdot\sigma)/2$, physicality requires a real vector $r_j$ with $|r_j|\le1$. Define

$$
t=2\pi-1,\qquad v=\pi r_0-(1-\pi)r_1.
$$

Then

$$
\Delta=\frac{tI+v\cdot\sigma}{2},\qquad
\lambda_\pm=\frac{t\pm|v|}{2},\qquad
\|\Delta\|_1=\max(|t|,|v|).
$$

There are distinct cases:

| Condition | One optimal rule | Meaning |
| --- | --- | --- |
| $|v|>|t|$ | $M_0=(I+\widehat v\cdot\sigma)/2$ | A measurement improves on the prior-only rule |
| $t\ge|v|$, $t>0$ | $M_0=I$ | Always guess zero |
| $t\le-|v|$, $t<0$ | $M_0=0$ | Always guess one |
| $t=0$, $v=0$ | Any valid $M_0$ | No measurement improves on success $1/2$ |

At $|v|=|t|$ a zero eigenspace makes the optimum nonunique. Do not manufacture a preferred measurement direction from a zero vector. A program can select a canonical effect while reporting this freedom.

The geometry concerns a decision operator, not a physical density state. In particular, $\Delta$ can have negative eigenvalues. Its vector $v$ must not be presented as the Bloch vector of a normalized output state.

## 4. Work through a nonorthogonal pair

Take equal priors and

$$
|\psi_0\rangle=|0\rangle,\qquad
|\psi_1\rangle=\frac35|0\rangle+\frac45|1\rangle.
$$

The second state is normalized, and its overlap with the first is $3/5$. The states are not orthogonal, so perfect single-copy discrimination is impossible.

Their density matrices and weighted difference are

$$
\rho_0=\begin{pmatrix}1&0\\0&0\end{pmatrix},\quad
\rho_1=\begin{pmatrix}9/25&12/25\\12/25&16/25\end{pmatrix},\quad
\Delta=\begin{pmatrix}8/25&-6/25\\-6/25&-8/25\end{pmatrix}.
$$

The eigenvalues of $\Delta$ are $\pm2/5$. Thus $\|\Delta\|_1=4/5$ and $P_{\rm opt}=9/10$. One optimal effect is

$$
M_0=\begin{pmatrix}9/10&-3/10\\-3/10&1/10\end{pmatrix},\qquad M_1=I-M_0.
$$

Verify that $M_0^2=M_0$, $\operatorname{Tr}M_0=1$, and its eigenvalues are zero and one. The off-diagonal entries are essential.

| Prepared state | Outcome zero | Outcome one |
| --- | --- | --- |
| $\rho_0$ | $9/10$ | $1/10$ |
| $\rho_1$ | $1/10$ | $9/10$ |

The average success is $9/10$. Measuring in the computational basis instead, with zero interpreted as label zero, succeeds with

$$
\frac12\cdot1+\frac12\cdot\frac{16}{25}=\frac{41}{50}.
$$

The optimal measurement gains $2/25$ over that particular measurement and $2/5$ over blind equal-prior guessing. These are different comparisons.

## 5. A strong prior changes the answer

Consider mixed states with Bloch vectors $r_0=(0,0,1/5)$ and $r_1=(0,0,-1/5)$. They are different states. Nevertheless, with $\pi=9/10$,

$$
t=4/5,\quad v=(0,0,1/5),\quad
\lambda_+=1/2,\quad\lambda_-=3/10.
$$

Both eigenvalues are positive. Choosing $M_0=I$ gives success $9/10$, and no measurement can do better.

Measuring $Z$ and mechanically identifying its outcomes with the two labels succeeds only with $3/5$. This does not mean that performing a measurement inevitably reduces the best possible success: you can ignore the outcome and still guess zero. The mistake is a poorly chosen decision rule.

At $\pi=3/5$, the eigenvalues become $1/5$ and zero. Every effect $M_0=\operatorname{diag}(1,u)$ with $0\le u\le1$ is optimal. A tiny change of the prior can cross this boundary. Exact sign tests distinguish the cases; a tolerance that rounds a small eigenvalue to zero cannot certify the boundary.

## 6. Trace distance has an operational meaning

For normalized states, define

$$
D(\rho_0,\rho_1)=\frac12\|\rho_0-\rho_1\|_1.
$$

For equal priors only,

$$
P_{\rm opt}=\frac{1+D(\rho_0,\rho_1)}2.
$$

For qubits, $D=|r_0-r_1|/2$. Equal states have distance zero; orthogonal supports have distance one. With unequal priors, return to the weighted difference rather than substituting the unweighted distance into the equal-prior formula.

A physical channel cannot increase trace distance. Intuitively, a measurement after a channel could already have been implemented as a measurement on the original system with the channel included in the apparatus. Processing cannot improve the optimum over all originally available measurements.

For the worked pair, completely dephasing in the $Z$ basis removes the off-diagonal entries. The remaining distance is $16/25$, so optimal success after dephasing is $41/50$. The information lost by this channel is exactly what the coherent measurement used to improve the result.

To see why complex phases matter, compare $|+i\rangle=(|0\rangle+i|1\rangle)/\sqrt2$ and $|-i\rangle$. A $Z$ measurement gives identical distributions; a $Y$ measurement distinguishes them perfectly. Equal distributions for one measurement do not prove equal quantum states.

## 7. Keep fidelity conventions separate

This book uses squared Uhlmann fidelity:

$$
F_2(\rho,\sigma)=\left(\operatorname{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\right)^2.
$$

For two qubits,

$$
F_2=\operatorname{Tr}(\rho\sigma)+2\sqrt{\det\rho\det\sigma}
=\frac{1+r\cdot s+\sqrt{(1-|r|^2)(1-|s|^2)}}2.
$$

For pure states, this is the squared overlap. Watrous uses the unsquared fidelity in [§3.2](https://cs.uwaterloo.ca/~watrous/TQI/TQI.3.pdf); translate conventions before importing an inequality.

With this book's convention, the Fuchs–van de Graaf bounds read

$$
1-\sqrt{F_2}\le D\le\sqrt{1-F_2}.
$$

For our pair, $F_2=9/25$, its square root is $3/5$, and $D=4/5$. The bounds give $2/5\le4/5\le4/5$. Fidelity is a similarity, trace distance is a distinguishability, and neither equals a general unequal-prior success probability.

Raw amplitude vectors must first define normalized density states. Scaling a vector by a nonzero scalar changes neither state nor these quantities. Do not silently apply formulas for normalized states to raw unnormalized amplitudes.

## 8. A success probability is not a number of bits

In the worked example the optimal binary decision makes symmetric errors with probability $1/10$. The mutual information of this particular classical input/output channel is

$$
I(X:Y)=1-h_2(1/10)\approx0.5310044\text{ bits},
$$

where $h_2(p)=-p\log_2p-(1-p)\log_2(1-p)$. A success probability of $0.9$ does not mean $0.9$ bits were extracted.

The average input state has eigenvalues $4/5$ and $1/5$. Its entropy is approximately $0.7219281$ bits. The relationship between this entropy, the ensemble label, accessible information and the Holevo bound requires its own argument. Computing the success probability alone does not solve that optimization or determine a channel capacity.

## 9. Exercises with explanations

**1. Identical states with an unequal prior.** If $\rho_0=\rho_1$ and $\pi=3/4$, what is optimal? **Explanation:** $\Delta=(1/2)\rho_0$, with trace norm $1/2$. Always guessing zero attains $3/4$. Equality of the states removes measurement information, not prior information.

**2. Check the measurement.** Multiply the worked $M_0$ by itself. **Explanation:** the top-left entry is $81/100+9/100=9/10$, and the off-diagonal entry is $-27/100-3/100=-3/10$. The remaining entries also reproduce $M_0$. This is a rank-one projector despite having fractional entries.

**3. Nonuniqueness at a threshold.** Use the mixed pair at $\pi=3/5$. **Explanation:** $\Delta=\operatorname{diag}(1/5,0)$. Its objective depends on the first diagonal entry of $M_0$ but not the second. An optimizer must not report a uniquely determined second eigenvector decision.

**4. A measurement that misses a phase.** Compare the two $Y$ eigenstates using $Z$. **Explanation:** both yield $(1/2,1/2)$. This demonstrates a limitation of that measurement, while the optimal measurement still succeeds with probability one.

**5. A difference below ordinary display resolution.** Compare $r_0=(0,0,1)$ and $r_1=(0,0,1-10^{-30})$ with equal priors. **Explanation:** $D=1/(2\cdot10^{30})$ and the success advantage over $1/2$ is $1/(4\cdot10^{30})$. A rounded display may show $0.5$ for both success and baseline. The exact positive advantage remains meaningful; display equality is not a proof of state equality.

**6. Change the experimental budget.** Would two copies necessarily have the same optimum as one? **Explanation:** the alternatives become $\rho_0^{\otimes2}$ and $\rho_1^{\otimes2}$, and collective measurements are allowed unless restricted. The single-copy formula still describes the appropriate larger matrices, but its value must be recomputed. A one-copy demonstration does not establish an asymptotic error exponent.

## 10. Connections to AI and further research

The decision operator plays a role analogous to a likelihood comparison in binary classification, while the admissible measurements constrain the observable features. Priors, loss functions, single-copy versus repeated observations, and restricted versus collective measurements define different learning tasks.

A numerical qubit experiment can verify a finite example, inspect the optimal effect, and expose a biased-prior threshold. It does not prove a multi-hypothesis discrimination theorem, characterize channel discrimination with reference assistance, or solve the asymptotic collective-measurement problem. Those extensions require additional mathematics and experiments.

Before moving on, explain why a pure-state overlap of $3/5$ permits success $9/10$ without permitting perfect discrimination, and why that success is still different from classical information measured in bits.
