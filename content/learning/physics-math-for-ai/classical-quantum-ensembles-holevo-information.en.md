## 1. Which variable are we trying to learn?

Alice chooses a classical label $X=x$ with probability $p_x$ and prepares the quantum state $\rho_x$. Bob receives the quantum system $Q$, but not the label. The ensemble is the entire list $\{p_x,\rho_x\}$, not just the average state. Different preparation lists can have the same average and different correlations with the label.

For example, equally likely $|0\rangle,|1\rangle$ and equally likely $|+\rangle,|-\rangle$ both average to $I_2/2$. A $Z$ measurement perfectly identifies the first label and tells us nothing about the second. The receiver's measurement must be specified before discussing its classical information.

Review [quantum measurement and backaction](/book/online/physics-math-for-ai/qubit-povm-instrument-backaction/) and [density matrices and spectra](/book/online/physics-math-for-ai/qubit-density-spectrum-entropy-dephasing/).

## 2. Keep the classical label as a register

Introduce orthogonal label states $|x\rangle_X$ and write the classical-quantum state

$$
\omega_{XQ}=\sum_x p_x|x\rangle\langle x|\otimes\rho_x,
\qquad \bar\rho=\sum_x p_x\rho_x.
$$

The matrix is block diagonal in $X$. It is not the coherent superposition $\sum_x\sqrt{p_x}|x\rangle|\psi_x\rangle$. In the classical register, alternatives are recorded as distinct labels without coherence between those alternatives.

The eigenvalues of block $x$ are $p_x\lambda_{xj}$, where $\lambda_{xj}$ are eigenvalues of $\rho_x$. Expanding their logarithms gives

$$
S(XQ)=H(p)+\sum_xp_xS(\rho_x).
$$

Tracing out $Q$ leaves the classical distribution, so $S(X)=H(p)$. Tracing out $X$ leaves $\bar\rho$, so $S(Q)=S(\bar\rho)$.

## 3. Holevo information is a joint information quantity

Substituting the three entropies into $I(X:Q)$ gives

$$
\chi=I(X:Q)=S(\bar\rho)-\sum_xp_xS(\rho_x).
$$

This expression subtracts the average internal mixedness of the prepared states from the entropy of their average. It measures the quantum system's correlation with the classical label. It is not simply the entropy of the average unless every prepared state is pure.

For a binary qubit ensemble, $0\le\chi\le\min\{H(X),1\}$. If the two prepared states are identical, $\chi=0$ even when $H(X)=1$. If the labels are certain, both label information and Holevo information vanish.

## 4. A measurement produces an ordinary probability table

For POVM effects $M_y$, the measured channel is

$$
p(y|x)=\operatorname{Tr}(M_y\rho_x),\qquad p(x,y)=p_xp(y|x).
$$

The classical mutual information is

$$
I(X:Y)=\sum_{x,y:p(x,y)>0}p(x,y)\log_2\frac{p(x,y)}{p_xp_y}.
$$

For binary outcomes, it is often easier to compute $H(Y)-H(Y|X)$. Zero-probability terms contribute zero by continuity; a label with zero prior contributes nothing. The probability table, including both row and column marginals, is the best place to check a numerical implementation.

Visualize this as two preparation blocks flowing into two outcome columns. Changing the measurement redistributes probability among the columns. It does not change the preparation priors or the average-state entropy.

## 5. Work through a nonorthogonal pair

Take equal priors and

$$
|\psi_0\rangle=|0\rangle,\qquad
|\psi_1\rangle=\tfrac35|0\rangle+\tfrac45|1\rangle.
$$

The average density matrix is

$$
\bar\rho=\frac1{25}\begin{pmatrix}17&6\\6&8\end{pmatrix}.
$$

Its trace is one and determinant is $4/25$, so its eigenvalues are $4/5,1/5$. Since each component state is pure, $\chi=h_2(1/5)\approx0.721928$ bit.

A minimum-error measurement uses

$$
M_0=\begin{pmatrix}9/10&-3/10\\-3/10&1/10\end{pmatrix},\qquad M_1=I-M_0.
$$

| Preparation | $p(Y=0|X)$ | $p(Y=1|X)$ |
|---|---:|---:|
| $X=0$ | $9/10$ | $1/10$ |
| $X=1$ | $1/10$ | $9/10$ |

The output marginal is uniform. Thus the selected measurement yields $I(X:Y)=1-h_2(1/10)\approx0.531004$ bit. Its correct-guess probability is $9/10$, which has a different meaning and unit.

## 6. Compare the measurement without changing the source

Measure the same ensemble in $Z$. The outcome-zero probabilities are $1$ and $9/25$. Hence $p(Y=0)=17/25$, and

$$
I_Z(X:Y)=h_2(17/25)-\tfrac12h_2(9/25).
$$

This gives a second, independently reproducible value. The source's Holevo information remains $h_2(1/5)$; the measurement changes only the accessible classical table. The average matrix's diagonal entries $17/25,8/25$ give $H(Y)$ for this measurement, not the spectrum entropy of $\bar\rho$.

A useful control will switch between the $Z$ and minimum-error measurements while keeping the ensemble fixed. Beside each table, show $H(Y)$, $H(Y|X)$, their difference, and the fixed $\chi$ bar.

## 7. Optimizing guessing and optimizing information are different

The accessible information is $I_{\mathrm{acc}}=\sup_{\{M_y\}}I(X:Y)$. A Helstrom measurement optimizes correct-guess probability. This fact alone does not prove that it maximizes mutual information.

An explicit example makes the distinction visible. Let the two states have Bloch vectors $(0,0,1/5)$ and $(0,0,-1/5)$, with priors $9/10,1/10$. Always guessing the first label is minimum-error optimal, with success $9/10$. Its canonical one-outcome measurement has mutual information zero.

A $Z$ measurement has different distributions $(3/5,2/5)$ and $(2/5,3/5)$, so it has positive mutual information. The receiver can still ignore its outcomes when making a minimum-error guess. Obtaining information and changing the most likely decision are separate issues.

## 8. Why the Holevo bound holds

Measurement is a quantum channel from $Q$ to a classical register $Y$. Applying this channel to $\omega_{XQ}$ cannot increase its mutual information with $X$. Consequently,

$$
I(X:Y)\le I_{\mathrm{acc}}\le\chi.
$$

The first inequality follows from the definition of a supremum; the second is Holevo's theorem. A full proof uses data processing for relative entropy. Numerical agreement for our examples illustrates the theorem but does not prove it for all ensembles or POVMs.

The chain gives an achievable value and an upper bound. A positive gap for one selected measurement does not establish the exact accessible information, nor does it prove no other measurement closes that gap.

## 9. Exercises with worked answers

1. **Identical states.** Both labels prepare $I_2/2$, with equal priors. **Answer:** $S(\bar\rho)=1$ and the average component entropy is one, so $\chi=0$. Every measurement table has identical rows.
2. **Orthogonal pure states.** Equal priors on $|0\rangle,|1\rangle$. **Answer:** $\chi=1$ and the $Z$ measurement achieves one bit with zero error.
3. **Average versus label.** Equal priors on $|+\rangle,|-\rangle$, measured in $Z$. **Answer:** the average state has entropy one, but both table rows are $(1/2,1/2)$, so this measurement gives zero information.
4. **Joint probability.** In Section 5, find $p(X=0,Y=1)$. **Answer:** $(1/2)(1/10)=1/20$, not $1/10$; include the prior.
5. **Decision versus information.** Why can the biased example give information without changing the optimal guess? **Answer:** posterior probabilities change but remain on the same side of the decision threshold. Mutual information detects dependence; the optimal label need not switch.
6. **An unjustified equality.** Does evaluating a minimum-error measurement prove $I_{\mathrm{acc}}=I(X:Y)$? **Answer:** no. The objective functions differ. An additional optimization theorem or certificate is needed.

## 10. Research and AI connections

Distinguish label entropy, representation information, task accuracy and a representation's theoretical information limit. The quantum ensemble provides a precise example where these quantities differ. An analogy to a neural classifier does not turn ordinary feature vectors into quantum density matrices.

Channel capacity additionally specifies allowed codewords, many channel uses, collective measurements and any shared resources. The one-use table above does not establish a channel coding theorem. Dense coding changes the shared-resource assumptions; it does not contradict the bound for this unassisted ensemble task.

The numerical foundation evaluates selected measurements with directed entropy intervals. Interactive controls, accessible-information optimization, coding proofs and production reader integration remain separate work. Primary definitions and the bound appear in [Watrous, Chapter 5, Section 5.3.3](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf).
