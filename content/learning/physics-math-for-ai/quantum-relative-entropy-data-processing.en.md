## 1. Comparing two hypotheses about a state

Suppose the actual state is $\rho$, while a model assumes $\sigma$. Umegaki relative entropy compares these ordered states. With base-two logarithms,

$$
D(\rho\|\sigma)=\operatorname{Tr}[\rho(\log_2\rho-\log_2\sigma)]
$$

when $\operatorname{supp}(\rho)\subseteq\operatorname{supp}(\sigma)$. Otherwise it is $+\infty$. It is nonnegative and vanishes exactly when the two normalized states agree. It is generally asymmetric and is not a metric.

A quantum support is the span of eigenvectors with positive eigenvalues. It is a subspace, not the set of nonzero matrix entries. Review [density matrices and spectra](/book/online/physics-math-for-ai/qubit-density-spectrum-entropy-dephasing/) before computing matrix logarithms.

## 2. A matrix logarithm is spectral

If $\sigma=\sum_j s_j|s_j\rangle\langle s_j|$ is full rank, then

$$
\log_2\sigma=\sum_j\log_2(s_j)|s_j\rangle\langle s_j|.
$$

Do not take logarithms of individual matrix entries. Entries can be negative or complex even when the operator is a valid density matrix. For a singular matrix the logarithm is evaluated on its support; the support condition determines whether the divergence is finite.

An equivalent calculation is

$$
D(\rho\|\sigma)=-S(\rho)-\sum_j\langle s_j|\rho|s_j\rangle\log_2s_j.
$$

The weights in the last sum are probabilities for measuring $\rho$ in the eigenbasis of $\sigma$. They are generally not eigenvalues of $\rho$.

## 3. Recover classical KL divergence when states commute

Take $\rho=\operatorname{diag}(4/5,1/5)$ and $\sigma=I_2/2$. These operators commute and share an eigenbasis. Thus

$$
D(\rho\|I_2/2)=\tfrac45\log_2\tfrac85+\tfrac15\log_2\tfrac25
=1-h_2(1/5)\approx0.278072.
$$

For commuting states the expression reduces to classical KL divergence between their eigenvalue distributions in a common basis. Merely reading the diagonals of arbitrary noncommuting states does not have this property.

Also note $D(I_2/2\|\rho)=-1-\tfrac12\log_2[(4/5)(1/5)]\approx0.321928$. Reversing the order changes the answer.

## 4. Nonzero overlap does not guarantee finite divergence

Consider $\rho=|0\rangle\langle0|$ and $\sigma=|+\rangle\langle+|$. Their overlap squared is $1/2$, but their one-dimensional supports are distinct. Neither support is contained in the other. Therefore $D(\rho\|\sigma)=+\infty$.

This does not imply one copy can be discriminated perfectly. Their trace distance is $1/\sqrt2$, below one. Divergence and minimum-error guessing quantify different tasks. A zero eigenvalue in the reference can make the divergence infinite without making the two states orthogonal.

If $\sigma$ is pure, a normalized qubit $\rho$ has support contained in its support only when $\rho=\sigma$. Near-pure full-rank reference states, however, give finite values. Numerical tolerances must not replace exact support classification.

## 5. A finite noncommuting example

Use

$$
\rho=\tfrac12(I+\tfrac35Z),\qquad
\sigma=\tfrac12(I+\tfrac45X).
$$

The first state has eigenvalues $4/5,1/5$ in the $Z$ basis. The second has eigenvalues $9/10,1/10$ in the $X$ basis. Since $\rho$ has zero $X$ expectation, its probability in either eigenvector of $\sigma$ is $1/2$. Consequently,

$$
D(\rho\|\sigma)=-h_2(1/5)-\tfrac12\log_2(9/100)
\approx1.0150374993.
$$

This is a useful test of noncommuting matrix logarithms: the diagonal of $\sigma$ in the computational basis is uniform, but substituting that diagonal would incorrectly return $1-h_2(1/5)$.

## 6. Process both hypotheses through the same channel

For every completely positive trace-preserving channel $\Phi$, data processing states

$$
D(\Phi(\rho)\|\Phi(\sigma))\le D(\rho\|\sigma).
$$

The same channel must act on both states. Otherwise one could replace only the reference with an unrelated state and change the comparison arbitrarily. The theorem includes partial traces, measurements and dephasing, but does not say the entropy of each individual state always decreases.

For complete $Z$ dephasing,

$$
\mathcal D_Z\!\begin{pmatrix}a&b\\b^*&1-a\end{pmatrix}
=\begin{pmatrix}a&0\\0&1-a\end{pmatrix}.
$$

It removes coherences in a specified basis. In Bloch coordinates, $(x,y,z)$ becomes $(0,0,z)$.

## 7. Compute the loss instead of just drawing a descending bar

For Section 5, $\rho$ is unchanged and $\sigma$ becomes $I_2/2$. Therefore

$$
D_{\mathrm{after}}=1-h_2(1/5)\approx0.2780719051,
\qquad D_{\mathrm{before}}-D_{\mathrm{after}}=\log_2(5/3)\approx0.7369655942.
$$

| Pair | Before $Z$ dephasing | After $Z$ dephasing |
|---|---:|---:|
| Noncommuting pair in Section 5 | $1.0150374993$ | $0.2780719051$ |
| $\lvert0\rangle$ versus $\lvert+\rangle$ | $+\infty$ | 1 |
| $\operatorname{diag}(4/5,1/5)$ versus $I_2/2$ | $0.2780719051$ | $0.2780719051$ |

For the singular pair, dephasing sends $|+\rangle\langle+|$ to $I_2/2$ while preserving $|0\rangle\langle0|$. Its divergence becomes one bit. Do not display an ordinary finite subtraction for an infinite input value; retain an explicit mathematical-infinity state.

A visualization should show the two matrices and supports, alongside two Bloch vectors and separate before/after values. A support warning explains an infinite value more accurately than a clipped vertical bar.

## 8. Equality concerns a pair, not a global inverse

If both states are diagonal in $Z$, dephasing preserves them and gives equality. Yet dephasing destroys the coherence of $|+\rangle$, so it is not invertible on all qubit states. Equality for one pair does not establish a global inverse of the channel.

Recovery theorems characterize equality more precisely under their stated support conditions. A recovery map depends on the reference and channel. This chapter does not compute a Petz map or prove approximate recovery bounds. Those are research extensions, not consequences certified by a pair of matching rounded numbers.

If two numerical intervals overlap, their true difference may be unresolved. Overlap alone is not a proof of equality. The diagonal example has an exact structural explanation, independent of floating-point rounding.

## 9. Exercises with worked answers

1. **Direction matters.** Compare the two divergences in Section 3. **Answer:** approximately $0.278072$ and $0.321928$; a metric would require symmetry.
2. **Support versus overlap.** Why is $D(|0\rangle\langle0|\||+\rangle\langle+|)$ infinite? **Answer:** distinct one-dimensional supports do not satisfy containment, although their vectors overlap.
3. **Entrywise error.** What answer follows from replacing Section 5's $\sigma$ by its diagonal? **Answer:** $0.278072$, which is the processed pair's divergence, not the original value $1.015037$.
4. **One channel.** Can data processing be applied after dephasing only $\rho$? **Answer:** not in that form; both hypotheses must undergo the same channel.
5. **Equality without invertibility.** Give a pair preserved by dephasing. **Answer:** any two $Z$-diagonal states; the channel still fails to recover arbitrary coherences.
6. **Near the boundary.** Let $\rho=|1\rangle\langle1|$ and $\sigma=(I+(1-10^{-30})Z)/2$. **Answer:** the reference is full rank and $D=\log_2(2\cdot10^{30})$, a large finite number. Rounding its smaller eigenvalue to zero would misclassify the problem.

## 10. From information loss to research questions

Classical representation learning often uses KL-based losses. Quantum relative entropy extends the comparison to operators with incompatible eigenbases. This comparison helps explain why a basis-dependent diagonal approximation can throw away the very difference a model is supposed to capture.

Divergence has an operational role in asymptotic hypothesis testing under specified error conventions. It is not itself a one-copy error probability. General coding, hypothesis-testing exponents, recovery guarantees and channel capacities need additional theorems. The qubit calculation here supplies concrete examples rather than those complete protocols.

The numerical foundation retains exact support classification and directed logarithm enclosures. Interactive controls and reader integration remain to be built. See the primary definitions and data-processing theorem in [Watrous, Chapter 5](https://cs.uwaterloo.ca/~watrous/TQI/TQI.5.pdf).
