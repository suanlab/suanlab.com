## 1. Copying a label is different from copying an unknown state

A classical bit can be copied with a controlled NOT: $|0\rangle|0\rangle\mapsto|00\rangle$ and $|1\rangle|0\rangle\mapsto|11\rangle$. The same device does not copy every superposition. The no-cloning theorem concerns one fixed physical operation that would copy each state in an unknown set. It does not prevent preparing many copies of a known state from its classical description.

The distinction matters for communication. An eavesdropper cannot generally keep a perfect copy of an unknown nonorthogonal signal while delivering another perfect copy to the receiver. This restriction motivates quantum key distribution, but it is not by itself a complete security proof.

Review [measurement and backaction](/book/online/physics-math-for-ai/qubit-povm-instrument-backaction/) and [circuit order and interference](/book/online/physics-math-for-ai/finite-qubit-circuit-order-interference/).

## 2. Derive the obstruction from inner products

Suppose a fixed blank state $|b\rangle$ and unitary $U$ could clone both $|\psi\rangle$ and $|\phi\rangle$. If no additional machine register is present, preservation of inner products would require

$$
s=\langle\psi|\phi\rangle
=\langle\psi\psi|\phi\phi\rangle=s^2.
$$

Allowing input-dependent global phases still gives $|s|=|s|^2$, which requires $|s|=0$ or $1$. Thus distinct nonorthogonal rays cannot both be cloned by this device. Orthogonal states do not encounter this obstruction; identical rays do not represent two distinct physical alternatives.

A discarded machine register does not remove the obstruction. Exact pure clones force the output to factor as $|\psi\psi\rangle|e_\psi\rangle$. An isometric dilation would imply $|s|=|s|^2|\langle e_\psi|e_\phi\rangle|\le|s|^2$. This is impossible for $0<|s|<1$. The argument applies to deterministic exact physical cloning, not approximate cloning with a specified error.

## 3. Test CNOT on a superposition

For $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$ and a blank $|0\rangle$,

$$
\operatorname{CNOT}(|\psi\rangle|0\rangle)=\alpha|00\rangle+\beta|11\rangle.
$$

The desired product clone would instead be

$$
|\psi\rangle^{\otimes2}=\alpha^2|00\rangle+\alpha\beta|01\rangle+\alpha\beta|10\rangle+\beta^2|11\rangle.
$$

For $|+\rangle$, the actual output is a Bell pair, not two $|+\rangle$ states. Each actual marginal is $I_2/2$; each desired marginal is $|+\rangle\langle+|$. A picture showing two correlated output labels can hide this difference. Compare density matrices, not just computational-basis sample strings.

For normalized complex amplitudes, each actual marginal is $\operatorname{diag}(|\alpha|^2,|\beta|^2)$. Its squared fidelity with the intended pure copy is $|\alpha|^4+|\beta|^4$, equal to $1/2$ for equal magnitudes. Relative phase has been lost locally.

## 4. The four BB84 signal states

Alice chooses a uniform random bit and an independent uniform random basis. The basis is either $Z$ or $X$:

| Basis | Bit 0 | Bit 1 |
|---|---|---|
| $Z$ | $\lvert0\rangle$ | $\lvert1\rangle$ |
| $X$ | $\lvert+\rangle$ | $\lvert-\rangle$ |

States within each basis are orthogonal. Across bases, every squared overlap is $1/2$. Bob independently chooses a uniform $Z$ or $X$ measurement. If his basis matches Alice's, an ideal noiseless transmission gives the correct bit. If it differs, his result is uniform and does not reveal Alice's bit.

The basis is eventually public; the raw key bit is not publicly announced at every retained position. The timing of announcements is part of the protocol.

## 5. Sifting changes the denominator

Alice and Bob announce their bases over an authenticated public classical channel and retain positions with matching bases. With the uniform independent choices above,

$$
P(\mathrm{sift})=\tfrac12.
$$

The sifted quantum bit error rate is

$$
q=\frac{P(\mathrm{Bob\ bit}\ne\mathrm{Alice\ bit},\ \mathrm{sift})}{P(\mathrm{sift})}.
$$

It is not the error fraction over all transmissions, which would include positions that the protocol discards. In a visualization, show the transmission stream first, then the sifting filter, then the retained error count. Keep the denominator visible.

Authentication prevents an active adversary from impersonating the parties in the classical discussion. A public channel that anybody can modify without detection is insufficient. Privacy of that discussion is a different requirement from authentication.

## 6. Enumerate an intercept-resend attack

In the ideal attack model, Eve intercepts an attacked signal, chooses a basis, measures it projectively and sends Bob the measured eigenstate. First suppose she attacks every signal and chooses $Z/X$ uniformly.

Condition on sifting. Eve uses Alice's basis with probability $1/2$: she learns the bit and introduces no error. With probability $1/2$ she uses the other basis: her bit is independent of Alice's, and Bob's matching-basis measurement is wrong with probability $1/2$. Therefore

$$
q=\tfrac12\cdot0+\tfrac12\cdot\tfrac12=\tfrac14.
$$

The numerical model explicitly enumerates Alice's basis/bit, Bob's basis/bit and Eve's basis/outcome. Including unattacked branches gives 80 rows, some with zero weight. Each row retains its exact probability; the total is one before sifting and one half after sifting.

## 7. Attack fraction, basis bias and Eve's information

If Eve attacks each signal independently with probability $f$, the ideal sifted QBER is $f/4$. After the public basis announcement, she knows which of her attacked positions used the correct basis. Half of attacked sifted bits are known exactly; the rest give no information in this model.

Consequently, her conditional information is $f/2$ bit per sifted bit, while her optimal bit-guess success is $1/2+f/4$. For $f=2/5$, these values are $q=1/10$, information $1/5$ bit and guess success $3/5$. Neither the information nor guess success is a secret-key rate.

Let Eve choose $Z$ with probability $e$. Then

$$
q_Z=\frac{f(1-e)}2,\qquad q_X=\frac{fe}2,\qquad q=\frac{q_Z+q_X}2=\frac f4.
$$

Choosing only $Z$ hides the disturbance in retained $Z$ positions but gives $q_X=f/2$. Basis-specific checks reveal structure that an aggregate average can conceal.

## 8. A sample can miss errors

For $m$ independent sampled sifted positions in this IID attack model,

$$
P(\mathrm{at\ least\ one\ sample\ error})=1-(1-q)^m.
$$

Under full intercept-resend and $m=20$, this is $1-(3/4)^{20}\approx0.996829$. Even then, a zero-error sample is possible. For $m=0$, the detection probability is zero.

This is an attack-conditional likelihood, not a posterior probability that Eve exists and not a general confidence certificate for secrecy. Finite sampling without replacement, background noise, correlated attacks and adversarial quantum memory require different analyses. Publicly revealed test bits must also be removed from the final key material.

## 9. Exercises with worked answers

1. **Two nonorthogonal states.** Can one deterministic device exactly clone both $|0\rangle$ and $|+\rangle$? **Answer:** no; their overlap magnitude is $1/\sqrt2$, which does not equal its square.
2. **CNOT output.** Input $|+\rangle|0\rangle$. What is either output marginal? **Answer:** $I_2/2$, whereas a perfect clone would have marginal $|+\rangle\langle+|$.
3. **Sifting.** What fraction is retained with independent uniform bases? **Answer:** $1/2$ in expectation, before parameter-estimation disclosures and other processing.
4. **Partial attack.** If $f=2/5$, find ideal sifted QBER. **Answer:** $1/10$; the unconditioned probability of a sifted error is $1/20$.
5. **Biased Eve.** Set $e=1$, $f=1$. **Answer:** $q_Z=0$, $q_X=1/2$, average $q=1/4$; checking only the $Z$ subset would miss the disturbance.
6. **No errors observed.** Does this imply a secure key? **Answer:** no. The sample may miss errors, and this model supplies neither parameter-estimation bounds nor a complete security proof.

## 10. What remains before this becomes a security protocol?

A full key-distribution analysis must specify device assumptions, authenticated discussion, parameter estimation, error-correction leakage, privacy amplification and a security definition with finite-key bounds. The ideal intercept-resend calculation proves none of those on its own. No universal QBER threshold or final key length is inferred here.

For AI research, the transferable habit is to state the observation model and conditioning event before comparing information and accuracy. The numerical model is an exact educational enumeration; its Worker, interactive controls and production reader registration remain to be implemented.

The historical primary source is [Bennett and Brassard's 1984 paper, archived in 2020](https://arxiv.org/abs/2003.06557). Its archive date is not the protocol's invention date. The calculations and examples above are developed for this course.
