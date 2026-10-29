## Distinguish uncertainty from model error

A source can be intrinsically unpredictable even when we model it perfectly. Conversely, an easy-to-predict source can be described by a poor model. Entropy measures uncertainty under a distribution; cross-entropy scores another model on that distribution; KL divergence measures the excess expected log cost. Confusing them makes it difficult to interpret a training loss.

We use Bernoulli distributions $P=(p,1-p)$ and $Q=(q,1-q)$ over the same two outcomes. All logarithms are natural, so information is measured in nats. Divide by $\log2$ to convert to bits.

## Define the three quantities

| Quantity | Formula and interpretation |
| --- | --- |
| Surprisal | $-\log P(x)$ for a realized outcome |
| Entropy | $H(P)=-\sum_xP(x)\log P(x)$ |
| Cross-entropy | $H(P,Q)=-\sum_xP(x)\log Q(x)$ |
| KL divergence | $D_{KL}(P\Vert Q)=\sum_xP(x)\log(P(x)/Q(x))$ |

Use the continuous extension $0\log0=0$. If $P(x)>0$ but $Q(x)=0$, the corresponding cross-entropy and KL are infinite. The lab displays infinity explicitly. Silently clipping probabilities changes the objective and should be documented when used in an implementation.

## Derive the decomposition

Subtract the entropy from cross-entropy and combine logarithms:

$$
H(P,Q)-H(P)=\sum_xP(x)\log\frac{P(x)}{Q(x)}=D_{KL}(P\Vert Q).
$$

For compatible support, the inequality $\log z\le z-1$ implies nonnegativity of KL by applying it to $Q(x)/P(x)$ and summing. Equality holds when distributions agree on the relevant support. KL is generally asymmetric and does not satisfy the triangle inequality, so it is not an ordinary distance metric.

## Work the Bernoulli example

Take $p=0.75,q=0.5$. Then

$$
H(P)=-0.75\log0.75-0.25\log0.25\approx0.562335,
$$

while $H(P,Q)=\log2\approx0.693147$. The difference is approximately 0.130812 nat. If $q$ becomes 0.75, KL becomes zero but entropy remains 0.562335. A matched model does not remove uncertainty in the source.

If $q=0$, a success that occurs with probability 0.75 receives zero model probability. Its cost is infinite. A finite plotting window cannot represent that value, so the curve remains the finite entropy curve while the separate cost readout and warning show the support mismatch.

## Understand the plotted object

Blue shows Bernoulli entropy as a function of the true probability. It reaches $\log2$ at $p=1/2$ and vanishes at deterministic endpoints. The orange marker locates the current true distribution. Changing $q$ changes cross-entropy and KL, not the entropy curve. The picture does not claim the entropy of the source changes merely because a model changes.

## Connect to prediction and learning

For a dataset, average negative log predicted probability is an empirical cross-entropy score. Under suitable sampling assumptions it estimates an expectation under the data-generating distribution. Minimizing that expectation over all admissible distributions would match the true distribution, but a restricted model class, finite data and imperfect optimization can prevent it.

Log-score improvements require a common evaluation protocol and outcome representation. Differential entropy for continuous variables has coordinate dependence and can be negative, unlike finite discrete entropy. One should not transfer every discrete statement unchanged to a density without checking the measure and transformation.

Information entropy is also not automatically thermodynamic entropy. A physical interpretation requires specified microscopic states, probabilities, energy, temperature and units. The statistical-mechanics unit supplies one concrete equilibrium model rather than treating all training losses as physical energy.

## Research checks and exercises

1. For $P=Q=(1/2,1/2)$, compute all three quantities. **Answer:** entropy and cross-entropy are $\log2$, KL is zero.
2. Give a support mismatch with infinite KL in one direction and finite KL in the other.
3. Explain why a lower empirical loss on reused training data does not establish improved held-out log score.
4. If probabilities are clipped at an epsilon, state exactly which objective has changed and test sensitivity to that epsilon.

For the historical formulation, see Shannon’s [A Mathematical Theory of Communication](https://www.nokia.com/bell-labs/publications-and-media/publications/a-mathematical-theory-of-communication/). The Bernoulli calculations and support examples above are independently worked examples for this textbook.
