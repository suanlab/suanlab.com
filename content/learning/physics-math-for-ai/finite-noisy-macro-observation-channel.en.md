## A noisy reading is a new random variable

Imagine a system occupying one of two labeled boxes. A sensor lights either a blue or an orange lamp. Sometimes the lamp correctly identifies the box; sometimes it reports the other color. The system state and the sensor reading are now two different random variables. A changing lamp does not by itself tell us that the underlying system changed state.

In a deterministic grouping, one input state always produces one label. Here the same input can produce several labels with specified probabilities. This adds an ingredient to the observation model: a conditional law for the instrument. We will follow preparation, observation and inference separately rather than treating them as interchangeable probability vectors.

The central question is not simply how unpredictable the lamp is. It is how much seeing that lamp helps identify the original state. A perfectly random lamp can tell us nothing; a lamp that always reverses the labels can still identify the state exactly once its rule is known.

## Specify preparation and instrument separately

Give the input states nonnegative weights w with a positive total. These weights specify the preparation p:

$$
W=\sum_iw_i>0,\qquad p_i=P(X=i)=\frac{w_i}{W}.
$$

For each input i, give the instrument a row of nonnegative weights a with a positive row total. Normalize each row separately to obtain its channel:

$$
A_i=\sum_ja_{ij}>0,\qquad K_{j\mid i}=P(Y=j\mid X=i)=\frac{a_{ij}}{A_i},\qquad \sum_jK_{j\mid i}=1.
$$

Rows answer different hypothetical questions: if input 0 were supplied, what would be read; if input 1 were supplied, what would be read? A row is not the probability of observing that input. Preparation determines how often a row is used.

Multiplying an entire row by a positive factor leaves that row's conditional law unchanged. Multiplying every preparation weight by a common factor similarly leaves p unchanged. Changing only one entry generally changes the corresponding law. Keeping these two kinds of controls separate prevents an instrument adjustment from being mistaken for a change in the system's preparation.

## Follow probability mass from rows into output columns

The joint probability names an input and its reading together. Multiply the input probability by the appropriate conditional probability:

$$
J_{ij}=P(X=i,Y=j)=p_iK_{j\mid i}.
$$

A row in the joint table sums to p, not necessarily to one. A channel row sums to one. Thus the joint table and the channel table can have the same shape while describing different quantities.

The sensor output is found by adding the joint masses entering each output column:

$$
R_j=P(Y=j)=\sum_iJ_{ij}=\sum_ip_iK_{j\mid i}.
$$

For a fair input and a sensor that flips labels with probability .1, the joint table makes the flow visible:

| Input | Reading 0 | Reading 1 | Row total |
| --- | ---: | ---: | ---: |
| 0 | .45 | .05 | .50 |
| 1 | .05 | .45 | .50 |
| Column total | .50 | .50 | 1 |

Each row starts with .5 units of probability mass. Its two arrows carry .45 and .05. Adding downward gives the reading probabilities. The width of an individual arrow records a joint event; it is not the same as the conditional .9 or .1 written on the sensor rule.

## Infer the state after a reading

After observing an output with positive probability, divide its joint column by that output's total. This is Bayes conditioning:

$$
B_{i\mid j}=P(X=i\mid Y=j)=\frac{J_{ij}}{R_j}=\frac{p_iK_{j\mid i}}{\sum_\ell p_\ell K_{j\mid\ell}}\qquad(R_j>0).
$$

For the fair-input table, reading 0 gives posterior (.9,.1), and reading 1 gives (.1,.9). These happen to resemble channel rows because of this symmetric preparation and instrument. Bayes conditioning does not generally mean transposing the channel matrix.

Read a column backward: among all trials that produced this lamp, what fraction began in each box? The prior input frequencies matter to that answer. If an output has zero probability, its posterior is not specified by this preparation and channel; the calculator reports it as undefined rather than normalizing a zero column.

## Separate instrument noise from hidden state uncertainty

Use natural logarithms and nats, with 0 ln 0=0. Joint entropy describes uncertainty about the pair:

$$
H(X,Y)=-\sum_{i,j}J_{ij}\ln J_{ij}.
$$

Factor the joint law in two ways: input then reading, or reading then hidden input. Splitting each logarithm gives two chain rules:

$$
H(X,Y)=H(X)+H(Y\mid X)=H(Y)+H(X\mid Y),\qquad
H(Y\mid X)=\sum_ip_iH(K_i),\quad H(X\mid Y)=\sum_{j:R_j>0}R_jH(B_j).
$$

The first conditional entropy averages instrument randomness with the input known. The second averages remaining input uncertainty after seeing the output. They answer different questions, even when a symmetric example makes their numerical values equal.

Rearrange the identity to compare input and output entropy correctly:

$$
H(X)=H(Y)+H(X\mid Y)-H(Y\mid X).
$$

In a deterministic grouping the last term vanishes. In a noisy observation it need not vanish, so H(Y) can exceed H(X). Leaving out the noise term would incorrectly apply a deterministic information-loss statement to a stochastic reading.

## Measure what the reading reveals

Mutual information compares the joint law with the product law that would hold if input and reading were independent. It also equals either entropy reduction:

$$
I(X;Y)=D(J\Vert p\otimes R)=\sum_{i,j:J_{ij}>0}J_{ij}\ln\frac{J_{ij}}{p_iR_j}
=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X).
$$

Whenever J is positive, its input and output marginals are positive too. Thus this finite-channel mutual information has no target-support infinity. Zero input or output coordinates require their usual zero contribution, not an invented logarithm of zero. The general bounds are:

$$
0\le I(X;Y)\le\min\{H(X),H(Y)\},\qquad I(X;Y)=0\iff J_{ij}=p_iR_j\ \text{for all }i,j.
$$

A high-entropy lamp is useful only to the extent that its variation is associated with the input. The comparison is symmetric as an information quantity, although the specified channel still describes an instrument operating from input to output.

For the general channel and conditional-information framework, see [Polyanskiy and Wu, MIT 6.441, chapter 2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/184197ca5d5418da2415d37e929860b9_MIT6_441S16_chapter_2.pdf). The concrete tables, priors and instruments here are specified finite examples; their physical realization is an additional question.

## Recover deterministic grouping as a special case

Suppose every prepared input produces exactly one output g(i). Then:

$$
K_{j\mid i}=\mathbf1\{j=g(i)\}\ \text{on the preparation support},\qquad H(Y\mid X)=0,\qquad I(X;Y)=H(Y).
$$

Use preparation weights (6,2,1,1), sending the first two states to output 0 and the last two to output 1. The output probabilities are (.8,.2). Output entropy is approximately .500402424 nats, while hidden state entropy after the reading is .588497552 nats. Together they recover H(X)≈1.088899975.

This is the previous deterministic partition inside the larger channel model. Giving every input a distinct output instead makes both conditional entropies zero and preserves H(X) completely. A deterministic row on an input with zero preparation mass does not affect these quantities; the requirement concerns rows actually used by the preparation.

## A random lamp can add uncertainty without information

Let both channel rows be (.5,.5), regardless of input. If the input is certainly state 0, then H(X)=0, but H(Y)=ln 2. The lamp is unpredictable even though the original state is known. Its entire entropy is instrument randomness: H(Y|X)=ln 2, while I(X;Y)=0.

With prior (.75,.25), the same lamp still has zero mutual information. Each positive-probability output leaves posterior (.75,.25). It tells us nothing that we did not know before. Now H(X|Y)=H(X), and the full identity accounts for the output's independent noise.

This example is especially useful when reading generative or measurement systems in AI. Diverse outputs need not be informative about a selected latent variable. Specify which input variable and which preparation the information comparison concerns; output diversity alone does not establish that relationship.

## A symmetric error is not the same as independent noise

For a binary sensor with flip probability e, the channel matrix is:

$$
K=\begin{pmatrix}1-e&e\\e&1-e\end{pmatrix},\qquad 0\le e\le1.
$$

Under a fair input, both readings are fair. The binary entropy h determines the two conditional entropies and the information:

$$
h(e)=-e\ln e-(1-e)\ln(1-e),\qquad
H(Y\mid X)=H(X\mid Y)=h(e),\qquad I(X;Y)=\ln2-h(e).
$$

At e=.1, each conditional entropy is approximately .325082973 nats and mutual information is .368064207 nats. A reading narrows down the state without identifying it perfectly. At e=.5 the rows coincide and the output is independent. At e=1 every label is reversed deterministically, so information is fully preserved.

Therefore increasing a quantity called error rate does not decrease information throughout the entire interval from zero to one. The instrument's rule matters. A known, always-reversed label can be decoded exactly; a fair random label cannot. Correcting a systematic encoding rule and reducing stochastic uncertainty are different tasks.

## An erasure tells you when information is missing

Give the sensor three outputs: 0, 1 and ?. Let an erasure occur with probability e independently of input; otherwise report the correct input:

$$
K=\begin{pmatrix}1-e&0&e\\0&1-e&e\end{pmatrix}.
$$

For a fair input and e=1/4, the joint table has a different structure from symmetric error:

| Input | Reading 0 | Reading 1 | Reading ? |
| --- | ---: | ---: | ---: |
| 0 | 3/8 | 0 | 1/8 |
| 1 | 0 | 3/8 | 1/8 |
| Column total | 3/8 | 3/8 | 1/4 |

Reading 0 or 1 identifies the input exactly. Reading ? leaves the original fair prior. The averaged quantities are:

$$
H(Y)=h(e)+(1-e)\ln2,\qquad H(Y\mid X)=h(e),\qquad H(X\mid Y)=e\ln2,\qquad I(X;Y)=(1-e)\ln2.
$$

At e=1/4, the retained information is .519860385 nats and hidden information is .173286795 nats. A third output can make output entropy exceed ln 2, even though mutual information cannot exceed the one-bit input entropy. The extra output uncertainty includes which trials were erased.

## The prior changes the meaning of the same reading

Keep the symmetric sensor with flip probability .1, but change the preparation to (.75,.25). Output probabilities become (.7,.3). Reading 0 has posterior (27/28,1/28), while reading 1 has posterior (1/4,3/4).

To obtain the second result, compare the joint masses .75×.1=.075 and .25×.9=.225, then divide by their sum .3. The sensor's reading 1 is .9 likely when input 1 is supplied, but the posterior probability of input 1 after this reading is .75. These are different conditional statements.

H(Y|X) remains h(.1) because both rows have that entropy. Mutual information changes to approximately .285781329 nats because the prior changes H(Y). Thus an instrument does not have a single input-independent mutual information value. Channel capacity would ask a separate optimization over allowed preparations; this lesson evaluates the explicitly supplied preparation instead.

## Keep impossible events and unused rules distinct

A declared output column can have total probability zero. It has no posterior determined by this model, and its weighted hidden entropy contribution is zero. A positive output too small for a binary64 display still has a well-defined posterior. Its positive sign and logarithm must not be replaced by a support zero.

An input of preparation probability zero is different. The instrument still has a specified conditional row for that hypothetical input, and that row may have positive entropy. Its contribution pH(K) to the current experiment is exactly zero. Changing this unused row leaves joint and observed information unchanged, while it can change the row's own conditional entropy.

The calculator requires complete, positive-total channel rows even for unused inputs. It does not repair an all-zero row by guessing a uniform rule. This keeps the instrument specification separate from which inputs happen to be prepared.

## Preserve small information quantities numerically

Rows can have different normalization denominators. Converting every row to rounded display probabilities before forming the joint law can lose a positive path, alter independence or erase a small association. The model keeps preparation, channel, joint, output and posterior laws as exact rationals of the supplied finite binary64 weights.

Logarithms remain approximations. Mutual information is evaluated as a sum of nonnegative Bregman terms, rather than subtracting two nearly equal entropy displays. Once these approximate information terms are formed, each is rounded to 192 significant binary digits with unrestricted exponent, and their dyadic sum is exact. The per-term relative rounding bound is:

$$
\left|\operatorname{round}_{192}(z)-z\right|\le2^{-192}z\qquad(z>0).
$$

This does not make the underlying transcendental logarithm exact. It also does not round probabilities early. The unrestricted exponent preserves a tiny positive information term that a fixed binary64 display cannot show. Exact zero independence is decided from the rational joint law, not by applying a tolerance to mutual information.

For example, preparation weights (MIN_VALUE,1024) with an identity sensor have a first-state probability below the smallest binary64 value. Its entropy and mutual information still round to a positive minimum subnormal value. Multiplying the original positive probability by its information factor before display preserves this contribution.

## Research extension: repeated readings can share the same error

Seeing the same answer twice can be fresh evidence or merely a copy of the first error. A single-reading channel does not distinguish those situations. To combine sensors or repeated observations, specify their joint conditional law, not just each instrument's marginal error rate.

Keep one fixed fair binary input X and error probability 0<e<1/2. Compare conditionally independent errors with one shared error reused by every reading. The input does not evolve between readings, and no clock, energy, implementation or reset cost is specified. Both models have exactly the same channel for each individual reading.

Each reading is the input xor its error bit. Independence must be stated conditional on X: even the independent-error model has readings correlated marginally through their common input.

$$
X\sim\operatorname{Bernoulli}(1/2),\qquad Y_j=X\mathbin{\mathrm{xor}}N_j,\quad P(N_j=1)=e,\quad0<e<1/2.
$$

Only the independent-error model permits multiplying the individual likelihoods. Reusing an error bit violates this factorization, even when every single sensor still has the advertised error probability.

$$
P(Y_1=y_1,\ldots,Y_n=y_n\mid X)=\prod_{j=1}^nP(Y_j=y_j\mid X)\quad\text{for conditionally independent errors}.
$$

For independent errors, the number K of ones in n readings is sufficient for inference about X. Conditional on K, permutations of the output pattern have equal likelihood under either input; discarding the order loses no information about X within this declared family.

$$
K=\sum_jY_j,\quad P(K=k\mid X=0)=\binom nk e^k(1-e)^{n-k},\quad P(K=k\mid X=1)=\binom nk(1-e)^k e^{n-k}.
$$

Bayes odds multiply by the likelihood ratio of the count. For a biased prior an extra prior-odds factor would be required; this equation uses the specified fair input.

$$
\frac{P(X=1\mid K=k)}{P(X=0\mid K=k)}=\left(\frac{1-e}{e}\right)^{2k-n}.
$$

Two agreeing ones at e=0.1 give a posterior of 81/82, while disagreement leaves the fair prior. More concentrated posteriors on some outcomes do not by themselves imply an improvement of every decision metric.

$$
n=2,\ e=1/10:\quad P(X=1\mid Y_1=Y_2=1)=81/82,\quad P(X=1\mid Y_1\ne Y_2)=1/2.
$$

The information chain rule measures genuinely new evidence through the conditional term. It does not say that total information equals twice the one-reading information. The second reading is partly redundant with the first.

$$
I(X;Y_1,Y_2)=I(X;Y_1)+I(X;Y_2\mid Y_1).
$$

For zero-one classification loss, optimal Bayes error sums the smaller joint mass for each count. The multiplicity is already inside the count law and must not be counted again. With e=0.1, one and two independent readings both have error 0.1 because of two-reading ties, even though mutual information increases.

$$
P_{\rm error}^{\rm MAP}=\sum_{k=0}^n\min\{P(X=0,K=k),P(X=1,K=k)\}.
$$

With a shared error, all readings are identical. Further copies reveal nothing beyond the first reading. After any number of agreeing ones the correct posterior is 1−e, not the increasingly confident independent-model posterior.

$$
N_1=\cdots=N_n=N:\quad Y_1=\cdots=Y_n,\quad I(X;Y_1,\ldots,Y_n)=\log2-h(e).
$$

The two models have identical single-reading channels but different joint agreement probabilities. Repeated agreement can therefore help diagnose the assumed dependence model if the experiment and input conditions are controlled.

$$
P(Y_1=Y_2\mid X)=(1-e)^2+e^2\quad\text{independent errors};\qquad P(Y_1=Y_2\mid X)=1\quad\text{shared error}.
$$

Compute information by averaging posterior uncertainty. Independent repeated readings cannot reveal more than the entropy of the one fixed input, however many are taken. These are information and decision quantities, not a thermodynamic measurement cost or a guarantee for sensors with unknown dependence.

$$
I_n=\log2-\sum_kP(K=k)h(P(X=1\mid K=k)),\qquad0\le I_n\le\log2,\quad h(p)=-p\log p-(1-p)\log(1-p).
$$

### Read information gain and false confidence

The upper panel compares total information for independent and shared errors at e=0.1. The lower panel compares the posterior after all-one readings. Applying the independent formula to shared-error data creates growing confidence without new evidence. The table separately reports optimal decision error, which need not improve at every n.

![Independent error readings gain information toward the input-entropy bound; shared-error copies add no information despite a falsely increasing independent-model posterior.](/assets/learning/micro-macro-repeated-sensors.en.svg)

### Reproduce four reading counts

Use a fair input and e=1/10. The code forms exact rational count-joint probabilities and approximates only the logarithms. Shared-model information and its all-one posterior remain unchanged; neither is inferred from the independent calculation.

| Readings n | Independent I | Independent MAP error | Shared I | All-one posterior: independent | All-one posterior: shared |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.36806421 | 0.10000000 | 0.36806421 | 0.90000000 | 0.90000000 |
| 2 | 0.51437472 | 0.10000000 | 0.36806421 | 0.98780488 | 0.90000000 |
| 3 | 0.59778242 | 0.02800000 | 0.36806421 | 0.99863014 | 0.90000000 |
| 5 | 0.66327904 | 0.00856000 | 0.36806421 | 0.99998307 | 0.90000000 |

```python
from fractions import Fraction as F
from math import comb, log
e = F(1,10)
def h(p):
    if p in [0,1]: return 0.0
    return -float(p)*log(float(p))-float(1-p)*log(float(1-p))
shared = log(2)-h(e)
for n in [1,2,3,5]:
    hidden, error = 0.0, F(0)
    for k in range(n+1):
        p0 = F(comb(n,k),2)*e**k*(1-e)**(n-k)
        p1 = F(comb(n,k),2)*(1-e)**k*e**(n-k)
        hidden += float(p0+p1)*h(p1/(p0+p1))
        error += min(p0,p1)
    posterior = (1-e)**n/((1-e)**n+e**n)
    print(f"{n} {log(2)-hidden:.8f} {float(error):.8f} {shared:.8f} "
          f"{float(posterior):.8f} {float(1-e):.8f}")
```

### Three research questions with explanations

#### 1. Why do two readings improve information but not the shown classification error?

**Explanation.** They change posterior uncertainty: agreeing readings are more decisive, while disagreements are tied. The averaged conditional entropy falls, so mutual information rises. With fair input and the stated zero-one loss, the tie contribution makes the two-reading Bayes error remain 0.1.

#### 2. Can identical one-reading channels justify multiplying likelihoods?

**Explanation.** No. Both error models have the same individual binary symmetric channel. Their joint conditional law differs. Shared errors make later readings copies of the first, so multiplying marginal likelihoods would invent evidence and overstate posterior confidence.

#### 3. Does summarizing by K always preserve the input information?

**Explanation.** Only under an appropriate specified family. Here independent identical error probabilities make the likelihood ratio depend solely on K, and the conditional distribution of order given K is input-independent. Unequal sensors, correlated errors or a changing input need a different sufficient representation.

### Source and research boundary

Conditional information, channel composition and chain rules follow [Polyanskiy and Wu, MIT 6.441, chapter 2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/184197ca5d5418da2415d37e929860b9_MIT6_441S16_chapter_2.pdf). The independent/shared comparison and Bayes-error examples are derived for this explicit finite experiment. Output diversity, repeated agreement and physical entropy production remain different claims.

## Worked exercises

### Exercise 1: Build and invert a joint table

Use a fair input and channel rows (.9,.1),(.1,.9). Find the joint table, the output probabilities and the posterior after reading 0.

**Answer:** Joint rows are (.45,.05) and (.05,.45). Each output has probability .5. Dividing the first column by .5 gives posterior (.9,.1). Channel rows sum to one; the joint rows each sum to the preparation probability .5.

### Exercise 2: Explain an unpredictable but uninformative reading

Prepare input 0 with certainty and let both channel rows be (.5,.5). Find H(X),H(Y),H(Y|X),H(X|Y) and mutual information.

**Answer:** H(X)=H(X|Y)=I=0. H(Y)=H(Y|X)=ln 2. The output entropy comes entirely from independent instrument randomness. Substituting both conditional terms into the chain identity recovers input entropy zero.

### Exercise 3: Compare random errors with a deterministic reversal

For a fair input, compute mutual information at binary flip probabilities .1,.5 and 1.

**Answer:** At .1 it is ln 2−h(.1)≈.368064207 nats. At .5 it is zero. At 1 it is ln 2 because the reversal is known and invertible. Error frequency alone does not determine whether a reading can be decoded.

### Exercise 4: Condition on an erasure

Use fair input and erasure probability 1/4. Give output probabilities, posterior after ?, and average hidden information.

**Answer:** Outputs have probabilities (3/8,3/8,1/4). The ? posterior is (.5,.5); other positive readings identify the state. Hidden information is (1/4)ln 2≈.173286795 nats, and retained information is (3/4)ln 2≈.519860385.

### Exercise 5: Use a biased prior in Bayes conditioning

For prior (.75,.25) and a symmetric .1-flip channel, what is P(X=1|Y=1)? Why is it not .9?

**Answer:** Joint masses for Y=1 are .075 and .225, so the posterior is .225/.3=.75. The .9 value is P(Y=1|X=1), a conditional statement in the other direction. The prior weights affect how many of the observed readings originate in each state.

### Exercise 6: Distinguish an absent output from an unused input

Suppose an output column has zero mass, while a preparation-zero input has a noisy specified row. What remains defined, and what contributes to the entropy averages?

**Answer:** The absent output's posterior is undefined, and its weighted hidden entropy contribution is zero. The unused input row remains a defined instrument rule and can have positive row entropy, but its weighted noise contribution is zero. Neither case licenses erasing an arbitrarily small positive probability elsewhere.

## Explore the experiment

Choose “Symmetric error .1” and compare three tables in order: channel, joint, then posterior. A channel row sums to one; the corresponding joint row sums to the preparation probability. Select output 1 to read its posterior, rather than treating a channel row as an inverse rule.

| Example | Compare | What remains fixed or changes |
| --- | --- | --- |
| Biased prior 3:1 | Posterior at Y=1 versus Symmetric error .1 | The channel stays fixed; the changed prior gives posterior input-1 probability .75. |
| Erasure probability 1/4 | Outputs Y=0,1,2 | Y=2 means ? and leaves the fair prior; the other readings identify the input. |
| Independent lamp | H(X), H(Y), H(Y|X) and I(X;Y) | A random lamp can add output uncertainty while revealing no input information. |
| Always reversed labels | Posterior and hidden conditional entropy | A known deterministic reversal still identifies each prepared input. |
| Unused input and output | Row entropy, weighted noise and posterior at Y=2 | An unused row stays defined; an impossible output has no determined posterior. |
| Positive information below probability range | Input probability, H(X) and I(X;Y), including status/log columns | A probability outside the linear display range can still contribute positive information. |

Change preparation and channel-row weights separately. Each row is normalized by its own total. In “Extended settings”, the number of channel rows must match the preparation length and all rows must have the same output count and a positive total. A zero preparation weight is allowed; an all-zero channel row is not repaired by guessing a sensor rule. Select an output after each change to compare the full joint column with its posterior.

The reader provides figures, guided experiments and quizzes. Calculations run in a worker so a costly input can be stopped or replaced; do not interpret a pending or failed calculation as zero information. The full 16×16 mixed extreme remains an expensive calculation, so its permitted size does not promise instant completion. An example scan compares different preparations or instruments and supplies no clock, heat budget or physical measurement cost. Later ensemble units add specific reservoir assumptions; classical channel information alone does not supply them.
