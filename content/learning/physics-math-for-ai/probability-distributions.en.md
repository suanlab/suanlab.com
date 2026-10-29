## A model of possible outcomes

A classifier emits a probability, yet the next label is one outcome, not a fraction of a label. Probability describes uncertainty across possible outcomes. A sample is what actually occurs. Keeping these separate is essential when interpreting confidence, measuring performance or constructing a likelihood.

We begin with independent binary trials. Success means the event being counted, not necessarily a beneficial result. If each trial has success probability $p$, its indicator is Bernoulli. The sum of $n$ such indicators is binomial only when trials are independent and share the same $p$.

## Outcomes, events and normalization

| Symbol | Meaning |
| --- | --- |
| $\Omega$ | Set of possible outcomes |
| $A$ | Event: a subset of outcomes |
| $X_i\in\{0,1\}$ | Indicator for trial $i$ |
| $K=\sum_iX_i$ | Number of successes |
| $p$ | Common success probability |

Probabilities are nonnegative and total probability is one. Disjoint alternatives add. Independence means the probability of joint events factors into a product; it is an assumption about the joint distribution, not a conclusion from equal marginal probabilities.

## Count the ways before multiplying

For $n=4$, the sequence 1100 has probability $p^2(1-p)^2$. There are six ways to place two successes in four positions. Thus

$$
P(K=k)=\binom nk p^k(1-p)^{n-k}.
$$

The combinatorial factor counts distinct ordered outcomes belonging to the same event. Omitting it confuses the probability of one sequence with the probability of a count.

At $p=1/2$, the masses for $k=0,1,2,3,4$ are $(1,4,6,4,1)/16$. They sum to one. The most probable count is 2, but observing 0 or 4 is possible. Probability 0.375 for count 2 does not mean every block of observations contains exactly 37.5% such counts.

## Mean and variation by indicators

Since $E[X_i]=p$, linearity gives $E[K]=np$. Independence makes cross-covariances zero, giving

$$
\operatorname{Var}(K)=np(1-p).
$$

For four fair trials, the mean is 2 and variance is 1. For $p=0.2$, the mean is 0.8 and variance 0.64. The mean need not be a possible observed integer. The next unit derives expectation and variance more generally.

At $p=0$ or 1 the count is deterministic and variance is zero. The lab evaluates these endpoints directly; it does not insert a hidden positive probability. A binomial model may be a poor description of correlated trials, changing conditions or an unknown mixture of populations.

## Conditional probability is another question

If an event $B$ has positive probability, $P(A|B)=P(A\cap B)/P(B)$. This changes the reference population. It does not generally equal $P(B|A)$, and conditioning does not itself establish causation.

For two fair independent trials, let $B$ mean at least one success. The possible sequences in $B$ are 01, 10 and 11, each equally likely. Therefore $P(K=2|B)=1/3$, although the unconditional probability is $1/4$. State the conditioning event before interpreting a probability in a dataset.

## Discrete mass versus continuous density

The experiment shows exact probability masses as bars, not sample frequencies. In a continuous model, probability is an integral of density over an interval. A density height can exceed one; the probability of an individual point is ordinarily zero for a continuous distribution. Replacing a histogram height with a point probability mixes different objects.

The sum-to-one readout is a normalization check. Alter $n$ and $p$ and calculate an endpoint mass by hand before looking at the bars. Axis ranges adapt; the numeric heights, rather than screen area across different settings, provide the comparison.

## AI applications and limits

Binary labels motivate Bernoulli likelihoods. A count of independent successes motivates a binomial model. Real evaluation examples may share users, locations or time periods, making independence questionable. Four repeated measurements of one object are not automatically four independent objects.

A useful research exercise is to compare the empirical variance of counts to $np(1-p)$ under controlled data collection. Excess variation can indicate dependence or changing success probabilities, but that observation alone does not identify which mechanism caused it. Keep the assumed sampling unit explicit.

## Exercises and reference

1. For $n=3,p=1/2$, write the four count probabilities. **Answer:** $(1,3,3,1)/8$.
2. Compute $P(K=0)$ for $n=4,p=0.2$. **Answer:** $0.8^4=0.4096$.
3. Explain why equal success probability does not imply independent trials.
4. Describe a data-collection scheme in which a binomial model would be inappropriate.

For probability axioms, conditional probability and random variables, see [MIT Introduction to Probability](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/). The lab’s finite distributions are direct calculations, not a dataset or empirical study.
