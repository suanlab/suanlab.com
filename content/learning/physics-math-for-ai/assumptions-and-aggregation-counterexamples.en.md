## 1. Name the population the metric represents

Suppose restart episode $i$ contains $n_i>0$ evaluated time points, and the paired improvement averaged within that episode is $d_i$. An episode-weighted average gives each restart one vote. A point-weighted average gives a longer restart more votes. These are different estimands when the observation counts vary.

## 2. State a theorem with its assumptions

For $N$ episodes, let bars denote arithmetic means over episodes. Define $\operatorname{Cov}_N(n,d)=N^{-1}\sum_i(n_i-\bar n)(d_i-\bar d)$. With $\bar n>0$,

$$
\frac{\sum_i n_i d_i}{\sum_i n_i}=\bar d+\frac{\operatorname{Cov}_N(n,d)}{\bar n}.
$$

Equal observation counts imply equal averages. Zero covariance is the general equality condition for this identity.

## 3. Prove the identity

Expand the centered product. The mixed terms simplify because the centered variables sum to zero. This gives $\operatorname{Cov}_N(n,d)=\overline{nd}-\bar n\bar d$. Divide by $\bar n$ and rearrange. The proof applies to any finite list satisfying the stated count condition, including lists with negative improvements.

## 4. Construct a counterexample

Take two episodes with counts $(1,9)$ and effects $(0.4,-0.1)$ K. The episode-weighted effect is $0.15$ K, but the point-weighted effect is $-0.05$ K. The longer episode has a worse effect and receives more weight. This refutes the claim that a positive episode-weighted improvement guarantees a positive point-weighted improvement.

## 5. Check the arithmetic and the general identity

```python
from math import isclose
from statistics import mean

counts = [1, 9]
effects = [.4, -.1]
macro = mean(effects)
micro = sum(n*d for n, d in zip(counts, effects)) / sum(counts)
covariance = mean((n-mean(counts))*(d-macro) for n, d in zip(counts, effects))
assert isclose(micro, macro + covariance/mean(counts))
assert isclose(macro, .15) and isclose(micro, -.05)
print("episode mean:", macro, "point mean:", micro)
```

A numerical example checks the arithmetic and gives a counterexample. The algebra supplies the general identity.

## 6. Report what the result supports

Choose the weighting based on the research question before evaluation. A typical episode and a typical observed time point are different targets. Report counts, the weighting rule, and sensitivity to a meaningful alternative. State whether the claim is an algebraic identity, a property under assumptions, or an empirical observation.
