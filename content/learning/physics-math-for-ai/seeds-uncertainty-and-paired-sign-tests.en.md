## 1. Say what each repetition varies

Different training seeds probe randomness in initialization, ordering and stochastic optimization under the chosen data and protocol. Resampling test episodes probes another source of uncertainty. Repeating the same time series at many timestamps creates correlated observations. Define the population and independent unit before computing an interval or test.

## 2. Choose a test for the stated effect

A paired sign test can examine whether an independent pair is more likely to favor the candidate than the baseline. Subtract a prespecified practical threshold from each paired effect, discard exact ties, and count $k$ positive differences among $n$ remaining pairs. For a continuous difference distribution, the null gives probability one-half to each sign. This concerns the sign probability or median, while the mean effect needs a suitable mean-based analysis. [NIST, Sign Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm)

## 3. Derive an exact one-sided tail

Under independent fair signs the positive count is binomial. For the alternative favoring positive differences,

$$
p_{\rm tail}=\sum_{j=k}^{n}\binom nj2^{-n}.
$$

Six positive differences out of six give $1/64=0.015625$. Five out of six give $7/64=0.109375$. The test retains signs and does not measure improvement magnitude.

## 4. Keep the analysis plan separate from outcome selection

Prespecify the primary outcome, threshold and one-sided direction when justified. If many independent null tests each have size 0.05, the probability of at least one false positive across $m$ tests is $1-0.95^m$. Dependence changes this expression. Use a declared multiple-testing procedure or fresh data for selected follow-up claims.

## 5. Enumerate the null signs exactly

```python
from itertools import product
from math import comb, isclose

def tail(n, k):
    return sum(comb(n, j) for j in range(k, n+1)) / 2**n

sign_patterns = list(product([-1, 1], repeat=6))
for k in (5, 6):
    enumerated = sum(sum(s > 0 for s in pattern) >= k for pattern in sign_patterns)/len(sign_patterns)
    assert isclose(tail(6, k), enumerated)
assert isclose(tail(6, 6), .015625)
assert isclose(tail(6, 5), .109375)
print("six positives:", tail(6, 6), "five positives:", tail(6, 5))
```

## 6. Report estimates and the experimental unit

Publish paired effect values, effect-size summaries, seed identifiers, the number of independent units, and the test assumptions. Report uncertainty over data and over training randomness according to what was actually varied. A small p-value is evidence against the stated null under its assumptions, and an operational decision also needs the magnitude and cost of the effect.
