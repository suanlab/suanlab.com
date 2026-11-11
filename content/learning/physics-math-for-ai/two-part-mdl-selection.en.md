## Transmit the model as well as the data

Likelihood rewards good fit. Allowing unlimited uncharged parameters can make memorization look optimal. Two-part MDL adds the cost of identifying a model and the cost of transmitting data under it:

$$
L(M,x)=L(M)+L(x\mid M).
$$

This experiment uses an ordered Bernoulli sequence of N observations with K ones. Four predeclared models have probabilities .5, .25, .5 and .75 and prefix headers 0, 10, 110 and 111. The duplicate .5 model isolates the effect of a longer header.

## Interpret the logarithmic cost

The probability of one ordered sequence is pᴷ(1−p)ᴺ⁻ᴷ, giving ideal data length

$$
L(x\mid p)=-K\log_2p-(N-K)\log_2(1-p).
$$

If only the aggregate count were transmitted, its probability would include a binomial coefficient. State what constitutes the data. The displayed costs may be fractional: this is an ideal log-cost comparison, not an implemented file codec. A concrete arithmetic coder requires finite-length overhead and termination rules.

## Worked example: N=16, K=4

Model .5 costs sixteen data bits plus one header bit, totaling seventeen. Model .25 costs about 12.98045 data bits plus two header bits, totaling 14.98045. The other .5 model has identical data cost but a three-bit header, totaling nineteen.

$$
L(x\mid .25)=-4\log_2(.25)-12\log_2(.75).
$$

Blue bars show total cost and orange bars data cost alone. With K=8, the short-header balanced model wins at seventeen bits. With K=12, the improved fit of .75 outweighs its header. Increasing the repetition multiplier preserves K/N and multiplies data log cost while charging the model header once.

## MLE and generalization

The lab also shows data cost at the MLE p̂=K/N. Comparing this uncharged cost directly with competitors' total costs gives fitted parameters a free pass. Specify parameter precision and model coding rules. Refined MDL and normalized maximum likelihood go beyond this finite four-model example.

A shorter description does not automatically establish robustness under distribution shift or a causal interpretation. AI comparisons still need data separation, consistent treatment of shared code and weights, and predictive evaluation.

## Two-part coding and a mixture combine models differently

The four header lengths ℓ=(1,2,3,3) satisfy Σ2⁻ˡ=1. For each model, lengths ceil(−log₂Pᵢ(x)) admit a prefix code because their Kraft sum is at most one. Concatenating the model header and its data code remains prefix-free. The resulting integer length is less than one bit above the displayed ideal cost, assuming external N and shared models/code assignments. The browser does not implement this codec.

Reading header lengths as prior masses and summing candidates gives a normalized mixture. The current two-part cost instead chooses the largest individual contribution:

$$
m(x)=\sum_{i=1}^4 2^{-\ell_i}P_i(x),\qquad
L_2(x)=-\log_2\max_i[2^{-\ell_i}P_i(x)],
$$
$$
L_2(x)-\log_2 4\le-\log_2 m(x)\le L_2(x).
$$

At N=16,K=4, two-part cost is 14.980450 bits and mixture cost 14.592679. The duplicate .5 model cannot win minimum selection with its longer header, but still contributes mixture mass. Changing duplicates or prior rules changes the comparison.

## Exact Bernoulli NML normalization and regret

Over all ordered strings of fixed length N, sum each string's maximized likelihood. There are binom(N,k) strings with k ones, giving

$$
C_N=\sum_{k=0}^N{N\choose k}(k/N)^k(1-k/N)^{N-k},\qquad
q_{\mathrm{NML}}(x)=\frac{P_{\widehat\theta(x)}(x)}{C_N}.
$$

Interpret endpoint 0⁰ as one. The resulting q_NML sums to one, whereas the unnormalized data-dependent MLE need not define one common distribution. For any normalized q, if P_MLE(x)/q(x)≤r for every x, summing P_MLE(x)≤r q(x) gives C_N≤r. Worst-case logarithmic regret is therefore at least log₂C_N. NML attains this lower bound by making the ratio exactly C_N for every string.

| N | C_N | Constant regret, bits |
|---|---|---|
| 4 | 3.218750 | 1.686501 |
| 8 | 4.245018 | 2.085771 |
| 16 | 5.704258 | 2.512039 |

Regret compares q's log loss with the best-fitting Bernoulli model's log loss on that same data. The fraction code checks normalization and identical regret ratios for every count. N is fixed and shared; these are ideal logarithmic lengths. A continuous model with an infinite normalizing integral cannot use the same definition unchanged.

**Exercise:** Why need the individually maximized likelihoods not sum to one? **Explanation:** A different model is chosen for each x. They are not one fixed model's probability table; normalization by C_N is required.

```python
from fractions import Fraction as F
from math import comb,log2

models=[(1,F(1,2)),(2,F(1,4)),(3,F(1,2)),(3,F(3,4))]
assert sum(F(1,2**length) for length,p in models)==1
N,K=16,4
parts=[F(1,2**length)*p**K*(1-p)**(N-K) for length,p in models]
mixture=sum(parts)
two_part=-log2(float(max(parts)))
mix_length=-log2(float(mixture))
assert two_part-2<=mix_length<=two_part
print('two-part ideal length, mixture ideal length:',round(two_part,6),round(mix_length,6))
for n in [4,8,16]:
    # Every ordered string with k ones has this maximized likelihood.
    maxima=[F(k,n)**k*(1-F(k,n))**(n-k) for k in range(n+1)]
    complexity=sum(comb(n,k)*maxima[k] for k in range(n+1))
    q=[v/complexity for v in maxima]
    assert sum(comb(n,k)*q[k] for k in range(n+1))==1
    assert all(maxima[k]/q[k]==complexity for k in range(n+1))
    print('N, NML normalizer, constant regret bits:',
          n,round(float(complexity),6),round(log2(float(complexity)),6))
```


## Exercises with explanations

1. What if the same p gets a header two bits longer? **Total cost increases by exactly two bits.** Fit is unchanged.
2. Is the header repeated when sample size grows? **Not here.** One model describes the longer sequence.
3. Is the winning cost literally a 14.98045-bit file? **No.** It is an ideal length; an actual codec has discrete output and overhead.

Further reading: [Grünwald and Roos, Minimum Description Length Revisited](https://arxiv.org/abs/1908.08484).
