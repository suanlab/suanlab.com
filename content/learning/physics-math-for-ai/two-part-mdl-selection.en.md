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

## Exercises with explanations

1. What if the same p gets a header two bits longer? **Total cost increases by exactly two bits.** Fit is unchanged.
2. Is the header repeated when sample size grows? **Not here.** One model describes the longer sequence.
3. Is the winning cost literally a 14.98045-bit file? **No.** It is an ideal length; an actual codec has discrete output and overhead.

Further reading: [Grünwald and Roos, Minimum Description Length Revisited](https://arxiv.org/abs/1908.08484).
