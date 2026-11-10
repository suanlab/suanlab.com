## One surprise versus a distribution average

A success with probability one quarter is rarer than a success from a fair coin. Self-information measures this rarity as −log₂ p(x): probability one quarter gives two bits, while one half gives one bit. It measures rarity under a probability model, rather than semantic importance or value.

$$
\imath(x)=-\log_2p(x),\qquad H(X)=E[\imath(X)]=-\sum_xp(x)\log_2p(x).
$$

## Work through success probability one quarter

Success has surprisal two bits; failure has −log₂(3/4)≈0.415 bits. Average these values with their occurrence probabilities, not equal weights. Entropy is 0.25×2+0.75×0.415≈0.811 bits. An observed outcome's information differs from the distribution's expected information.

| Outcome | Probability | Surprisal in bits | Entropy contribution |
|---|---|---|---|
| Failure zero | 0.75 | About 0.415 | About 0.311 |
| Success one | 0.25 | 2 | 0.5 |

## Why infinity and zero occur together

At p=0, selecting success produces infinite surprisal, but its entropy contribution is zero. This is not ordinary multiplication of zero by infinity. It uses the continuous extension of −t log t at zero. If an actually observed outcome was excluded by the model, inspect the model rather than treating a small model entropy as evidence of success.

$$
\lim_{t\downarrow0}-t\log_2t=0,\qquad \imath(x)=\infty\ \text{if }p(x)=0.
$$

## Read the binary entropy curve

Blue plots h₂(p) across success probabilities, and the dot marks the selected p. Changing only the observed outcome changes its surprisal, while entropy and the curve remain fixed. The maximum is one bit at p=.5. Deterministic endpoints have zero entropy.

## Connections to AI and coding

A probabilistic predictor's log loss has the same form as self-information. Ideal coding lengths also connect to minus log probability. Actual prefix-code lengths are integers and may include headers and block overhead, so this number is not an exact file length. This chapter concerns finite discrete probabilities; differential entropy involving densities is treated separately.

## Exercises with explanations

1. What is surprisal for probability one eighth? **Three bits.**
2. Does changing the selected outcome at fixed p=.25 change entropy? **No: the distribution is unchanged.**
3. What is the weighted entropy term for an impossible outcome? **Zero by continuous extension.**

Further reading: [Shannon, A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf).
