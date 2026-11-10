## Why send the same bit repeatedly?

Encode zero as 000 and one as 111. A received 001 is one position away from 000 and two away from 111, so choose zero. Hamming distance counts differing positions. Redundancy creates space around valid codewords in which corrupted messages remain distinguishable.

An odd-length repetition code has two codewords at distance n. It uniquely corrects any t=(n−1)/2 flips. This worst-case statement does not assume a probability model.

$$
d_{\min}=n,\qquad 2t<d_{\min},\qquad R=1/n.
$$

## Probabilities require a noise model

Assume independent BSC flips with probability ε. The number of errors K is binomial. Majority decoding fails when K exceeds half the block. A particular error mask has no combinatorial factor; summing all masks with the same error count introduces that factor.

$$
\begin{aligned}
P(K=k)&={n\choose k}\varepsilon^k(1-\varepsilon)^{n-k},\\
P_e&=\sum_{k=(n+1)/2}^{n}P(K=k).
\end{aligned}
$$

At ε=.1, three repetitions give 3×.01×.9+.001=.028 error. Five give .00856 but consume five channel uses per information bit. Their rates 1/3 and 1/5 are below the channel capacity of about .531. Lower error alone does not establish an efficient code.

## Read the distribution and transcript separately

Blue points show binomial masses; orange points mark majority failures. The table shows one selected received mask, using its lowest n bits. Changing this transcript does not change the theoretical block error rate.

For n=5, sent zero and mask 3, two positions are corrupted. Decoding succeeds. This individual word has probability .1²×.9³=.00729, distinct from the full block failure probability .00856.

## Boundaries and AI systems

At ε=.5 majority error is .5. Above .5, invert outputs or otherwise incorporate the known channel; naive majority becomes inappropriate. The lab deliberately retains majority to expose this mismatch.

Correlated device failures invalidate the binomial calculation. Similarly, multiple replicas in distributed learning need not fail independently. Evaluate shared failure causes, bandwidth and latency as well as replication count.

## Exercises with explanations

1. How many flips can length seven guarantee correcting? **Three.** Radius-three Hamming balls around the two codewords do not overlap.
2. What happens with four flips at length seven? **Majority selects the wrong bit.** Repeating that decoder cannot restore missing evidence.
3. Can .028 be used without independence? **No.** If all three bits flip together with probability .1, each marginal error remains .1 but block error is .1.

Further reading: [MIT MAS.160, Error Correction](https://ocw.mit.edu/courses/mas-160-signals-systems-and-information-for-media-technology-fall-2007/94144b6cc83df4bea4f9cef22c91baf4_1017_error.pdf).
