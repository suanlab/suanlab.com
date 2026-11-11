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

## Distance guarantees and a Chernoff error bound

If radius-t Hamming balls around two codewords overlap, the triangle inequality puts their centers at distance at most 2t. Thus minimum distance d>2t makes these balls disjoint and permits unique correction of every error inside them. This geometric proof needs no independence. Independence enters when computing the probability of leaving a ball.

For odd n≥3, let K sum independent Bernoulli errors and q=(n+1)/(2n). For 0<ε<1/2 and λ>0, Markov's inequality and the MGF give P(K≥nq)≤$e^{-\lambda nq}(1-\varepsilon+\varepsilon e^\lambda)^n$. Substituting optimal λ=ln[q(1−ε)/(ε(1−q))] yields

$$
P_e\le e^{-nD(q\Vert\varepsilon)},\qquad
D(q\Vert\varepsilon)=q\ln\frac q\varepsilon+(1-q)\ln\frac{1-q}{1-\varepsilon}.
$$

Replacing the threshold by 1/2 gives a looser bound. The expression above uses the actual odd-n failure threshold. Its exponent uses natural logarithms, distinct from the bit units of communication rate.

| n, ε=.1 | Exact independent error | Chernoff bound | Common-flip error |
|---|---|---|---|
| 3 | .028000000 | .060750000 | .1 |
| 5 | .008560000 | .023437500 | .1 |
| 9 | .000890920 | .003177332 | .1 |

When every position flips together, each marginal error rate is still .1 but Pₑ=.1. The final column does not satisfy the independent binomial model. The code compares exact-fraction tails with the exponential bound.

At fixed ε<.5, longer repetition reduces independent error while rate 1/n tends to zero. This is different from approaching positive capacity while improving reliability.

Numeric range is separate too. For n=9,ε=10⁻¹⁰⁰, failure probability is positive but below binary64 representation range. Affected selected probabilities show a dash and numeric-range explanation, distinct from exact zero at ε=0.

**Exercise:** Does increasing repetition length increase channel capacity? **Explanation:** No. Capacity belongs to the channel law. Repetition changes the chosen code's rate and error.

```python
from fractions import Fraction as F
from math import comb, log, exp

epsilon=F(1,10)
for n in [3,5,9]:
    threshold=(n+1)//2
    exact=sum(F(comb(n,k))*epsilon**k*(1-epsilon)**(n-k)
              for k in range(threshold,n+1))
    q=threshold/n
    divergence=q*log(q/float(epsilon))+(1-q)*log((1-q)/(1-float(epsilon)))
    bound=exp(-n*divergence)
    assert float(exact)<=bound
    # Correlated model: all positions flip together with probability epsilon.
    correlated=epsilon
    print('n, rate, exact independent error, Chernoff bound, common-flip error:',
          n,round(1/n,6),round(float(exact),9),round(bound,9),float(correlated))
```


## Exercises with explanations

1. How many flips can length seven guarantee correcting? **Three.** Radius-three Hamming balls around the two codewords do not overlap.
2. What happens with four flips at length seven? **Majority selects the wrong bit.** Repeating that decoder cannot restore missing evidence.
3. Can .028 be used without independence? **No.** If all three bits flip together with probability .1, each marginal error remains .1 but block error is .1.

Further reading: [MIT MAS.160, Error Correction](https://ocw.mit.edu/courses/mas-160-signals-systems-and-information-for-media-technology-fall-2007/94144b6cc83df4bea4f9cef22c91baf4_1017_error.pdf).
