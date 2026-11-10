## A channel is a conditional law

Consider a corrupted storage bit or a representation sent between GPUs. Sending X does not guarantee receiving the same Y. The channel W(y|x) specifies the probability of an output **after fixing an input**. It is different from P(X), the distribution of symbols sent.

We assume a known stationary memoryless channel, independent noise across uses, no input cost constraint, and a receiver that knows the channel law. Time variation, correlated noise and continuous neural representations require additional models.

$$
P(X=x,Y=y)=P(X=x)W(y\mid x).
$$

## Flips and erasures carry different information

A binary symmetric channel (BSC) flips each bit with probability ε. Receiving 1 does not reveal whether 1 was sent or 0 flipped. A binary erasure channel (BEC) returns a question mark with probability ε. The receiver loses the value but **knows which position was lost**. Equal noise probabilities therefore need not mean equal information loss.

| Channel | Outputs given input 0 | Outputs given input 1 |
| --- | --- | --- |
| BSC | 0: 1−ε, 1: ε | 0: ε, 1: 1−ε |
| BEC | 0: 1−ε, ?: ε | 1: 1−ε, ?: ε |

## Information at an input law and capacity

Let p=P(X=1). Binary entropy h₂ uses base-two logarithms and the convention 0 log₂0=0. In the BSC, the output-one probability is q=ε+(1−2ε)p. Conditional output uncertainty is h₂(ε), giving

$$
\begin{aligned}
I_{\mathrm{BSC}}(p)&=h_2(q)-h_2(\varepsilon),\\
I_{\mathrm{BEC}}(p)&=(1-\varepsilon)h_2(p).
\end{aligned}
$$

Always sending zero communicates no new message, so I=0 even through a useful channel. **Capacity also optimizes the input distribution.** Both symmetric channels achieve their maximum at p=.5.

$$
\begin{aligned}
C&=\max_{P_X}I(X;Y),\\
C_{\mathrm{BSC}}&=1-h_2(\varepsilon),\\
C_{\mathrm{BEC}}&=1-\varepsilon.
\end{aligned}
$$

## Worked example and visualization

For p=.5 and ε=.1, the BSC joint table is .45, .05, .05, .45 for assignments (0,0), (0,1), (1,0), (1,1). The output is uniform. Since h₂(.1)≈.469, information and capacity both equal about .531 bits/use. The BEC at the same noise probability retains exact values on 90% of positions and has capacity .9.

Change p to .2. The BSC output probability becomes q=.26 and information falls to about .358, while capacity remains .531. The blue curve varies input probability, the orange dashed line marks capacity, and the orange dot marks the selected input. **A fixed channel can be used less efficiently by a biased source.**

## What the coding theorem guarantees

For a memoryless channel and rate R<C, there exists a sequence of codes whose average error tends to zero as block length grows. This is an asymptotic existence statement. It does not guarantee zero error for a short block or good behavior for an arbitrary code.

Bits/use also differs from bits/second; converting requires a channel-use frequency. For AI systems, compression rate, reconstruction distortion, channel corruption and latency are separate quantities. The real-valued dimension of an embedding cannot simply be equated with this binary-channel capacity. Error-correcting codes will make the tradeoff between actual code length and error probability concrete.

## Exercises with explanations

1. What is BSC capacity at ε=1? **One bit/use.** Every output is inverted, and a second inversion recovers the input exactly. Capacity vanishes at ε=.5, where output and input are independent.
2. Does a BEC with ε=1 behave the same? **No.** Every output is an erasure, so information and capacity are zero.
3. Does I=0 at p=0 imply C=0? **No.** A constant input carries no message; capacity compares all allowed input distributions.

Further reading: [MIT 6.441, Lecture 8: Channel Capacity](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/resources/mit6_441s10_lec08/).
