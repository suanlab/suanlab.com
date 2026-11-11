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

## Why symmetric inputs optimize capacity, and a converse

For the BSC, conditional output entropy h₂(ε) is fixed. Binary output entropy is at most one; p=1/2 makes the output fair and attains that bound. Hence capacity is 1−h₂(ε). For the BEC, the erasure event is independent of X, and a non-erased output reveals X exactly. Retained information is (1−ε)H(X), again maximized by a fair input. Relabeling BSC outputs replaces ε>.5 by 1−ε without changing capacity.

Now send one of M equiprobable messages in n uses, with decoder average error Pₑ. Assume a finite-alphabet memoryless channel without feedback or input cost constraints. Fano gives H(message|Yⁿ)≤1+Pₑ log₂M. Data processing, entropy subadditivity and conditional independence of channel noise also give

$$
I(J;Y^n)\le I(X^n;Y^n)
\le\sum_{i=1}^n I(X_i;Y_i)\le nC.
$$

Here J is the random message and M is its alphabet size. Combining these bounds with H(message)=log₂M yields, for R=(log₂M)/n>0,

$$
(1-P_e)\log_2 M\le nC+1,
\qquad P_e\ge\max\left(0,1-\frac CR-\frac1{nR}\right).
$$

For example ε=.5 gives C=0. At n=100,R=1/2, the bound is Pₑ≥.98. Longer codes cannot create reliable positive-rate communication through an input-independent channel. More generally, holding R>C makes this lower bound stay positive. This is a finite-length weak converse. It neither proves achievability below capacity nor replaces a strong converse asserting error tending to one above capacity.

## Compute small information without cancellation

The mathematically correct h₂(q)−h₂(ε) can lose a small positive difference. At ε=.5+2⁻⁴⁰, C≈2.39×10⁻²⁴ bits/use, but the previous implementation returned zero. At p=10⁻³⁰⁰,ε=.1, information is also positive, about 2.54×10⁻³⁰⁰. The calculator now shares the stable channel-KL and log1p entropy routines verified in the information-theory lessons. Rounded table cells or bars need not reveal such small dependence. When a selected positive I falls below numeric representation range, the readout shows a dash and numeric-range explanation and withholds its point. Exact independence at ε=.5 and constant-input zeros remain distinct.

**Exercise:** With C=0, can sending one fixed message have zero error? **Explanation:** Yes. M=1 gives R=0, outside this positive-rate bound. State the message alphabet and error criterion together.

## Exercises with explanations

1. What is BSC capacity at ε=1? **One bit/use.** Every output is inverted, and a second inversion recovers the input exactly. Capacity vanishes at ε=.5, where output and input are independent.
2. Does a BEC with ε=1 behave the same? **No.** Every output is an erasure, so information and capacity are zero.
3. Does I=0 at p=0 imply C=0? **No.** A constant input carries no message; capacity compares all allowed input distributions.

Further reading: [MIT 6.441, Lecture 8: Channel Capacity](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/resources/mit6_441s10_lec08/).
