## How much compression is possible with errors?

Lossless compression must recover the source exactly. Lossy compression permits discarding information within a specified distortion budget. This lab assumes an iid fair binary source and Hamming cost d(x,y)=1{x≠y}. Distortion is the average mismatched-bit fraction. Different sources or distortion measures require different results.

$$
R(D)=\inf_{P_{Y|X}:E[d(X,Y)]\leq D}I(X;Y).
$$

## The exact binary boundary

Here R(D)=1−h₂(D) up to D=.5 and zero thereafter. At D=.1, rate is about .531 bits per symbol. Zero distortion requires one bit per fair source bit. With budget at least .5, always reconstructing zero achieves distortion .5 without sending information.

$$
R(D)=\begin{cases}1-h_2(D),&0\leq D\leq1/2,\\0,&1/2\leq D\leq1.\end{cases}
$$

## Why this boundary appears

For actual error d≤D≤.5, the binary Fano relation gives H(X|Y)≤h₂(d)≤h₂(D), hence I(X;Y)≥1−h₂(D). A binary symmetric test channel with crossover D reaches the same distortion and information. The rate–distortion coding theorem connects this information optimization to operational long-block coding; the graph itself does not construct a block codec.

## Separate the budget from a selected channel

Blue is optimal R(D) over all allowed test channels. Orange is 1−h₂(ε) for a symmetric test channel. The curves match through ε=.5, but larger flips are inefficient. At ε=.75, channel information is about .189 bits while the optimum for that distortion budget is zero.

Inspect feasibility separately: actual distortion ε must not exceed D. A channel with D=.1 and ε=.25 uses less information but violates the constraint. Blue and orange dots represent different controls.

## Connections and limits

The example motivates studying representation information alongside reconstruction cost. Its closed form does not automatically describe semantic image or text loss, or arbitrary neural networks. Units are asymptotic bits per symbol, not exact single-file byte lengths or guarantees for every sample. Specify source, reproduction alphabet, distortion and average-versus-worst-case conditions first.

## Exercises with explanations

1. What is R(.5)? **Zero.**
2. Does the ε=1 channel have zero information? **No: perfect inversion is invertible and retains one bit.**
3. Is a low-information channel outside the distortion budget a valid solution? **No: it must satisfy the constraint.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
