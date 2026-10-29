## Vanishing values, persistent area

For n≥2 on [0,1], define a continuous triangular spike:

$$
f_n(x)=n\max(0,1-|nx-1|).
$$

Its peak is at 1/n, height n, support [0,2/n], and area one. Every fixed positive x eventually lies outside the support; zero always has value zero. Therefore the pointwise limit is zero everywhere, yet

$$
\lim_n\int_0^1 f_n(x)\,dx=1\ne0=\int_0^1\lim_n f_n(x)\,dx.
$$

Existence of all finite integrals and a pointwise limit is insufficient. The plot includes exact spike vertices so a coarse grid cannot silently miss its mass.

## Uniform convergence is sufficient on finite intervals

For integrable functions converging uniformly,

$$
\left|\int_a^b(f_n-f)\right|\le(b-a)\|f_n-f\|_\infty\to0.
$$

The spike fails this condition: its supremum is n. Uniform convergence is sufficient, not the only possible justification.

## Dominated convergence adds different control

For measurable functions on a measure space, almost-everywhere convergence together with |fₙ|≤g for one integrable g implies convergence of integrals. Read all assumptions: measurability, almost everywhere, absolute value, a common dominator and integrability.

The comparison x/n is dominated by one on [0,1] and has integral 1/(2n)→0. It is also uniformly convergent. A general dominated-convergence proof belongs to the later measure-theory development.

## Why no common integrable dominator exists here

Any g dominating every spike would have integral at least one on [0,2/n]. But an integrable function’s integral over sets of vanishing length tends to zero, a contradiction. This rules out every integrable dominator, not just the constant one.

Mass disappearing from a pointwise limit connects to weak convergence of measures, but the current model establishes only an ordinary function-limit interchange failure.

## AI interpretation and exercises

Moving a limit through an expectation, integrating a series termwise or differentiating an expected loss requires assumptions. Dominated convergence for limits does not automatically justify differentiation; difference quotients or derivatives need suitable additional control.

1. At n=32, spike height and area are **32 and one**.
2. The integral of its pointwise limit is **zero**.
3. At n=8 the comparison integral is **1/16**.
4. A grid missing the spike demonstrates **sampling failure, not a theorem**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
