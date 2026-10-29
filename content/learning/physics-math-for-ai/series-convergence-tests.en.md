## Terms and partial sums are different sequences

Series convergence means the partial sums S_N approach a finite limit. Convergence implies vanishing terms because aₙ=Sₙ−Sₙ₋₁:

$$
\sum_{n=1}^{\infty}a_n\text{ converges}\implies a_n\to0.
$$

The converse fails. Harmonic terms 1/n tend to zero while their sum diverges. Small individual contributions do not guarantee bounded accumulation.

## Prove harmonic divergence

The block from 2ᵏ+1 through 2ᵏ⁺¹ contains 2ᵏ terms, each at least 1/2ᵏ⁺¹:

$$
\sum_{n=2^k+1}^{2^{k+1}}\frac1n\ge\frac12.
$$

Infinitely many such blocks make the partial sums unbounded. This argument covers arbitrary sizes, unlike observing a finite plot through N=128.

## Alternation supplies error control

If positive terms decrease monotonically to zero, their alternating series converges and the error is at most the next term. For reciprocal terms,

$$
\left|\log2-\sum_{n=1}^N\frac{(-1)^{n+1}}n\right|\le\frac1{N+1}.
$$

Even and odd partial sums squeeze the limit from opposite sides. The value ln2 can be derived by integrating a finite geometric expansion of 1/(1+x) and bounding its remainder. This convergence is conditional because the absolute-value series is harmonic.

## Integral comparison and absolute convergence

For the decreasing positive function x⁻²,

$$
0\le\sum_{n=N+1}^{\infty}\frac1{n^2}\le\int_N^{\infty}\frac{dx}{x^2}=\frac1N.
$$

This proves convergence. The lab uses the known value π²/6 as a reference, but does not prove that evaluation here. Distinguish proving convergence from computing a closed-form sum; harmonic analysis provides a later route to the evaluation.

Absolute convergence permits arbitrary rearrangements without changing the sum. Conditional convergence does not. Finite floating-point summation order introduces a separate numerical issue.

## Read the experiment and connect to AI

The horizontal axis counts terms; the blue values are actual partial sums. Harmonic mode has no finite reference sum. A — is not zero. At N=8, alternating error is bounded by 1/9 and the reciprocal-square tail by 1/8.

Accumulated error, learning-rate schedules and infinite feature expansions require these distinctions. Divergence of Σ1/n and convergence of Σ1/n² appear in some stochastic-approximation step conditions, but those two facts alone do not prove every optimizer converges.

## Exercises and answers

1. Terms that fail to vanish imply **series divergence**.
2. The alternating harmonic series is **not absolutely convergent**.
3. A sufficient N for the reciprocal-square tail bound to be ≤0.01 is **100**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
