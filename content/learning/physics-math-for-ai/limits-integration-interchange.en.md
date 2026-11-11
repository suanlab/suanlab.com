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

## AI interpretation

Moving a limit through an expectation, integrating a series termwise or differentiating an expected loss requires assumptions. Dominated convergence for limits does not automatically justify differentiation; difference quotients or derivatives need suitable additional control.

## Estimate the same area from random inputs

Let $U_1,\ldots,U_M$ be independent uniform inputs on $[0,1]$, with $M\ge1$. The sample mean $\widehat I_{n,M}=M^{-1}\sum_i f_n(U_i)$ estimates the integral of the same triangular spike. For integer $n\ge2$, symmetry of its two linear sides gives

$$
\mathbb E[f_n(U)]=1,\qquad
\mathbb E[f_n(U)^2]=\frac{2n}{3},\qquad
\operatorname{Var}(\widehat I_{n,M})=\frac{2n/3-1}{M}.
$$

For the second moment, substitute $v=nx$ on the rising side and double the result, obtaining $2n\int_0^1v^2\,dv=2n/3$. Independence of the samples makes variance of their mean equal to single-sample variance divided by M. The estimate is unbiased for each fixed n and M, while its variance grows with n at a fixed batch size.

A sample has a positive value only inside $(0,2/n)$. Thus the probability that every sample misses the spike is $(1-2/n)^M$. With batch size eight,

| Index n | Probability of an all-zero batch | Variance of the sample mean |
|---|---|---|
| 4 | 0.003906 | 5/24 |
| 16 | 0.343609 | 29/24 |
| 64 | 0.775700 | 125/24 |

An all-zero observed batch has empirical variance zero too, although the population variance is positive. Record how the inputs were sampled when interpreting a small error bar or a plot with no visible mass.

## Keep the two limits explicit

Fix M and reuse the same finite set of uniform draws while n increases. Almost surely their minimum is positive. Eventually $2/n$ is smaller than that minimum, so every sampled value is zero and $\widehat I_{n,M}\to0$ almost surely. Its expectation nevertheless stays one. The increasing rare values prevent exchange of this limit and expectation.

For a fixed n, increasing M reduces variance. If both vary and $M_n/n\to\infty$, the displayed variance tends to zero. Since the estimator is unbiased, this gives convergence in mean square to one. These are different limiting procedures. Specify which parameters are fixed and which grow when arguing that a larger simulation or batch resolves the approximation.

## Change the proposal and preserve the target integral

Choose independent $X_i$ uniformly on $(0,2/n)$, whose density there is $q_n=n/2$. To estimate the original integral, average the weighted values $f_n(X_i)/q_n(X_i)$. The general identity requires the proposal to cover all nonzero integrand contributions. [Owen, Importance sampling, §9.1 and Theorem 9.1](https://artowen.su.domains/mc/Ch-var-is.pdf)

Here $V=nX_i-1$ is uniform on $(-1,1)$ and $H=1-|V|$ is uniform on $[0,1]$. For $0\le h\le1$, the two outer intervals give $\Pr(H\le h)=h$. Therefore

$$
\frac{f_n(X_i)}{q_n(X_i)}=2H,\qquad
\mathbb E[2H]=1,\qquad
\operatorname{Var}\!\left(\frac1M\sum_i2H_i\right)=\frac1{3M}.
$$

The weighted estimator's variance is independent of n. Averaging the unweighted spike values under this proposal instead has expectation $n/2$. Restricting the proposal further to $(0,1/n)$ misses half the area and gives weighted expectation $1/2$. Correct weights and adequate support are both required. The support is known in this constructed example; a research problem may require learning or locating it.

## Check moments with a finite exact calculation

Let K count positive observations under uniform sampling on $[0,1]$. It has the binomial distribution with success probability $2/n$. Given $K=k$, the k positive heights are independent $nH_j$, with $H_j$ uniform on $[0,1]$. Since their mean and variance are $1/2$ and $1/12$,

$$
\mathbb E[\widehat I\mid K=k]=\frac{nk}{2M},\qquad
\mathbb E[\widehat I^2\mid K=k]
=\frac{n^2}{M^2}\left(\frac{k}{12}+\frac{k^2}{4}\right).
$$

Sum these conditional moments over the M+1 possible counts. The Python example uses exact rational arithmetic, so its identities do not depend on a random seed or a favorable finite simulation. It also checks the moments by integrating each side and its square with Simpson's rule at the known breakpoints 0, 1/n and 2/n. For degree-at-most-three polynomials, interval width times (left value + four times midpoint value + right value)/6 is the exact integral, as checking the four monomials shows.

```python
from fractions import Fraction
from math import comb


def spike(index, x):
    return index * max(0, 1 - abs(index * x - 1))


batch_size = 8
for index in [4, 16, 64]:
    # Simpson's rule is exact on each linear side and its quadratic square.
    area = integral_squared = Fraction(0)
    for left, right in [(Fraction(0), Fraction(1, index)),
                        (Fraction(1, index), Fraction(2, index))]:
        values = [spike(index, x) for x in [left, (left + right) / 2, right]]
        piece_area = (right - left) * (values[0] + 4*values[1] + values[2]) / 6
        piece_squared = (right - left) * (
            values[0]**2 + 4*values[1]**2 + values[2]**2
        ) / 6
        if left == 0:
            half_area = piece_area
        area += piece_area
        integral_squared += piece_squared
    assert area == 1 and half_area == Fraction(1, 2)
    assert integral_squared == Fraction(2 * index, 3)

    hit_probability = Fraction(2, index)
    mean = estimate_second = Fraction(0)
    for hits in range(batch_size + 1):
        probability = (
            comb(batch_size, hits)
            * hit_probability**hits
            * (1 - hit_probability)**(batch_size - hits)
        )
        conditional_mean = Fraction(index * hits, 2 * batch_size)
        conditional_second = index**2 * (
            Fraction(hits, 12 * batch_size**2)
            + Fraction(hits**2, 4 * batch_size**2)
        )
        mean += probability * conditional_mean
        estimate_second += probability * conditional_second
    variance = estimate_second - mean**2
    missed = (1 - hit_probability)**batch_size
    assert mean == area
    assert variance == (integral_squared - area**2) / batch_size

    proposal_density = Fraction(index, 2)
    importance_variance = (
        integral_squared / proposal_density - area**2
    ) / batch_size
    assert importance_variance == Fraction(1, 3 * batch_size)
    assert variance / importance_variance == 2 * index - 3
    assert proposal_density * area == Fraction(index, 2)
    print(index, round(float(missed), 6), variance, importance_variance)
```

Each row reports n, missed-batch probability, original variance, and support-proposal variance. The last variance is always `1/24` for M=8. The displayed probabilities are rounded; the moment checks use exact fractions. The weighting and support failures are checked through their changed expectation, even when a single observed batch might appear plausible.

## Connect the calculation to an expected gradient

Consider the scalar loss $\ell_n(\theta,x)=\theta^2/2-\theta f_n(x)$. Its population objective and gradient are

$$
L_n(\theta)=\frac{\theta^2}{2}-\theta,\qquad
L_n'(\theta)=\theta-1.
$$

For each fixed n, the sample gradient is $\theta-\widehat I_{n,M}$ and is unbiased, with variance $(2n/3-1)/M$. At $\theta=0$, an all-zero batch returns gradient zero while the population gradient is minus one. The probability in the table quantifies this event. Convergence claims for an optimization procedure additionally need the update rule, step sizes and sampling schedule. For this loss the quadratic term is evaluated directly; importance weighting is applied to the spike integral.

For fixed n, on a local parameter interval $|\theta|\le C$, the derivative satisfies $|\theta-f_n(x)|\le C+n$. This finite constant is an integrable local envelope, consistent with differentiating the expected loss here. Its dependence on n leaves the earlier limit-interchange counterexample in force.

A research extension can vary n and M, retain all-zero batches, compare the two proposals under equal evaluation budgets, and deliberately shorten or misplace the proposal support. Report the target measure, weighting rule, bias, variance and cost. Separate exchanging a limit with expectation from differentiating at a fixed n. The triangles and finite moment audit here are independently constructed teaching examples.

## Exercises and answers

1. At n=32, spike height and area are **32 and one**.
2. The integral of its pointwise limit is **zero**.
3. At n=8 the comparison integral is **1/16**.
4. A grid missing the spike demonstrates **sampling failure, not a theorem**.

5. Does an unbiased estimator have small variance at every n? **No. Its variance here is $(2n/3-1)/M$, which grows at fixed M.**
6. Why does a proposal on half the support remain biased after weighting? **Its expectation integrates only that half, whose area is $1/2$.**

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
