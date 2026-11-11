## The evaluation point changes a stochastic sum

Ordinary smooth calculus often hides the difference between evaluating an integrand at the left endpoint and near the middle of a small interval. Brownian increments have a nonvanishing accumulated square, so that difference survives refinement. The lab uses the same Brownian realization for both sums, eliminating an unrelated change of sample as an explanation.

For integrand W, the blue cumulative sum uses the left value W_i. The orange sum averages W_i and W_(i+1). This symmetric endpoint rule is the midpoint convention for this particular linear integrand, rather than a claim that every numerical midpoint formula is interchangeable for arbitrary stochastic equations.

## Derive two exact finite-grid identities

Expanding the square gives W_next²-W_i²=2 W_i dW_i+(dW_i)². Summing from W0=0 yields

$$
L_N=\sum_i W_i\Delta W_i=\frac12(W_T^2-QV_N),\qquad
S_N=\sum_i\frac{W_i+W_{i+1}}2\Delta W_i=\frac12W_T^2.
$$

These are algebraic identities on every finite path, with any increment values. Their validity does not require the observed QV to equal T. The code computes the sums term by term and reports the independently evaluated identities so numerical agreement can be checked.

## Pass to the Brownian limiting integral

Brownian quadratic variation converges to elapsed time under the appropriate refinement. The corresponding integrals are

$$
\int_0^T W_t\,dW_t=\frac12(W_T^2-T),\qquad
\int_0^T W_t\circ dW_t=\frac12W_T^2.
$$

The circle denotes the Stratonovich convention. Its chain rule resembles ordinary calculus in this example; the Itô version includes the second-order correction. Replacing a finite observed QV by T before stating which quantity is computed would erase the discretization error rather than measure it.

## General chain rules and drift conversion

For a smooth function F and an Itô SDE dX=b dt+sigma dW, the chain rule includes a second derivative:

$$
dF(X)=F'(X)b\,dt+F'(X)\sigma\,dW+\tfrac12F''(X)\sigma^2\,dt.
$$

Under the corresponding smoothness assumptions, Stratonovich notation uses the ordinary chain-rule form. In one dimension the same process has drifts related by b_I=b_S+sigma sigma'/2. A constant diffusion has no drift-conversion term, yet a nonlinear F still has an Itô chain-rule correction. These are different statements; the W-squared example isolates the latter.

## An exact mean-square error is available

The left-sum error against the Itô endpoint integral equals (T-QV_N)/2. On a uniform Brownian grid its mean is zero and its mean-square value is T²/(2N). This is an expectation across Gaussian paths. One observed squared error can be larger or smaller, and grid refinements preserve the endpoint without forcing QV errors to decrease monotonically.

At T=1,N=64 the theoretical MSE is 1/128. At T=2,N=16 it is 1/8. The scaling is quadratic in T and inverse in N, which is different from the sample path's square-root time scaling.

## Adaptedness is what makes the isometry work

Assume EΣHᵢ²Δtᵢ<∞. For a simple predictable integrand Hᵢ measurable before ΔWᵢ, each term HᵢΔWᵢ has conditional mean zero. Cross terms vanish by conditioning on the later increment, and the diagonal term is E[Hᵢ²]Δtᵢ. Therefore

$$
E\left[\left(\sum_iH_i\Delta W_i\right)^2\right]
=\sum_i E[H_i^2]\Delta t_i.
$$

Completing predictable simple processes in L²(dt×dP) constructs the Itô integral and extends this identity. It is not an identity for arbitrary future-dependent integrands. Taking Hᵢ=W_T for every interval gives W_TΣΔWᵢ=W_T², whose expectation is T rather than zero. The missing hypothesis is adaptedness, not a numerical accuracy issue.

For a finite symmetric walk with N increments ±1/√N, the predictable left sum with Hᵢ=Wᵢ has second moment (N−1)/(2N). This discrete martingale example is exactly enumerable. Its deterministic QV equals one; its finite increment distribution is not Gaussian.

| N | E[left sum squared] |
|---|---:|
| 2 | 1/4 |
| 4 | 3/8 |
| 8 | 7/16 |

```python
from fractions import Fraction as F
from itertools import product
for N in (2, 4, 8):
    second = anticipated_mean = isometry = F(0)
    for increments in product((-1,1), repeat=N):
        state, left, diagonal = 0, 0, 0
        for dw in increments:
            left += state*dw
            diagonal += state*state
            state += dw
        second += F(left*left, N*N*2**N)
        isometry += F(diagonal, N*N*2**N)
        anticipated_mean += F(state*state, N*2**N)
    assert second==isometry==F(N-1,2*N) and anticipated_mean==1
    print(N, second, anticipated_mean)
```

The code uses unscaled ±1 increments and normalizes the sum by N, so exact fractions avoid irrational arithmetic. The anticipated sum is W_T²/N and has mean one. In research code, moving a state evaluation to the next step can change the stochastic integral being approximated even when ordinary deterministic quadrature would tolerate that change.

## Exercise: distinguish three corrections

For the displayed realization calculate S_N-L_N, QV_N/2 and T/2. The first two agree up to rounding; the third is the limiting correction and generally differs on a finite grid. Then replace every increment on paper by a deterministic smooth path increment. The finite identities remain true, but the limiting QV behavior changes. Explain why the stochastic calculus distinction comes from the process law as well as the summation convention.

Further reading: [Lawler's stochastic-integral notes](https://www.math.uchicago.edu/~lawler/finbook.pdf). The discrete identities and MSE experiment are derived independently.
