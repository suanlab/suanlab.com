## One-step defects and accumulated error

A local defect starts from the exact state and takes just one numerical step. A global error follows the actual numerical trajectory, where each next step starts from an already perturbed state. Confusing these definitions creates an apparent discrepancy of one order. This chapter uses y′=-λy with y(0)=1, so the exact exponential is available independently of every numerical method.

Blue displays terminal global error and orange displays an unscaled local defect. The horizontal axis is step size h, not physical time. Each plotted point comes from a separate completed computation at the same final T.

## State the convention explicitly

$$
d_h=|\Phi_h(1)-e^{-\lambda h}|,\qquad
E_h=|\Phi_h^N(1)-e^{-\lambda T}|,\qquad Nh=T.
$$

Here the local defect is not divided by h. Some texts use the name local truncation error for a divided quantity, so inspect the formula before comparing stated orders. With sufficient regularity, a method of global order p typically has an unscaled one-step defect of order h^{p+1}. Roughly T/h injections accumulate, provided error propagation is stable, giving a global error of order h^p. This argument cannot be applied unchanged to unstable propagation.

## Measure ratios rather than prescribing a curve

The experiment repeats actual integration with N, 2N and 4N steps. Initial data and final time remain identical. The measured ratios are

$$
p_{obs}=\log_2(E_h/E_{h/2}),\qquad
q_{obs}=\log_2(d_h/d_{h/2}).
$$

For Euler, expect approximately one and two; for midpoint, two and three; for RK4, four and five. Finite step sizes retain higher-order terms, so an observed order need not be an exact integer. Zero errors make logarithmic ratios undefined. Near roundoff, cancellations can make ratios fluctuate rather than converge. Inspect both successive ratios and the absolute error sizes.

## A discrete Gronwall estimate separates three error sources

Assume the numerical one-step map is Lipschitz with factor at most 1+Lh on the region visited, the unscaled local defect is at most Ch^(p+1), and each computed step adds a perturbation of size at most ε. These are uniform hypotheses over the fixed interval T=Nh. Repeatedly applying the error recurrence gives

$$
|e_N|\le(1+Lh)^N|e_0|+
(Ch^{p+1}+\varepsilon)\sum_{j=0}^{N-1}(1+Lh)^j.
$$

For L>0, use (1+Lh)^N≤e^(LT) to bound the added terms by (Chᵖ+ε/h)(e^(LT)−1)/L. At L=0 the factor is T. Local consistency, stable propagation and evaluation accuracy are separate inputs; reducing h does not remove a fixed per-step perturbation budget.

With C=1, minimizing the simplified budget hᵖ+ε/h gives h=(ε/p)^(1/(p+1)). This is a worst-case assumed-error model, not a prediction that every machine error has the same sign or size. Near a roundoff plateau, observed error ratios cannot certify the nominal order.

| p | Balancing step for C=1 |
|---|---|
| 1 | ε^(1/2) |
| 2 | (ε/2)^(1/3) |
| 4 | (ε/4)^(1/5) |

```python
from fractions import Fraction as F
from math import exp
for N in (4,8,16):
    h=F(1,N);L=F(1);C=F(1);eps=F(1,10**8);p=2
    error=F(0)
    for _ in range(N):error=(1+L*h)*error+C*h**(p+1)+eps
    geometric=sum((1+L*h)**j for j in range(N))
    assert error==(C*h**(p+1)+eps)*geometric
    upper=(float(C*h**p)+float(eps/h))*(exp(1)-1)
    assert float(error)<=upper
    print(N,float(error),upper)
```

The rational recursion checks accumulation without using measured errors to fit an assumed order. In the live decay example, a stable expm1 comparison retains small analytic changes. The resulting measured defects still include the arithmetic used to execute each stage; they are not replaced by the Taylor leading term.

## Worked example and exercise

For λ=T=1, Euler with N=2 has global error approximately 0.117879. With N=4 it has error approximately 0.051473. The first observed order is about log₂(0.117879/0.051473)=1.195, already informative but not exactly one. The one-step defect at h=1/2 is |1/2-exp(-1/2)|≈0.106531, a different quantity.

Exercise: derive the leading Euler local defect. Solution: expand exp(-λh)=1-λh+λ²h²/2+O(h³). Subtract the Euler step 1-λh and take absolute value, obtaining approximately λ²h²/2. Halving h therefore reduces this defect by about four. Using that single defect as the terminal global error would omit accumulation and propagation.

## Reporting errors in AI research

When training through ODE models or physical simulators, equal step counts can correspond to different physical horizons. Fix the problem, T and error norm before comparing methods and cost. Adaptive solver tolerances are different settings from a fixed h and should not be presented as direct guarantees on terminal error. A convergence table supports computational verification; it does not by itself validate the physical model or the data assumptions.

Reference: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
