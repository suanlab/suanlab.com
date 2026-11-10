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

## Worked example and exercise

For λ=T=1, Euler with N=2 has global error approximately 0.117879. With N=4 it has error approximately 0.051473. The first observed order is about log₂(0.117879/0.051473)=1.195, already informative but not exactly one. The one-step defect at h=1/2 is |1/2-exp(-1/2)|≈0.106531, a different quantity.

Exercise: derive the leading Euler local defect. Solution: expand exp(-λh)=1-λh+λ²h²/2+O(h³). Subtract the Euler step 1-λh and take absolute value, obtaining approximately λ²h²/2. Halving h therefore reduces this defect by about four. Using that single defect as the terminal global error would omit accumulation and propagation.

## Reporting errors in AI research

When training through ODE models or physical simulators, equal step counts can correspond to different physical horizons. Fix the problem, T and error norm before comparing methods and cost. Adaptive solver tolerances are different settings from a fixed h and should not be presented as direct guarantees on terminal error. A convergence table supports computational verification; it does not by itself validate the physical model or the data assumptions.

Reference: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
