## A flow maps initial data into later states

Fix a differential equation and let S_t(a) denote its state at time t from initial value a. A central question is whether adding initial values before evolution gives the same result as adding evolved states. The lab compares S_t(a+b) in blue with S_t(a)+S_t(b) in orange, keeping the equation and its forcing fixed.

This is a test of superposition, not simply a test of whether two plotted curves look similar. At time zero both expressions agree by construction. Later agreement must come from the structure of the equation or a special parameter choice.

## Homogeneous linear evolution preserves addition

For y'=r y,

$$
S_t(a)=ae^{rt},\qquad S_t(a+b)=S_t(a)+S_t(b).
$$

The same idea extends to matrix exponentials for linear systems. Exact arithmetic gives zero superposition defect. The lab also reports the expanded floating-point subtraction, which can contain a tiny rounding remainder even when the analytic identity is exact. Numerical noise should not be mistaken for mathematical nonlinearity.

## Constant forcing makes the flow affine

For y'=r y+f with fixed f,

$$
S_t(a)=ae^{rt}+f\phi(r,t),\qquad
\phi(r,t)=\begin{cases}(e^{rt}-1)/r&r\ne0,\\t&r=0.\end{cases}
$$

Consequently, the superposition defect is minus f times phi. Each separately evolved solution has received the forcing once, so adding them counts it twice. The equation is nonhomogeneous linear, but its map from initial value to final state is affine. Differences of two solutions still satisfy the homogeneous equation.

Near r=0, directly subtracting one from exp(rt) loses digits. The implementation uses expm1 and a small-argument series, including the limiting value t. This connects analytic solution formulas to numerical stability rather than treating exact mathematics and reliable computation as unrelated tasks.

## Logistic evolution introduces state-dependent saturation

For y'=r y(1-y),

$$
S_t(a)=\frac{a}{a+(1-a)e^{-rt}}.
$$

The controls restrict a and b so each and their sum remain strictly between zero and one. This avoids an unintended finite-time pole in the negative-rate, above-one regime. Within the specified domain the solution exists for the displayed times and is generally nonlinear in the initial value.

At r=1, a=0.2, b=0.3 and t=1, evolving 0.5 differs from adding the evolutions of 0.2 and 0.3. At r=0 every state stays constant, so this special logistic parameter happens to preserve addition without making the general family linear.

## Exercise: locate the extra forcing

Take r=1 and f=1/2. Write out the three forced solutions and cancel their initial-value terms. The defect is -(e^t-1)/2. At r=0 it becomes -t/2. Explain why subtracting two forced solutions removes f, while adding them does not. Finally compare with the homogeneous preset and distinguish its analytic zero from any displayed expanded rounding remainder.

Further reading: [MIT linear and nonlinear differential equations](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). These three flow comparisons are derived explicitly.
