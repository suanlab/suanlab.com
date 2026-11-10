## Keep every iterate strictly inside

Consider minimizing cx on the closed interval [0,1]. A log barrier replaces the boundary constraints with an objective defined only on the strict interior:

$$
B_t(x)=tcx-\log x-\log(1-x),\qquad0<x<1.
$$

Here t=2 to the selected exponent. As t grows, the original objective dominates the barrier. Positive c pulls toward zero and negative c toward one. For c=0 every feasible point solves the original problem, while the barrier uniquely selects the midpoint.

## The exact central point is a reference

Differentiation gives B′=tc−1/x+1/(1−x) and B″=1/x²+1/(1−x)²>0. With C=tc≥0, the unique stationary point can be evaluated stably as

$$
x_t=\frac{2}{C+2+\sqrt{C^2+4}}.
$$

For C<0 use symmetry, x(C)=1−x(−C). This rationalized formula avoids subtracting nearly equal large numbers. The blue reference point is this analytic minimizer; it is not substituted for the computed Newton iterate.

## Actual damped Newton steps

Starting from the chosen interior point, each iteration proposes Δx=−B′/B″. A full step can cross a boundary. The algorithm first limits it to 99 percent of the distance to that boundary, then halves the step until the Armijo decrease condition with coefficient 0.01 holds.

The table records actual accepted step sizes, backtracks and iterates. Orange connects these discrete iterates on the normalized barrier B/t. It is not a continuous dynamical trajectory. The plotted curve covers only [0.001,0.999]; the objective diverges at the true boundaries, and the chart includes any computed points beyond that finite plotting window.

## A central gap and a current certificate

At the exact central point define λL=1/(txₜ), λU=1/[t(1−xₜ)]. They are nonnegative and satisfy c−λL+λU=0, so −λU is a feasible dual lower bound. Complementarity gives

$$
cx_t-(-\lambda_U)=\lambda_Lx_t+\lambda_U(1-x_t)=2/t.
$$

This identity applies to the exact central point. A finite Newton iterate usually has a nonzero stationarity residual. Its own reciprocal-slack multipliers need not satisfy dual stationarity and cannot automatically supply a lower bound.

The lab instead uses the exact central dual certificate for every current feasible iterate. Thus its current gap is cx_current+λU, which can differ from 2/t. Both quantities are shown separately. They bound objective error, rather than assert that the objective error equals the bound.

## Exercise and stopping decisions

With c=1 and t=1 the central point is approximately 0.381966 and its gap is two. At t=16 it is approximately 0.058609 with gap 0.125. Now choose t=256, start at 0.5 and request zero steps. The current gap is approximately 0.503922 while the central gap is only 0.0078125. Increasing t alone did not solve the barrier subproblem.

Repeat with negative c and observe the symmetry near the upper boundary. At c=0 the true objective error is already zero everywhere, even though the central certificate gap is positive. A stopping rule must distinguish a sufficient bound from exact error, and check both inner-solve accuracy and the barrier parameter.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
