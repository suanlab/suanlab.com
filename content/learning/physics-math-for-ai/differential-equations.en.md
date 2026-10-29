## A law of change is not yet a trajectory

A differential equation specifies an instantaneous rate. An initial condition selects a trajectory. A numerical method then approximates that trajectory at discrete times. These are three different objects. Neural differential equations and differentiable simulators rely on this distinction: learning a continuous law does not remove numerical error in its solver.

Use the scalar initial-value problem $x'(t)=-kx(t)$ with $k>0$ and $x(0)=x_0$. Interpret time in chosen units and $k$ in inverse time. The state may be dimensionless; the product $k\Delta t$ is always dimensionless.

## Solve the continuous problem

The exact solution is

$$
x(t)=x_0e^{-kt}.
$$

Differentiating verifies the equation and evaluating at zero verifies the initial condition. The right-hand side is globally Lipschitz in the state, so this initial-value problem has a unique solution. For positive initial state, it stays positive and decays toward zero.

This is first-order relaxation, not an undamped Newtonian oscillator. Gradient flow of a quadratic shares this equation after a mathematical identification, but a physical interpretation still needs units, dissipation and a specified system.

## Construct forward Euler

Approximate the short-time change using the current derivative:

$$
x_{n+1}=x_n+\Delta t(-kx_n)=(1-k\Delta t)x_n.
$$

For $x_0=1,k=1,\Delta t=0.25$, the first Euler value is 0.75. The exact value at time 0.25 is approximately 0.778801. After 20 steps, time is 5: exact state is about 0.006738, while Euler gives $0.75^{20}\approx0.003171$.

The method uses the old state for the slope. Replacing it with the new unknown state creates implicit Euler, a different method with different stability behavior.

## Derive the stability condition

Euler’s repeated multiplier is $r=1-k\Delta t$. Its numerical state tends to zero when $|r|<1$, equivalent to $0<k\Delta t<2$. For $0<k\Delta t\le1$, nonnegative states remain nonnegative. Between 1 and 2, the numerical solution alternates signs while shrinking even though the exact positive solution never changes sign.

At $k\Delta t=2$, the multiplier is $-1$ and the magnitude does not decay. Above 2, the numerical magnitude grows. The unstable preset uses $k=3,\Delta t=1$, so every step multiplies by $-2$. This growth is numerical instability, not a property of the exact decay law.

## Accuracy requires a fixed comparison time

For a smooth solution, forward Euler has local one-step truncation error of order $\Delta t^2$ and global error of order $\Delta t$ over a fixed finite time interval under appropriate stability conditions. The orders concern a refinement process, not every arbitrary pair of experiments.

The interactive lab fixes the number of steps at 20. Decreasing the step changes the terminal time too. A smaller final error at a different time is not by itself a valid order-of-convergence measurement. For a convergence study, fix $T$, use step counts $T/h,T/(h/2),T/(h/4)$ and compare all states at $T$.

## Read the plot and diagnose failure

Blue is the exact solution and orange is Euler evaluated at the same displayed times. Numeric values are not clipped to force a stable-looking curve. When the numerical state grows, auto-scaling can make the exact solution appear flat; inspect the terminal error and dimensionless multiplier.

Separate three questions: does the continuous equation describe the system, does the numerical method approximate its solution, and do the inferred parameters fit observations? A successful fit can compensate for discretization error by learning the wrong continuous parameter.

## AI and research connections

A solver inside a learned model becomes part of the computational objective. Step size, tolerance, interpolation and gradient computation can affect results. Compare learned parameters and predictions across solver refinements, not only training loss at one resolution. Record whether gradients refer to the discretized solver or an independently derived continuous sensitivity.

This unit treats one linear ordinary differential equation. Stiff systems, partial differential equations, boundary conditions and adaptive solvers require additional analysis; passing this example does not validate a general scientific simulator.

## Exercises and reference

1. For $k=2,h=0.25$, calculate the multiplier. **Answer:** 0.5.
2. For $k=1,h=1.5$, explain why a positive initial value changes sign numerically.
3. Design a fixed-final-time refinement test and predict the error ratio for first-order convergence.
4. Explain why learning $k$ with a coarse solver can bias the inferred physical law.

See [MIT Differential Equations](https://ocw.mit.edu/courses/18-03-differential-equations-spring-2010/download/) for Euler’s method and continuous models. The exact solution and stability calculation here are worked directly for the specified test equation.
