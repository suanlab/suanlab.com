## One optimal-control problem, two useful viewpoints

A value function asks what the least remaining cost would be from every possible time and state. Pontryagin's principle follows a state trajectory together with an auxiliary sensitivity called the costate. For a smooth value function, the two viewpoints meet: the costate along an optimal trajectory is the state gradient of the value function.

This statement needs assumptions. A general optimal-control value function may have corners, and a costate is not automatically a gradient that exists everywhere. The present linear, convex, scalar example is smooth enough to derive the connection explicitly and prove global optimality. It provides a checkable reference before nonsmooth HJB equations or learned value approximations are introduced.

## Fix the time and leave the terminal state free

Use the continuous integrator and objective

$$
\dot x=u,\quad x(0)=x_0,\qquad
J(u)=\int_0^T(qx(t)^2+ru(t)^2)\,dt+p x(T)^2.
$$

The terminal time T is fixed. The terminal state is free and evaluated by a cost, rather than constrained to zero. Parameters q,p are nonnegative and r is positive. There are no input limits, state constraints, disturbances or observation noise. The cost contains no factors of one half, so derivatives produce factors of two.

The state and input have different units: input is a state change per unit time. Their curves can share a numerical plot, but comparing their magnitudes does not compare two efficiencies or energies. Similarly, an optimal-control Hamiltonian is not automatically a physical Hamiltonian or mechanical energy.

## From a short Bellman step to HJB

Define V(t,x) as the minimum remaining running plus terminal cost. Over a short interval, Bellman's principle compares the immediate cost with V at the resulting later state. Expanding a smooth V and dividing by the interval gives

$$
V_t+\min_u\left(qx^2+ru^2+V_xu\right)=0,
\qquad V(T,x)=p x^2.
$$

The boundary condition belongs to the whole terminal value function. It does not say V(T,x)=0 for every x unless p=0. Minimizing the strictly convex quadratic in u gives u*=-V_x/(2r).

Try V(t,x)=P(t)x². Substitution gives

$$
u^*(t,x)=-\frac{P(t)}{r}x,\qquad
\dot P=\frac{P^2}{r}-q,\qquad P(T)=p.
$$

This is a continuous-time differential equation solved from a terminal condition. It is not the discrete finite-horizon Riccati recurrence from the previous chapter. The derivative sign also changes if one switches from t to remaining time T-t; keep the time variable explicit.

## Derive the costate and its terminal condition

Use the **minimum** Hamiltonian

$$
H(x,u,\lambda)=qx^2+ru^2+\lambda u.
$$

Introduce the dynamics constraint through the augmented integral H-lambda*xdot. Varying u gives 2ru+lambda=0. Integrating the variation in x by parts gives the interior condition lambda_dot=-2qx and the free-endpoint boundary term (2p*x(T)-lambda(T))*delta x(T). Since the terminal variation is free,

$$
\dot x=u,\qquad \dot\lambda=-2qx,\qquad
u=-\frac{\lambda}{2r},\qquad \lambda(T)=2p x(T).
$$

The initial state and terminal costate condition are specified at opposite ends of the interval: this is a boundary-value problem. A source using a maximum-Hamiltonian convention can use the opposite costate sign. Do not mix its terminal sign with the minimum convention here. Because T is fixed, we do not impose the free-time condition H(T)=0.

Now set lambda=2P*x along the proposed optimal trajectory. Differentiation gives lambda_dot=2*Pdot*x+2*P*u=-2q*x, and the terminal relation follows from P(T)=p. Thus lambda=V_x, with matching state, control and boundary equations. The costate is a sensitivity of remaining optimal cost to state, not an extra physical state measured by a sensor.

## What the visualization displays

At the selected time, the blue bowl is V(t,x) for hypothetical states across the horizontal axis. The dot is the actual optimal state at that time. The orange line is the tangent there; its slope is lambda. A tangent can be negative away from its contact point even though the nonnegative optimal cost cannot. It is a local linear approximation, not a second value function.

The lower plot shows the actual optimal state and input over continuous time. The table lists the costate, value gradient and remaining cost along that trajectory. Moving the selected-time control updates the value-function view and contact point. When P is constant, the bowl stays unchanged while the point moves. Selecting a different time does not change the underlying optimal trajectory when the other parameters stay fixed.

## Compute the solution without a false zero-cost regularizer

Let tau=T-t and k=sqrt(q/r). Define S(tau)=sinh(k*tau)/k when k is nonzero, with the continuous limit S(tau)=tau at k=0. The implementation evaluates this through tau*sinh(k*tau)/(k*tau), using its exact limit when the argument is zero. Then define

$$
F(\tau)=\cosh(k\tau)+\frac{p}{r}S(\tau),\qquad
G(\tau)=qS(\tau)+p\cosh(k\tau).
$$

The solution is

$$
P(t)=\frac{G(T-t)}{F(T-t)},\quad
x(t)=x_0\frac{F(T-t)}{F(T)},\quad
u(t)=-\frac{x_0G(T-t)}{rF(T)}.
$$

Here F is positive on the stated domain, F_tau=G/r and G_tau=qF. These identities verify the Riccati, state and costate equations directly. They also retain q=0 and p=0 without replacing either by a hidden positive epsilon. In particular, when q=0,

$$
x(t)=x_0\frac{r+p(T-t)}{r+pT},\qquad
u(t)=-\frac{p x_0}{r+pT}.
$$

The displayed quadrature cost is a Simpson approximation to running plus terminal cost; it is compared with the exact value P(0)*x0². Independent tests additionally solve the state–costate boundary problem by RK4 shooting and differentiate V numerically. Finite residual samples are numerical checks, not the general proof below.

## Why this candidate is globally optimal

For any admissible trajectory with the same initial state, differentiate P(t)*x(t)² and use the Riccati equation:

$$
qx^2+ru^2+\frac{d}{dt}\left(Px^2\right)
=r\left(u+\frac{P}{r}x\right)^2.
$$

Integrating and using P(T)=p yields

$$
J(u)-P(0)x_0^2
=\int_0^T r\left(u+\frac{P}{r}x\right)^2dt\geq0.
$$

The proposed feedback makes the square vanish and attains the lower bound. This is a sufficiency argument over all admissible finite-cost controls, not just a test of one plotted trajectory. Pontryagin conditions alone are generally necessary conditions and can identify nonoptimal candidates in other problems. Here linear dynamics, convex cost and this verification identity establish the stronger result.

## Research extension: when the value has a corner

The native quadratic problem has a smooth value, but this is special. In an auxiliary bounded integrator $\dot x=u$, $|u|\le1$, with remaining time $\tau$, zero running cost, and terminal cost $|x(T)|$, the exact value is

$$V(x,t)=\max(|x|-(T-t),0).$$

The triangle inequality gives $|x(T)|\ge\max(|x|-\tau,0)$. A constant control $u=-\operatorname{sign}(x)\min(1,|x|/\tau)$ attains the bound. There are corners at $x=\pm\tau$; forcing a single classical gradient there is incorrect. Away from corners, $V_t-|V_x|=0$. Viscosity solutions extend HJB through such corners using smooth test functions touching from above or below; uniqueness additionally needs a comparison principle and correct terminal data.

| $x$, with $\tau=1$ | Value | An optimal constant input |
|---|---:|---:|
| -3/2 | 1/2 | 1 |
| 0 | 0 | 0 |
| 3/2 | 1/2 | -1 |

![Exact nonsmooth value curves at three remaining times; the flat reachable region widens with time and meets linear tails at genuine corners.](/assets/learning/control-nonsmooth-value-en.svg)

```python
from fractions import Fraction as F
for x in [F(-2),F(-1),F(-1,2),F(0),F(1,2),F(1),F(2)]:
    for tau in [F(1,4),F(1,2),F(1)]:
        sign=(x>0)-(x<0)
        u=-sign*min(F(1),abs(x)/tau)
        assert abs(u)<=1
        assert abs(x+tau*u)==max(abs(x)-tau,0)
print('21 bounded-input optimality checks passed')
```

A sampled neural PDE residual cannot by itself certify a global viscosity solution. This auxiliary terminal-absolute-value problem does not replace the native smooth quadratic costate calculation.

## Worked exercise and solution

Set q=0,r=p=T=x0=1. The optimal input is constant -0.5, so x(t)=1-t/2 and x(T)=0.5. Running cost is 0.25 and terminal cost is 0.25, giving J=0.5. The costate is constant 1, consistent with lambda(T)=2*1*0.5. At t=0.5, P=2/3, x=0.75 and V_x=2*(2/3)*0.75=1. The Hamiltonian is -0.25, illustrating why imposing H(T)=0 would be wrong for this fixed-time problem.

For a second exercise set q=r=T=x0=1,p=0. The terminal conditions give lambda(T)=u(T)=0, but x(T)=1/cosh(1), about 0.648054. The optimal cost is tanh(1), about 0.761594. A free endpoint with zero terminal penalty does not require zero terminal state. Check the flat terminal bowl and the nonzero state in the third guided example.

## Connection to AI research

Neural value functions can be trained against HJB residuals, and differentiable optimal control can use costate or adjoint sensitivities. Correct dynamics, Hamiltonian sign, boundary data and regularity assumptions remain essential. A low residual on sampled points does not automatically establish a global solution, and nonsmooth value functions require an appropriate generalized framework. This analytic model supplies exact values and sensitivities for testing those larger methods.

## Primary references

Daniel Liberzon, [Principle of optimality](https://liberzon.csl.illinois.edu/teaching/cvoc/node94.html), [HJB equation](https://liberzon.csl.illinois.edu/teaching/cvoc/node95.html) and [Hamiltonian sign conventions](https://liberzon.csl.illinois.edu/teaching/cvoc/node58.html), provide the control-theory setting. The fixed-time scalar derivation, entire-function evaluation, visualization, sufficiency identity and numerical experiments here are original teaching constructions.
