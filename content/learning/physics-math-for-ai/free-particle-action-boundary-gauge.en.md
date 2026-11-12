## A path is the object we compare

Imagine two photographs of a cart on a straight, frictionless track. The first fixes its position at time zero and the second fixes its position at time T. A straight line in a position–time diagram describes constant velocity. A bowed curve also connects the photographs, but it describes acceleration between them. The variational question compares these entire candidate paths while keeping the physical experiment's specified boundary data fixed.

A candidate path is a differentiable function, not automatically a solution of Newton's equation. This distinction matters for AI models that output trajectories: matching two observations does not establish that a predicted path obeys the force law. We will compute a scalar action, its response to changes of path, and the residual of the differential equation separately.

## Separate physical time from a path perturbation

Let mass m and duration T be strictly positive. Write s=t/T, so the dimensionless coordinate s runs from zero to one. Position is measured in meters, velocity in meters per second, and physical time in seconds. Choose

$$
f_1(s)=s(1-s),\qquad f_2(s)=s(1-s)(2s-1),
$$

$$
q(s)=a+(b-a)s+A f_1(s)+B f_2(s).
$$

Both shape functions vanish at the endpoints. Therefore changing A or B preserves q(0)=a and q(T)=b. Their amplitudes have units of meters. The first produces a symmetric bow and the second an antisymmetric distortion around the midpoint. Time derivatives require the chain rule: velocity is q_s/T and acceleration is q_ss/T². Forgetting these factors changes both the physical equation and the units of action.

## Define the Lagrangian and action

For a free particle use L₀=m v²/2. Its units are joules. The action is the time integral

$$
S_0[q]=\int_0^T L_0\,dt
=\frac{m}{2T}\left[(b-a)^2+\frac{A^2}{3}+\frac{B^2}{5}\right].
$$

Action has units of joule seconds. It is not the energy at the cursor. To derive this expression, differentiate the polynomial, square q_s, and integrate from zero to one with dt=T ds. Cross terms vanish: each shape derivative integrates to zero, and the product of the two shape derivatives is odd about the midpoint. Their squared integrals are 1/3 and 1/5. These cancellations explain the diagonal quadratic rather than merely providing a convenient formula.

## Locate the stationary path and prove the minimum

The amplitude derivatives are mA/(3T) and mB/(5T). Both vanish only when A=B=0. The corresponding path has constant velocity (b−a)/T and zero acceleration. The Hessian is diagonal with positive entries m/(3T) and m/(5T), so this path is the unique minimum within our family.

The free-particle claim also holds beyond this family. Write any admissible path as q_cl+η, where η has zero endpoints and square-integrable derivative. Expanding the action gives a cross term proportional to the integral of ηdot. That integral is η(T)−η(0)=0. Consequently,

$$
S_0[q_{\mathrm{cl}}+\eta]-S_0[q_{\mathrm{cl}}]
=\frac m2\int_0^T\dot\eta^2\,dt\geq0.
$$

Equality forces η to be constant almost everywhere; its zero endpoints then force η=0. This is a full fixed-endpoint minimum proof. The two-amplitude calculation alone would establish only a restricted-family result.

## Add a total time derivative

Define Fγ(q)=γq²/2, where γ has units kg/s. Its time derivative is γqv and has energy units. A second Lagrangian is

$$
L_\gamma=L_0+\frac{dF_\gamma}{dt}
=L_0+\gamma qv,\qquad
S_\gamma=S_0+\frac\gamma2(b^2-a^2).
$$

For fixed endpoints the added quantity is constant over the competing paths. It changes action values without changing amplitude gradients or the stationary path. Negative action is possible even when kinetic energy is positive. The zero of action is not a universal detector of physical motion, energy, or stability.

The canonical momentum becomes pγ=mv+γq, whereas mechanical momentum remains mv. Direct differentiation shows d(pγ)/dt−∂Lγ/∂q=m qddot: the γ terms cancel. The Hamiltonian pγv−Lγ equals kinetic energy here because Fγ has no explicit time dependence. Along an arbitrary accelerating trial path this quantity need not be conserved.

## State which endpoint changes are allowed

Let a variation be

$$
\xi(s)=u_0(1-s)+u_1s+C f_1(s)+D f_2(s),
\qquad q_\epsilon=q+\epsilon\xi.
$$

Epsilon is dimensionless and labels different functions. It does not advance the cart in time. The variation fixes the original endpoints only if u₀=u₁=0. If either is nonzero, it compares a different boundary problem. For fixed physical endpoint times, integration by parts gives

$$
\delta S_\gamma=[(mv+\gamma q)\xi]_0^T
-\int_0^T m\ddot q\,\xi\,dt.
$$

At the constant-velocity path the interior integral is zero. The boundary term can remain nonzero when endpoint positions are allowed to change. Adding a total derivative preserves the fixed-endpoint problem, but freely moving endpoints require the natural boundary conditions to be transformed as well. We are not varying endpoint times in this lesson.

## Compute an exact finite perturbation

Write d=b−a and Δu=u₁−u₀. Differentiating with respect to epsilon at zero gives

$$
\delta S_\gamma=\frac mT d\,\Delta u
+\gamma(bu_1-au_0)+\frac{mAC}{3T}+\frac{mBD}{5T}.
$$

The second variation is

$$
\delta^2S_\gamma=\frac mT\left[(\Delta u)^2+\frac{C^2}{3}+\frac{D^2}{5}\right]
+\gamma(u_1^2-u_0^2).
$$

Because the action is quadratic, S(qε)=S(q)+εδS+ε²δ²S/2 exactly. There is no neglected cubic term in this example. Gradients with respect to A and B have units Js/m; derivatives with respect to dimensionless epsilon have units Js. A zero directional derivative can result from choosing a direction orthogonal to the gradient. It does not by itself certify stationarity under every admissible variation.

## Worked example: the same path with different action

Set m=T=1, a=0, b=1, A=B=0. Velocity is 1 m/s, kinetic energy is 1/2 J, and S₀=1/2 Js. Choose a fixed-endpoint bow C=1, D=0. The first variation is zero and the action increase is ε²/6 Js. At ε=1/2 the varied action is 13/24 Js.

Now choose γ=−2 kg/s. The boundary shift is −1 Js, so the original path has action −1/2 Js. Velocity, kinetic energy, and the Newton residual are unchanged. Finally let u₀=0, u₁=1, C=D=0. The first variation is 1 Js for γ=0 and −1 Js for γ=−2. This is an endpoint change, so fixed-endpoint invariance does not apply.

## Read the interactive diagrams

Compare position against the constant-velocity reference before reading a single action number. Then inspect velocity and the whole-path Newton residual. Keep the Lagrangian in joules and cumulative action in joule seconds on distinct axes. An action-versus-epsilon plot describes a family of trajectories, so its horizontal axis must never be labeled time.

The calculation uses reduced rational arithmetic for bounded rational controls. Polynomial action and its integration-by-parts identity are exact for those controls. An independent five-node Boole quadrature integrates the degree-five gauged Lagrangian exactly. Floating-point plotting is a display of these values, not a numerical dynamics integrator.

## Research extension: transform terminal costs with the action gauge

Suppose the second photograph is a preference rather than an exact endpoint. The optimizer may choose a nearby final position and pay a penalty. A boundary term that was constant in the fixed-endpoint problem now varies across candidates. Keeping the interior Newton equation unchanged is not enough to keep this optimization task unchanged.

### Define a soft endpoint with consistent units

Fix $a$ and the physical times $0,T$, with $m,T,\rho>0$. Allow the final position $b$ to vary over the whole real line, without a terminal position bound. Use an action-valued terminal cost:

$$
\mathcal J_\gamma[q]=\int_0^T\left(\frac m2\dot q^2+\gamma q\dot q\right)dt+\Phi_0(b),\qquad
q(0)=a,\quad b=q(T)\in\mathbb R\ \text{free},\qquad
\Phi_0(b)=\frac\rho2(b-b_{\rm target})^2.
$$

Both $\gamma$ and $\rho$ have units kg/s, so the integral and terminal cost have units J s. The weight $\rho$ is not a spring stiffness in N/m. The terminal penalty is part of the boundary-value objective; it does not add a force to the free-particle equation $m\ddot q=0$. This is a separate Python companion to the native prescribed-path experiment.

### Reduce the entire path problem to one endpoint

For any admissible path with a square-integrable velocity, subtract the straight path joining its own endpoints. The remaining variation has zero endpoint traces. The kinetic cross term integrates to zero, giving the exact decomposition:

$$
q(t)=a+\frac{b-a}{T}t+\eta(t),\quad\eta\in H_0^1(0,T),\qquad
\mathcal J_\gamma[q]=j_\gamma(b)+\frac m2\int_0^T\dot\eta^2dt,
\quad j_\gamma(b)=\frac{m(b-a)^2}{2T}+\frac\gamma2(b^2-a^2)+\frac\rho2(b-b_{\rm target})^2.
$$

For every fixed $b$, the extra integral is nonnegative and vanishes only for $\eta=0$. Thus minimizing $j_\gamma(b)$ solves the full path problem, rather than only a chosen polynomial family. Its linear and quadratic coefficients are:

$$
h_\gamma=\frac mT+\gamma+\rho,\qquad
d=\frac{ma}{T}+\rho b_{\rm target},\qquad
j_\gamma'(b)=h_\gamma b-d,\qquad j_\gamma''(b)=h_\gamma.
$$

Integration by parts also gives the natural terminal condition, because $\delta q(0)=0$ while $\delta q(T)$ is arbitrary:

$$
m\dot q(T)+\gamma b+\rho(b-b_{\rm target})=0,\qquad
b_{\rm stat}=\frac d{h_\gamma}\ (h_\gamma\ne0),\qquad
j_\gamma(b_{\rm stat}+z)-j_\gamma(b_{\rm stat})=\frac{h_\gamma}2z^2.
$$

If $h_\gamma>0$, the endpoint and the full path have one global minimum. If $h_\gamma<0$, the endpoint objective is unbounded below: its stationary point is a maximum only within the straight-path endpoint family. The full functional is a saddle there, since a fixed-endpoint bow increases kinetic action. If $h_\gamma=0$ and $d\ne0$, the remaining linear objective is unbounded below; if both vanish, every straight endpoint path minimizes the same constant value. A vanishing first variation alone cannot distinguish these cases.

### Keep the same task by correcting the terminal cost

The action gauge contributes $\gamma(b^2-a^2)/2$. Subtract this same boundary quantity from the terminal cost:

$$
\Phi_\gamma(b)=\Phi_0(b)-\frac\gamma2(b^2-a^2),\qquad
\mathcal J_\gamma^{\rm corrected}[q]
=\int_0^T L_\gamma\,dt+\Phi_\gamma(b)
=\mathcal J_0[q]\quad\text{for every admissible }q.
$$

This is pointwise equality of objectives for all admissible paths, not merely equality at an optimum. The initial-position term is constant but is retained to make the numerical values equal too. The corrected natural condition and minimizer are:

$$
m\dot q(T)+\rho(b-b_{\rm target})=0,\qquad
b_{\rm min}^{\rm corrected}
=\frac{ma/T+\rho b_{\rm target}}{m/T+\rho},\qquad \frac mT+\rho>0.
$$

The separate corrected terminal term need not remain nonnegative; the total objective remains exactly the original positive quadratic problem. Treating each gauge-dependent piece as an independently meaningful physical energy would miss that equality.

![Free-endpoint objective curves and terminal gauge compensation](/assets/learning/analytical-terminal-cost.en.svg)

For $m=1$ kg, $T=1$ s, $a=0$, $b_{\rm target}=1$ m and $\rho=1$ kg/s:

| $\gamma$ / (kg/s) | $h_\gamma$ / (kg/s) | Uncorrected minimizing $b$ / m | Corrected minimizing $b$ / m |
|---|---|---|---|
| -3 | -1 | No minimum | 1/2 |
| -2 | 0 | No minimum | 1/2 |
| -1 | 1 | 1 | 1/2 |
| 0 | 2 | 1/2 | 1/2 |
| 1 | 3 | 1/3 | 1/2 |

At $\gamma=-3$, the stationary endpoint is $b=-1$ m, but it is not a minimum. All straight paths still have zero Newton residual. For AI trajectory objectives, a learned Lagrangian gauge and a soft observation cost must be transformed together if they are intended to describe the same task.

### Execute and explain the boundary problem

This exact rational cell reproduces the table and the pointwise cancellation.

```python
from fractions import Fraction as Q

# SI: m=1 kg, T=1 s, rho=1 kg/s, a=0 m, target=1 m.
def reduced(b, gamma):
    b, gamma = Q(b), Q(gamma)
    return b*b/2+gamma*b*b/2+(b-1)**2/2

for gamma in map(Q, [-3, -2, -1, 0, 1]):
    h = 2+gamma
    minimum = Q(1)/h if h > 0 else None
    corrected_minimum = Q(1, 2)
    for b in map(Q, [-1, 0, 1, 2]):
        assert reduced(b, gamma)-gamma*b*b/2 == reduced(b, 0)
    if h:
        stationary = Q(1)/h
        assert h*stationary-1 == 0
        assert reduced(stationary+1, gamma)-reduced(stationary, gamma) == h/2
    print(gamma, h, minimum, corrected_minimum)
# None means no minimum in THESE examples (the linear drive is nonzero).
# A stationary endpoint with h<0 is not a minimum of the functional.
```

**Research questions with answers**

- Why does a total derivative affect the uncorrected optimizer? **Answer:** The free endpoint makes its boundary value variable, and the original terminal cost was left unchanged.
- Is a concave reduced endpoint maximum a maximum over all paths? **Answer:** No. Fixed-endpoint shape variations increase kinetic action, so the full stationary point is a saddle.
- Does $h_\gamma=0$ always mean no minimizer? **Answer:** No. With $d=0$, every straight endpoint path is a minimizer; with $d\ne0$, the linear objective is unbounded below.

## Exercises with explanations

1. With m=T=1, a=0, b=1 and A=B=1, calculate S₀. Answer: (1+1/3+1/5)/2=23/30 Js. Positive excess action shows this trial is above the constant-velocity path.
2. Apply γ=−2 to that example. Answer: subtract 1 Js to obtain −7/30 Js. The excess relative to the equally shifted reference is still 4/15 Js.
3. Set A=1, B=0 and choose C=0, D=1 with fixed endpoints. Answer: δS=0, but the A gradient is nonzero. One flat directional derivative does not make the trial stationary.
4. Set A=0, B=1 and inspect s=1/2. Answer: the cubic's acceleration vanishes there, but is nonzero away from the midpoint. Use the integral of the squared Newton residual to test the whole path.
5. Double T while retaining all positions and γ=0. Answer: all velocities halve, kinetic energy quarters, and action halves because integration lasts twice as long. This checks the chain-rule factors.
6. Explain why zero Newton residual is insufficient when endpoints are free. Answer: the boundary term must also vanish for all permitted endpoint variations; this imposes canonical momentum conditions at the free endpoints.

## Connection to AI research

Trajectory learning often combines endpoint fitting with differential-equation penalties. This example shows why their meanings differ and why one sampled residual can miss a violation. When a learned Lagrangian is identifiable only up to a total derivative, compare the induced equations and explicitly chosen boundary problem rather than raw action values. Later lessons derive general Lagrange equations and extend these ideas to multiple coordinates and constraints.
