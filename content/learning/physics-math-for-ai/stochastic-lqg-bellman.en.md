## Choose a policy before seeing the next disturbance

Stochastic control chooses actions using available information, not future noise. The lab uses a fully observed scalar system X_next=aX+u+sigma Z with independent standard normal Z. Stage cost is (X²+R u²)/2 and terminal cost is qF X_N²/2. R is positive, controls are unconstrained and the current state is known exactly.

These assumptions matter. Partial observation requires an information-state treatment, constraints can change the optimizer, and state-dependent noise can change the recursion. The result below is a concrete solvable case rather than a universal feedback formula.

## Bellman recursion separates curvature and noise offset

Assume V_n(x)=P_n x²/2+c_n, with terminal P_N=qF and c_N=0. Conditional expectation of the next quadratic value gives a scalar convex minimization. Its solution is

$$
u_n=-K_nx,\quad K_n=\frac{aP_{n+1}}{R+P_{n+1}},\qquad
P_n=1+\frac{a^2R P_{n+1}}{R+P_{n+1}},\quad
c_n=c_{n+1}+\frac{\sigma^2P_{n+1}}2.
$$

The code executes this backward recursion. The noise increases the additive value offset but does not alter the feedback gains under the stated fully observed additive-noise assumptions. Removing noise therefore reduces optimal cost without necessarily changing the optimal action at a given state.

## Verify costs through a second forward calculation

For feedback u=-Kx, forward moments satisfy m_next=(a-K)m and v_next=(a-K)²v+sigma². The expected stage cost is (1+R K²)(m²+v)/2. Summing these costs and the terminal term independently checks V0(x0), without a Monte Carlo estimate that might agree only within sampling error.

Blue shows the selected policy's cumulative expected cost, orange the optimal policy's counterpart. A zero-control policy can spend less at an early stage while accumulating more total horizon cost. Optimality of the full objective does not mean the optimal curve must be lower at every intermediate stage.

## Work through one noisy decision

Set a=R=qF=sigma=x0=1 and N=1. The gain is one half and the action is minus one half. The current expected cost is 0.625; the next state has mean one half and variance one, giving terminal expected cost 0.625. Total is 1.25. Zero control instead costs 1.5. Removing noise keeps the gain and reduces the optimal cost to 0.75.

## Connection to a continuous-time HJB equation

For controlled diffusion dX=b(X,u)dt+sigma(X,u)dW, smooth value functions formally obey

$$
V_t+\min_u\left\{\ell(x,u)+b(x,u)V_x+\tfrac12\sigma(x,u)^2V_{xx}\right\}=0,
\qquad V(T,x)=g(x).
$$

This is the infinitesimal conditional-expectation principle, with a diffusion generator rather than a deterministic derivative. Classical verification needs suitable regularity, admissible adapted controls and integrability. The finite-horizon discrete experiment proves its own Bellman identities; it does not bypass these hypotheses for a general continuous problem.

## Exercise: expose an information advantage

Explain why choosing u after observing the next Z would solve a different, artificially easier problem. Then reproduce the one-stage Bellman derivative and its positive curvature R+P_next. Check the special terminal weight zero: the last action is zero because no future state penalty remains. Finally describe how a continuous-time HJB equation relates to the same conditional minimization principle, while noting that a discrete finite-horizon recursion is not itself a general continuous-time solver.

Further reading: [Boyd's stochastic linear-quadratic control notes](https://ee363.stanford.edu/archive/lectures/stoch_lqr.pdf). The scalar model, half-cost convention and independent forward check are derived here.
