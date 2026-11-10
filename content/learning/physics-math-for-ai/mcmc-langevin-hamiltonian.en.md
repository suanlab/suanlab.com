## A sampler has proposals, states and an invariant target

The target here is standard normal with potential U(x)=x²/2. The lab actually runs three algorithms rather than drawing a target density and calling it a simulation. The blue trace records chain states, including repeated states after rejection. Orange is target mean zero, not a prediction that a finite empirical average must equal zero.

Reported moments use the retained states after burn-in. Their sample count is explicit, and an empty retained window has no moment estimate. These finite statistics can reveal a problem but cannot by themselves prove invariance or convergence.

## Unadjusted Langevin has a known finite-step bias

Euler discretization of dX=-X dt+sqrt(2)dW gives

$$
X_{n+1}=(1-h)X_n+\sqrt{2h}Z_n,\qquad
V_h=\frac1{1-h/2},\quad 0<h<2.
$$

The variance follows by solving V=(1-h)²V+2h. At h=0.5 it is 4/3 rather than one. Stability does not remove discretization bias. At h at least two, the finite stationary variance formula is not valid, so the readout leaves it absent and identifies the unstable step range.

## Metropolis corrects a symmetric random walk

Propose y=x+sqrt(2h)Z and accept with probability min(1,exp(U(x)-U(y))). Symmetry of the proposal makes this the target-density ratio; an asymmetric proposal would require its forward and reverse densities as well. Rejection retains x, and the displayed acceptance fraction counts actual decisions rather than treating every proposal as a state transition.

## HMC adds momentum and numerical dynamics

HMC refreshes an independent standard normal momentum, then performs actual leapfrog updates for Hamiltonian H=(q²+p²)/2. Full-step positions and momenta appear in an equal-scale phase diagram. Connecting them guides the discrete integration and does not assert an exact continuous orbit.

$$
p_{1/2}=p-\epsilon q/2,\qquad
q'=q+\epsilon p_{1/2},\qquad
p'=p_{1/2}-\epsilon q'/2.
$$

The reversible, volume-preserving proposal with momentum reversal is corrected by min(1,exp(-Delta H)). The diagram shows the last proposal even if it was rejected. A large unstable integrator step can yield a dramatic trajectory and almost no useful accepted movement; a visually interesting proposal is not evidence of good sampling.

## Exercise: separate three diagnostics

At h=0.5 verify the ULA variance recurrence yields 4/3. In random-walk mode with sixty proposals and burn-in ten, explain why there are fifty-one retained states: the initial state also belongs to the trace. In HMC mode compare the last proposal endpoint with the final accepted chain state and the energy error. Finally state why high acceptance, accurate finite sample moments and a proof of invariant target are three different pieces of evidence.

Further reading: [Betancourt's conceptual HMC introduction](https://arxiv.org/abs/1701.02434). The Gaussian-target implementations and finite-step bias example are independently constructed.
