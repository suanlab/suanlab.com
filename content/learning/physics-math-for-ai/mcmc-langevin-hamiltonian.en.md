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

## Leapfrog preserves a modified quadratic, not the target energy

For the harmonic target, one leapfrog step is a linear map

$$
M_\epsilon=\begin{pmatrix}1-\epsilon^2/2&\epsilon\\
-\epsilon(1-\epsilon^2/4)&1-\epsilon^2/2\end{pmatrix}.
$$

Its determinant is one and trace is 2−ε². For 0<ε<2 the eigenvalues lie on the unit circle and the positive quadratic H̃=[(1−ε²/4)q²+p²]/2 is exactly invariant in exact arithmetic: Mᵀdiag(1−ε²/4,1)M equals that same diagonal matrix. Target H=(q²+p²)/2 is different, with ΔH=ε²(q_L²−q₀²)/8. This explains bounded oscillatory energy error without claiming exact target-energy conservation. At ε=2 the quadratic is degenerate; above two the map has an unstable real eigenvalue.

![Exact-arithmetic harmonic leapfrog energy oscillations for three stable steps, compared with their initial target energy.](/assets/learning/stochastic-shadow-en.svg)

| ε | Modified q² coefficient | Linear stability |
|---|---:|---|
| 1/2 | 15/16 | stable |
| 1 | 3/4 | stable |
| 3/2 | 7/16 | stable |

```python
from fractions import Fraction as F
for eps in (F(1,2), F(1), F(3,2)):
    q, p = F(7,10), F(-3,10)
    q0 = q
    c = 1-eps*eps/4
    initial = (q*q+p*p)/2
    shadow = (c*q*q+p*p)/2
    for _ in range(8):
        half = p-eps*q/2
        q += eps*half
        p = half-eps*q/2
        assert (c*q*q+p*p)/2 == shadow
        assert (q*q+p*p)/2-initial == eps*eps*(q*q-q0*q0)/8
    assert (1-eps*eps/2)**2+eps*eps*c == 1
    print(eps, c, 'energy change:', float((q*q+p*p)/2-initial))
```

The rational code verifies the invariant and energy error without a finite-difference Jacobian. Momentum flip makes the reversible deterministic proposal an involution; area preservation and the Metropolis ratio then preserve the extended target. Momentum refresh and retained rejections are part of the sampler. Integrator stability alone does not prove useful mixing or eliminate the need for acceptance correction.

## Exercise: separate three diagnostics

At h=0.5 verify the ULA variance recurrence yields 4/3. In random-walk mode with sixty proposals and burn-in ten, explain why there are fifty-one retained states: the initial state also belongs to the trace. In HMC mode compare the last proposal endpoint with the final accepted chain state and the energy error. Finally state why high acceptance, accurate finite sample moments and a proof of invariant target are three different pieces of evidence.

Further reading: [Betancourt's conceptual HMC introduction](https://arxiv.org/abs/1701.02434). The Gaussian-target implementations and finite-step bias example are independently constructed.
