## Reverse the clock and the law together

Fix a forward terminal time T and define reverse elapsed time s, with physical forward time t=T-s. A reversed process must begin from the actual forward terminal distribution p_T. Starting from an unrelated standard Gaussian generally changes the problem even if that distribution is convenient for simulation.

The Markov chapter already showed the finite-state analogue: Bayes reverse rows depend on the endpoint marginal and are undefined outside its support. Continuous diffusions likewise require regularity and density conditions. This experiment stays within constant positive diffusion and smooth positive Gaussian marginals, avoiding a silent application at degenerate or singular states.

## Derive the score-corrected OU drift

For forward drift b(x)=-theta x and diffusion amplitude sigma, using an increasing reverse clock gives

$$
\widetilde b(x,s)=\theta x+\sigma^2\partial_x\log p_{T-s}(x),
\qquad \partial_x\log p_t(x)=-\frac{x-m_t}{v_t}.
$$

Blue plots this drift; orange plots only theta x, the naive negation. The density score accounts for which current states are likely under the forward law. It is not optional noise cancellation. More general state-dependent diffusions require additional care; the constant-diffusion formula should not be copied unchanged into those models.

## Verify reversed moments independently

The reversed marginal is Gaussian with mean m(T-s) and variance v(T-s). Its clock derivatives are theta m and 2theta v-sigma². Taking moments of the corrected affine drift gives exactly those derivatives:

$$
\frac{dm}{ds}=\theta m,\qquad
\frac{dv}{ds}=2(\theta-\sigma^2/v)v+\sigma^2
=2\theta v-\sigma^2.
$$

The code compares these routes instead of simply declaring the reverse path valid. Decreasing reverse marginal variance does not require a negative diffusion coefficient; the score drift overcomes positive stochastic spreading.

## Stationary reversibility is not drift negation

At theta=1, sigma=sqrt(2) and stationary variance one, the score is -x. The reverse drift is x-2x=-x, identical to the forward drift. Naive negation would produce +x and move away from the stationary law. This example exposes why reversing random dynamics differs from reversing a deterministic velocity field.

With theta zero and an initial standard Gaussian, the forward drift vanishes but the density spreads. Reversal still needs a nonzero score drift. At physical time zero the displayed reverse drift is -x for sigma one.

## Same marginals do not imply the same path law

The forward Fokker–Planck flux for constant diffusion is bp−σ²pₓ/2. A deterministic continuity equation with velocity b−σ²score/2 has the same flux and hence, under the required well-posedness conditions, the same marginals. In increasing reverse time, its velocity is θx+σ²score/2 for OU. The reverse SDE instead uses θx+σ²score and retains Brownian noise. The half factor is essential: removing noise from the reverse SDE without changing its drift does not preserve its marginals.

For Gaussian marginals, the deterministic reverse flow is explicitly

$$
x(s)=m_{T-s}+\sqrt{v_{T-s}/v_T}\,(x_T-m_T).
$$

Differentiating gives m′ᵣ+(v′ᵣ/(2v))(x−m), exactly the reverse probability-flow velocity. It maps the terminal Gaussian into the requested earlier Gaussian. The noisy reverse SDE has the same mean and variance but different joint laws across times. Neither construction authorizes starting from an unrelated terminal distribution.

| θ=0, σ=v=1, m=0 | Increasing reverse-time drift |
|---|---|
| Reverse SDE, with noise | −x |
| Probability-flow ODE, without noise | −x/2 |
| Naive sign reversal | 0 |

```python
from math import exp, sqrt
theta, sigma, m0, v0, T = .7, 1., .8, .4, 1.
def moments(t):
    decay = exp(-theta*t)
    return m0*decay, v0*decay**2+sigma**2*(1-decay**2)/(2*theta)
mT, vT = moments(T)
def flow(s, terminal):
    m, v = moments(T-s)
    return m+sqrt(v/vT)*(terminal-mT)
for s in (.25, .5, .75):
    m, v = moments(T-s)
    assert abs(flow(s,mT)-m) < 1e-14
    scale = sqrt(v/vT)
    assert abs(scale*scale*vT-v) < 1e-14
    x = flow(s, 1.2)
    velocity = theta*x-sigma**2*(x-m)/(2*v)
    difference = (flow(s+1e-5,1.2)-flow(s-1e-5,1.2))/2e-5
    assert abs(difference-velocity) < 1e-9
    print(s, x, velocity)
```

This auxiliary exact Gaussian map is not an additional live ODE solver. The code verifies its transported moments and velocity by an independent time difference. General score models also need approximation-error analysis and numerical integration; equality of ideal marginals does not prove that a discretized learned sampler reproduces them.

## Exercise: identify the required terminal law

Use nonstationary initial mean one and variance one quarter. Compute the forward mean and variance at T=1 and use that Gaussian as the reverse initial law. Then choose reverse fraction one half and determine the physical time. Explain why substituting a standard Gaussian or evaluating the score at s instead of T-s would fail the stated marginal moment equations.

Further reading: [Song et al., score-based SDE modeling](https://arxiv.org/abs/2011.13456). The scalar Gaussian reversal and moment checks are derived independently here.
