## Evolve a law rather than a single realization

An SDE describes random trajectories, while a Fokker–Planck equation describes their marginal probability density. Neither object is a substitute for the other. For OU drift -theta x and constant diffusion amplitude sigma, write probability conservation as rho_t+J_x=0, where J combines drift transport and diffusion down the density gradient.

The experiment starts from a Gaussian with mean m0 and positive standard deviation s0. Positive sigma and s0 keep the displayed marginals smooth and positive, including at initial time. A point-mass start would need different treatment at its singular initial endpoint and is outside these controls.

## Derive the Gaussian moments

The marginal remains Gaussian, with

$$
m(t)=m_0e^{-\theta t},\qquad
v(t)=s_0^2e^{-2\theta t}+\sigma^2t f(2\theta t),
\qquad f(z)=\frac{1-e^{-z}}z,\quad f(0)=1.
$$

Its moment equations are m'=-theta m and v'=sigma²-2theta v. The zero-theta limit is ordinary diffusion with fixed mean and variance s0²+sigma²t. If theta is positive, variance approaches sigma²/(2theta), making the balance between confinement and noise visible.

## Differentiate density and flux independently

The Gaussian density is rho=exp(-(x-m)²/(2v))/sqrt(2pi v). Its derivatives provide the flux and an independent conservation check:

$$
J=-\theta x\rho-\frac{\sigma^2}{2}\rho_x,\qquad
\rho_t=\rho\left(-\frac{v'}{2v}+\frac{(x-m)m'}v+
\frac{(x-m)^2v'}{2v^2}\right).
$$

The code evaluates J_x from rho, rho_x and rho_xx separately rather than defining it as minus rho_t. The displayed residual can therefore detect inconsistent formulas. A local flux can be nonzero even though total probability is conserved; conservation involves its divergence and boundary behavior.

## A plotting window is not the probability domain

Blue and orange show current and initial densities on a finite window covering both. The whole-line mass is analytically one. Gaussian tails outside the crop are omitted from the picture without renormalizing it. An integral of the visible curve over that window is therefore not exactly the same quantity as whole-line normalization.

For theta=1, sigma=sqrt(2), m0=0 and s0=1, the standard Gaussian is stationary and drift flux cancels diffusion flux. Stationary density does not mean sample paths stop moving; they continue fluctuating while their marginal law remains unchanged.

## A distributional Lyapunov function: relative entropy

For θ>0, the OU stationary law is π=N(0,v*), v*=σ²/(2θ). Its Gaussian marginal has relative entropy D=KL(N(m,v)||π). Put r=v/v*. Direct Gaussian integration and the moment equations give

$$
D=\tfrac12(m^2/v_*+r-1-\log r),\qquad
D'=-\theta\left(m^2/v_*+\frac{(r-1)^2}{r}\right)\le0.
$$

This is stronger information than conservation of total mass. Since log r−1+1/r≥0 for r>0, the dissipation is at least 2θD, so D(t)≤e^(−2θt)D(0). The inequality follows by differentiating log r−1+1/r, which has its minimum zero at r=1. It is a concrete Gaussian entropy estimate; general densities need an appropriate functional inequality and justified integrations by parts.

| m, r | Relative entropy D |
|---|---|
| 0, 1 | 0 |
| 1, 1 with v*=1 | 1/2 |
| 0, 2 | (1−log2)/2 |

```python
from math import exp, log
theta, stationary, m0, v0 = 1., 1., 1., .25
def entropy(m, v):
    r = v/stationary
    return .5*(m*m/stationary+r-1-log(r))
initial = entropy(m0, v0)
for t in (0., .5, 1.):
    m = m0*exp(-theta*t)
    v = stationary+(v0-stationary)*exp(-2*theta*t)
    r = v/stationary
    D = entropy(m, v)
    dissipation = theta*(m*m/stationary+(r-1)**2/r)
    assert D <= exp(-2*theta*t)*initial+1e-14
    assert dissipation >= 2*theta*D-1e-14
    print(t, D, dissipation)
```

The code checks the decay bound and nonnegative dissipation. In the live density calculation, flux is evaluated from the SDE coefficients as ρ[−θm+(σ²/(2v)−θ)(x−m)], with its spatial derivative obtained by the product rule. This equals drift plus diffusive flux analytically but retains a tiny moving-mean contribution when those two large terms cancel numerically. The separate density-time derivative remains a conservation check; these floating diagnostics are not interval certificates.

## Exercise: stationary and freely diffusing cases

Verify v'=0 for the stationary standard Gaussian. Then set theta zero, sigma one, m0 one and s0 one half: at t=1, mean is one and variance is 1.25. Calculate rho_x at a point above the mean and explain the sign of the diffusion contribution to J. Finally identify what boundary assumption lets the integrated whole-line probability remain one.

Further reading: [Lawler's diffusion and heat-equation notes](https://www.math.uchicago.edu/~lawler/finbook.pdf). The Gaussian derivatives and flux checks are independently implemented.
