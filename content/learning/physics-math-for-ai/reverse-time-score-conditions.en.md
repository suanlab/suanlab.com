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

## Exercise: identify the required terminal law

Use nonstationary initial mean one and variance one quarter. Compute the forward mean and variance at T=1 and use that Gaussian as the reverse initial law. Then choose reverse fraction one half and determine the physical time. Explain why substituting a standard Gaussian or evaluating the score at s instead of T-s would fail the stated marginal moment equations.

Further reading: [Song et al., score-based SDE modeling](https://arxiv.org/abs/2011.13456). The scalar Gaussian reversal and moment checks are derived independently here.
