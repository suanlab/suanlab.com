## Separate parameters from noise

Write a Gaussian variable as μ+σε instead of treating sampling as an opaque operation. The base noise ε follows a parameter-independent standard normal law. Hold the same ε fixed while differentiating to track parameter effects. The noise control sets a deterministic value rather than drawing a fresh sample, so the plot and table are not a random-sampling experiment.

$$
z=\mu+\sigma\epsilon,\quad\epsilon\sim N(0,1),\qquad\ell(z)=\tfrac12(z-1)^2.
$$

## Differentiate one noise path

The chain rule gives derivative z−1 with respect to μ and (z−1)ε with respect to σ. At μ=0,σ=1,ε=0, the sample is zero, loss is one half and gradient is (−1,0). Changing ε to one makes the sample equal the target, giving zero sample loss and gradient. This does not establish that the distribution-level objective has been minimized.

$$
\partial_\mu\ell=z-1,\qquad\partial_\sigma\ell=(z-1)\epsilon.
$$

## Compare with the expected gradient

Using noise mean zero and second moment one gives an exact expected quadratic loss. The default expected loss is one and its gradient is (−1,1), differing from the zero scale derivative at ε=0. Averaging the two table nodes ε=−1 and one reproduces the moments needed for this quadratic loss and gradient. Those nodes are not Gaussian samples and do not integrate arbitrary functions exactly.

$$
E\ell=\tfrac12[(\mu-1)^2+\sigma^2],\qquad\nabla E\ell=(\mu-1,\sigma).
$$

## Conditions for exchanging expectation and differentiation

Reparameterization alone does not justify exchanging differentiation and expectation. Check differentiability and suitable integrable domination or other sufficient conditions. Polynomial losses and finite Gaussian moments justify the exchange here. At σ=0 the law collapses to a point, but the path expression and expected quadratic loss remain evaluable. Their derivatives can be read through a smooth signed-scale extension or as one-sided derivatives in the allowed domain.

$$
\nabla_\theta E_\epsilon[\ell(g_\theta(\epsilon))]=E_\epsilon[\nabla_\theta\ell(g_\theta(\epsilon))]\quad\text{under suitable conditions}.
$$

## AI connections and exercises

Continuous latent-variable models such as VAEs use pathwise derivatives to estimate gradients of expected objectives. This quadratic example explains the mechanism without training a VAE or evaluating a full ELBO.

1. What is expected loss when ε=1 gives zero sample loss? **Still one at μ=0,σ=1.**
2. May noise change while differentiating the same path? **No: hold it fixed.**
3. Is the two-node check exact for every function? **No: only the required matched moments justify it here.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).

[Kingma–Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
