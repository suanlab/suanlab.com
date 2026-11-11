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

## Derive the gradient estimator's variance

An unbiased gradient can still fluctuate substantially. Write m=μ−1 and keep the standard normal noise law. The two pathwise gradient components are gμ=m+σε and gσ=mε+σε². Expanding their squares and using Eε=Eε³=0, Eε²=1 and Eε⁴=3 gives

$$
E[g_\mu]=m,\quad E[g_\sigma]=\sigma,\qquad
\operatorname{Var}(g_\mu)=\sigma^2,\quad
\operatorname{Var}(g_\sigma)=m^2+2\sigma^2.
$$

In particular, correct expected gradients do not establish correct gradient uncertainty. The deterministic nodes ε=−1,+1 have fourth moment one, not three. At μ=0,σ=1, their scale gradients are 2 and 0, with mean one and variance one. The Gaussian scale-gradient variance is three. The original two-node check is exact for the expected quadratic loss and gradient, but cannot be used as a Gaussian variance estimate.

## Compare an antithetic pair at the same evaluation budget

Use one random Gaussian ε and evaluate both ε and −ε. Their average cancels odd terms:

$$
\bar g^{\mathrm{anti}}_\mu=m,\qquad
\bar g^{\mathrm{anti}}_\sigma=\sigma\epsilon^2,\qquad
\operatorname{Var}(\bar g^{\mathrm{anti}}_\sigma)=2\sigma^2.
$$

This uses two loss-gradient evaluations. A fair comparator is the average of two independent noise paths, whose variances are σ²/2 and (m²+2σ²)/2. Antithetic pairing removes the location-gradient variance, but improves the scale-gradient variance only when m²>2σ². Symmetry of the noise does not guarantee improvement for every component.

Each result lists (location-gradient variance, scale-gradient variance).

| $(\mu,\sigma)$ | 2 iid paths | ε, −ε pair |
|---|---|---|
| $(0,1)$ | $(1/2,3/2)$ | $(0,2)$ |
| $(-2,1)$ | $(1/2,11/2)$ | $(0,2)$ |
| $(0,0)$ | $(0,1/2)$ | $(0,0)$ |

At σ=0, the expected scale derivative is zero, but the single-path derivative mε can still vary. As above, interpret this boundary through a smooth signed-scale extension or a one-sided derivative. The displayed noise control is deterministic; none of these variance claims is an empirical estimate from its three table rows.

## A threshold shows why almost-everywhere path derivatives are insufficient

Let U be uniform on [0,1] and Bθ=1{U<θ} for 0<θ<1. Then E[Bθ]=θ, so its derivative is one. For almost every fixed U, however, Bθ is locally constant in θ and its path derivative is zero. Exchanging derivative and expectation would give a false result:

$$
\frac{d}{d\theta}E[B_\theta]=1,\qquad
E\!\left[\frac{d}{d\theta}B_\theta\right]=0.
$$

For positive h<1−θ, the forward difference quotient is 1/h on an interval of U of width h and zero elsewhere. Its expectation stays one while its almost-everywhere limit is zero. The shrinking interval and growing height prevent the required domination. Writing a discrete sample as a function of parameter-independent noise therefore does not make naive pathwise differentiation valid. A discrete score estimator or a justified alternative requires its own derivation.

## Compute moments without mistaking quadrature for sampling

The code evaluates polynomial Gaussian moments exactly with fractions, compares the two evaluation budgets and integrates the threshold's rectangular difference quotient exactly. It samples no random paths and does not train a latent-variable model. Its variance analysis extends the quadratic example; the interactive controls and original three noise nodes remain unchanged.

```python
from fractions import Fraction as Q
from math import prod

def gaussian_moment(k):
    return Q(0) if k % 2 else Q(prod(range(1, k, 2)))

def expected(poly):
    return sum(c*gaussian_moment(k) for k, c in enumerate(poly))

def multiply(a, b):
    out = [Q(0)] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out

def variance(poly):
    return expected(multiply(poly, poly)) - expected(poly)**2

for mu, sigma in [(Q(0), Q(1)), (Q(-2), Q(1)), (Q(0), Q(0))]:
    m = mu-1
    g_mu, g_sigma = [m, sigma], [Q(0), m, sigma]
    raw = (variance(g_mu), variance(g_sigma))
    iid_two = tuple(v/2 for v in raw)
    antithetic = (variance([m]), variance([Q(0), Q(0), sigma]))
    assert (expected(g_mu), expected(g_sigma)) == (m, sigma)
    assert raw == (sigma**2, m*m+2*sigma**2)
    assert antithetic == (0, 2*sigma**2)
    print("mu, sigma:", mu, sigma, "raw:", raw,
          "two independent:", iid_two, "one antithetic pair:", antithetic)
assert variance([Q(0), Q(-1), Q(1)]) == 3
node_gradients = [Q(2), Q(0)]  # epsilon = -1, +1 at mu=0, sigma=1.
node_mean = sum(node_gradients)/2
node_variance = sum((g-node_mean)**2 for g in node_gradients)/2
assert node_mean == 1 and node_variance == 1  # Not Gaussian variance 3.
for h in [Q(1, 4), Q(1, 16), Q(1, 64)]:
    interval_width, difference_quotient = h, 1/h
    assert interval_width*difference_quotient == 1
    print("threshold difference quotient:", h, difference_quotient,
          "integrated value:", interval_width*difference_quotient)
```

## Further questions and research use

1. Does a zero variance location-gradient pair prove that both components improved? **No: the first table row has worse scale-gradient variance than two independent paths.**
2. Why is checking the first two noise moments insufficient for gradient variance here? **Squaring the scale gradient introduces the fourth noise moment.**
3. What should an estimator comparison fix? **The objective, parameter values, unbiasedness assumptions and number of model evaluations. A lower variance at a different cost is a different comparison.**

The differentiable-noise construction is described in [Kingma–Welling, §§2.3–2.4](https://arxiv.org/html/1312.6114v11). The quadratic variance, equal-budget comparison and threshold counterexample above are direct derivations, not a reproduction of the paper's training experiments.

## AI connections and exercises

Continuous latent-variable models such as VAEs use pathwise derivatives to estimate gradients of expected objectives. This quadratic example explains the mechanism without training a VAE or evaluating a full ELBO.

1. What is expected loss when ε=1 gives zero sample loss? **Still one at μ=0,σ=1.**
2. May noise change while differentiating the same path? **No: hold it fixed.**
3. Is the two-node check exact for every function? **No: only the required matched moments justify it here.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).

[Kingma–Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
