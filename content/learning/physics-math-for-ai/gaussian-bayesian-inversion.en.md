## 1. Turn a noisy observation into a distribution

Let $x$ be an unknown scalar, with a Gaussian prior $x\sim\mathcal N(\mu_0,v_0)$. A sensor measures $y=sx+\eta$ with known strength $s$ and independent noise $\eta\sim\mathcal N(0,r)$. Bayes' rule multiplies the prior density by the likelihood and normalizes the product. The resulting posterior summarizes uncertainty conditional on the measured value and the assumed model. [Stuart, Inverse Problems from a Bayesian Perspective](https://doi.org/10.1017/S0962492910000061)

## 2. Complete the square

The negative log posterior, up to an additive constant, is

$$
\frac{(x-\mu_0)^2}{2v_0}+\frac{(y-sx)^2}{2r}.
$$

The coefficient of $x^2$ gives the precision and the coefficient of $x$ determines the posterior mean:

$$
v=\left(v_0^{-1}+s^2/r\right)^{-1},\qquad
\mu=v\left(\mu_0/v_0+sy/r\right).
$$

Both the prior and the observation contribute precision. The maximum a posteriori estimate equals the mean in this Gaussian model. With zero prior mean the same optimizer minimizes $(y-sx)^2+(r/v_0)x^2$, giving a Tikhonov penalty with an explicit probabilistic interpretation.

## 3. Compare informative and weak observations

Take $\mu_0=0$, $v_0=1$, $s=0.5$, $r=0.25$, and $y=1$. The posterior mean and variance are both easy to calculate: $\mu=1$ and $v=0.5$. At $s=0$ the likelihood is constant as a function of $x$ and the posterior equals the prior. Smaller $r$ increases the information contributed by a nonzero sensor.

## 4. Predict a new reading

For an independent follow-up reading $y_*=s_*x+\eta_*$ with variance $r_*$, the predictive distribution is

$$
y_*\mid y\sim\mathcal N(s_*\mu,\ s_*^2v+r_*).
$$

The first variance term represents parameter uncertainty. The second represents new observation noise. A credible interval for $x$ and a prediction interval for $y_*$ answer different questions.

## 5. Verify the posterior by numerical integration

```python
import numpy as np

x = np.linspace(-8, 8, 16001)
prior_mean, prior_variance = 0., 1.
strength, noise_variance, observation = .5, .25, 1.
weights = np.exp(-.5 * ((x-prior_mean)**2 / prior_variance
                        + (observation-strength*x)**2 / noise_variance))
weights /= weights.sum()
mean = np.dot(weights, x)
variance = np.dot(weights, (x-mean)**2)
assert np.isclose(mean, 1.)
assert np.isclose(variance, .5)
print(mean, variance, "predictive variance:", strength**2 * variance + noise_variance)
```

## 6. Check how the assumptions influence inference

Compare posterior estimates for plausible prior scales and noise levels. Strong prior influence is expected when the observation operator is weak. Use withheld observations to check predictive residuals. A narrow posterior is conditional on the assumed model and can be misleading when the forward physics or noise distribution is inaccurate.
