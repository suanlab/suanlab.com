## Put parameter uncertainty into the model

Seven successes in ten trials suggest a probability near 0.7, but finite observations leave many parameter values plausible. Bayesian inference specifies a prior distribution over the parameter and updates it with likelihood. The posterior is conditional on both the observed data and the assumed model; it is not an assumption-free statement about reality.

We use a beta prior and a Bernoulli likelihood because their algebra is inspectable. More complicated models may require numerical integration or sampling. Those approximations introduce additional errors that this exact conjugate example does not contain.

## Notation and density

| Symbol | Role |
| --- | --- |
| $p$ | Unknown Bernoulli success probability |
| $\alpha,\beta>0$ | Prior shape parameters |
| $D$ | Fixed observations: seven successes and three failures |
| $\pi(p)$ | Prior density |
| $\pi(p|D)$ | Posterior density |

A beta density is proportional to $p^{\alpha-1}(1-p)^{\beta-1}$. Its normalizer is the beta integral. A density value is not a point probability; probability is obtained by integrating over an interval. The lab uses positive integer shapes for simple exact normalization, while the distribution itself allows positive real shapes.

## Derive the update

Bayes’ rule multiplies prior by likelihood and divides by evidence:

$$
\pi(p|D)=\frac{P(D|p)\pi(p)}{\int_0^1P(D|u)\pi(u)\,du}.
$$

Multiplying powers gives $p^{\alpha+7-1}(1-p)^{\beta+3-1}$, so

$$
p|D\sim\operatorname{Beta}(\alpha+7,\beta+3).
$$

The denominator makes this a normalized density over parameters. Unlike likelihood normalized to peak one, its integral is one. The evidence depends on the prior and observation model; comparing evidence from incompatible representations without accounting for constants can be misleading.

## Uniform-prior worked example

Start with Beta(1,1), uniform on $[0,1]$. The posterior is Beta(8,4). Its mean is $8/12=2/3$, and its variance is

$$
\frac{8\cdot4}{12^2\cdot13}=\frac{2}{117}\approx0.017094.
$$

The MLE remains 0.7. The posterior mean differs because it combines the prior with the finite observations. The posterior mode is a different summary again: here it is $7/10=0.7$. State which summary you report rather than calling all of them “the probability estimate.”

## Prediction integrates the uncertainty

For a new trial under the same model,

$$
P(X_{\rm new}=1|D)=\int_0^1p\,\pi(p|D)\,dp=2/3.
$$

For one binary outcome this equals the posterior mean. Joint predictions for multiple future trials retain dependence induced by the shared uncertain parameter. Replacing $p$ by its mean inside every nonlinear expression is generally not the same as integrating over the posterior.

## Prior sensitivity is part of the analysis

Beta(9,1) followed by the same data gives Beta(16,4), with mean 0.8. Beta(1,9) gives Beta(8,12), with mean 0.4. These differences are not an arithmetic error: the priors express different assumptions. Calling a prior “uninformative” requires a reference parameterization and problem context; uniformity is not invariant under arbitrary transformations.

The lab plots prior and posterior densities on common axes. Peaks can exceed one. The tails near 0 and 1 are not fully sampled by the interior plotting grid; moments use analytical formulas rather than numerical integration of the displayed line. This distinction prevents a drawing resolution from silently changing the inference.

## AI connections, assumptions and checks

Bayesian prediction appears in uncertainty-aware models, latent-variable inference and inverse problems. A narrow posterior can still be confidently wrong under model misspecification or biased observations. Parameter uncertainty does not automatically include all distribution shifts, sensor failures or alternative scientific models.

For research, vary defensible priors, check predictive behavior on held-out observations and distinguish exact calculations from approximate inference. Report whether an interval is a posterior credible interval or a repeated-sampling confidence interval; they have different definitions. This lab displays moments, not a claim that those two interval concepts are interchangeable.

## Exercises and reference

1. Starting with Beta(2,2), update with 7 successes and 3 failures. **Answer:** Beta(9,5), mean $9/14$.
2. Explain why multiplying a likelihood by a prior still requires normalization.
3. Compare posterior mean and mode for the uniform-prior example.
4. State a sampling failure that changing the prior alone would not fix.

For Bayes’ rule and inference foundations, see [MIT Introduction to Probability](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/). All posterior calculations here are exact for the specified beta-Bernoulli model, not a general guarantee for Bayesian AI systems.
