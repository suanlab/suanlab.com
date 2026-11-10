## Likelihood holds the data fixed

Observe k successes in ten Bernoulli trials. Likelihood holds those data fixed and varies p. Ordered-sample and count likelihoods differ by a p-independent binomial factor, so they have the same maximizer. Likelihood need not integrate to one over p and is not a posterior distribution. The blue curve is a relative likelihood scaled to peak at one.

$$
L(p)\propto p^k(1-p)^{10-k},\qquad\hat p_{\mathrm{MLE}}=k/10.
$$

## A Beta prior and conjugate update

A Beta(α,β) prior is proportional to p^(α−1)(1−p)^(β−1). Multiplication by the likelihood gives posterior Beta(k+α,10−k+β). Controls restrict prior shapes to at least one, making the interior or single-boundary mode straightforward. The lab does not cover all boundary-mode behavior of Beta priors with shapes below one.

$$
p\mid x\sim\operatorname{Beta}(A,B),\quad A=k+\alpha,\ B=10-k+\beta.
$$

## Read mode and mean separately

With k=7 and α=β=2, MLE is 0.7. The Beta(9,5) posterior has MAP 8/12 and mean 9/14. These are distinct summaries. MAP is a density mode in the chosen p coordinate and generally changes under nonlinear reparameterization through a Jacobian. Posterior mean minimizes posterior squared-error risk, while other losses may favor other summaries.

$$
\hat p_{\mathrm{MAP}}=\frac{A-1}{A+B-2},\qquad E[p\mid x]=\frac A{A+B}.
$$

## Uniform priors and boundary cases

Under a uniform prior, MAP equals MLE, though the posterior mean can differ. With zero successes, both maxima lie at zero while the posterior mean is 1/12. The orange posterior profile is independently peak-scaled, so comparing areas under these two displayed curves is not a probability calculation. A boundary estimate does not prove the parameter value with certainty.

## Exercises with explanations

1. What is normalized to form a posterior density? **The prior–likelihood product over p.**
2. What is the default posterior mean? **9/14, about 0.643.**
3. Must MAP and mean agree under a uniform prior? **No: MAP can equal MLE while the mean differs.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
