## Reverse the probability question

Probability starts with a model and asks what data might occur. Estimation starts with observed data and asks which parameter values explain it. The same algebra can appear in both directions, but its interpretation changes. A likelihood is a function of parameters for fixed observations; it is not automatically a probability distribution over those parameters.

We observe $n$ independent Bernoulli outcomes with $k$ successes and assume one common unknown success probability $p$. This assumption is part of the model. Dependence, selection bias or changing probabilities can invalidate the intended interpretation even when optimization is performed correctly.

## Define the likelihood

For an ordered sequence, multiplying the Bernoulli probabilities gives

$$
L(p)=p^k(1-p)^{n-k},\qquad 0\le p\le1.
$$

If only the count is observed, a binomial coefficient multiplies this expression. That coefficient is independent of $p$, so it does not change the maximizing parameter. It does change the numerical probability of the observed event, which matters if one compares different data representations.

| Object | Meaning |
| --- | --- |
| $n,k$ | Fixed observed counts |
| $p$ | Candidate parameter |
| $\hat p$ | Estimate computed from data |
| $\ell(p)$ | Log likelihood |
| SE | Sampling variability approximation for an estimator |

## Derive the maximum

For $0<k<n$, take the log:

$$
\ell(p)=k\log p+(n-k)\log(1-p),
\quad \ell'(p)=\frac{k}{p}-\frac{n-k}{1-p}.
$$

Setting the derivative to zero gives $\hat p=k/n$. The second derivative is negative on the interior, so the stationary point is the unique interior maximum. For $k=0$ or $k=n$, inspect the boundary directly instead of dividing by a zero count or pretending the maximum must be interior.

For seven successes in ten trials, $\hat p=0.7$. At $p=0.5$, the ordered-data log likelihood is approximately $-6.931472$, while at 0.7 it is approximately $-6.108643$. Higher is better here. A negative log likelihood used as a loss reverses that comparison.

## Estimate variability is another calculation

If the model is correct, $\hat p=K/n$ has mean $p$ and variance $p(1-p)/n$. Replacing the unknown $p$ with its estimate yields the plug-in standard error

$$
\widehat{SE}=\sqrt{\hat p(1-\hat p)/n}.
$$

At 7/10 this is about 0.144914. This is not the standard deviation of a single label, nor a posterior probability for $p$. It approximates variability across repeated samples under the assumed sampling model.

At ten successes in ten trials, the MLE is 1 and the plug-in standard error is zero. Finite data have not proven that failure is impossible. Boundary behavior exposes a weakness of this approximation. Interval methods need their own assumptions and coverage analysis; multiplying any displayed standard error by 1.96 is not universally justified.

## Read the likelihood curve

The plotted curve is divided by its maximum, so its peak is one. Its integral need not be one. Blue is relative likelihood, not a prior or posterior density. The candidate parameter affects the readout, while the full curve continues to represent all candidate values for the fixed data.

The second preset intentionally keeps seven successes but increases total trials to 100. It represents seven successes and 93 failures, not a tenfold repetition of 7/10. Its MLE is 0.07. For a tenfold repetition with the same success proportion, one would need 70 successes in 100 trials. Check counts before interpreting a “larger dataset” comparison.

## From likelihood to AI losses

Binary cross-entropy is the negative Bernoulli log likelihood for given labels and predicted probabilities. A neural network links those probabilities to parameters through a function; backpropagation differentiates the composed objective. This connection requires an explicit observation model. Minimizing a loss does not establish calibration, causal correctness or representativeness of the dataset.

A research evaluation should record the sampling unit, exclusions, missing outcomes and whether the same data were used to select the model. Optimization error, model misspecification and sampling uncertainty are different sources of failure. More training steps reduce neither selection bias nor data leakage by themselves.

## Exercises and further study

1. For 3 successes in 12 trials, calculate the MLE. **Answer:** 0.25.
2. Why can the binomial coefficient be omitted when optimizing $p$ for fixed $n,k$?
3. Explain why a likelihood normalized to peak one is not a probability density.
4. Construct a boundary example where plug-in standard error is misleading.

For estimation and sampling distributions, see [MIT Introduction to Probability, Inference and Limit Theorems](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/). The next unit introduces an explicit distribution over parameters rather than silently treating likelihood as one.
