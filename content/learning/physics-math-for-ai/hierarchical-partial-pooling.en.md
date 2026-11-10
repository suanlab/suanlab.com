## Estimating many small groups

Consider means for hospitals, cities or datasets. Small-group means fluctuate considerably. Complete pooling erases differences; separate estimation discards common information. A hierarchical model assumes group parameters come from a shared distribution and creates partial pooling between these extremes.

$$
\theta_j\sim N(0,\tau^2),\qquad \bar Y_j\mid\theta_j\sim N(\theta_j,1/n).
$$

## Add precisions to obtain the posterior

Individual observation variance is known to be one and each group contains n observations. Prior precision is 1/τ²; sample-mean precision is n. Completing the square gives posterior variance and mean.

$$
v=(n+\tau^{-2})^{-1},\qquad m_j=v n\bar y_j=w\bar y_j,\qquad w=\frac{n\tau^2}{1+n\tau^2}.
$$

With n=4, τ=1 and observed group means two and minus one, w=0.8. Posterior means are 1.6 and −0.8; both posterior variances are 0.2. Blue marks show observed means, orange marks posterior means. Their heights encode means; uncertainty is reported separately in the readout and table.

## Change the pooling strength

A smaller τ asserts that group differences are smaller, pulling estimates more strongly toward zero. At τ=0.25, w=0.2 and the observed mean two becomes posterior mean 0.4. Increasing n strengthens the group data and moves w toward one. The weight follows from a specified probabilistic model rather than an arbitrary averaging rule.

## Understand what is conditioned on

The population center zero and τ are fixed inputs. This lab neither estimates nor integrates a hyperparameter posterior. Conditional posterior covariance between the groups is therefore zero. Making the shared center unknown and inferring it jointly can introduce posterior dependence. Also distinguish empirical Bayes, which estimates hyperparameters and plugs them in, from full Bayes, which integrates their posterior uncertainty.

For a new group with no observations, parameter variance remains τ². Do not substitute the smaller conditional posterior variance v of an observed group. This distinction matters in AI applications involving task-specific performance or regional effects with both shared structure and individual differences.

## Exercises with explanations

1. What is data weight at n=4,τ=1? **0.8.**
2. Does reducing τ increase reliance on group data? **No: it increases shrinkage toward the common center.**
3. Does the plot incorporate hyperparameter uncertainty? **No: it is a conditional calculation with fixed hyperparameters.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
