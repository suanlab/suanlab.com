## Expected rates versus finite observed slopes

Estimate true mean zero from independent-model Rademacher observations ±sigma, optionally adding fixed estimator bias b. Each repetition generates the longest sample and uses prefixes N=2,4,...,128. Points at different N within one repetition are coupled, not independent observations.

Blue averages realized squared estimator errors. Orange is the model-derived expectation. Enabling the prediction-noise floor adds exact independent test-noise variance conditional on each learned estimator. Those values are not presented as observed test samples.

## Derive the expected risk

The sample mean has mean zero and variance sigma²/N, while bias is fixed:

$$
E[(\overline X+b)^2]=b^2+\frac{\sigma^2}N.
$$

Predicting an independent noisy label adds sigma², giving total risk b²+sigma²/N+sigma². Conditional risk is evaluated exactly for each realized estimator and then averaged across actual repetitions. Observations are not forced to equal the expectation.

Metric choice changes exponents. If mean-estimation MSE is proportional to 1/N, RMSE is proportional to 1/sqrt(N). Comparing these directly with a confidence radius or deviation probability from the previous chapter mixes different definitions.

## Floors and observation windows

For R(N)=C+A/N, its continuous log-log tangent is

$$
\frac{d\log R}{d\log N}=-\frac{A/N}{C+A/N}.
$$

It is minus one without a floor, but approaches zero when positive C dominates. Thus bias, irreducible noise and the chosen N window can change apparent scaling for the same estimator. Finite-repetition slopes may even leave this theoretical interval because of sampling variation.

## Fit actual positive observations

Ordinary least squares is applied only to positive repetition-average risks:

$$
\widehat s=\frac{\sum_i(\log N_i-\overline{\log N})(\log \widehat R_i-\overline{\log\widehat R})}{\sum_i(\log N_i-\overline{\log N})^2}.
$$

Fewer than two positive points leaves the slope undefined. Zero risk is not replaced with an arbitrary small positive value to manufacture a fit. Prefix dependence also prevents automatically reporting an independent-residual standard error. Assess seeds, repetitions, fit windows and the estimation procedure together.

## Worked example and exercise

With sigma one, zero bias and no prediction floor, expected risk at N=128 is 1/128=.0078125. Including independent prediction noise makes it 1.0078125. A nearly flat large-N curve in the latter case reflects its floor, not an estimator suddenly ceasing to learn.

Exercise: set sigma=b=0. What are risk and log slope? Solution: all risks are zero, so logarithms and the fitted slope are undefined. For sigma zero but bias one, all risks equal one, giving slope zero when at least two N values exist. These cases should not be conflated as the same flat success curve.

## Scaling claims in new AI research

A good finite-window power-law fit is not a universal neural scaling theorem. Record task, model, data distribution, metric, compute, floors and observation range, and compare alternative explanations. In this transparent example, the exact 1/N expectation follows from assumptions while the blue curve comes from actual repetitions. Preserving that distinction is essential to interpreting broader research claims.

References: [MIT learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/) and [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
