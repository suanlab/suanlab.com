## Separate sources of predictive uncertainty

Consider posterior θ~Beta(α,β) and future Y|θ~Bernoulli(θ). Variation remaining even with known θ differs from uncertainty about θ itself. This decomposition is relative to the chosen probabilistic model; it is not a unique universal classification of uncertainty in arbitrary neural networks.

$$
\operatorname{Var}(Y\mid x)=E[\theta(1-\theta)\mid x]+\operatorname{Var}(\theta\mid x).
$$

## Same total variance, different decomposition

Write s=α+β. The terms are αβ/[s(s+1)] and αβ/[s²(s+1)], summing to p(1−p) with p=α/s. Beta(2,2) gives 0.2+0.05=0.25; Beta(12,12) gives 0.24+0.01=0.25. A more concentrated posterior can retain the same total single-trial variance.

One Bernoulli predictive distribution is determined entirely by its mean. Total binary variance therefore cannot distinguish every posterior with that mean. Inspect the parameter distribution or joint predictions of several future observations as well. An increase in expected conditional variance does not mean increased uncertainty about the parameter.

## Calibration compares predictions with outcome frequencies

The calibration plot uses a separate fixed evaluation dataset of fifty labels. Five bins each contain ten labels, with positive counts one, three, five, seven and nine. Matched predictions are 0.1,0.3,0.5,0.7,0.9. The second mode predicts 0.02,0.1,0.5,0.9,0.98 against the same labels, making more extreme claims.

$$
\operatorname{ECE}=\sum_b\frac{n_b}{N}|\bar p_b-\bar y_b|,\qquad
\operatorname{Brier}=\frac1N\sum_i(p_i-y_i)^2.
$$

Matched-bin ECE is zero and Brier score 0.17. Overconfident-bin ECE is 0.112 and Brier score 0.18856. Blue plots prediction against observed fraction; orange is the agreement diagonal. Changing Beta shapes does not change these fixed evaluation bins. The two panels are separate illustrations, not one shared generative experiment.

## Limits and research practice

Zero empirical ECE establishes agreement in these five bins, not calibration at every probability, within every subgroup or after distribution shift. ECE depends on bin choices and sample size. Calibration alone also does not establish discrimination: always predicting an overall positive rate can fail to distinguish individuals.

In AI evaluation, document held-out data, temporal or group splits, label definitions and sample sizes. Data used to fit a calibrator such as temperature scaling should be separated from final evaluation data. This lab fits no calibrator; it compares fixed numerical examples.

## Exercises with explanations

1. What is parameter variance for Beta(12,12)? **0.01.**
2. Does ECE zero mean every label was predicted correctly? **No: it measures bin-frequency agreement.**
3. Is minimizing predictive variance enough? **No: assess model adequacy, calibration and predictive performance together.**

Further reading: [Stan User’s Guide, Posterior Predictive Checks](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html).
