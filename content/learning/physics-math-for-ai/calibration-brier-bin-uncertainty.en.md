## Accuracy, probabilities and uncertainty differ

A prediction p=.9 does not mean every label is correct. A probability near ninety percent over repeated comparable cases differs from success on an individual label. Use three equally weighted strata, known Bernoulli probabilities q and base predictions [.15,.5,.85]. Positive temperature scaling divides logits and changes confidence without moving hard predictions across the .5 threshold here.

Blue shows true q at each predicted p; orange is the calibration diagonal. Actual sampled label frequencies appear separately in the table. Known q is a teaching oracle, not an assertion that real probabilities are directly observable.

## Exact conditional Brier decomposition

For Bernoulli Y,

$$
E[(p-Y)^2]=(p-q)^2+q(1-q).
$$

The first term is probability-prediction error; the second is intrinsic label variance. Average these conditional stratum terms. They are not simply renamed terms from a full Murphy decomposition with another binning or marginal variance. Even p=q has positive expected Brier unless q is zero or one.

## Uncertainty in observed bin frequencies

Generate N actual labels in each stratum. Their mean satisfies

$$
E[\widehat q]=q,\qquad\operatorname{Var}(\widehat q)=\frac{q(1-q)}N.
$$

For q=.5 and N=20, variance is .0125 and standard deviation approximately .1118. Observed frequencies can differ from p even under exact calibration. This is repeated-sampling variation under the declared model, not certainty about an individual label or an automatic parameter credible interval.

## Merging bins can hide errors

ECE here averages absolute differences between bin mean predictions and true or observed probabilities. Merge mode changes diagnostic bins only, retaining the actual predictions and Brier score. Reverse true probabilities to [.85,.5,.15]. Both endpoint strata have error .7, but global mean prediction and probability are both .5:

$$
\mathrm{ECE}_{separate}=\frac{.7+0+.7}{3}\approx .4667,\qquad
\mathrm{ECE}_{merged}=0.
$$

One perfect aggregate point does not establish within-stratum calibration. Absolute value after averaging differs from averaging absolute values. Brier remains approximately .495 in this miscalibrated example.

## Worked example and exercise

For q=p=[.15,.5,.85], probability-error contribution is zero and mean intrinsic variance is (.1275+.25+.1275)/3≈.16833. Hard accuracy is (.85+.5+.85)/3≈.73333. Perfect calibration is therefore different from perfect accuracy.

Exercise: lower temperature to .5. What changes? Solution: probabilities become more extreme but stay on the same sides of .5, retaining hard accuracy. They depart from the true q, making the squared probability-error term positive and increasing expected Brier. A finite empirical score additionally contains sampling variation.

## Reporting AI probability evaluations

Report accuracy, proper scores, bin definitions, sample counts and uncertainty together. Repeated selection of bins or temperature on the same evaluation data creates a selection procedure requiring separate assessment. The known-q decomposition is a transparent teaching reference, not an oracle available automatically in real datasets.

References: [MIT statistical learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/) and [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
