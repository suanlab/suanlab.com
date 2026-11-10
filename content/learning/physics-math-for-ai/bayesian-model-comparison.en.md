## Priors within and between models

M0 fixes θ=0.5. M1 assigns θ a Beta(a,a) prior. We also need separate prior probabilities P(M0) and P(M1). The Beta prior averages parameters within M1; model probabilities weight the competing models themselves.

## Compare evidence for the same data event

Observe one ordered sample with k successes in n=8 trials. M0 evidence is 0.5⁸; M1 evidence is a Beta-function ratio. Their ratio is the Bayes factor. Using the count event would multiply both evidences by the same binomial coefficient, leaving that ratio unchanged.

$$
\operatorname{BF}_{10}=\frac{p(x\mid M_1)}{p(x\mid M_0)},\qquad
p(x\mid M_1)=\frac{B(a+k,a+8-k)}{B(a,a)}.
$$

For uniform alternative a=1 and four successes, BF≈0.4063 favors the balanced point model. With eight successes, BF=256/9≈28.44 favors M1. A more flexible model need not have higher evidence: evidence averages rather than maximizing likelihood.

## Update odds, then convert to probability

Multiply prior odds by BF to obtain posterior odds. Convert odds O to probability using O/(1+O). With equal model priors, the default posterior P(M1) is about 0.2889; eight successes give about 0.9660. Keeping eight successes but reducing prior P(M1) to 0.1 lowers the posterior to about 0.7596.

$$
\frac{P(M_1\mid x)}{P(M_0\mid x)}=\operatorname{BF}_{10}\frac{P(M_1)}{P(M_0)}.
$$

The plot shows posterior P(M1) across possible success counts in blue and prior probability in orange. The k control selects the observed row's readout while leaving the full comparison curve unchanged. Alternative shape a and model prior probability change the curve itself.

## Relative comparison and adequacy

These probabilities condition on the two-model candidate set. Both models may miss the data-collection mechanism or be inadequate. A high posterior model probability is not universal certification of truth. Inspect prior sensitivity, posterior predictive checks and held-out prediction. AIC and cross-validation optimize different criteria and do not automatically equal this evidence calculation.

## Exercises with explanations

1. What are posterior odds under equal model priors? **The Bayes factor.**
2. Does BF alone specify posterior probability? **Prior model odds are also needed.**
3. What happens to a common binomial coefficient? **It cancels from the evidence ratio.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
