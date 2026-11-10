## Identify independent units before counting measurements

Uncertainty depends on covariances as well as measurement count. Repeats from one subject, dataset or training run may not be independent. The lab compares three designs with known unit marginal variance and input correlation. Correlation is a model assumption rather than an estimate from observed data. Paired and independent-group modes target a mean difference; clustered mode targets a single population mean.

$$
\operatorname{Var}\!\left(\frac1N\sum_iY_i\right)=\frac1{N^2}\sum_{i,j}\operatorname{Cov}(Y_i,Y_j).
$$

## Shared noise can cancel in a paired difference

For n independent pairs with unit marginal variances and within-pair correlation ρ, difference variance is 2(1−ρ). At n=10,ρ=0.8 the mean-difference standard error is 0.2. Independent groups with n measurements each give √(2/n), about 0.447. Both use 2n measurements but have different covariance structures. The repeat-count control is inactive in these two modes.

$$
\operatorname{Var}(\overline{Y_B-Y_A})=\frac{2(1-\rho)}n,\qquad\operatorname{Var}(\bar Y_B-\bar Y_A)=\frac2n\ \text{for independent groups}.
$$

## Clustered repeats do not add independent units

With n independent clusters, k measurements each and common within-cluster correlation ρ≥0, mean variance is [1+(k−1)ρ]/(nk). It exceeds the naive independence variance by the design effect. At n=10,k=4,ρ=0.8, DE=3.4 and standard errors are approximately 0.292 versus 0.158. Effective sample size is a variance-equivalent count for this specific mean model, not a universal count for every estimator.

$$
\mathrm{DE}=1+(k-1)\rho,\qquad n_{\mathrm{eff}}=\frac{nk}{\mathrm{DE}}.
$$

## Randomization, blocking and evaluation splits

At cluster correlation one, repeated noise is identical and effective sample size equals n. Zero paired-difference error at perfect correlation likewise reflects a strong model assumption rather than a real-world guarantee. Distinguish random assignment for reducing confounding from blocking or pairing for variance reduction, and distinguish assignment from random population sampling. AI evaluations should record reused data, seeds, selected outcomes, held-out splits and planned comparisons.

## Exercises with explanations

1. What is the cluster design effect at ρ=0? **One.**
2. Does repeat correlation affect variance when k=1? **No: there is only one measurement per cluster.**
3. Do changed seeds on one test set capture every population uncertainty? **No: identify which sources were independently replicated.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
