## When the sampling law is unknown

A statistic’s sampling distribution depends on an often unknown population law. The nonparametric bootstrap substitutes the empirical distribution of observed data. For (0,1,2), each observed row gets mass one third. Resamples also have size three and use replacement, so a row may appear repeatedly.

$$
\widehat P_3=\frac13\sum_{i=1}^3\delta_{x_i},\qquad X_1^*,X_2^*,X_3^*\mid x\overset{\mathrm{iid}}\sim\widehat P_3.
$$

## Enumerate a small case completely

Three row choices at each of three positions give twenty-seven ordered resamples. The lab enumerates all of them. Duplicate observed values still correspond to separate row choices; equal statistic values are aggregated afterward. For the mean, outputs range from zero to two in steps of one third with counts (1,3,6,7,6,3,1). This law is exact conditional on the data, not the exact population sampling law.

$$
P^*(T^*=t\mid x)=\frac{\#\{\text{ordered resamples giving }t\}}{27}.
$$

## Bias, standard error and nonlinear statistics

For a mean, the conditional bootstrap expectation equals the original mean, giving zero bootstrap bias. Its standard error for (0,1,2) is √(2/9). With data (0,0,3) and the median, the observed statistic is zero but the bootstrap median equals three with probability 7/27. Its conditional expectation and bootstrap bias estimate are 7/9. This is an empirical-law calculation rather than exact knowledge of population bias.

$$
\widehat{\mathrm{bias}}^*=E^*[T^*]-T(x),\qquad\widehat{\mathrm{SE}}^*=\sqrt{\operatorname{Var}^*(T^*)}.
$$

## Limits of percentile intervals

Endpoints use the inverse empirical CDF of resampled statistics: choose the first sorted value reaching q, without interpolation. The selected 80% or 95% is a percentile level, not a demonstrated coverage guarantee for this tiny sample. Constant data give zero bootstrap standard error without proving a constant population. Time series or clustered data require considering resampling units that preserve dependence.

$$
[\,\widehat F_*^{-1}(\alpha/2),\;\widehat F_*^{-1}(1-\alpha/2)\,].
$$

## Exercises with explanations

1. Are identical-valued row choices counted separately? **Yes, then statistic outputs are aggregated.**
2. Is shuffling every observation once the same procedure? **No: this uses replacement.**
3. Is nominal percentile coverage automatic at small n? **No: additional theory and diagnostics are needed.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[Wasserman, All of Statistics — author’s page](https://stat.cmu.edu/~larry/all-of-statistics/index.html).
