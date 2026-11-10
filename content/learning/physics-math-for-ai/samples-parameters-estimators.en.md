## Distinguish language before and after observation

A model parameter p specifies the law generating repeated samples. In a frequentist sampling model it is fixed, while the observations are random before collection. The observed vector is one realization. An estimator is a function T of the random sample; an estimate T(x) is the number obtained after observation. Keeping these roles distinct prevents confusing a fixed parameter with a statistic whose value varies across samples.

$$
X_i\overset{\mathrm{iid}}\sim\operatorname{Bernoulli}(p),\qquad T(X)=\bar X=\frac1n\sum_{i=1}^nX_i.
$$

## Change the fixed data and model separately

Read the first n low bits of the mask as observations. With n=4 and mask three, the data are (1,1,0,0), giving estimate 0.5. Changing p from 0.5 to 0.8 leaves those fixed data unchanged but changes their model probability and the repeated-sampling law. A success observed under p=0 has zero model probability; this identifies incompatibility rather than changing the recorded observation.

$$
T(x)=\frac24=0.5,\qquad E_p[T]=p,\quad\operatorname{Var}_p(T)=\frac{p(1-p)}n.
$$

## An ordered sample and a count are different events

One ordered sample has probability p^k(1−p)^(n−k). The event of observing k successes combines multiple ordered samples and therefore includes a binomial coefficient. The defaults give 1/16 for the ordered vector and 6/16 for its count. Bars display the theoretical sampling probabilities of possible estimates, not a simulated trajectory; the orange marker locates the fixed observed estimate.

$$
P_p(S=k)={n\choose k}p^k(1-p)^{n-k},\qquad S=nT.
$$

## Connections to learning and evaluation

Standard error summarizes variability across repetitions of an estimation procedure. A realized estimate need not equal its sampling expectation. Doubling independent sample count halves variance here and multiplies standard error by 1/√2. Evaluation examples with dependence or selection effects require additional analysis; the blue sampling law assumes the stated iid model.

## Exercises with explanations

1. What are the first eight bits of mask fifteen? **(1,1,1,1,0,0,0,0).**
2. Can p change while the observed estimate stays fixed? **Yes: model and observation are separate.**
3. What is mean variance for n=4,p=0.5? **0.0625.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
