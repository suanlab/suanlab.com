## One number cannot describe every kind of uncertainty

Two predictors may have the same average error yet very different variability. Expectation locates the weighted center; variance describes squared spread around it. Neither alone specifies a distribution. This unit derives both quantities on a distribution small enough to inspect completely.

Let $X$ take values $-1,0,2$ with probabilities $1/4,1/2,1/4$. The lab accepts nonnegative weights and normalizes them. A weight is not a probability until its normalization is stated.

## Definitions and units

| Quantity | Definition |
| --- | --- |
| Mean $\mu$ | $E[X]=\sum_xxp(x)$ |
| Second moment | $E[X^2]=\sum_xx^2p(x)$ |
| Variance | $E[(X-\mu)^2]$ |
| Standard deviation | $\sqrt{\operatorname{Var}(X)}$ |

The finite distribution here has every required moment. For unbounded variables, expectation and variance require integrability conditions; a formula cannot create a finite moment when its defining sum or integral diverges. If $X$ has meters as units, its mean and standard deviation have meters, while its variance has square meters.

## Work the example by hand

The mean is $(-1)/4+0/2+2/4=0.25$. Squaring first gives $E[X^2]=1/4+4/4=1.25$. The mean squared is only $0.0625$.

Expand the definition to obtain

$$
E[(X-\mu)^2]=E[X^2]-2\mu E[X]+\mu^2
=E[X^2]-\mu^2=1.1875.
$$

The standard deviation is about 1.089725. The mean 0.25 is not one of the possible values of $X$, and it need not be. It is a weighted center, not a guaranteed observation.

You can check variance directly: squared deviations are $1.5625,0.0625,3.0625$, whose weighted sum is 1.1875. Agreement of two computations is a useful arithmetic check; the expansion supplies the general identity.

## What changes under shifting and scaling?

Linearity gives $E[aX+b]=aE[X]+b$. Subtracting this mean shows

$$
\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X).
$$

A constant shift changes location but not spread. Multiplying by two doubles standard deviation and multiplies variance by four. This is why reporting variance without units can hide a scale change.

For two variables, covariance is $E[(X-E[X])(Y-E[Y])]$. Then $\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y)$. Adding variances without checking the covariance can substantially understate uncertainty. Independence with finite second moments implies zero covariance; the converse is generally false.

## Expected loss and the best constant prediction

Suppose a constant predictor $a$ is evaluated using squared error. Expanding around the mean gives

$$
E[(X-a)^2]=\operatorname{Var}(X)+(a-\mu)^2.
$$

The cross term vanishes because $E[X-\mu]=0$. Therefore the mean minimizes expected squared error among constant predictions. This is a precisely scoped result: it depends on squared loss and a finite second moment. Absolute loss leads to a median instead, so a different objective can select a different prediction.

## Read the lab correctly

Blue bars show probabilities; the orange line marks the mean’s horizontal location. Its height is a marker, not extra probability mass. Changing weights changes both center and spread. In the single-point preset, $X=2$ with certainty and variance is zero. In the asymmetric preset, probabilities $(0.6,0.2,0.2)$ give mean $-0.2$ and variance 1.36.

These are population quantities from a specified distribution. Estimating variance from samples is a separate problem. The common sample denominator $n-1$ is justified under assumptions about independent identically distributed observations, not by changing the definition of population variance.

## Exercises and research connection

1. Calculate the mean and variance of $2X+3$. **Answer:** 3.5 and 4.75.
2. Explain why $E[X^2]\ge E[X]^2$ when the second moment is finite. **Answer:** their difference is a nonnegative variance.
3. Construct two distributions with equal means and different variances.
4. Decide which error quantity an experiment actually reports: variance across examples, variance across seeds or uncertainty of an estimated mean. These are different sampling questions.

For expectations, moments and covariance, see [MIT Introduction to Probability](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/). When extending the example to real data, report the sampling unit and dependence assumptions before attaching an uncertainty interpretation.
