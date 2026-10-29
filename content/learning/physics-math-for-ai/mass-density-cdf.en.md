## A random variable maps outcomes to numbers

A random variable is a measurable function mapping outcomes to numbers. Its law records probabilities of sets of numerical values. Discrete distributions can assign positive point masses; distributions with a Lebesgue density give every individual point probability zero. Compare a Bernoulli law with success probability one quarter against a uniform law on [0,w]. The width control applies only to the uniform mode.

$$
F_X(x)=P(X\le x),\qquad p_X(x)=P(X=x)\quad\text{for a discrete law}.
$$

## Density is not probability itself

Uniform density is 1/w inside its support. At w=0.5 it equals two, exceeding one. It measures probability per unit length, so integrate it over a region to obtain a probability. The accumulated area up to x=0.25 is two times 0.25, or 0.5, while the point probability remains zero. A density-based likelihood can likewise exceed one without invalidating a model.

$$
P(a<X\le b)=\int_a^bf_X(t)\,dt,\qquad\int_{\mathbb R}f_X(t)\,dt=1.
$$

## The CDF handles both cases

A CDF is nondecreasing and right-continuous, with limits zero and one at the two infinities. The Bernoulli CDF jumps to 0.75 at zero and to one at one. Open circles indicate excluded endpoint values; filled circles indicate actual CDF values. A jump measures point mass. The uniform CDF instead rises continuously and linearly on its support. Read the table’s mass and density columns separately.

$$
P(X=x)=F_X(x)-F_X(x^-).
$$

## Conditions behind differentiation

For an absolutely continuous law, the CDF derivative equals a density almost everywhere. A continuous CDF need not have a density: singular continuous laws exist. Mixed laws can combine atoms with a continuous part. These two examples therefore do not partition all possible distributions into discrete and density-based cases. The uniform density includes its endpoints by convention; changing finitely many density values would not change integrated probabilities.

$$
F_X(x)=\int_{-\infty}^xf_X(t)\,dt\quad\text{for absolutely continuous laws}.
$$

## Exercises with explanations

1. What is P(X=0.25) for Uniform[0,0.5]? **Zero.**
2. What is F(1)−F(1−) for Bernoulli(1/4)? **0.25.**
3. What is the CDF above the support? **One.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
