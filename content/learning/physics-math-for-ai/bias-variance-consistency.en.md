## Compute the risk of an estimation procedure

Squared-error risk averages estimation error over repeated samples at a fixed parameter. One observed error does not determine procedure risk. The lab shrinks a binomial sample proportion toward one half. At λ=0 it is the ordinary proportion; at λ=1 it always returns one half. The plot transports the same sample probabilities to different estimator values, aggregating coincident outputs.

$$
T_\lambda=(1-\lambda)\frac Sn+\frac\lambda2,\qquad R_p(T)=E_p[(T-p)^2].
$$

## An exact bias–variance decomposition

Centering the estimator removes the expected cross term and gives an exact decomposition. Bias is λ(1/2−p) and variance is (1−λ)²p(1−p)/n. Shrinkage lowers variance but can increase bias away from one half. For p=0.8,n=10,λ=0.5, variance 0.004 plus squared bias 0.0225 gives risk 0.0265, worse than the unshrunk risk 0.016.

$$
E[(T-p)^2]=\operatorname{Var}(T)+(E[T]-p)^2.
$$

## Finite-sample improvement is possible

Changing p to 0.6 gives risk 0.006+0.0025=0.0085 under the same n and λ, below the unshrunk 0.024. Gains depend on parameter and shrinkage level. At λ=1 the law collapses to one point, yet error persists unless p equals one half. Small variance alone does not establish accuracy.

$$
\operatorname{Bias}_p(T_\lambda)=\lambda(1/2-p),\quad\operatorname{Var}_p(T_\lambda)=(1-\lambda)^2\frac{p(1-p)}n.
$$

## Consistency differs from unbiasedness

With fixed λ and increasing n, variance vanishes but squared bias can remain. Only λ=0 is consistent for every p among these fixed choices, although other λ work at p=1/2. Choosing λ_n tending to zero makes both components vanish in this example. Unbiasedness alone does not imply consistency. Vanishing mean squared error is a sufficient condition for convergence in probability.

$$
\lim_{n\to\infty}R_p(T_\lambda)=\lambda^2(1/2-p)^2.
$$

For example, an estimator that always uses only X1 stays unbiased as n grows but its Bernoulli variance does not shrink, preventing consistency at interior parameters.

## Exercises with explanations

1. What is limiting risk at p=0.8,λ=0.5? **0.0225.**
2. What is variance at λ=1? **Zero, though risk may be positive.**
3. Is shrinkage always beneficial? **No: inspect risk and assumptions.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
