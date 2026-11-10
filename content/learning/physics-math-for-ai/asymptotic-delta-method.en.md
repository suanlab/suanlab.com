## Variation after a nonlinear transformation

An asymptotically normal estimator can be transformed through a differentiable function using a first-order Taylor approximation. Here iid N(μ,1) observations give an exact finite-n sample-mean law N(μ,1/n). The graph evaluates a deterministic grid of base-noise values rather than drawing samples. Blue is the actual transform and orange its local linearization at μ.

$$
\sqrt n(\widehat\theta_n-\theta)\Rightarrow N(0,V)\implies\sqrt n(g(\widehat\theta_n)-g(\theta))\Rightarrow N(0,[g^\prime(\theta)]^2V).
$$

## Compare exact square moments

For squaring, g′(μ)=2μ and first-order variance is 4μ²/n. Exact Gaussian moments add 2/n², and the expectation is μ²+1/n rather than μ². At μ=1,n=10, exact variance is 0.42 versus first-order 0.4. An asymptotic approximation is not an identity for finite n.

$$
E[\bar X^2]=\mu^2+\frac1n,\qquad\operatorname{Var}(\bar X^2)=\frac{4\mu^2}n+\frac2{n^2}.
$$

## A zero derivative is not zero true variance

At μ=0 the first-order variance is zero, while exact variance is 2/n². Degeneracy on the √n scale is correct, but the n scale reveals a chi-square law with one degree of freedom. Here that law is exact at finite n. The flat orange line and varying blue curve show why a higher-order expansion and different scaling can be necessary.

$$
\mu=0:\qquad n\bar X^2\sim\chi_1^2,\qquad\operatorname{Var}(\bar X^2)=2/n^2.
$$

## The exponential transform and its scope

For exp, the derivative is nonzero but curvature still creates bias and approximation error. Exact expectation is exp(μ+1/(2n)); exact variance is (exp(1/n)−1)exp(2μ+1/n). The first-order variance is exp(2μ)/n. Large displayed noise values can reveal poor linearization, including negative linearized values for a genuinely nonnegative square. The method is local and asymptotic, not a uniform finite-sample tail guarantee.

$$
\operatorname{Var}(e^{\bar X})=(e^{1/n}-1)e^{2\mu+1/n},\qquad V_{\mathrm{delta}}=e^{2\mu}/n.
$$

Convergence in distribution alone does not guarantee moment convergence. Here the finite moments are calculated directly under the specified Gaussian law and compared with the first-order approximation.

## Exercises with explanations

1. What is exact square variance at μ=0,n=10? **0.02.**
2. Is the squared mean unbiased for μ²? **No: bias is 1/n.**
3. Do the exact formulas survive every non-Gaussian model? **No: distinguish asymptotic theory from the specified exact law.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[Wasserman, All of Statistics — author’s page](https://stat.cmu.edu/~larry/all-of-statistics/index.html).
