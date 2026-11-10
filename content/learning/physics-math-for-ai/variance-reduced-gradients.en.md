## Reduce noise without changing the mean direction

A control variate subtracts a correlated quantity with known expectation, then adds that expectation back. The goal is to reduce variance while preserving the desired mean. In finite-sum optimization, evaluating the same component gradient at two nearby points provides such a correlation.

The lab uses fᵢ(x)=hᵢ(x−aᵢ)²/2 with h=(1,2,3,4) and a=(−1,1,2,0). The full gradient is G(x)=2.5x−1.75, so the optimizer is x*=0.7. Individual gradients need not vanish there: the component minimizers disagree.

## Derive the SVRG estimator

Compute the full gradient once at an anchor x̃, then sample a uniform component and form

$$
v_i(x,\widetilde x)=g_i(x)-g_i(\widetilde x)+G(\widetilde x),
\qquad\mathbb E_i[v_i]=G(x).
$$

The two component gradients must use the same index. Subtracting an unrelated component can preserve the mean under extra independence assumptions but changes the covariance and loses the intended cancellation.

For this quadratic family the target-dependent terms cancel exactly:

$$
v_i=h_i(x-\widetilde x)+2.5\widetilde x-1.75,
\qquad\operatorname{Var}(v_i)=1.25(x-\widetilde x)^2.
$$

The factor 1.25 is the population variance of h=(1,2,3,4). The anchor-distance formula is specific to this example, not a universal SVRG variance identity.

## Three illuminating anchors

At x=x̃=0.7 all SVRG estimates are zero. Ordinary SGD gradients are (1.7,−0.6,−3.9,2.8), with mean zero and variance 6.575. This nonzero optimum noise is possible because the component objectives do not interpolate a shared minimizer.

Keep x=0.7 and move the anchor to −1. The SVRG variance becomes 1.25·1.7²=3.6125, still unbiased but noisier than with the matching anchor. An arbitrarily stale anchor can even make variance larger than ordinary SGD variance; variance reduction is not a pointwise guarantee for every possible pair.

Now choose x=x̃=2. SVRG variance is zero, but every estimate equals 3.25. The direction is exact yet nonzero: zero variance is not an optimality certificate.

## Account for work and outer loops

The diagram compares four possible component estimates at one pair of points. It is not a full SVRG training run. Computing G(x̃) costs four component gradients here. An inner update generally evaluates gradients at both x and the anchor unless anchor information is cached. The full-gradient cost must be amortized across several updates, with a specified anchor-refresh schedule.

Convergence rates require assumptions such as smooth component losses and appropriate convexity or nonconvex stationarity targets. A lower one-step variance alone does not specify the step size, total work or final accuracy.

## Exercise and explained answer

At x=0.7,x̃=−1 and selected index zero, g₁(x)=1.7. The SVRG estimate is 1.7−g₁(−1)+G(−1)=−2.55. With η=0.1 the SGD next point is 0.53 while the SVRG next point is 0.955. Explain why either individual step can move away from the optimizer even though both estimators have mean zero there. Verify all four estimates average to G(x).

Further reading: [Johnson and Zhang, Predictive Variance Reduction](https://papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html).
