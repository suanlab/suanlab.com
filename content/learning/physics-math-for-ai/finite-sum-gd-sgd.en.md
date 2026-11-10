## A finite dataset makes the randomness explicit

Let four targets be yᵢ=1+σzᵢ with z=(−2,−1,1,2). The empirical objective averages four squared losses, rather than adding an unspecified noise term to an arbitrary neural network:

$$
F(x)=\frac14\sum_{i=1}^4\frac{(x-y_i)^2}{2}
=\frac{(x-1)^2}{2}+\frac{5\sigma^2}{4}.
$$

The optimizer is one. The irreducible constant in this empirical objective is not parameter error; the objective excess is (x−1)²/2. With a uniform random index, the component gradient x−yᵢ has conditional expectation x−1. Its conditional noise variance is 2.5σ², independent of the current x in this particular family.

## GD and SGD use different information

GD averages all four component gradients before updating. SGD selects one component. Writing eₖ=xₖ−1 gives

$$
e^{GD}_{k+1}=(1-\eta)e^{GD}_k,\qquad
e^{SGD}_{k+1}=(1-\eta)e^{SGD}_k+\eta\sigma z_{i_k}.
$$

Blue and orange plot these errors, not the full empirical loss. Setting σ=0 makes all components identical and both trajectories coincide. A seeded pseudo-random generator makes the orange illustration reproducible. Reusing a seed is not evidence of independent sampling or a statistical guarantee.

## Exact moments under an iid model

The table separately assumes that each index is independently uniform and independent of the previous iterate. Under precisely those assumptions, noise has conditional mean zero and the cross term vanishes:

$$
m_{k+1}=(1-\eta)m_k,\qquad
v_{k+1}=(1-\eta)^2v_k+2.5\eta^2\sigma^2,
\qquad\mathbb E[e_k^2]=m_k^2+v_k.
$$

These are analytic distributional moments, not statistics estimated from the displayed path. A single path can look better or worse than its expected squared error. A mean error near zero can coexist with a substantial spread around the optimizer.

## Constant steps and stability

For 0<η<2 the deterministic error contracts. With nonzero σ and a constant step, its stochastic variance instead approaches 2.5ησ²/(2−η). This noise floor concerns this quadratic and iid model; it is not a universal formula for arbitrary training.

At η=2, deterministic errors alternate sign without decreasing in magnitude. Each iteration adds 10σ² to the variance. At η=0 there is no movement, so an apparently stable trajectory does not imply optimization. Decreasing steps can reduce persistent noise under suitable summability, unbiasedness and variance conditions, which must be stated before claiming convergence.

## Exercise and explained answer

Choose initial x=−1, η=0.2, σ=1 and ten updates. Mean error is −2·0.8¹⁰, approximately −0.214748. Variance is approximately 0.274575, so MSE is approximately 0.320692. Half that MSE is expected objective excess. Turning σ off leaves MSE approximately 0.046117. Explain why the difference comes from variance rather than a different expected gradient.

In AI research, specify the finite-sum weighting, replacement scheme, filtration and variance assumptions. Random reshuffling and correlated batches need not obey the iid moment recurrence above.

Further reading: [Bottou, Curtis and Nocedal, Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838).
