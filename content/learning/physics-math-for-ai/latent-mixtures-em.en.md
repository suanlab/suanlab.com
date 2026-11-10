## Two groups with hidden membership

Suppose observations −2,−1,1,2 come from a mixture of two Gaussian components. Membership z is unobserved. Summing over memberships gives a weighted sum of component densities. This lab fixes both mixture weights at one half and both variances at one, estimating only the means.

$$
p(x_i\mid\mu)=\tfrac12\phi(x_i-\mu_1)+\tfrac12\phi(x_i-\mu_2).
$$

## E step: assign soft responsibilities

At current means, calculate each observation's posterior membership probability rᵢⱼ. This differs from assigning a hard zero-or-one label using distance alone. An observation explained by both components shares responsibility between them.

$$
r_{ij}=\frac{\frac12\phi(x_i-\mu_j)}{\sum_{l=1}^{2}\frac12\phi(x_i-\mu_l)}.
$$

## M step: update weighted means

Hold responsibilities fixed and update each mean using its assigned probability mass as weights. Alternating these exact E and M steps does not decrease observed log likelihood in this specified model. Rounded calculations or approximate optimization in other implementations require separate numerical checks.

$$
\mu_j^{\mathrm{new}}=\frac{\sum_i r_{ij}x_i}{\sum_i r_{ij}},\qquad \ell(\mu)=\sum_i\log p(x_i\mid\mu).
$$

Starting from means −1 and one, four updates produce approximately −1.4404 and 1.4404. Log likelihood rises from about −7.1582 to −6.8400. The horizontal axis is iteration number, not spatial position. The vertical axis is log likelihood gain from initialization. Blue shows the gain and orange the zero reference, making small improvements visible. Original log likelihood values remain in the readout. Table responsibilities are recalculated at the final updated means.

## Initialization and symmetry

If both means start at zero, all responsibilities equal one half. The weighted means are again zero, so the procedure remains stationary. Its likelihood is lower than the separated solution, yet the iteration cannot escape this symmetric initialization. Nondecreasing likelihood does not guarantee global optimization.

Swapping component labels preserves the mixture density; component numbers carry no intrinsic identity. Freely estimating Gaussian variances introduces further singularity issues, which are excluded here by fixing variance at one. EM in this lab is a point-estimation algorithm for two means, not a calculation of their full Bayesian posterior.

## Exercises with explanations

1. What do responsibilities sum to for each observation? **One.**
2. Do identical initial means automatically separate? **Not in this symmetric example.**
3. Does increasing likelihood establish a global optimum? **No: consider different initializations and model adequacy.**

Further reading: [Stan User’s Guide, Finite Mixtures](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html).
