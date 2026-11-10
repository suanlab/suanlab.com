## Turn posterior approximation into optimization

When a posterior is difficult to compute, choose a tractable distribution family q and approximate it. This lab exposes the mechanism in a two-state latent model. The joint masses are p(x,z=1)=0.4π and p(x,z=0)=0.4(1−π). Evidence is therefore 0.4 and the exact posterior success probability is π. Changing π also changes the model's joint masses.

## Read ELBO as log joint plus entropy

The variational candidate is q(z=1)=r. Average log joint probability under q and add entropy. Expanding the definitions yields the identity relating evidence, ELBO and KL divergence.

$$
\mathcal L(q)=\sum_zq(z)\log p(x,z)-\sum_zq(z)\log q(z),
\qquad\log p(x)=\mathcal L(q)+\operatorname{KL}(q\Vert p(z\mid x)).
$$

Nonnegative KL makes ELBO a lower bound. For fixed data and model, log evidence does not depend on r, so maximizing ELBO minimizes this directed KL. Jointly training model parameters is a further setting, separate from this fixed-model demonstration.

## Check the numbers

At π=0.7 and r=0.5, KL≈0.08718, log evidence≈−0.91629 and ELBO≈−1.00347. Compare negative numbers carefully: −1.00347 is the smaller lower bound. Blue plots ELBO against r; the orange horizontal line is log evidence. The dot marks the selected r.

$$
\operatorname{KL}=r\log\frac r\pi+(1-r)\log\frac{1-r}{1-\pi}.
$$

## Family restrictions and KL direction

The family of all Bernoulli q distributions contains the exact posterior, so the gap reaches zero at r=π. A mean-field family that removes dependence between variables need not contain the exact posterior. Approximation error can remain even after successful optimization.

At r=1, q ignores state zero, yet KL(q||posterior)=log(1/π) remains finite. Zero-weight terms use the continuous convention 0 log 0=0. In the reverse direction, KL(posterior||q) is infinite because the posterior assigns positive mass where q is zero. Reversing the direction changes the problem. This small example does not guarantee approximation quality in high dimensions.

## Exercises with explanations

1. What is ELBO at r=π? **It equals log evidence.**
2. Does changing q increase fixed-model evidence? **No: it changes the bound and gap.**
3. Is KL a symmetric distance? **No: direction and support matter.**

Further reading: [Blei, Kucukelbir & McAuliffe, Variational Inference](https://arxiv.org/abs/1601.00670).
