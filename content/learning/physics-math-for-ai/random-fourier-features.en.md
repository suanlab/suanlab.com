## Replace a kernel with finite features

The Gaussian RBF k(x,y)=exp(−(x−y)²/(2ℓ²)) approaches one for nearby points. Instead of storing the full kernel matrix, build explicit frequency features. This chapter uses a paired cosine/sine map for one-dimensional inputs.

$$
z(x)=\frac1{\sqrt m}(\cos\omega_1x,\sin\omega_1x,\ldots,\cos\omega_mx,\sin\omega_mx).
$$

There are m frequencies and 2m coordinates. Do not mix its normalization or variance with a different random-phase cosine-only construction.

## Trigonometry and the Gaussian characteristic function

The identity cos a cos b+sin a sin b=cos(a−b) makes the inner product an average of cos(ωⱼr), where r=x−y. Under ideal independent N(0,ℓ⁻²) frequencies, the Gaussian characteristic function gives the RBF expectation.

$$
z(x)^\top z(y)=\frac1m\sum_j\cos(\omega_jr),\qquad E[\cos(\omega r)]=e^{-r^2/(2\ell^2)}.
$$

## Exact pointwise sampling variance

Using cos²u=(1+cos2u)/2 yields a closed form. With s=r/ℓ:

$$
\operatorname{Var}(\hat k(r))=\frac{(1-e^{-s^2})^2}{2m}.
$$

Variance decreases as 1/m, but one seed's realized error need not decrease at every separation. At r=0, all cosines equal one, so estimate and zero variance are exact. A sampling standard error differs from observed absolute error and does not certify simultaneous accuracy over an input region.

## Inspect a reproducible realization

Blue is exact RBF; orange uses fixed pseudo-Gaussian frequencies. Reproducibility comes from a pseudo-random generator and Box–Muller transform, distinct from the ideal iid sampling law behind the variance formula. Increasing m retains a prefix of the same seed's frequencies.

Negative estimates are not clipped. An explicit-feature Gram matrix is positive semidefinite even though individual off-diagonal inner products can be negative. Forcing every estimated RBF entry positive defines a different estimator.

## Connections and limitations

The example motivates kernel approximation and explicit-feature learning. A smaller lengthscale changes more rapidly and may be harder to approximate at fixed m. Uniform guarantees also require data dimension, domain, accuracy and probability assumptions. This one-dimensional illustration alone does not establish general learning performance.

## Exercises with explanations

1. How many coordinates at m=32? **Sixty-four.**
2. What is error at r=0? **Exactly zero.**
3. Does a negative entry rule out a PSD Gram matrix? **No: matrix PSD differs from entrywise positivity.**

Further reading: [Rahimi & Recht, Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf).
