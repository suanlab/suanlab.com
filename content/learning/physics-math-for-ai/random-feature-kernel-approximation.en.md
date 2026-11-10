## Replace a kernel with explicit features

Small feature vectors can approximate kernel methods without storing a large Gram matrix. This extends the earlier high-dimensional random-feature lesson from one pair to function sections and a complete small Gram structure. Accuracy and PSD structure are separate properties.

The RBF kernel is a Gaussian characteristic function:

$$
k(x,z)=\exp\!\left(-\frac{(x-z)^2}{2\ell^2}\right)=E_\omega\cos(\omega(x-z)),\quad\omega\sim N(0,\ell^{-2}).
$$

## Pair cosine and sine

For each frequency, use two coordinates scaled by 1/√m:

$$
\begin{aligned}
\phi_m(x)&=\frac1{\sqrt m}(\cos\omega_jx,\sin\omega_jx)_{j=1}^m,\\
\widetilde k(x,z)&=\phi_m(x)^T\phi_m(z)=\frac1m\sum_j\cos(\omega_j(x-z)).
\end{aligned}
$$

Four frequencies therefore mean eight coordinates. The diagonal is one by the squared sine/cosine identity. A feature matrix Z produces Gram ZZᵀ, whose quadratic forms are nonnegative even when approximation error remains.

## What the experiment measures

Blue is the exact section k(x,z); orange is its feature approximation. The table shows actual frequencies and query features. Gram Frobenius error concerns only inputs −1,0,1. The displayed-grid maximum concerns 121 points, not a continuous supremum or learning risk.

A seed fixes one reproducible pseudorandom transcript. Unbiasedness and concentration statements concern ideal independent Gaussian draws; they do not certify this one seed. Increasing m need not monotonically reduce its error at every step.

## Computational and statistical tradeoffs

Feature dimension changes storage, computation and the approximate function space. Replacing kernel ridge with feature ridge can also change regularization and data fit. Specify kernel, domain, sample size, target tolerance and probability assumptions before claiming enough features.

## Exercises with explanations

1. Does a negative approximate kernel value violate PSD? **No.** PSD constrains Gram quadratic forms, not each entry's sign.
2. How many coordinates at m=32? **Sixty-four**, counting both coordinates per frequency.
3. Does matching one section imply identical learning results? **No.** Consider the full Gram error and stability of the regularized solve.

Further reading: [Rahimi and Recht, Random Features](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf).
