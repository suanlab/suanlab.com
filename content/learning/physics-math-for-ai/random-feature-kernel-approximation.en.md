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

## Derive an error statement with its probability space

Fix d=x−z and draw ideal independent ω_j∼N(0,ℓ⁻²). Put u=d²/ℓ². The characteristic function and cos²t=(1+cos2t)/2 give

$$
E\widetilde k=k,\qquad
E(\widetilde k-k)^2=\frac{(1-e^{-u})^2}{2m}.
$$

Independence divides the single-frequency variance by m. At d=0 the variance is exactly zero; the paired-feature diagonal is exact for every draw. At d=ℓ the table gives expected squared error over frequencies, not an error guarantee for the live seed.

| Frequencies m | Ideal expected squared error |
|---|---|
| 4 | .04994705 |
| 16 | .01248676 |
| 64 | .00312169 |
| 256 | .00078042 |

Because each cosine is in [−1,1], Hoeffding gives 2exp(−mε²/2) for one pair's absolute-error tail. A union bound over a fixed n-point dataset gives at most 2n²exp(−mε²/2) for any Gram entry to exceed ε. For n=3,ε=.2,δ=.05, m=295 is sufficient for this conservative bound. This exceeds the lab's maximum 32; it is a separate theoretical comparison. Neither a continuum supremum nor the fixed pseudorandom transcript follows from this finite-set statement. The code independently integrates the Gaussian cosine moments.

## Connect kernel error to the regularized training prediction

For PSD Gram matrices G and G̃ and λ>0, their smoothers satisfy the resolvent identity

$$
\widetilde S-S=\lambda(\widetilde G+\lambda I)^{-1}
(\widetilde G-G)(G+\lambda I)^{-1},
\qquad\|\widetilde S-S\|_{\rm op}\le\frac{\|\widetilde G-G\|_{\rm op}}\lambda.
$$

Both inverse norms are at most 1/λ. Hence training prediction error is at most ‖G̃−G‖op‖y‖/λ, and entrywise error ε implies the sufficient bound nε‖y‖/λ. Weak regularization amplifies approximation sensitivity. This controls predictions at these training inputs, not off-data kernel sections or population generalization. At λ=0 this argument provides no bound.

**Exercise:** Does PSD of the approximate Gram imply small prediction error? **Explanation:** No. PSD preserves a valid quadratic form, while closeness and regularized-solve stability require the additional estimates above.

```python
from math import exp, cos, sqrt, pi, ceil, log

def simpson(f,a,b,n=20000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if i%2 else 2)*f(a+i*h) for i in range(1,n)))/3

# Independent integration of the first and second moments at gap/length=1.
density=lambda w: exp(-w*w/2)/sqrt(2*pi)
first=simpson(lambda w: cos(w)*density(w),-10,10)
second=simpson(lambda w: cos(w)**2*density(w),-10,10)
var=(1-exp(-1))**2/2
assert abs(first-exp(-.5))<1e-11
assert abs(second-first*first-var)<1e-11
for m in [4,16,64,256]: print(m,round(var/m,8))
n,epsilon,delta=3,.2,.05
print('sufficient ideal iid frequencies:',ceil(2/epsilon**2*log(2*n*n/delta)))
# This bound is for n^2 fixed pairs, not all points in a continuum or a fixed PRNG seed.
```

## Exercises with explanations

1. Does a negative approximate kernel value violate PSD? **No.** PSD constrains Gram quadratic forms, not each entry's sign.
2. How many coordinates at m=32? **Sixty-four**, counting both coordinates per frequency.
3. Does matching one section imply identical learning results? **No.** Consider the full Gram error and stability of the regularized solve.

Further reading: [Rahimi and Recht, Random Features](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf).
