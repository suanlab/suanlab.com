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

## From one-pair variance to a fixed finite-set guarantee

Take N≥2 fixed one-dimensional points, a lengthscale ℓ>0, and ideal independent ω_j~N(0,ℓ⁻²) frequencies chosen after the points. Keep the paired cos/sin map with m frequencies. For an error threshold η>0, each pair estimate averages m cosines in [−1,1], so Hoeffding gives

$$
P(|\widehat k(x,y)-k(x,y)|\geq\eta)
\leq2\exp(-m\eta^2/2).
$$

A union bound over L=N(N−1)/2 distinct pairs bounds the probability of maximum entry error exceeding η by 2L exp(−mη²/2). A sufficient frequency count is m≥(2/η²)ln(2L/δ), for 0<δ<1. Pairs may depend on one another through shared frequencies; the required independence concerns frequency draws. This finite-set guarantee is different from a uniform guarantee on a continuous interval.

## Convert entry error into a Gram operator bound

Let K and K̂ be exact and approximate Gram matrices on the same N points, with E=K̂−K. Each paired feature has squared norm one, giving K_ii=K̂_ii=1. If all off-diagonal errors are at most η, then for a unit vector v,

$$
|v^\top Ev|\leq\eta\sum_{i\ne j}|v_i||v_j|
=\eta\left[\left(\sum_i|v_i|\right)^2-1\right]
\leq(N-1)\eta.
$$

Hence ||E||₂≤(N−1)η. More points can amplify the same entry error when considering the whole linear operation. Both Gram matrices are PSD; for K̂ this follows directly from cᵀK̂c=||Σc_i z(x_i)||²≥0.

## Ridge coefficients depend on regularization scale

For λ>0 define α=(K+λI)⁻¹y and α̂=(K̂+λI)⁻¹y. This convention uses λ rather than Nλ. The inverse identity and the PSD inverse bounds, each at most 1/λ, give

$$
(\widehat K+\lambda I)^{-1}-(K+\lambda I)^{-1}
=-(\widehat K+\lambda I)^{-1}E(K+\lambda I)^{-1},
$$
$$
\|\widehat\alpha-\alpha\|_2
\leq\frac{\|E\|_2\|y\|_2}{\lambda^2}
\leq\frac{(N-1)\eta\|y\|_2}{\lambda^2}.
$$

This controls coefficients at the training points. Prediction at a new point also requires that point's kernel-vector error; the formula alone does not establish generalization performance. Small λ can strongly amplify a given kernel approximation error.

## Compare both errors in a small fixed example

Fix x=(0,.5,1,1.5), ℓ=1 and y=(1,−1,1,−1). Entry below is maximum Gram entry error; Coeff is ||α̂−α||₂ at λ=1. A separate fixed Python frequency prefix is used, so its values need not match the JavaScript seed in the screen.

| m | Entry | Coeff |
|---|---|---|
| 2 | 0.49891 | 0.19455 |
| 8 | 0.33936 | 0.10495 |
| 32 | 0.12520 | 0.02672 |

The decrease in this realization does not guarantee monotonic error for every seed and input. The code compares feature inner products with cosine averages and solves systems at λ=.1,1,10, checking the deterministic bound numerically. It is a floating-point check, not a rigorous rounding-error certificate for every matrix.

```python
from math import cos, sin, exp, sqrt
from random import Random

def solve(matrix, vector):
    n = len(vector)
    a = [list(row)+[value] for row, value in zip(matrix, vector)]
    for j in range(n):
        pivot = a[j][j]
        assert pivot > 0  # These unpivoted pivots are positive for the SPD systems below.
        a[j] = [v/pivot for v in a[j]]
        for i in range(n):
            if i != j:
                factor = a[i][j]
                a[i] = [v-factor*w for v, w in zip(a[i], a[j])]
    return [row[-1] for row in a]

def norm(v):
    return sqrt(sum(x*x for x in v))

x, y = [0, .5, 1, 1.5], [1, -1, 1, -1]
N = len(x)
K = [[exp(-.5*(a-b)**2) for b in x] for a in x]
rng = Random(7)
omega = [rng.gauss(0, 1) for _ in range(32)]
for m in [2, 8, 32]:
    z = [[v/sqrt(m) for w in omega[:m] for v in [cos(w*a), sin(w*a)]] for a in x]
    approx = [[sum(a*b for a, b in zip(zi, zj)) for zj in z] for zi in z]
    cosine = [[sum(cos(w*(a-b)) for w in omega[:m])/m for b in x] for a in x]
    assert max(abs(approx[i][j]-cosine[i][j]) for i in range(N) for j in range(N)) < 1e-12
    error = max(abs(approx[i][j]-K[i][j]) for i in range(N) for j in range(N))
    for lam in [.1, 1, 10]:
        exact_system = [[K[i][j]+lam*(i == j) for j in range(N)] for i in range(N)]
        approx_system = [[approx[i][j]+lam*(i == j) for j in range(N)] for i in range(N)]
        exact, estimated = solve(exact_system, y), solve(approx_system, y)
        difference = norm([a-b for a, b in zip(exact, estimated)])
        bound = (N-1)*error*norm(y)/lam**2
        assert difference <= bound+1e-12
        if lam == 1:
            print('m, max entry error, coefficient error at lambda=1, upper bound:',
                  m, error, difference, bound)
```

## Questions for research use

1. Is m the number of feature coordinates? **Here m counts frequencies; there are 2m coordinates.**
2. Does small entry error imply coefficient error of the same size? **No: account for amplification by N and λ.**
3. What if evaluation points are selected after inspecting the frequencies? **Recheck the fixed finite-set assumptions.**

For background, see [Rahimi–Recht, Random Features for Large-Scale Kernel Machines §3](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf). The constants here use the paired map's [−1,1] average, and the finite Gram and ridge bounds are derived explicitly.

## Exercises with explanations

1. How many coordinates at m=32? **Sixty-four.**
2. What is error at r=0? **Exactly zero.**
3. Does a negative entry rule out a PSD Gram matrix? **No: matrix PSD differs from entrywise positivity.**

Further reading: [Rahimi & Recht, Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf).
