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

## Derive a higher-order scalar delta limit

Suppose √n(θ̂_n−θ) converges in distribution to a real random variable Z. Let g:R→R be measurable and C^r in a neighborhood of θ for a positive integer r, with derivatives of orders 1 through r−1 equal to zero and g^(r)(θ) nonzero. Taylor's remainder gives

$$
g(\theta+h)-g(\theta)=\frac{g^{(r)}(\theta)}{r!}h^r+o(|h|^r),\qquad
n^{r/2}\{g(\widehat\theta_n)-g(\theta)\}\Rightarrow\frac{g^{(r)}(\theta)}{r!}Z^r.
$$

Why does the remainder disappear? Convergence in distribution makes √n(θ̂_n−θ) bounded in probability and hence θ̂_n→θ in probability. The remainder coefficient tends to zero in probability, while [√n(θ̂_n−θ)]^r remains bounded in probability. Their product tends to zero. The continuous mapping theorem handles the leading power. This proves the displayed limit; it does not infer convergence of moments from convergence in law.

For r=2 and Z∼N(0,V) with V>0, the limit is g″(θ)Vχ1²/2, rather than a nondegenerate normal law. The sign of g″ can make this a negative multiple of a chi-square variable. The page's square-at-zero example has g″=2 and V=1.

## Study a shrinking neighborhood of the stationary point

Let the data in row n be iid N(μ_n,1) with μ_n=c/√n for fixed real c. The sample mean has the exact representation (c+Z)/√n, with Z standard normal in distribution. Thus

$$
n\bar X_n^2\overset d=(Z+c)^2,\qquad
n(\bar X_n^2-\mu_n^2)\overset d=Z^2+2cZ.
$$

The first variable is noncentral chi-square with one degree of freedom and noncentrality c². Its mean is 1+c² and variance 2+4c². Squaring introduces bias 1/n on the original scale, even in this shrinking neighborhood.

The table gives n² times the actual variance and its first-order approximation.

| c | Exact | First order |
|---|---|---|
| 0 | 2 | 0 |
| $1/2$ | 3 | 1 |
| 2 | 18 | 16 |

The exact variance is (2+4c²)/n² and the linearized variance is 4c²/n². Their ratio is fixed as n grows. For c=1/2 the first-order value remains only one third of the exact value. Increasing n alone therefore does not make this relative variance error vanish uniformly near μ=0. A fixed nonzero-μ asymptotic statement cannot silently be treated as such a uniform guarantee.

## Compute the moments instead of assuming a normal error bar

Expand (Z+c)² and its square, using E[Z]=E[Z³]=0, E[Z²]=1 and E[Z⁴]=3. Subtracting the squared mean leaves 2+4c². These finite Gaussian moments justify the exact table; the higher-order distributional theorem by itself would not justify a variance calculation.

The code evaluates polynomial Gaussian moments with fractions and checks pure-power scaling algebraically on fixed base values. It does not draw random samples. The current controls still set a mean and sample size directly; the linked sequence μ_n=c/√n is a separate executable experiment.

```python
from fractions import Fraction as Q
from math import prod

def moment(k):
    return Q(0) if k % 2 else Q(prod(range(1, k, 2)))

def expected(poly):
    return sum(a*moment(k) for k, a in enumerate(poly))

def square(poly):
    out = [Q(0)]*(2*len(poly)-1)
    for i, a in enumerate(poly):
        for j, b in enumerate(poly):
            out[i+j] += a*b
    return out

for c in [Q(0), Q(1, 2), Q(2)]:
    transformed = [c*c, 2*c, Q(1)]  # (Z+c)^2, Z standard normal.
    mean = expected(transformed)
    variance = expected(square(transformed))-mean**2
    linear_variance = 4*c*c
    assert mean == 1+c*c and variance == 2+4*c*c
    for n in [10, 100]:
        exact = variance/(n*n)
        first_order = linear_variance/(n*n)
        assert exact-first_order == Q(2, n*n)
        print("c, n, exact variance, first order:", c, n, exact, first_order)
    print("scaled variance ratio:", linear_variance/variance)
for r in [2, 3, 4]:
    for scale in [2, 4, 8]:
        for z in [Q(-1), Q(1, 2), Q(2)]:
            estimate = z/scale  # n=scale^2, theta=0, g(x)=x^r.
            assert scale**r * estimate**r == z**r
print("Higher-order power scaling checked algebraically on fixed base values.")
```

## Further questions and research use

1. Can zero first derivative justify declaring the transformed estimator deterministic? **No: a higher-order term can determine its fluctuations.**
2. Does the higher-order limit automatically supply a valid normal confidence interval? **No: its law may be a polynomial of a Gaussian rather than Gaussian.**
3. What should an asymptotic claim specify near a stationary point? **The parameter sequence, scaling, limiting law and any moment or uniformity assumptions.**

For Taylor-based distribution and moment approximations, see [MIT 18.655, Lecture 16, Theorem 5.3.3 and the variance discussion](https://ocw.mit.edu/courses/18-655-mathematical-statistics-spring-2016/2ca793c6ea6c4a6fece39c0da3440ba2_MIT18_655S16_LecNote16.pdf). The scalar rth-order and local-Gaussian calculations above are derived explicitly.

## Exercises with explanations

1. What is exact square variance at μ=0,n=10? **0.02.**
2. Is the squared mean unbiased for μ²? **No: bias is 1/n.**
3. Do the exact formulas survive every non-Gaussian model? **No: distinguish asymptotic theory from the specified exact law.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[Wasserman, All of Statistics — author’s page](https://stat.cmu.edu/~larry/all-of-statistics/index.html).
