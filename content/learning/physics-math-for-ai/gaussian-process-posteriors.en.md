## A consistent distribution over finite function evaluations

A Gaussian process makes function values jointly Gaussian at every finite input set. It needs a mean function and a PSD covariance kernel. This lab uses a zero-mean, unit-amplitude RBF prior with independent Gaussian observation noise.

$$
f(X)\sim N(0,K),\qquad y=f(X)+\epsilon,\quad\epsilon\sim N(0,\sigma^2I).
$$

The control is noise **variance**, not standard deviation. Adding a standard deviation directly to the Gram diagonal specifies a different model.

## Condition a joint Gaussian

For a query x*, define k*=k(X,x*) and k**=k(x*,x*). Conditional Gaussian formulas give

$$
\begin{aligned}
m_*&=k_*^T(K+\sigma^2I)^{-1}y,\\
v_*&=k_{**}-k_*^T(K+\sigma^2I)^{-1}k_*.
\end{aligned}
$$

Under unit amplitude and the previous sum-loss convention, the mean equals kernel ridge with λ=σ². The GP additionally supplies function uncertainty and a prior interpretation.

## Latent function versus new observation

Latent variance excludes new measurement noise. A fresh independent noisy observation has variance

$$
\operatorname{Var}(y_*\mid y)=v_*+\sigma^2.
$$

At noise variance .1, the two readouts always differ by .1. Noiseless observed locations have zero latent variance. Away from observations variance often returns toward prior uncertainty, but need not increase monotonically throughout a finite interval.

The implementation uses linear solves and stable conditional covariance differences, described below. Negative computed variance raises an error; no diagonal jitter is added.

## Interpret the interval honestly

Blue is the posterior mean and orange mean ±2√v*. Under the specified Gaussian model, a single fixed query has about 95.45% conditional mass inside this interval. It is not a simultaneous 95% band or a distribution-free guarantee.

With kernel and noise fixed, changing labels changes means but not posterior variances. Heteroscedastic data and uncertain learned hyperparameters require a richer model.

## Derive conditioning by an independent residual

Write A=K+σ²I and assume it is positive definite. In the joint Gaussian vector (y,f*), define z=f*−k*ᵀA⁻¹y. Then Cov(z,y)=0. Joint Gaussianity turns uncorrelatedness into independence, so conditioning on y leaves z's distribution unchanged. Its mean is zero and variance k**−k*ᵀA⁻¹k*, proving both prediction formulas. The block covariance's positive semidefiniteness also gives a nonnegative Schur complement.

For several queries, replace k* by the cross-covariance matrix: off-diagonal posterior covariances matter. Pointwise intervals do not automatically provide simultaneous coverage. Even two independent standard normals fall together in their individual ±2 intervals with probability erf(√2)²≈.9111 rather than .9545. This independent two-variable example is an illustration, not a claim that two actual GP queries are independent.

## Keep tiny positive variances separate from exact interpolation

The formula is mathematically correct, but subtracting two nearly equal unit quantities can lose the entire variance. With the live length scale one, the following values can be computed independently with 450-digit Decimal arithmetic.

| Noise variance | Query x | Latent variance |
|---|---|---|
| 0 | 0 | Exactly 0 |
| 10⁻³⁰ | 0 | Approximately 10⁻³⁰ |
| 0 | 10⁻⁸ | Approximately 1.49081872×10⁻¹⁷ |
| 0 | 10⁻¹⁰⁰ | Approximately 1.49081872×10⁻²⁰¹ |

The implementation first conditions on the nearest noiseless observation, expressing small covariance differences with expm1, then conditions on the other two points. It adds noise by the resolvent identity

$$
v_\lambda(x)=v_0(x)+\lambda k_x^T K^{-1}(K+\lambda I)^{-1}k_x,
\qquad\lambda=\sigma^2.
$$

This keeps the stated noise variance explicit even when adding it to a unit matrix diagonal rounds away. It uses positive-definite solves at the fixed three distinct inputs and adds no jitter. The identity follows by subtracting the two inverse matrices; it is not an additional probabilistic assumption.

A mathematically positive variance below binary64 range is displayed as — with a numeric-range note, including its interval half-width and new-observation variance. Exact noiseless training locations retain zero. A positive width is not evidence that the kernel or noise assumptions fit real data.

**Exercise:** Why does zero covariance imply independence in the proof? **Explanation:** The residual and observations are jointly Gaussian. This implication fails for general non-Gaussian variables.

```python
from decimal import Decimal as D, localcontext

def solve(A,b):
    a=[list(row)+[v] for row,v in zip(A,b)]
    n=len(a)
    for j in range(n):
        pivot=a[j][j]
        a[j]=[v/pivot for v in a[j]]
        for i in range(n):
            if i!=j:
                scale=a[i][j]
                a[i]=[v-scale*w for v,w in zip(a[i],a[j])]
    return [row[-1] for row in a]

with localcontext() as ctx:
    ctx.prec=450
    nodes=[D(-1),D(0),D(1)]
    def kernel(x,z): return (-(x-z)**2/2).exp()
    for noise,x in [(D(0),D(0)),(D('1e-30'),D(0)),(D(0),D('1e-8')),(D(0),D('1e-100'))]:
        A=[[kernel(u,v)+(noise if i==j else 0) for j,v in enumerate(nodes)] for i,u in enumerate(nodes)]
        k=[kernel(x,u) for u in nodes]
        w=solve(A,k)
        variance=1-sum(v*t for v,t in zip(k,w))
        if noise==0 and x==0: assert abs(variance)<D('1e-440')
        else: assert variance>0
        print('noise, x, latent variance:',noise,x,format(variance,'.8E'))
```

## Exercises with explanations

1. Do matching GP and ridge means imply identical predictive information? **No.** GP uncertainty and assumptions add information.
2. Is variance .1 the same as standard deviation .1? **No:** the latter has variance .01.
3. Must changed labels alter the variance? **Not here.** The fixed-kernel conditional variance does not depend on y.

Further reading: [Rasmussen and Williams, Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
