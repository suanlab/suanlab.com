## 1. Ask what a small field changes

In the quartic Gibbs chapter we computed a mean and a covariance at a fixed external field. Here we vary the field and read both observables from one function. Imagine tuning a source until a material reaches a target average. In the source description you specify the control and observe the mean. The Legendre transform lets you specify the mean and recover the source.

## 2. The connected generating function

Let $A_0(x)$ be the dimensionless zero-source action on a finite lattice. With $h$ coupled as $h\cdot x$, define

$$
Z(h)=\int e^{-A_0(x)+h\cdot x}\,dx,\qquad W(h)=\log Z(h)-\log Z(0).
$$

Thus $e^{W(h)}$ is the moment generating function of the zero-source Gibbs measure. For the positive quartic model, a quartic envelope bounds the integrand and its polynomial derivatives uniformly for sources in compact sets. This justifies differentiation under the integral. [Tong, Statistical Field Theory §2.2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2).

## 3. Curvature measures response

Differentiating $W$ once inserts $x_i$. Differentiating the resulting ratio again also differentiates its denominator.

$$
\partial_i W=m_i=\mathbb E_h[x_i],\qquad
\partial_jm_i=\partial_{ij} W
 =\mathbb E_h[x_i x_j]-m_i m_j=\chi_{ij}.
$$

The subtraction removes the contribution from the means. For any direction $v$, $v^\mathsf T\chi v=\operatorname{Var}_h(v\cdot x)$. This proves convexity. Positive density throughout a finite-dimensional Euclidean space makes that variance positive for $v\ne 0$, because the linear observable varies across the support. The fourth derivative in a single direction is the fourth central moment minus three times the variance squared.

## 4. Trade source coordinates for mean coordinates

Define the Legendre–Fenchel transform

$$
\Gamma(m)=\sup_h\{h\cdot m-W(h)\}.
$$

At an interior mean in the range of $\nabla W$, stationarity gives $m=\nabla W(h)$. Strict convexity makes the corresponding source unique. The tangent plane of $W$ at that source supports its graph, and the value of the transform is the negative intercept of that plane.

Differentiate $\Gamma=h(m)\cdot m-W(h(m))$. Terms containing $dh$ cancel since $\nabla W(h)=m$. Differentiating again and using the inverse function theorem gives

$$
\nabla_m\Gamma=h,\qquad
\nabla_m^2\Gamma=\chi(h)^{-1}.
$$

The inverse appears in the space of means. The microscopic Hessian $\nabla_x^2 A_0$ acts in configuration space and can have negative eigenvalues even when this effective curvature is positive.

## 5. Work through two exact examples

A centered Gaussian with positive definite precision $K$ gives $W(h)=h^\mathsf T K^{-1}h/2$ and $\Gamma(m)=m^\mathsf T K m/2$. Its inverse response is constant.

For one unbiased spin $s\in\{-1,1\}$, $W(h)=\log\cosh h$, $m=\tanh h$, and $\chi=1-m^2$. Inside $(-1,1)$, $h=\operatorname{artanh}m$ and

$$
\Gamma(m)=\tfrac{1+m}{2}\log(1+m)+\tfrac{1-m}{2}\log(1-m).
$$

The transform reaches $\log 2$ at either endpoint by continuity. A finite source reaches only interior means. Approaching saturation makes $\chi$ tend to zero and the effective curvature diverge. Outside $[-1,1]$ the transform is infinite.

## 6. A numerical quartic example

This Python exercise takes the mean as input. A bracketed solve searches for the source that produces it. Gauss–Legendre quadrature on a finite interval supplies numerical estimates. Compare different intervals and resolutions as in the preceding chapter.

```python
import numpy as np

def response(radius, panels):
    nodes, weights = np.polynomial.legendre.leggauss(panels)
    x, w = radius * nodes, radius * weights
    base = np.log(w) - 0.25*x**4 + 0.5*x**2

    def observe(h):
        logw = base + h*x
        offset = np.max(logw)
        p = np.exp(logw - offset)
        total = p.sum()
        p /= total
        mean = p @ x
        return offset + np.log(total), mean, p @ (x-mean)**2

    return observe

def invert(target, observe):
    lo, hi = -2.0, 2.0
    assert observe(lo)[1] <= target <= observe(hi)[1]
    for _ in range(50):
        mid = (lo + hi)/2
        if observe(mid)[1] < target:
            lo = mid
        else:
            hi = mid
    h = (lo + hi)/2
    logz, mean, chi = observe(h)
    return h, h*mean - logz + observe(0)[0], 1/chi

observe = response(6, 96)
target, step = 0.6, 1e-3
h, gamma, curvature = invert(target, observe)
slope = (invert(target+step, observe)[1] - invert(target-step, observe)[1])/(2*step)
np.testing.assert_allclose(slope, h, atol=2e-7, rtol=0)
np.testing.assert_allclose(
    [h, gamma, curvature], invert(target, response(7, 128)), atol=2e-8, rtol=0
)
print(h, gamma, curvature)
```

## 7. What to check

- At zero source the even quartic model has zero mean and positive susceptibility. Explain the different symmetry of $x$ and $x^2$.
- At fixed target mean, read the source from the slope of $\Gamma$. For two targets, predict the ordering of their sources before calculating.
- Compare the quartic bare curvature $A_0''(0)=-1$ with $\Gamma''(0)>0$. The first is local in configuration space. The second includes fluctuations across both wells.
- In an energy-based model with a fixed reference measure, a source parameter likewise tilts the distribution. The mean gives the gradient of the log normalizer. The covariance gives its response matrix.
