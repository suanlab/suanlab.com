## 1. Decide which response to preserve

An effective theory keeps the degrees of freedom and observables needed at the scales being probed. Imagine driving a light coordinate slowly while it is coupled to a stiff coordinate. The stiff coordinate still affects the light response. Integrating it out expresses that effect in the parameters and derivative terms of a light-only theory. [Burgess, Introduction to Effective Field Theory](https://arxiv.org/abs/hep-th/0701053)

## 2. Eliminate a Gaussian heavy mode exactly

For each Euclidean frequency $\omega$, take the real quadratic block
$$
Q(\omega)=
\begin{pmatrix}
\omega^2+m^2 & g\\
g & \omega^2+M^2
\end{pmatrix},
\qquad 0<m<M,\quad g^2<m^2M^2.
$$

The stability condition makes the block positive definite even at zero frequency. Complete the square in the heavy variable. Integrating its Gaussian gives a normalization factor independent of the light amplitude and a light kernel
$$
Q_{\rm light}(\omega)=\omega^2+m^2-\frac{g^2}{\omega^2+M^2}.
$$
The light variance or static source response for that mode is $1/Q_{\rm light}$. The determinant factor contributes to the absolute partition function.

## 3. Match a local derivative expansion

Set $x=\omega^2/M^2$. The heavy propagator contributes a factor $(1+x)^{-1}$. Expanding around $\omega=0$ gives a constant, a term proportional to $\omega^2$, and then terms with four or more derivatives. The first two light coefficients are
$$
Q_{\rm light}(0)=m^2-g^2/M^2,\qquad
\left.\frac{dQ_{\rm light}}{d(\omega^2)}\right|_0=1+g^2/M^4.
$$
Matching means choosing the low-frequency coefficients of the effective model to reproduce the full theory to the requested order. Use the matched theory to predict other low-frequency responses and compare them against exact values where available.

## 4. Estimate the error before using the result

Retain powers $x^j$ through $j=p$ in the heavy term. The finite geometric identity gives the exact kernel error
$$
|Q_{\rm light}-Q_p|=\frac{g^2}{M^2}\frac{x^{p+1}}{1+x}.
$$
Each additional order reduces the error by a factor $x$ for $0<x<1$. For $x>1$ the error grows with the order. The pole at $x=-1$ sets the Taylor radius, even though positive Euclidean frequencies have no pole.

The response error also depends on the two kernels: $|Q_{\rm light}^{-1}-Q_p^{-1}|=|Q_{\rm light}-Q_p|/(Q_{\rm light}Q_p)$ when both kernels are positive. A small kernel error can have a large response effect near a soft mode. A nonpositive truncated kernel cannot define a Gaussian probability.

## 5. Inspect the hierarchy and the matching convention

Use $m=1$, $M=4$ and $g=2$. The model is stable, and $Q_{\rm light}(0)=0.75$. At $\omega=M/2$, the error in the matched kernel is multiplied by $1/4$ at each extra order. At $\omega=2M$ it is multiplied by 4. The ratio $g/(mM)$ is a convenient way to vary mixing while retaining stability. Increasing $M$ at fixed $g$ decreases the static shift. Increasing $M$ while holding $g/(mM)$ fixed leaves a finite static shift.

```python
from math import isclose

m, M, g = 1.0, 4.0, 2.0
def kernels(omega, order):
    x = (omega / M)**2
    strength = (g / M)**2
    exact = omega**2 + m**2 - strength/(1+x)
    matched = omega**2 + m**2 - strength*sum((-x)**j for j in range(order+1))
    return exact, matched, strength*x**(order+1)/(1+x)

assert kernels(0, 0)[0] == 0.75
for ratio in [0.25, 0.5, 1, 2]:
    for order in range(5):
        exact, approx, error = kernels(ratio*M, order)
        assert isclose(abs(exact-approx), error, rel_tol=1e-8, abs_tol=1e-14)
        if approx > 0:
            assert isclose(abs(1/exact-1/approx), error/(exact*approx), rel_tol=1e-8, abs_tol=1e-14)
    print(ratio, kernels(ratio*M, 2))
```

## 6. Apply the same checks to a new effective model

- Specify the low-frequency observables and the normalization that matter.
- Match the retained coefficients and inspect the next-order error over the intended range.
- Repeat with a stronger mixing ratio near the stability boundary and compare kernel and response errors.
- When fitting a reduced statistical model, separate the parameters adjusted on matching data from predictions tested on other scales.
