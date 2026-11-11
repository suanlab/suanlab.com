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

## A family can prevent a zero approximation error

The Bernoulli family above contains its exact posterior. To isolate the effect of removing dependence, consider a normalized bivariate Gaussian target p. This is a separate example assuming the exact posterior has the specified form. For |ρ|<1 its covariance Σ is positive definite. Candidates q can change only their mean m and positive diagonal variances d_1,d_2.

$$
p=N(0,\Sigma),\quad \Sigma=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix},\qquad
q=N(m,D),\quad D=\operatorname{diag}(d_1,d_2).
$$

Subtract the two Gaussian log densities and average under q. Writing Λ=Σ^{-1} for precision gives

$$
\operatorname{KL}(q\Vert p)=\frac12\left[
\operatorname{tr}(\Lambda D)+m^\top\Lambda m-2+
\log\frac{\det\Sigma}{d_1d_2}\right].
$$

Positive definiteness makes the mean term uniquely minimal at m=0. The term depending on d_i is (Λ_ii d_i−log d_i)/2. Its derivative is (Λ_ii−1/d_i)/2 and its second derivative is 1/(2d_i²)>0. Thus the global optimum within this diagonal Gaussian family uses reciprocal diagonal precision.

$$
d_i^*=\Lambda_{ii}^{-1}=1-\rho^2,\qquad
\min_q\operatorname{KL}(q\Vert p)=-\frac12\log(1-\rho^2).
$$

## Picture a tilted ellipse and a smaller circle

Each exact marginal variance is one, while each optimal q variance is 1−ρ². For ρ≠0, q does not match the marginal variances. Its variance instead equals the target Gaussian's conditional variance Var(Z_1|Z_2). This is a consequence of the family and KL direction, even with successful optimization.

For positive ρ, the target contours form an ellipse along the positive diagonal. Expanding q's independent circle to marginal standard deviation one would place mass along the opposite diagonal where p has little density. The optimum for KL(q||p) uses a smaller circle. This describes the specified Gaussian family; it is not a theorem that every variational method always reduces variance.

![At correlation 0.9, the exact blue ellipse, smaller orange forward-KL circle and dashed reverse-KL circle have different covariance matrices.](/assets/learning/bayes-mean-field-en.svg)

The diagram fixes ρ=0.9. Each contour has Mahalanobis radius two for its own distribution; these are density contours, not a claim of 95% coverage.

| ρ | Var(q) | KL |
|---|---|---|
| 0 | 1 | 0 |
| $1/2$ | $3/4$ | 0.14384 |
| $9/10$ | $19/100$ | 0.83037 |

Changing the sign of ρ reverses the ellipse's tilt but preserves these variances and minimum KL values. At |ρ|=1 the target is singular, so the density and inverse-matrix formulas no longer apply.

## Reverse direction changes the optimal approximation

Minimizing KL(p||q) over the same family gives mean zero and both variances one. Direct expansion gives variance terms (1/d_i+log d_i)/2, minimized at d_i=1. This direction matches the exact marginal moments but still sets covariance to zero. In this two-dimensional example the two minimum KL numbers happen to agree, even though the optimizing distributions differ. Comparing one number alone misses that distinction.

The code evaluates Gaussian log-density expectations in both directions and compares perturbed mean and variance candidates. Finite candidate checks do not replace the algebraic global-optimum proof. The Bernoulli controls do not manipulate this separate Gaussian example.

```python
from fractions import Fraction as Q
from math import log, isclose
from itertools import product

def forward(rho, mean, d):
    det = 1-rho*rho
    quad = (mean[0]**2-2*rho*mean[0]*mean[1]+mean[1]**2)/det
    return .5*((sum(d)/det)+quad-2+log(float(det/(d[0]*d[1]))))

def reverse(rho, mean, d):
    det = 1-rho*rho
    return .5*(sum((1+m*m)/v for m, v in zip(mean, d))-2
               +log(float(d[0]*d[1]/det)))

for rho in [Q(0), Q(1, 2), Q(9, 10), Q(-9, 10)]:
    variance = 1-rho*rho
    gap = -.5*log(float(variance))
    assert isclose(forward(rho, [0, 0], [variance]*2), gap, abs_tol=1e-12)
    assert isclose(reverse(rho, [0, 0], [Q(1)]*2), gap, abs_tol=1e-12)
    for m1, m2 in product([Q(-1, 2), Q(0), Q(1, 2)], repeat=2):
        for a, b in product([Q(1, 2), Q(1), Q(2)], repeat=2):
            assert forward(rho, [m1, m2], [a*variance, b*variance]) >= gap-1e-12
            assert reverse(rho, [m1, m2], [a, b]) >= gap-1e-12
    print("rho, forward-optimal variance, reverse-optimal variance, gap:",
          rho, variance, 1, gap)
```

## Questions for research use

1. Does completed ELBO optimization imply zero posterior error? **No, if the family excludes the exact posterior.**
2. Can this q variance be reported as the exact marginal variance? **No: here 1−ρ² differs from one.**
3. What belongs in an approximation comparison? **The target, coordinates, family, KL direction and remaining optimization error.**

For background, see [Blei et al., §2.3 and Figure 1](https://arxiv.org/html/1601.00670v9#S2.SS3). The unit-marginal optimum and reverse-direction comparison above are computed directly from log densities.

## Exercises with explanations

1. What is ELBO at r=π? **It equals log evidence.**
2. Does changing q increase fixed-model evidence? **No: it changes the bound and gap.**
3. Is KL a symmetric distance? **No: direction and support matter.**

Further reading: [Blei, Kucukelbir & McAuliffe, Variational Inference](https://arxiv.org/abs/1601.00670).
