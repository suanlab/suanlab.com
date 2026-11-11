## Residue is not pole order

In the Laurent expansion around an isolated singularity,

$$
f(z)=\sum_{n=-\infty}^{\infty}c_n(z-a)^n,\qquad
\operatorname{Res}(f,a)=c_{-1}.
$$

The simple pole 1/(z−a) has residue one. The double pole 1/(z−a)² has residue zero. Pole order is not the multiplier to insert into a contour integral.

## Winding and the residue theorem

For a contour avoiding a,

$$
\operatorname{Ind}(\gamma,a)=\frac1{2\pi i}\oint_\gamma\frac{dz}{z-a}.
$$

A simple CCW circle has index one inside and zero outside. Multiple turns and reversed orientation change the signed integer. Under the relevant holomorphicity and isolated-singularity assumptions,

$$
\oint_\gamma f(z)\,dz=2\pi i\sum_a\operatorname{Ind}(\gamma,a)\operatorname{Res}(f,a).
$$

The lab certifies inside/outside by |a|<R, not a visual proximity guess.

## Three locations

With a=0,R=1,m=1 the integral is 2πi. Moving the pole to two gives zero. At a=1 the pole lies on the contour and the ordinary integral is undefined.

A finite midpoint sum may avoid directly sampling that point, but cannot rescue the ordinary integral. The model rejects on-contour poles before quadrature. Principal values and indentations require separate prescriptions and are not silently substituted.

## Why higher pure poles integrate to zero

For m≥2, a primitive on the punctured domain is (z−a)¹⁻ᵐ/(1−m). Every closed contour avoiding a therefore integrates to zero, matching its zero residue. The primitive’s singularity at a does not contradict the result on valid paths.

## Numerical and physical interpretation

Near-contour poles create rapidly varying integrands and can make fixed-panel quadrature inaccurate. Compare analytic value, numerical value, error and pole-to-contour distance separately. A numerical discrepancy need not refute the theorem.

The open marker locates an excluded input pole, not a plotted function value. Orientation is fixed CCW. Connections from response poles to stability or resonance require additional transform and dynamical assumptions.

## Quantify the midpoint error near a pole

Consider the simple pole with R>0, a≥0 and ρ=a/R≠1. The normalized CCW integral I/(2πi) is one for ρ<1 and zero for ρ>1. At the lab’s midpoint nodes w_k=exp(2πi(k+1/2)/N), the normalized finite quadrature is

$$
Q_N(\rho)=\frac1N\sum_{k=0}^{N-1}\frac{w_k}{w_k-\rho}
=\frac1{1+\rho^N}.
$$

Here N is a positive integer. For ρ<1, expand w/(w−ρ)=Σⱼ≥₀ρʲw⁻ʲ. The average of w⁻ʲ vanishes unless j is a multiple of N, when it is $(-1)^{j/N}$. The surviving geometric series is 1−ρᴺ+ρ²ᴺ−⋯. For ρ>1, use −Σⱼ≥₁(w/ρ)ʲ instead; the surviving series gives the same rational expression. Both derivations use convergent series on their respective domains.

Thus the normalized absolute error is ρᴺ/(1+ρᴺ) inside, and 1/(1+ρᴺ) outside. This is the error after division by 2π, not the absolute integral error shown in the lab.

| N | Q_N(0.99), exact target 1 | Inside normalized error | Q_N(1.01), exact target 0 |
|---|---|---|---|
| 32 | 0.579717 | 0.420283 | 0.421063 |
| 128 | 0.783545 | 0.216455 | 0.218634 |
| 512 | 0.994210 | 0.005790 | 0.006093 |

A finite sum at ρ=1 has the continuous value 1/2, since none of these midpoint nodes is 1. The ordinary integral is nevertheless undefined there. The production lab correctly rejects that pole before quadrature; the finite sum cannot replace the domain check.

## Choose resolution together with the pole distance

For fixed ρ>0, ρ≠1, a target normalized error ε∈(0,1/2) is achieved when

$$
N\ge\frac{\log((1-\varepsilon)/\varepsilon)}{|\log\rho|}.
$$

At ρ=0 the midpoint rule is exact. Near ρ=1, |log ρ| is small and more nodes are needed. There is no uniform convergence as the pole approaches the contour with the resolution: choosing ρ_N=exp(−c/N), c>0, leaves error 1/(1+exp c) for every N.

For a diagonal matrix A=diag(0.99,1.01), entrywise residues give the unit-circle resolvent projector

$$
P=\frac1{2\pi i}\oint_{|z|=1}(zI-A)^{-1}\,dz
=\operatorname{diag}(1,0).
$$

Midpoint quadrature instead gives P_N=diag(Q_N(0.99),Q_N(1.01)). Its entries approach a spectral selection, but at finite N it need not satisfy P_N²=P_N. This diagonal example illustrates contour-based spectral filters; general nonnormal matrices require further conditioning analysis.

## Reproduce the quadrature and filter in Python

The code computes midpoint sums independently from the rational formula. Exact fractions provide the table and projector defect, and numerical sums check the contour parametrization. It also tests a pole approaching the contour and the error-budget formula. The raw boundary sum is labeled separately from a valid integral.

```python
from cmath import exp
from fractions import Fraction as Q
from math import pi, ceil, log

def midpoint(rho, n):
    total = 0j
    for k in range(n):
        w = exp(2j*pi*(k+0.5)/n)
        total += w/(w-rho)
    return total/n

for n in [32, 128, 512]:
    inside = 1/(1+Q(99, 100)**n)
    outside = 1/(1+Q(101, 100)**n)
    for rho, expected in [(Q(99, 100), inside), (Q(101, 100), outside)]:
        assert abs(midpoint(float(rho), n) - float(expected)) < 2e-11
    assert 0 < inside < 1 and 0 < outside < 1
    defect = max(abs(inside*inside-inside), abs(outside*outside-outside))
    assert defect > 0
    print(n, float(inside), float(1-inside), float(outside), float(defect))

for n in [32, 128, 512]:
    assert abs(midpoint(exp(-1/n).real, n) - 1/(1+exp(-1).real)) < 2e-11

# A finite raw sum at the pole is NOT an ordinary contour integral.
assert abs(midpoint(1.0, 512) - 0.5) < 2e-11

epsilon = Q(1, 10000)
for rho in [Q(99, 100), Q(101, 100)]:
    n = ceil(log(float((1-epsilon)/epsilon))/abs(log(float(rho))))
    estimate = 1/(1+rho**n)
    error = 1-estimate if rho < 1 else estimate
    assert error <= epsilon
    print('tolerance', rho, n, float(error))
```

Source connection: [Trefethen and Weideman, The Exponentially Convergent Trapezoidal Rule](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2014_149.pdf), §3 and §18, discusses periodic aliasing and resolvent quadrature. The shifted-node formula, table and diagonal filter here are independently derived; the paper’s other applications and experiments are outside this reproduction.

## Exercises and answers

1. The pure triple pole has **residue zero**.
2. An outside simple pole contributes **zero**.
3. An on-contour pole has **no ordinary reference integral**, so reporting zero error would be misleading.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
