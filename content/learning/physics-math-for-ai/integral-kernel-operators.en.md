## A continuous analogue of matrix multiplication

An integral operator weights input f(t) by a kernel and integrates to produce a new function. The column index becomes continuous, but the measure and function space must be specified.

$$
Tf(x)=\int_0^1K(x,t)f(t)\,dt.
$$

A square-integrable two-variable kernel defines a bounded compact Hilbert–Schmidt operator on L2. This is sufficient, not a characterization of every bounded integral map.

## An exactly solvable rank-one kernel

Let g(x)=1+a x and K(x,t)=g(x)g(t). Then Tf=g⟨g,f⟩, so every output lies in span(g). Since g is nonzero, rank is one. The operator is self-adjoint and positive because ⟨Tf,f⟩=|⟨g,f⟩|².

$$
Tg=\|g\|_2^2g,\qquad \lambda=\|g\|_2^2=1+a+a^2/3.
$$

Functions orthogonal to g form its kernel. The sole nonzero eigenvalue equals both operator and Hilbert–Schmidt norms; the squared kernel integral factors as (∫g²)².

## Separate quadrature from the operator

For f=1+b t, q=∫gf=1+(a+b)/2+ab/3. Using midpoints tᵢ=(i+.5)/N gives coefficient bias −ab/(12N²).

$$
T_Nf(x)=g(x)\frac1N\sum_i g(t_i)f(t_i).
$$

At a=b=1,N=2, exact q is 7/3 and midpoint qₙ is 37/16, an error 1/48. At N=8 it becomes 1/768. Blue and orange compare exact and approximate outputs. If a or b vanishes, the integrand is at most linear and midpoint integration is exact.

## Keep quadrature weights in the matrix

The Nyström matrix has entries K(tᵢ,tⱼ)/N. Omitting 1/N mis-scales its spectrum. Its only possible nonzero eigenvalue is ∑g(tᵢ)²/N, differing from λ by −a²/(12N²). If every sampled g vanishes, the quadrature matrix is zero. At a=−2,N=1, g(.5)=0, so the original rank-one operator has a rank-zero discretization. Refining this matrix changes the approximation, not the underlying operator.

This connects to kernels, Gaussian processes and neural operators. General attention weights need not share this kernel's symmetry or positivity.

## Why Hilbert–Schmidt kernels give bounded compact operators

On real L²[0,1], Cauchy–Schwarz first in t and then integration in x gives ‖Tf‖₂≤‖K‖L²([0,1]²)‖f‖₂. Approximate a two-variable L² kernel by finite sums of rectangular step functions. Each separated term a(x)b(t) gives finite rank. Kernel L² error controls operator-norm error, so T is a norm limit of finite-rank operators and is compact. This is not a converse characterizing every bounded operator.

## Separate point quadrature from an operator on all L²

The screen's f=1+b t is smooth with defined point values, making its midpoint calculation valid on that family. But f(t_i) is undefined for a general L² equivalence class and remains unbounded in L² norm even on continuous functions. Treating the quadrature formula as automatically bounded on all L² would contradict the dual-space counterexample.

Instead let P_N be the orthogonal cell-average projection and set g_N=P_Ng, K_N(x,t)=g_N(x)g_N(t). Then T_N^cell f=g_N⟨g_N,f⟩ is bounded on all L². For linear g, cell averages equal midpoint values. In the orthonormal cell basis its matrix is g(t_i)g(t_j)/N, matching the Nyström matrix. This identifies cell-average input coordinates and step-function output reconstruction; it does not identify that output with the screen's continuous g(x)q_N at every x.

$$
\|T-T_N^{\mathrm{cell}}\|
\le(\|g\|_2+\|g_N\|_2)\|g-g_N\|_2,
\qquad \|g-g_N\|_2^2=\frac{a^2}{12N^2}.
$$

The vectors g_N and e=g−g_N are orthogonal. Writing p=‖g_N‖,d=‖e‖, the difference has matrix [[0,pd],[pd,d²]] in those directions, with norm (d²+sqrt(d⁴+4p²d²))/2. The same value extends to the rank-one case p=0.

| N, a=−2,b=2 | Cell operator norm error | Upper bound | Smooth-input coefficient error |
|---|---|---|---|
| 1 | .333333 | .333333 | .333333 |
| 2 | .191898 | .311004 | .083333 |
| 4 | .091773 | .164020 | .020833 |
| 8 | .044026 | .083007 | .005208 |

Fraction cell moments and the 2×2 characteristic equation check the table. Operator error here is O(1/N), whereas quadrature coefficient error for the selected smooth input is O(1/N²). The latter does not establish the same rate for the full operator.

For small a,b, subtracting q_N−q can also expose rounding noise. At a=b=10⁻⁸,N=16 the actual quadrature error is about 3.26×10⁻²⁰, but subtraction previously gave about 2.22×10⁻¹⁶. The error readout now evaluates the absolute analytic bias −ab/(12N²). Rounded coefficients can coincide while this bias remains positive. Unrepresentable positive values show a dash and explanation; actual zeros at a=0 or b=0 remain zero.

**Exercise:** Does obtaining the same midpoint matrix identify the original function-space operator? **Explanation:** No. Specify what input coefficients mean and how outputs reconstruct. Point samples and cell averages are different observations on general L².

```python
from fractions import Fraction as F
from math import sqrt

# g=1-2x; g_N is its cell average, equal to its midpoint value.
mu=F(1,3)
for N in [1,2,4,8]:
    midpoints=[F(2*i+1,2*N) for i in range(N)]
    values=[1-2*t for t in midpoints]
    projected_norm_squared=sum(v*v/N for v in values)
    error_squared=F(1,3*N*N)
    assert projected_norm_squared+error_squared==mu
    # In the orthogonal directions g_N and g-g_N, T-T_N has this 2x2 form.
    cross=sqrt(float(projected_norm_squared*error_squared))
    d=float(error_squared)
    exact_operator_error=(d+sqrt(d*d+4*cross*cross))/2
    upper=(sqrt(float(mu))+sqrt(float(projected_norm_squared)))*sqrt(d)
    assert exact_operator_error<=upper+1e-12
    assert abs(cross*cross+d*exact_operator_error-exact_operator_error**2)<1e-12
    # For the separate smooth input f=1+2t, midpoint coefficient bias is +1/(3N^2).
    exact_coefficient=F(-1,3)
    midpoint=sum((1-2*t)*(1+2*t)/N for t in midpoints)
    assert midpoint-exact_coefficient==error_squared
    print('N, bounded cell-operator norm error, its upper bound, smooth-input quadrature bias:',
          N,round(exact_operator_error,6),round(upper,6),round(float(error_squared),6))
```


## Exercises with explanations

1. Operator norm at a=0? **One**, the constant-kernel integration map.
2. Nonzero eigenvalue at a=−1? **1/3**, the integral of (1−x)².
3. Does rank one make the domain finite dimensional? **No.** L2 remains infinite dimensional; only the range is one dimensional.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
