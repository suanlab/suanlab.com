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

## Exercises with explanations

1. Operator norm at a=0? **One**, the constant-kernel integration map.
2. Nonzero eigenvalue at a=−1? **1/3**, the integral of (1−x)².
3. Does rank one make the domain finite dimensional? **No.** L2 remains infinite dimensional; only the range is one dimensional.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
