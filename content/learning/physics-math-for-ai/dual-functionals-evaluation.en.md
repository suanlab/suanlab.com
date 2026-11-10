## Continuous scalar measurements of functions

A linear functional sends vectors or functions to scalars. The continuous dual X* contains bounded linear functionals, with norm given by the largest output magnitude on the unit ball. Arbitrary algebraic linear maps need not belong to this dual.

$$
\|\ell\|_{X^*}=\sup_{\|f\|_X\le1}|\ell(f)|.
$$

## Riesz representation in a Hilbert space

Every continuous linear functional on a real Hilbert space has a unique inner-product representative g. In L2[0,1], it is integral gf. Cauchy–Schwarz supplies the upper bound; for nonzero g, input g/‖g‖ attains it.

$$
\ell_g(f)=\int_0^1gf,\qquad \|\ell_g\|=\|g\|_2.
$$

For g=1+a t and f=1+b t, the pairing is 1+(a+b)/2+ab/3. At a=b=1, pairing and Cauchy–Schwarz bound equal 7/3, while the dual norm is √(7/3). A functional's value differs from its norm.

## Two problems with point evaluation

An L2 equivalence class ignores null-set changes, so f(0) is not well defined. Even restricting to continuous representatives does not make point evaluation bounded in L2 norm.

$$
v_m(t)=\max(1-mt,0),\quad v_m(0)=1,\quad \|v_m\|_2=\frac1{\sqrt{3m}}.
$$

The orange triangle narrows without losing its height. At m=4 and sixteen, norms are about .289 and .144. The point-value-to-norm ratio √(3m) grows without bound. No common C can guarantee |f(0)|≤C‖f‖₂.

## Connect to kernels and sensors

Point evaluation can become continuous under a different norm or stronger regularity. In an RKHS it has a representative kₓ with f(x)=⟨f,kₓ⟩. General L2 is not automatically an RKHS. Sobolev point evaluation also depends on dimension and differentiation order.

An AI observation model must specify the function space and measurement stability. A finite-width weighted integral is continuous when its weight belongs to L2, as the Riesz example illustrates.

## Exercises with explanations

1. Norm for g=1? **One**, since the interval has unit length.
2. Is the pairing of 1+t with itself its norm? **It is the squared norm:** distinguish 7/3 from √(7/3).
3. Does the m=16 picture prove unboundedness? **No.** The exact ratio as m tends to infinity supplies the proof.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
