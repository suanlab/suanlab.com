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

## Exercises and answers

1. The pure triple pole has **residue zero**.
2. An outside simple pole contributes **zero**.
3. An on-contour pole has **no ordinary reference integral**, so reporting zero error would be misleading.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
