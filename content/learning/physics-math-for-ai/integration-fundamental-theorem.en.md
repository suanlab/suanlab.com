## Accumulate small contributions

For dimensionless f(t)=t² on [0,x], N equal subintervals have width h=x/N. The left Riemann sum is

$$
L_N=h\sum_{k=0}^{N-1}(kh)^2.
$$

At x=1,N=4, this is (0+1+4+9)/64=7/32=0.21875. A finite sum differs from the integral defined by a refinement limit. Since t² increases on this interval, the left sum underestimates the integral.

## What integrability means

Upper and lower Riemann sums use the supremum and infimum on each small interval. Integrability means their gap can be made arbitrarily small by suitable partitions. Continuous functions on compact intervals satisfy this through uniform continuity.

$$
F(x)=\int_0^x t^2\,dt=\frac{x^3}{3}.
$$

At x=1 the exact value is one third. Computing decimal approximations and proving existence of an integral are different tasks. Some discontinuous functions are also integrable; continuity is sufficient, not necessary.

## The local fundamental-theorem argument

At a continuity point of f,

$$
\frac{F(x+h)-F(x)}h=\frac1h\int_x^{x+h}f(t)\,dt\longrightarrow f(x).
$$

Continuity controls f(t) near f(x) throughout the short interval. Therefore F′=f. Conversely, an antiderivative G gives ∫ₐᵇf=G(b)−G(a) under the theorem’s assumptions. “Integration and differentiation undo each other” requires those assumptions.

## Bound the finite-sum error

On [0,x], f′(t)=2t≤2x. Each left-rectangle error is at most ∫₀ʰ2xs ds=xh². Summing over N intervals gives

$$
0\le F(x)-L_N\le\frac{x^3}{N}.
$$

A bound may exceed actual error. The lab reports the directly computed sum, exact integral, actual error and bound separately. Refine N to compare them. A zero-width interval has zero accumulation even though the function need not be identically zero.

## Interpret the curves and AI connection

Blue is the integrand; orange is its accumulation from zero. With physical quantities their units generally differ. This example is dimensionless; sharing axes does not make the quantities identical.

Expected losses, continuous-time dynamics and cumulative hazards use integrals. Specify the measure, distribution and numerical partition or sampling procedure rather than treating all integral estimates as the same operation.

## Exercises and answers

1. At x=2 the exact integral is **8/3**.
2. At x=1,N=4 the bound is **one quarter**, larger than actual error.
3. Do left sums always underestimate? **No; monotonicity explains this example.**

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
