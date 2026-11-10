## A sample table supports several different questions

Given function samples on [0,1], interpolation asks for an approximate value between nodes, while quadrature asks for an approximate integral. These tasks reuse data but do not produce interchangeable objects. The lab samples either exp(x) or x^4 at an even number n=2m of equally spaced panels, requiring n+1 values.

Blue shows the smooth function and orange connects adjacent samples with straight segments. The interpolant agrees at nodes but generally differs between them. The integral of this polyline is exactly the composite trapezoid rule. Simpson uses the same function samples with different weights and does not integrate the orange polyline.

## Derive the local interpolation estimate

Between neighboring nodes xi and xi+h, the linear interpolant is a convex combination of the endpoint values. A second-derivative remainder gives

$$
f(x)-I_hf(x)=\frac{f''(\xi)}2(x-x_i)(x-x_i-h),\qquad
\|f-I_hf\|_\infty\le M_2h^2/8.
$$

Here ξ in the remainder denotes an intermediate point, and M2 bounds the absolute second derivative over the whole interval. For the convex examples, the chord lies above the curve. A query at a grid node has zero interpolation error even though the global upper bound remains positive.

## Same samples, different quadrature weights

For h=1/n, write fj=f(jh). Composite trapezoid and Simpson are

$$
T=h\left(\frac{f_0+f_n}2+\sum_{j=1}^{n-1}f_j\right),\qquad
S=\frac h3\left(f_0+f_n+4\sum_{j\text{ odd}}f_j+2\sum_{\substack{j\text{ even}\\0<j<n}}f_j\right).
$$

On this unit interval, absolute error bounds are M2 h²/12 for trapezoid and M4 h^4/180 for Simpson. The latter assumes a bounded fourth derivative and an even panel count. Higher order is therefore a statement with regularity and grid assumptions, not an automatic property of every sample sequence.

## Work through three samples

For f=x^4 and n=2, the values are (0,1/16,1). Simpson gives 5/24, whereas the exact integral is 1/5. The error is 1/120, exactly the stated bound because the fourth derivative is constantly 24. At query x=1/4, the linear interpolant is 1/32, while the true value is 1/256.

For exp(x), M2=M4=e and the exact integral is e-1. With the same three samples, Simpson's integral error is about 0.000579, compared with about 0.03565 for trapezoid. This comparison fixes sample count rather than giving one rule additional evaluations.

## Exercise: accuracy, bounds and cost

Double the panel count for x^4 using Simpson. Its error decreases by a factor of sixteen; the number of samples grows from n+1 to 2n+1. Explain why the positive interpolation bound can coexist with zero error at a node. Finally describe why the fourth-order bound cannot simply be transferred to a function with a kink: the derivative assumption used in its derivation has failed.

Further reading: [SciPy composite Simpson documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html). The error constants and examples are derived here.
