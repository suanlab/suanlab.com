## Match local derivative information

For exp(x), every derivative is exp(x) and every derivative at zero is one. Its Taylor polynomial is

$$
T_n(x)=\sum_{k=0}^n\frac{x^k}{k!}.
$$

Degree zero matches value, one matches slope, and two adds curvature. At x=1, T₂=2.5 versus e≈2.71828. A degree label alone does not guarantee accuracy throughout the domain.

## Include the remainder in the claim

Under sufficient differentiability between zero and x, some intermediate ξ gives

$$
f(x)-T_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}x^{n+1}.
$$

For the exponential, the derivative magnitude along that interval is at most e^|x|:

$$
|R_n(x)|\le e^{|x|}\frac{|x|^{n+1}}{(n+1)!}.
$$

This bound does not require finding ξ. At n=2,x=1 it is e/6≈0.453, while actual error is about 0.218. A valid upper bound need not be tight.

## Finite Taylor theory versus infinite series

For fixed x the exponential bound tends to zero as degree grows, proving equality with its infinite Taylor series. A smooth function need not equal its Taylor series. The example f(0)=0 and f(x)=exp(−1/x²) for x≠0 has all derivatives zero at the origin but positive nearby values. A full derivative-induction proof is left as an advanced exercise.

Distinguish finite-order differentiability assumptions from analyticity. Existence of arbitrarily high derivatives alone does not guarantee vanishing remainder.

## Read and reproduce the comparison

The plot compares the functions on [−2,2]. The probe control selects one numeric evaluation, not the expansion center. At degree eight and x=1 the error is about 3.06×10⁻⁶. At zero, even degree zero is exact.

Compute each next term by multiplying the previous term by x/k. The lab uses this recurrence rather than recalculating factorials. Display rounding differs from mathematical approximation error; a tiny displayed error is not automatically exact zero.

## AI connections and exercises

Local loss models, Newton steps and curvature-based uncertainty use Taylor reasoning. Far from the expansion point, or where higher derivatives grow, a quadratic model can fail. Specify the region and remainder control.

1. **T₃=1+x+x²/2+x³/6.**
2. Does a large bound force large actual error? **No; the bound can be loose.**
3. The error at x=0 for degree zero is **zero**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
