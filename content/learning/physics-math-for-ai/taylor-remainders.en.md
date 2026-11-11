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

For fixed x the exponential bound tends to zero as degree grows, proving equality with its infinite Taylor series. A smooth function need not equal its Taylor series. The example f(0)=0 and f(x)=exp(−1/x²) for x≠0 has all derivatives zero at the origin but positive nearby values. The derivative induction is proved below.

Distinguish finite-order differentiability assumptions from analyticity. Existence of arbitrarily high derivatives alone does not guarantee vanishing remainder.

## Prove that every derivative vanishes at the origin

Define $g(0)=0$ and $g(x)=e^{-1/x^2}$ for $x\ne0$. First show that every polynomial times $e^{-t^2}$ vanishes as $|t|\to\infty$. For an integer $K>d/2$, the positive-term expansion of the exponential gives $e^{t^2}\ge |t|^{2K}/K!$. Hence

$$
|t|^d e^{-t^2}\le K!\,|t|^{d-2K}\longrightarrow0.
$$

A polynomial is bounded by a constant times a suitable power for $|t|\ge1$, so the same conclusion holds for it. This is a bound for every sufficiently large input, with no floating-point underflow involved.

For $x\ne0$, put $t=1/x$ and define polynomials by

$$
P_0(t)=1,\qquad P_{m+1}(t)=-t^2P_m'(t)+2t^3P_m(t).
$$

The chain rule gives the next derivative of $P_m(1/x)e^{-1/x^2}$. To check the origin too, extend this expression by zero there. It is continuous by the preceding decay bound, and its difference quotient at zero is

$$
\frac{P_m(1/h)e^{-1/h^2}}h=tP_m(t)e^{-t^2}\longrightarrow0.
$$

Induction now proves that g is infinitely differentiable and $g^{(m)}(0)=0$ for every m. Its Taylor series at zero is the zero series, while $g(x)>0$ for every nonzero x. All derivatives at one point can therefore coexist with failure of equality to the local Taylor series.

## Check the derivative polynomials separately

The recurrence gives the following first four polynomials. Read this table as the flat-function example; the interactive plot illustrates the exponential and its finite Taylor polynomial.

| Derivative order m | Polynomial P_m(t) |
|---|---|
| 0 | 1 |
| 1 | 2t³ |
| 2 | 4t⁶−6t⁴ |
| 3 | 8t⁹−36t⁷+24t⁵ |

The integer-coefficient calculation below reproduces the table. It checks the finite polynomial arithmetic. The induction and decay bound establish the statement for all derivative orders and arbitrarily small nonzero x.

```python
from collections import defaultdict

polynomial = {0: 1}  # exponent -> integer coefficient
expected = [
    {0: 1},
    {3: 2},
    {4: -6, 6: 4},
    {5: 24, 7: -36, 9: 8},
]
for order in range(4):
    assert polynomial == expected[order]
    print(order, sorted(polynomial.items()))
    following = defaultdict(int)
    for exponent, coefficient in polynomial.items():
        following[exponent + 1] -= exponent * coefficient
        following[exponent + 3] += 2 * coefficient
    polynomial = {power: value for power, value in following.items() if value}
```

## Read and reproduce the comparison

The plot compares the functions on [−2,2]. The probe control selects one numeric evaluation, not the expansion center. At degree eight and x=1 the error is about 3.06×10⁻⁶. At zero, even degree zero is exact.

Compute each next term by multiplying the previous term by x/k. The lab uses this recurrence rather than recalculating factorials. Display rounding differs from mathematical approximation error; a tiny displayed error is not automatically exact zero.

## AI connections and exercises

Local loss models, Newton steps and curvature-based uncertainty use Taylor reasoning. Far from the expansion point, or where higher derivatives grow, a quadratic model can fail. Specify the region and remainder control.

1. **T₃=1+x+x²/2+x³/6.**
2. Does a large bound force large actual error? **No; the bound can be loose.**
3. The error at x=0 for degree zero is **zero**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
