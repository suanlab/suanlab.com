## A finite grid behind a familiar decimal

A computer stores a finite set of binary fractions, rather than every real number. In a normalized binade [2^e,2^(e+1)), precision p includes the leading one, so adjacent values are separated by 2^(e-p+1). Increasing the exponent increases absolute spacing even when the number of significant bits stays fixed. This is why a useful absolute tolerance at one scale can fail at another.

The lab deliberately uses a toy format with unbounded exponents and p significant bits. It implements round to nearest, with an exact midpoint going to an even significand integer. It does not simulate IEEE exponent limits, subnormals or exceptions. The separate exponential unit studies binary64 range failures.

$$
\operatorname{fl}_p(x)=\underset{y\in\mathcal F_p}{\operatorname{nearest}}x,
\qquad u=2^{-p},\qquad
|\operatorname{fl}_p(x)-x|\le u|x|.
$$

The relative bound assumes a normalized, nonzero result without exponent limits. It describes one rounding operation, not the accumulated error of an arbitrary algorithm.

## Input representation and accumulation are different errors

First round the input to s0=fl(x), then add the exactly representable binary increment d=2^-k repeatedly. The reference at step n is s0+nd. In the available control range this dyadic reference is exactly representable in binary64. Consequently, the plotted error measures the toy arithmetic alone; |s0-x| is a separate readout.

At p=4 the spacing above one is 1/8. Adding 1/32 to one rounds back to one. Sixteen sequential additions therefore still produce one, although the reference is 1.5. This absorption is not a failure of the real-number identity n times d equals nd: each intermediate operation has changed the calculation.

## Compensation carries information forward

The compensation variable estimates the low part lost during addition:

$$
y=\operatorname{fl}(d-c),\quad t=\operatorname{fl}(s+y),\quad
c=\operatorname{fl}(\operatorname{fl}(t-s)-y),\quad s=t.
$$

Every displayed operation is rounded to the selected precision. Compensation recovers 1.5 in the example, but it is not a universal exact-sum algorithm. It cannot reconstruct information absent from the original input. Blue and orange show absolute errors, so an orange curve at zero indicates agreement with the stored-input reference, rather than exact agreement with an arbitrary decimal input.

## From one rounding to a sum's condition number

Assume normalized rounding fl(a+b)=(a+b)(1+δ), |δ|≤u, with no range failure. Expanding sequential addition assigns a product of such factors to each stored input. If ku<1, the product differs from one by at most γₖ=ku/(1−ku). One way to bound the upper product is (1+u)ᵏ≤1/(1−ku); induction also bounds the lower product. Thus for n stored terms,

$$
|\widehat S-S|\le\gamma_{n-1}\sum_i|x_i|,\qquad
\frac{|\widehat S-S|}{|S|}\le\gamma_{n-1}
\frac{\sum_i|x_i|}{|\sum_i x_i|}\quad(S\ne0).
$$

The last ratio is the sum's condition number under componentwise relative input perturbations. Cancellation can make it large even when each arithmetic operation satisfies its relative bound. This theorem concerns stored inputs; it does not remove decimal representation error or promise an exact compensated sum.

For p=4, summing (1,d,−1) sequentially illustrates absorption:

| d | Sum condition | Absolute error |
|---|---:|---:|
| 1/8 | 17 | 0 |
| 1/32 | 65 | 1/32 |
| 1/128 | 257 | 1/128 |

```python
from fractions import Fraction as F
def rounding(x, p):
    if not x: return x
    sign, a = (1 if x > 0 else -1), abs(x)
    e = a.numerator.bit_length()-a.denominator.bit_length()
    if a < F(2)**e: e -= 1
    step = F(2)**(e-p+1)
    scaled = a/step
    k = scaled.numerator//scaled.denominator
    remainder = scaled-k
    if remainder > F(1, 2) or (remainder == F(1, 2) and k % 2): k += 1
    return sign*k*step
for d in (F(1, 8), F(1, 32), F(1, 128)):
    terms = (F(1), d, F(-1))
    value = terms[0]
    for x in terms[1:]: value = rounding(value+x, 4)
    condition = sum(map(abs, terms))/abs(sum(terms))
    bound = F(1, 7)*sum(map(abs, terms))  # gamma_2, u=1/16.
    error = abs(value-d)
    assert error <= bound
    print(d, condition, error)
```

The independent rational rounding routine avoids using a floating logarithm to decide a binade. The site's helper also corrects a rounded logarithm at binary64 binade boundaries; supporting 53 bits in that helper does not extend the lab's 3–24-bit controls.

## Exercise: a tie and a boundary

At p=4, round 1.0625 and 1.1875 by hand. The surrounding representable values have spacing 0.125. The first midpoint chooses 1 because its significand integer is eight, an even number. The second chooses 1.25 because ten is even. Also round 1.9375: the midpoint chooses two, crossing into the next binade.

Now set the addition count to zero and the input to 0.1. Both accumulation errors are zero while input representation error remains positive. Explain this outcome before increasing the precision. In AI experiments this distinction matters when interpreting reproducibility, reductions, tolerances and apparently inconsistent gradients.

Further reading: [Python's floating-point explanation](https://docs.python.org/3/tutorial/floatingpoint.html). The toy rounding and worked examples here are independently constructed.
