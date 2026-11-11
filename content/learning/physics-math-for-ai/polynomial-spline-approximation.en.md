## A representation differs from a chosen approximation rule

On [0,1], x² is already a quadratic, so best degree-two polynomial error is zero. A particular approximation procedure need not choose those optimal coefficients. We compare two concrete rules: the Bernstein operator and uniform-knot linear interpolation.

Bernstein order n names the nominal basis order. Its result on this target simplifies to a quadratic even when n is larger; the operator still uses n+1 sample weights.

## Interpret Bernstein weights as probabilities

For K~Binomial(n,x), Bₙf(x)=E[f(K/n)]. Apply the second-moment identity:

$$
\begin{aligned}
B_nf(x)&=\sum_{k=0}^n f(k/n){n\choose k}x^k(1-x)^{n-k},\\
B_n(x^2)&=x^2+\frac{x(1-x)}n.
\end{aligned}
$$

Error is nonnegative and maximal at the midpoint, with value 1/(4n). Orders four and eight give 1/16 and 1/32. These are errors of this procedure, not best polynomial errors.

## Local linear splines have a different rate

Connect target values at knots xⱼ=j/n. Convexity puts each segment above the target:

$$
S_nf(x)-x^2=(x-x_j)(x_{j+1}-x),\quad x\in[x_j,x_{j+1}].
$$

The midpoint maximum is h²/4=1/(4n²). Four intervals give 1/64; eight give 1/256. Blue is the target and orange the chosen approximation. The dot marks the query. Zero error at knots does not eliminate between-knot error.

## General rates require regularity

Approximation existence and convergence rate are different claims. The linear-interpolation bound h²‖f''‖∞/8 needs a bounded second derivative. Corners and discontinuities need separate analysis. General Bernstein convergence also uses continuity and domain assumptions.

For AI comparisons, distinguish hypothesis space, coefficient-selection algorithm, data and evaluation norm. The same number of stored values can support different global or local constructions.

## Prove convergence for every continuous target on the interval

Let f be continuous on [0,1], M=‖f‖∞, and ω_f(δ)=sup{|f(u)−f(v)|: |u−v|≤δ}. Uniform continuity makes ω_f(δ) tend to zero. Split E|f(K/n)−f(x)| according to whether |K/n−x|≤δ. On the near event use ω_f(δ); on the far event use 2M and Chebyshev's inequality. Since Var(K/n)=x(1−x)/n≤1/(4n),

$$
\|B_nf-f\|_\infty\le\omega_f(\delta)+\frac{M}{2n\delta^2}.
$$

First choose δ to control the modulus, then n to control the second term. This proves uniform convergence and hence polynomial density on this interval. It supplies no common rate for every continuous function. For an L-Lipschitz target, Cauchy–Schwarz gives the sharper bound L/(2√n). In contrast, the exact x² error is 1/(4n); a valid general bound need not be sharp for a smooth target.

## Derive a local spline bound and compare like quantities

For f∈C²[a,b] and its linear interpolant I_hf, the interpolation remainder at an interior x is f″(ξ)(x−a)(x−b)/2 for some ξ∈(a,b). Subtract a multiple of (t−a)(t−b) so that the difference has three zeros; applying Rolle twice proves this formula. Consequently

$$
\|f-I_hf\|_{L^\infty[a,b]}\le\frac{(b-a)^2}{8}\|f''\|_{L^\infty[a,b]}.
$$

The table compares the exact errors for x² with the Lipschitz bound 1/√n, using L=2. Its last column is a theorem's upper bound, not a third implemented approximation rule. Fraction arithmetic independently sums Bernstein weights and integrates no sampled error curve.

| n | Bernstein exact error | Linear spline exact error | Lipschitz upper bound |
|---|---|---|---|
| 2 | .125 | .0625 | .707107 |
| 4 | .0625 | .015625 | .5 |
| 8 | .03125 | .00390625 | .353553 |
| 16 | .015625 | .0009765625 | .25 |

**Exercise:** Does continuity alone justify the spline's O(h²) rate? **Explanation:** No. Uniform convergence follows from the modulus of continuity, but this second-order bound requires the stated second derivative control.

The live query error uses the positive error identities above rather than subtracting two rounded heights. At n=16,x=1−2⁻⁵³ both rules have positive error about 6.94×10⁻¹⁸, which height subtraction can erase. A positive error below numeric range is shown as — with a note, and that query marker is withheld. Exact zeros at endpoints and the relevant spline knots remain zero. The code checks this difference using exact fractions of the binary input.

```python
from fractions import Fraction as F
from math import comb, sqrt

for n in [2,4,8,16]:
    # Direct Bernstein sum at the midpoint, not the closed error formula.
    value=sum((F(k,n)**2*comb(n,k)*F(1,2)**n for k in range(n+1)),F(0))
    bernstein_error=value-F(1,4)
    spline_midpoint=(F(0)**2+F(1,n)**2)/2-F(1,2*n)**2
    assert bernstein_error==F(1,4*n)
    assert spline_midpoint==F(1,4*n*n)
    # f(x)=x^2 has Lipschitz constant 2 on [0,1].
    lipschitz_bound=1/sqrt(n)
    print(n,float(bernstein_error),float(spline_midpoint),round(lipschitz_bound,6))

x=F.from_float(1-2**-53)
errors=[x*(1-x)/16,(x-F(15,16))*(1-x)]
assert all(error>0 for error in errors)
print('Exact query errors near one:',*[float(error) for error in errors])
```

## Exercises with explanations

1. What happens at n=1? **Both rules give x**, the line through the endpoints.
2. Does nonzero Bernstein error mean x² is outside the degree-four space? **No.** The procedure does not select the best element.
3. What does evaluation only at spline knots miss? **Between-knot error**, maximal at each interval midpoint here.

Further reading: [MIT 18.102, Functional Approximation Background](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
