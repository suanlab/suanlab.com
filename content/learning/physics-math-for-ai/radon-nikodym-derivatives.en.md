## Compare measures against one reference

Q is absolutely continuous with respect to P if every P-null event is also Q-null. The direction Q≪P matters. For probability measures satisfying this condition, a nonnegative measurable function w converts Q into weighted P integration. This Radon–Nikodym derivative is a relative mass density, rather than an ordinary coordinate derivative.

$$
Q\ll P\iff P(A)=0\Rightarrow Q(A)=0,\qquad Q(A)=\int_A w\,dP.
$$

## Ratios on a finite space

The defaults are P=(1/2,1/2,0,0) and Q=(1/4,3/4,0,0). Ratios at the first two points are one half and three halves. At the final points, the table leaves the ratio undefined instead of evaluating 0/0. The chosen global version sets those values to zero. Other nonnegative values on P-null points give the same Q. The chosen derivative integrates to one under P.

$$
w_i=\frac{Q_i}{P_i}\quad(P_i>0),\qquad\int w\,dP=Q(\Omega)=1.
$$

## Change the measure of an expectation

For φ(ω)=ω, the default Q expectation is 0.75, equal to the P expectation of wφ. The full-support example gives 1.5 by both calculations. The identity applies to nonnegative functions or signed functions satisfying the relevant integrability conditions. General sigma-finite versions exist, but this lab works with finite probability measures.

$$
E_Q[\varphi]=E_P\!\left[\frac{dQ}{dP}\varphi\right].
$$

## Weights cannot create missing support

In failure mode Q puts mass one quarter at point two while P assigns zero. No weighting of P can create mass there, so no global derivative exists. Ratios on positive-P points remain candidate entries only; the global derivative integral and changed expectation stay undefined. Importance sampling must check this support condition. Even when it holds, finite or small estimator variance needs separate assumptions.

$$
P(\{2\})=0<Q(\{2\})\implies Q\not\ll P.
$$

## Absolute continuity is an existence condition, not a variance bound

Let Q≪P, w=dQ/dP and h be Q-integrable. With N independent samples from P, the ordinary importance-sampling average has expectation I=E_Q[h]. Its finite-variance formula requires a second-moment condition in addition to absolute continuity:

$$
\widehat I_N=\frac1N\sum_{i=1}^Nw(\omega_i)h(\omega_i),\qquad
\operatorname{Var}(\widehat I_N)=\frac{E_P[w^2h^2]-I^2}{N}.
$$

The condition is E_P[w²h²]<∞. Integrability of h under Q ensures the first absolute moment of wh under P, but does not imply this second moment. The identity concerns the stated average of weighted values; a self-normalized ratio is a different estimator.

For a continuous example on (0,1), let P be uniform and Q have density q(x)=1/(2√x). This integrates to one and is absolutely continuous with respect to P. Even with bounded h=1, the weighted value has mean one and infinite second moment:

$$
E_P[w]=\int_0^1\frac{dx}{2\sqrt x}=1,\qquad
E_P[w^2]=\frac14\int_0^1\frac{dx}{x}=\infty.
$$

Every finite sample has finite weights almost surely, so observing finite sample variances cannot establish the missing population moment. The usual finite-variance CLT is unavailable for this average. For h=1 a self-normalized estimator would be identically one; the counterexample is specifically about the unnormalized weighted average above.

## Clipping changes the expectation as well as the tail

For K≥1/2, clip the weight to min(w,K). Its transition point is x=1/(4K²). Integrating the constant segment and then the original weight gives

$$
E_P[\min(w,K)]=1-\frac1{4K},\qquad
E_P[\min(w,K)^2]=\frac14+\frac14\log(4K^2).
$$

The clipped estimator now has finite variance, but its expectation for this normalization integral is below one. Tail control and unbiasedness are separate claims. This exact bias calculation does not state that all forms of clipping or self-normalization have the same bias.

## Derive an optimal proposal under explicit assumptions

Suppose q and p are target and proposal densities against a common reference measure. Assume 0<C=E_Q|h|<∞ and |h|>0 Q-almost surely, and require Q≪P. Cauchy–Schwarz gives

$$
C^2=\left(\int\frac{|qh|}{\sqrt p}\sqrt p\right)^2
\le\int\frac{q^2h^2}{p},\qquad
p_* = \frac{q|h|}{C}.
$$

The integrands are evaluated on p>0, with zero contribution where q=0. The proposed p* integrates to one, retains Q's support under the nonzero-h assumption and attains equality. Thus the smallest single-sample variance is C²−I². For a nonzero fixed-sign h this is zero, because the weighted value becomes constant. Knowing C or sampling exactly from p* may itself be difficult; this is a theoretical optimum, not an automatically available algorithm. If h vanishes on a positive-Q set, the stated support argument must be revised.

For Q=(1/4,3/4) and h=(1,3), I=C=5/2. The rows compare a uniform proposal, P=Q and the optimal proposal:

| $P(0)$ | $P(1)$ | $\operatorname{Var}(wh)$ |
|---|---|---|
| $1/2$ | $1/2$ | 4 |
| $1/4$ | $3/4$ | $3/4$ |
| $1/10$ | $9/10$ | 0 |

## Evaluate finite proposals and the continuous formulas

The code computes finite weighted moments with exact fractions. For the continuous example it evaluates the derived restricted and clipped integrals; it does not infer their infinite-limit behavior from numerical sampling. The existing four-point RN controls continue to check existence and changed expectations; these variance examples are separate exercises.

```python
from fractions import Fraction as Q
from math import log

q, h = (Q(1, 4), Q(3, 4)), (Q(1), Q(3))
I = sum(a*b for a, b in zip(q, h))
C = sum(a*abs(b) for a, b in zip(q, h))
optimal = tuple(a*abs(b)/C for a, b in zip(q, h))
assert optimal == (Q(1, 10), Q(9, 10)) and I == Q(5, 2)
for p, expected_variance in [((Q(1, 2), Q(1, 2)), Q(4)),
                             (q, Q(3, 4)), (optimal, Q(0))]:
    values = tuple(qi*hi/pi for qi, hi, pi in zip(q, h, p))
    mean = sum(pi*value for pi, value in zip(p, values))
    second = sum(pi*value**2 for pi, value in zip(p, values))
    assert mean == I and second >= C*C
    assert second-I*I == expected_variance
    print("proposal, weighted values, variance:", p, values, second-I*I)
for m in [1, 2, 4, 8]:
    # Uniform proposal, w(x)=1/(2 sqrt(x)), restricted to x >= 4**(-m).
    retained_mean = 1-Q(1, 2**m)
    restricted_second = m*log(2)/2
    print("lower cutoff:", Q(1, 4**m), "mean:", retained_mean,
          "second moment:", restricted_second)
for K in [1, 2, 4, 16]:
    clipped_mean = 1-Q(1, 4*K)
    clipped_second = .25+.25*log(4*K*K)
    assert clipped_mean < 1 and clipped_second-float(clipped_mean)**2 > 0
    print("clipping:", K, "mean:", clipped_mean,
          "finite variance:", clipped_second-float(clipped_mean)**2)
```

## Further questions and research use

1. Why can h=1 still produce a noisy weighted average? **The average estimates the integral through random weights; the target function is constant but those weights are not.**
2. Does finite weight at every sampled point imply finite population variance? **No: the integral near zero above diverges.**
3. What should an importance-sampling report distinguish? **Support, first and second moments, clipping bias, estimator definition and sampling budget.**

For measure-change existence, see [Durrett, Theorem A.4.8 and §1.6](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf). The variance, clipping and proposal comparisons above follow directly from the stated laws and Cauchy–Schwarz. The density q also appears in [the squared-variable transformation](../random-variable-transformations/).

## Exercises with explanations

1. What is the derivative value on a P-null point? **It depends on the chosen version.**
2. Do ratios on some points guarantee a global derivative? **No: absolute continuity is required.**
3. With densities p,q against the same reference, what is the ratio? **q/p where p>0, subject to the support condition.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
