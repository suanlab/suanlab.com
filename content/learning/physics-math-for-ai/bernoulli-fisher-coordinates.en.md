## Fisher information averages over the model

For an interior Bernoulli distribution, an outcome y∈{0,1} has probability pʸ(1−p)¹⁻ʸ. The score is the derivative of log probability with respect to the chosen parameter. In probability coordinates,

$$
s_p(y)=\frac{y-p}{p(1-p)},\qquad
\mathbb E_p[s_p]=0,\qquad
I_p=\mathbb E_p[s_p^2]=\frac{1}{p(1-p)}.
$$

The expectation uses the model distribution at the current parameter. A squared score from one observed label is not generally this expectation. At p=0.2, label one gives score five and square twenty-five; label zero gives score −1.25 and square 1.5625. The model-weighted average is 0.2·25+0.8·1.5625=6.25.

## Information changes coordinates with the tangent

Let θ=log[p/(1−p)] be the logit. Since dp/dθ=p(1−p), the logit score is y−p and its Fisher information is p(1−p). Different numerical information values do not mean different statistical geometry:

$$
I_\theta=\left(\frac{dp}{d\theta}\right)^2 I_p,
\qquad I_p\,dp^2=I_\theta\,d\theta^2.
$$

The metric coefficient must be read together with the tangent coordinate. A large probability-coordinate coefficient near a boundary reflects coordinate stretching; interpreting it as universally greater learning difficulty ignores the parameterization.

## Local KL supplies the quadratic metric

For a small probability displacement Δ, the KL expansion is

$$
D_{KL}(\operatorname{Ber}(p)\|\operatorname{Ber}(p+\Delta))
=\frac12 I_p\Delta^2+O(\Delta^3).
$$

The lab keeps query probabilities interior. Blue is exact n-sample KL for independent observations; orange is the Fisher quadratic. The diagram zooms to q within 0.01 of p and scales the vertical axis to the local KL values. Their agreement is local, not an identity at every displacement. A stable log remainder removes the analytically canceling linear terms before numerical evaluation.

At p=0.5,Δ=0.01,n=1, exact KL is approximately 0.000200040 and the quadratic is 0.0002. At p=0.2 the asymmetry of the distribution makes the cubic correction more visible.

## Independence and regularity matter

For n independent observations with the same parameter, scores add and zero-mean cross terms vanish, giving total information nI. Correlated observations require their joint likelihood instead. Support changes, boundary parameters and singular models can invalidate familiar score or Hessian identities unless their regularity conditions are checked.

The lab's label control changes the observed score and outer product. It does not change true Fisher or the KL curve. The count control scales independent information and KL, while still showing one illustrative observed label in the table.

For a requested displacement smaller than the spacing of floating-point probabilities, the lab evaluates KL directly from Δ using logarithmic remainders. For example Δ=10⁻²⁰ at p=1/2 retains KL≈2×10⁻⁴⁰ even though the separately rounded query coordinate p+Δ equals p. These are different numerical evaluations of the same real-valued query, not KL between two identical stored probabilities.

## From model information to an estimation lower bound

For independent Bernoulli observations with interior p, let T be an unbiased estimator of p based on n observations and S the total score. The finite sample space permits differentiation of its expectation term by term. Differentiating EₚT=p gives Eₚ[(T−p)S]=1, while EₚS=0 and EₚS²=n/[p(1−p)]. Cauchy–Schwarz then gives

$$
1\le\operatorname{Var}_p(T)\,\frac{n}{p(1-p)},\qquad
\operatorname{Var}_p(T)\ge\frac{p(1-p)}n.
$$

The sample mean attains equality: its centered value is p(1−p)S/n. Information therefore connects local likelihood sensitivity to an achievable variance in this regular model. The conclusion concerns unbiased estimators at every interior parameter, not the error of an arbitrary fitted neural model. Biased estimators require a bias-derivative version; boundary parameters need separate analysis.

At p=1/5, enumeration over the binomial count gives:

| n | Information | Variance | Lower bound |
|---|---:|---:|---:|
| 1 | 25/4 | 4/25 | 4/25 |
| 5 | 125/4 | 4/125 | 4/125 |
| 10 | 125/2 | 2/125 | 2/125 |

```python
from fractions import Fraction as F
from math import comb
p = F(1, 5)
for n in (1, 5, 10):
    rows = [(F(k, n), (F(k)-n*p)/(p*(1-p)),
             comb(n, k)*p**k*(1-p)**(n-k)) for k in range(n+1)]
    assert sum(w for _, _, w in rows) == 1
    assert sum(t*w for t, _, w in rows) == p
    variance = sum((t-p)**2*w for t, _, w in rows)
    information = sum(s*s*w for _, s, w in rows)
    assert sum((t-p)*s*w for t, s, w in rows) == 1
    assert variance*information == 1
    print(n, information, variance, 1/information)
```

The code sums the exact count distribution, rather than simulating a small dataset or identifying one observed score with Fisher information. A single label's squared score can still differ from the model expectation while the inequality holds.

## Exercise and explained answer

Set p=0.2 and n=10. Total probability-coordinate Fisher is 62.5, while per-sample logit Fisher is 0.16. Switch the label from one to zero and verify only observed-score quantities change. Explain why an empirical gradient outer-product matrix computed using dataset labels can differ from a model-expectation Fisher matrix, even when both are positive semidefinite.

Further reading: [Martens, New Insights and Perspectives on the Natural Gradient Method](https://www.jmlr.org/papers/v21/17-678.html).
