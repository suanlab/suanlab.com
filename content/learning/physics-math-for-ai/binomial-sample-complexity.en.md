## Sufficient data versus exact failure probability

Estimate p using an IID Bernoulli mean. Required data depend on error tolerance epsilon, failure probability delta and the desired guarantee. This experiment compares exact binomial event sums with Hoeffding's distribution-free upper bound. Blue is the finite exact tail and orange the bound clipped at probability one.

The event |K/N-p|≥epsilon includes its boundary. Integer success counts make that boundary important at small N. Log-domain probabilities avoid underflow at extreme p, while floating ties are handled at scaled machine precision.

## Compute the finite event

$$
P\left(\left|\frac KN-p\right|\ge\epsilon\right)
=\sum_{k\le\lfloor N(p-\epsilon)\rfloor\ \mathrm{or}\ k\ge\lceil N(p+\epsilon)\rceil}
{N\choose k}p^k(1-p)^{N-k}.
$$

N is at most 128 here. Endpoint probabilities zero and one are handled as point masses. No prescribed smooth curve replaces the computed sum. Exact within a model still has floating-point rounding limits, so independent factorial formulas and normalization are checked.

## Derive a sufficient Hoeffding count

For independent variables in [0,1],

$$
P(|\overline X-p|\ge\epsilon)\le2e^{-2N\epsilon^2}.
$$

Requiring this bound to be at most delta gives

$$
N\ge\left\lceil\frac{\log(2/\delta)}{2\epsilon^2}\right\rceil.
$$

This is sufficient for every p, not the exact necessary minimum for a particular distribution. The sufficient count can exceed the plotted range. Changing delta changes the confidence target, but not the actual tail at fixed N,p,epsilon.

## A lower bound asks what every estimator must face

An upper bound for the sample mean does not establish that its sample count is necessary. Compare two possible IID worlds p_minus=1/2−2epsilon and p_plus=1/2+2epsilon, with 0<epsilon≤1/8. If one estimator has deviation probability at most delta<1/2 in both worlds, thresholding its estimate at one half constructs a test with both errors at most delta.

KL adds over independent observations. Data processing to the test output and monotonicity of binary KL imply

$$
N\,\mathrm{kl}(p_-\Vert p_+)\ge\mathrm{kl}(1-\delta\Vert\delta)=(1-2\delta)\log\frac{1-\delta}{\delta}.
$$

Indeed the event “choose minus” has probability at least 1−delta under the minus world and at most delta under the plus world. Coarsening cannot increase information. Here the denominator kl(p_minus||p_plus)=4epsilon*log((1/2+2epsilon)/(1/2−2epsilon)) is of order epsilon². Hence confidence over both worlds demands order log(1/delta)/epsilon² observations as well, with different constants from Hoeffding's sufficient count.

| epsilon, delta=1/20 | Necessary two-world lower count | Hoeffding sufficient count |
|---|---:|---:|
| 1/16 | 21 | 473 |
| 1/32 | 85 | 1889 |
| 1/64 | 339 | 7555 |

```python
from math import log,ceil,comb
from fractions import Fraction as F
def kl(p,q):return float(p)*log(float(p/q))+float(1-p)*log(float((1-p)/(1-q)))
delta=F(1,20)
for eps in (F(1,16),F(1,32),F(1,64)):
    minus=F(1,2)-2*eps;plus=F(1,2)+2*eps
    lower=ceil(kl(1-delta,delta)/kl(minus,plus))
    sufficient=ceil(log(float(2/delta))/(2*float(eps)**2))
    print(eps,lower,sufficient)
n=6;p=F(1,4);q=F(3,4)
def choose_minus(r):return sum((comb(n,k)*r**k*(1-r)**(n-k) for k in range(n//2+1)),F(0))
a,b=choose_minus(p),choose_minus(q)
assert kl(a,b)<=n*kl(p,q)+1e-14
print('test KL and full-data KL:',kl(a,b),n*kl(p,q))
```

The necessary and sufficient counts have different quantifiers: the lower argument applies to every estimator required to work in both candidate worlds, whereas the upper constructs a guarantee for the sample mean over all Bernoulli p. The auxiliary code also checks KL contraction for a finite majority test. The live plot still shows N≤128 and its declared scaled-precision boundary policy. Integer-tail oscillations, an unknown p, estimator risk and information-theoretic impossibility should remain separate concepts.

## Worked example and exercise

For p=.5, epsilon=.2 and N=3, counts one and two have error 1/6 and are excluded. Only zero and three contribute, giving probability 2/8=.25. At N=4, counts one and three also qualify because their error is .25. The tail becomes (1+4+4+1)/16=.625. More samples do not force every finite-n tail to decrease monotonically.

Exercise: obtain a sufficient count for epsilon=.2 and delta=.05. Solution: log(40)/.08≈46.11, so choose 47. The bound controls every N≥47. In contrast, finding a first exact-curve crossing below .05 does not establish that every subsequent N remains below it.

Also distinguish an exact curve at known p from estimating an unknown p. All-zero observations do not prove p is zero. For example, at p=.1, ten zero observations occur with probability .9^10, approximately .349. The interface p is an oracle setting defining the world, not a true parameter inferred automatically from the data.

## Report AI data requirements precisely

Specify loss, deviation event, confidence, IID assumptions and the distribution range when discussing sample complexity. Dependence, selection bias and distribution shift can invalidate an unchanged application. Estimator variance, RMSE and deviation probability are different quantities, so their exponents cannot be compared indiscriminately. The next chapter separates theoretical expectations from finite-window slopes in observed learning curves.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
