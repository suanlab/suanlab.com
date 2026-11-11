## Define the objectives behind the three error names

Simply naming approximation, estimation and optimization error can suggest that every displayed term must be positive. First specify the oracle and objective being compared. Here x is uniform on {-1,0,1}, the true regression function is f(x)=x+c x², and labels add independent centered Rademacher noise ±sigma. Loss is half the squared error.

The hypothesis has one coefficient multiplying either a constant feature or x. Actual GD runs on the sample quadratic, while the exact sample ERM and population class oracle are solved separately. Blue is population risk and orange is empirical risk: they average over different objects.

## An exact telescoping identity

Bayes risk is sigma²/2. Let w_H minimize population risk within the class, and w_S minimize empirical risk. For the current iterate,

$$
R(w_k)-R_{Bayes}=[R(w_H)-R_{Bayes}]
+[R(w_S)-R(w_H)]+[R(w_k)-R(w_S)].
$$

The first term is approximation error and the second is this sample's estimation effect. The final population optimization effect can be negative. In contrast, the same-objective empirical gap Rhat(w_k)-Rhat(w_S) is nonnegative. Replacing the last population term with that empirical gap would generally invalidate the identity.

## Actual quadratic optimization

For feature phi, sample curvature is A=mean(phi²) and the linear term is B=mean(phi y):

$$
w_{k+1}=w_k-\eta(Aw_k-B),\qquad w_S=B/A\quad(A>0).
$$

When all sampled features are zero, A=B=0 and the empirical objective is flat. The declared tie rule chooses w_S=0, and GD stays at zero. Having informative features in the population does not mean a particular sample contains them. No hidden ridge is added to change this degenerate case.

## Early stopping is a spectral filter with a bias cost

For a separate intercept problem, let Y=mu+sigma*sign with independent fair signs and minimize half-squared empirical loss from w0=0. With 0<eta<1, solving the scalar recurrence gives w_k=g_k*Ybar, where g_k=1−(1−eta)^k. The population oracle is mu. Independence and centered noise give the expected excess above Bayes risk:

$$
E[R(w_k)-R_{Bayes}]=\frac12(1-g_k)^2\mu^2+\frac12g_k^2\frac{\sigma^2}{N}.
$$

The first term is shrinkage bias and the second is estimator variance. Differentiating in a freely chosen g gives g*=mu²/(mu²+sigma²/N), when the denominator is positive. An available integer iteration need not realize that value. Mu and noise variance are oracle quantities here; the formula is not a data-only stopping rule.

For mu=1/4, sigma=1, N=4 and eta=1/5, one step improves expected excess, while running to ERM makes it larger. This is an expectation over all sixteen noise sequences, rather than a chosen favorable seed.

| Iteration | Expected excess above Bayes |
|---|---:|
| 0 | 1/32 |
| 1 | 1/40 |
| Limit | 1/8 |

```python
from itertools import product
from fractions import Fraction as F
N=4;mu=F(1,4);eta=F(1,5)
for k in (0,1,8):
    gain=1-(1-eta)**k
    exact=sum(((gain*(mu+F(sum(signs),N))-mu)**2/2 for signs in product((-1,1),repeat=N)),F(0))/2**N
    formula=((1-gain)**2*mu**2+gain**2/N)/2
    assert exact==formula
    print(k,exact,float(exact))
assert ((1-(1-eta))**2/N+(1-eta)**2*mu**2)/2==F(1,40)
assert F(1,2*N)==F(1,8)
```

In the live quadratic, population effects are now evaluated by centered identities: approximation is 1/3+c²/9 for constants or c²/3 for the x feature; estimation is A_pop*(w_S−w_H)²/2. The population optimization effect factors the difference of two squares and remains signed. The actual sample, ERM tie rule and GD recurrence are retained. These analytic diagnostics can retain a positive 3.3×10^(−17) approximation even when two displayed risks both round to the same noise floor. `expandedEffects` preserves the directly subtracted values for comparison; the diagnostics are not measured differences between rounded display strings.

## Worked example and exercise

For c=0, sigma=1, N=2 and seed nine, the actual observations are (-1,-2),(1,2). Linear-feature ERM is w=2. Population risk is (w-1)²/3+1/2, so the class oracle is one and the estimation effect is 1/3. Starting at zero with eta=1/2, one GD step reaches w=1. Its population optimization effect is -1/3, even though the empirical optimization gap remains 1/2.

Exercise: compute constant-class approximation error at c=1. Solution: true values are [0,0,2], with mean 2/3. Half their mean squared residual is 4/9. For the linear feature, the oracle coefficient is one and approximation error is 1/3. More optimization iterations cannot remove a representation restriction.

## Interpret early stopping carefully

The example proves that early stopping can outperform sample ERM in population risk, not that it always does. Report the optimized objective, evaluation law, hypothesis class, initialization and step settings together. Identifying the three effects in real research requires an oracle or independent estimation and experimental design. This finite world provides transparent definitions and checks, not a universal recipe for choosing stopping time.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
