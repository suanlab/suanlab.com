## Changing group proportions changes risk

Define a world with group labels zero and one. Training group-one probability is alpha and target probability is beta. Conditional labels stay fixed; only group proportions change. Classifier zero always predicts zero, classifier one always predicts one, and classifier two predicts an independent fair coin. Its group loss .5 is expected randomized loss, not one realized coin outcome.

Actual training group counts select the sample ERM. Blue shows selected-classifier risk across target group proportions, orange the robust choice, and markers identify current beta. Distinguish oracle risk in the fully declared world from risk identifiable from observed samples.

## Importance weights require support

For training group probabilities P_g and target probabilities Q_g,

$$
R_Q(h)=E_P\left[\frac{Q_G}{P_G}\ell(h,G)\right]
$$

requires positive training mass wherever target mass is positive, together with the maintained conditional-label assumptions. When P_g=0<Q_g, the ratio is undefined and loss there is not identified from samples alone without additional model assumptions. Importance estimation is then unavailable. The oracle risk remains computable only because the whole teaching world is specified; it does not resolve the observational identification problem.

Even with support, large weights can create high estimation variance. One finite weighted average need not equal target risk exactly. Unbiasedness concerns expectation over repeated training samples.

## Total-variation ambiguity

For two-point distributions, TV distance is the absolute difference in group-one mass. Around a declared population center alpha,

$$
q\in[\max(0,\alpha-\rho),\min(1,\alpha+\rho)],\qquad
R_q(h)=(1-q)\ell_{h0}+q\ell_{h1}.
$$

Risk is affine, so its maximum occurs at an interval endpoint. Compute both endpoints and choose the least-worst of the three classifiers. This ambiguity set is a declared model, not an automatically data-certified confidence set. A target outside the interval is not covered by its robustness claim.

## Turn a declared shift radius into a statistical guarantee

Suppose alpha is unknown and qhat is its empirical group proportion. With probability at least 1−delta, Hoeffding gives |qhat−alpha|≤e=sqrt(log(2/delta)/(2N)). If a target is additionally assumed to obey |beta−alpha|≤rho, the triangle inequality places it in the interval centered at qhat with radius rho+e, clipped to [0,1]. Solving the robust problem on that enlarged interval provides a confidence statement only under both assumptions. A chosen radius alone does not certify the unknown target.

Allow an independent constant prediction coin with probability u of label one. Its expected risk is u+q*(1−2u). If the allowed q interval is wholly below one half, u=0 is optimal; wholly above it, u=1 is optimal. If it straddles one half, q=1/2 forces every u to risk 1/2, and u=1/2 attains that value everywhere. Thus the live three classifiers already suffice for this continuum of constant randomized predictors in this particular world.

| Allowed group-one interval | Optimal coin u | Worst risk |
|---|---:|---:|
| [1/10,2/5] | 0 | 2/5 |
| [3/5,9/10] | 1 | 2/5 |
| [1/10,9/10] | 1/2 | 1/2 |

```python
from fractions import Fraction as F
for lo,hi in ((F(1,10),F(2,5)),(F(3,5),F(9,10)),(F(1,10),F(9,10))):
    risk=lambda u,q:u+q*(1-2*u)
    choices=[F(0),F(1),F(1,2)]
    best=min(choices,key=lambda u:max(risk(u,lo),risk(u,hi)))
    optimum=max(risk(best,lo),risk(best,hi))
    assert all(max(risk(F(k,100),lo),risk(F(k,100),hi))>=optimum for k in range(101))
    print(lo,hi,best,optimum)
alpha=F(4,5);rho=F(1,10);error=F(1,4)
for k in range(9):
    center=F(k,8)
    if abs(center-alpha)<=error:
        lo=max(0,center-rho-error);hi=min(1,center+rho+error)
        assert lo<=max(0,alpha-rho)<=min(1,alpha+rho)<=hi
P=F(1,10);Q=F(9,10);N=8
variance=(Q*Q/P-Q*Q)/N
assert variance==F(729,800)
print('importance variance:',variance)
```

The code verifies interval containment whenever its declared estimation event holds. It does not assert that radius 1/4 has a particular confidence at N=8. Importance weighting also has a computable variance here: for deterministic group losses l_g and support, Var(Rhat_IS)=[sum Q_g²*l_g²/P_g−R_Q²]/N. Rare training mass can make that large despite unbiasedness. Neither a support condition nor this finite-class minimax result protects against arbitrary changes within groups.

## Worked example and exercise

At alpha=.9, always-one has training population risk .1. Moving to beta=.1 raises target risk to .9. Radius .8 gives interval [.1,1]. Worst risks are one for always-zero, .9 for always-one and .5 for fair prediction, so the finite-class robust choice is fair prediction.

Exercise: compute weights for alpha=.9 and beta=.1. Solution: group zero receives .9/.1=9, while group one receives .1/.9=1/9. The rare training group gets a large weight. With alpha=1 and beta=0, the first denominator is zero and must not be replaced with an invented large number.

## Interpret AI robustness claims

Report the allowed shift, conditional-label assumptions and how the radius was obtained. Worst-case risk belongs to the declared ambiguity set and class, not every conceivable real-world change. Record training averages, target oracles, importance estimates, finite-sample uncertainty and minimax objectives separately. A transparent toy calculation helps identify those requirements without granting universal robustness.

References: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) and [MIT learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/).
