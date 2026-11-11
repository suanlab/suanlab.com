## Labelings realized on fixed points

VC dimension does not simply count parameters. It asks whether every binary labeling can be represented on the same points. Here the class consists of increasing thresholds on the real line. Place n distinct points in order: predictions are minus one before the threshold and plus one afterward. The upper matrix explicitly lists every row from threshold zero through n.

Columns are ordered points and rows are hypotheses. Signs appear numerically as well as by color. The query mask uses only its n low bits. Blue below shows each hypothesis's mismatch count with that query, exposing labelings that the class cannot realize.

## Prove the VC dimension

Moving a threshold generates exactly n+1 labelings. Shattering requires all 2^n labelings on the same points:

$$
\Pi_H(n)=n+1,\qquad \mathrm{VC}(H)=1.
$$

One point can receive either sign by moving the threshold across it. But no pair x₁<x₂ can receive (+1,-1). Thus no two-point set is shattered, and neither can a larger set be shattered. This argument applies to arbitrary ordered pairs, not merely the finite arrangement displayed in the interface.

## Rademacher complexity averages something else

For fixed points, introduce independent fair signs sigma_i. For each sign vector, maximize correlation with a hypothesis and then average that maximum:

$$
\widehat{\mathfrak R}(H)=E_\sigma\left[\max_{h\in H}\frac1n\sum_i\sigma_i h(x_i)\right]
\le\min\left(1,\sqrt{\frac{2\log(n+1)}n}\right).
$$

Because n≤10, every sign vector is actually enumerated. The n+1 hypothesis rows and 2^n sign vectors are different sets. This finite-class upper bound concerns [-1,1]-valued functions. A generalization statement additionally needs the appropriate loss class, IID and confidence assumptions. The displayed complexity is not directly an error bound for every possible loss.

## Random-walk maxima connect label geometry to loss complexity

For a sign vector define S_t=sum of its first t signs and S_0=0. Threshold t has correlation S_n−2S_t. Therefore the exact empirical complexity also equals

$$
\widehat{\mathfrak R}(H)=\frac1nE_\sigma[S_n-2\min_{0\le t\le n}S_t]=-\frac2nE_\sigma\min_t S_t.
$$

This is a maximum over thresholds inside an expectation, not the maximum of average correlations. The code checks both forms by every random walk, using rational sums. The finite matrix makes an optimization over hypotheses into a path-minimum calculation.

For fixed binary labels y_i and signs encoded as ±1, the 0–1 loss is (1−y_i h_i)/2. The hypothesis-independent constant has zero Rademacher expectation, and −sigma_i*y_i are again independent fair signs. Consequently the loss-class empirical complexity is exactly half the classifier complexity on these points. Generalization concerns that loss class.

| Distinct ordered points n | Exact classifier complexity |
|---|---:|
| 1 | 1 |
| 2 | 3/4 |
| 4 | 19/32 |

```python
from itertools import product
from fractions import Fraction as F
for n in (1,2,4):
    hs=[[-1 if i<t else 1 for i in range(n)] for t in range(n+1)]
    total=walk=loss_total=F(0);labels=[(-1)**i for i in range(n)]
    for signs in product((-1,1),repeat=n):
        path=[0]
        for sign in signs:path.append(path[-1]+sign)
        best=max(sum(a*b for a,b in zip(signs,h)) for h in hs)
        assert best==path[-1]-2*min(path)
        total+=F(best,n);walk+=F(-2*min(path),n)
        loss_total+=max(F(sum(s*(1-y*v) for s,y,v in zip(signs,labels,h)),2*n) for h in hs)
    rad=total/2**n
    assert total==walk and loss_total==total/2
    print(n,rad)
```

Under IID samples, the ghost-sample symmetrization argument bounds E sup_h(R−Rhat) by twice the expected loss complexity. Inserting empirical ERM and a fixed population oracle then bounds expected ERM excess by that same quantity: the oracle's signed empirical gap has expectation zero. A confidence statement needs concentration as well. Random sampled inputs may repeat, in which case the loss transformation still holds but the n+1 distinct-point labeling count cannot be reused unchanged. The live exhaustive calculation is conditional on its fixed, distinct points.

## Worked example and exercise

At n=2 the hypotheses are (++),(-+),(--). Signs (++),(-+),(--) each admit perfect correlation one. For (+-), maximum correlation is zero. Hence exact empirical Rademacher complexity is (1+1+1+0)/4=0.75. Query mask one is precisely the unrealizable (+-) labeling.

Exercise: derive the n=1 value. Solution: both plus and minus hypotheses are available. Either sign therefore has a matching hypothesis with correlation one, so the expectation is one. The point is shattered. VC dimension one and complexity one at this sample size are numerically equal but have different definitions and interpretations.

## Representation and generalization in AI

Realizable labelings, ability to match random signs and risk on actual data are distinct diagnostics. Even many parameters can define a constrained effective class depending on geometry and architecture. Connect the finite matrix with the general shattering argument and the assumptions of a risk bound. This threshold example does not establish the VC dimension of an arbitrary neural architecture.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
