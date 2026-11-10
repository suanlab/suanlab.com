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

## Worked example and exercise

At n=2 the hypotheses are (++),(-+),(--). Signs (++),(-+),(--) each admit perfect correlation one. For (+-), maximum correlation is zero. Hence exact empirical Rademacher complexity is (1+1+1+0)/4=0.75. Query mask one is precisely the unrealizable (+-) labeling.

Exercise: derive the n=1 value. Solution: both plus and minus hypotheses are available. Either sign therefore has a matching hypothesis with correlation one, so the expectation is one. The point is shattered. VC dimension one and complexity one at this sample size are numerically equal but have different definitions and interpretations.

## Representation and generalization in AI

Realizable labelings, ability to match random signs and risk on actual data are distinct diagnostics. Even many parameters can define a constrained effective class depending on geometry and architecture. Connect the finite matrix with the general shattering argument and the assumptions of a risk bound. This threshold example does not establish the VC dimension of an arbitrary neural architecture.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
