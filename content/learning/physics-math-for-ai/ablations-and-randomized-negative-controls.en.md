## 1. State which mechanism the intervention targets

An ablation removes or changes a component. A negative control disrupts the mechanism being claimed while preserving selected nuisance properties. Specify whether weights are frozen or the model is retrained, which distribution is preserved, and what outcome is expected if the explanation is correct. Randomization tests have exposed explanations insensitive to learned parameters or training information. [Adebayo et al., Sanity Checks for Saliency Maps](https://research.google/pubs/sanity-checks-for-saliency-maps/)

## 2. Build an inspectable predictive system

Let $X,Z,N$ be independent signs taking values −1 and +1 equally often. Define $Y=0.7X+0.3Z+0.1N$. The full predictor is $0.7X+0.3Z$. A predictor without the second feature is $0.7X$. Their mean squared errors are 0.01 and 0.10 respectively because the independent omitted terms contribute additive variances.

## 3. Randomize a feature while freezing the predictor

Replace $Z$ with an independent sign $Z'$ at inference. The frozen full predictor becomes $0.7X+0.3Z'$. Its residual is $0.3(Z-Z')+0.1N$. The error is $0.09+0.09+0.01=0.19$. The randomized feature has the same marginal distribution as the original feature while its relationship to the target is removed.

## 4. Change the training intervention explicitly

If the predictor is refitted in the population using independent $Z'$, the optimal coefficient of that feature is zero. The refitted predictor returns to $0.7X$ with error 0.10. Frozen randomization probes dependence of the deployed predictor. Refitting probes what information the learning procedure can use under a changed input relation.

For independent centered signs, each sign has second moment one and every cross moment between distinct signs is zero. Expanding the residual square therefore gives

$$
\mathbb E[(0.3(Z-Z')+0.1N)^2]=0.3^2(1+1)+0.1^2=0.19.
$$

After population refitting, consider predictors $aX+bZ'$. The same orthogonality gives

$$
\mathbb E[(Y-aX-bZ')^2]=(0.7-a)^2+b^2+0.3^2+0.1^2.
$$

The two variable terms are nonnegative and vanish at $a=0.7,b=0$, proving the stated optimum of 0.10 within this linear predictor family. Correlated signs or a feature-dependent intervention change the cross terms and require a new calculation.

## 5. Enumerate every case

```python
from itertools import product
from math import isclose

errors = {"full": [], "removed": [], "frozen_random": [], "refitted_random": []}
for x, z, z_random, n in product([-1, 1], repeat=4):
    y = .7*x+.3*z+.1*n
    predictions = {
        "full": .7*x+.3*z,
        "removed": .7*x,
        "frozen_random": .7*x+.3*z_random,
        "refitted_random": .7*x,
    }
    for name, prediction in predictions.items():
        errors[name].append((y-prediction)**2)
means = {name: sum(values)/len(values) for name, values in errors.items()}
for name, expected in [("full", .01), ("removed", .10), ("frozen_random", .19), ("refitted_random", .10)]:
    assert isclose(means[name], expected)
print(means)
```

## 6. Interpret the control under its assumptions

A performance drop under frozen randomization establishes dependence on the disrupted input in that experiment. It can also reflect an unrealistic input combination. Use additional controls for distribution shift, corrupted calibration and implementation errors. Report the exact intervention and the retraining policy so readers can determine which mechanism was tested.
