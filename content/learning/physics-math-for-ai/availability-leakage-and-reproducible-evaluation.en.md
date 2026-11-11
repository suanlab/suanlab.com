## 1. Define the information available at decision time

For a forecast at time $t$, record the timestamp when each feature becomes available, including reporting delays and later revisions. A value labeled with date $t$ may be published after the decision. Define the target horizon and the time at which training labels would have been available. [Kapoor and Narayanan, Leakage and the Reproducibility Crisis](https://doi.org/10.1016/j.patter.2023.100804)

## 2. Build a process with limited predictability

Let independent centered unit-variance variables be $X,U,N$. Define a future signal $F=\rho X+\sqrt{1-\rho^2}U$ and target $Y=0.8F+0.6N$. At decision time only $X$ is available. The best linear prediction from it is $0.8\rho X$, with mean squared error $0.64(1-\rho^2)+0.36$.

## 3. Measure the effect of a leaked future feature

An evaluator that exposes $F$ permits prediction $0.8F$ with MSE 0.36. Exposing the target itself gives zero error. At $\rho=0$ the legitimate MSE is 1, so the apparent gains can be entirely explained by unavailable information. Larger $\rho$ creates genuine predictive information in the present feature.

## 4. Make the split match the question

For forecasting, simulate what was available at each training and test decision. A test feature can legitimately use earlier observations if the deployment system has them. Fit normalizers and feature selection using the allowed training information. For a new-machine claim, split by machine and check identifiers or near-duplicate records that can reveal test labels indirectly.

For the available-information predictor, expand the residual as

$$
Y-0.8\rho X=0.8\sqrt{1-\rho^2}\,U+0.6N,\qquad |\rho|\le1.
$$

Independent centered unit-variance residual terms give

$$
\mathbb E[(Y-0.8\rho X)^2]=0.64(1-\rho^2)+0.36.
$$

The residual also has conditional mean zero given $X$, by independence. For any square-integrable predictor $g(X)$, its risk equals this residual risk plus $\mathbb E[(0.8\rho X-g(X))^2]$. This proves optimality with the specified available information. At $\rho=0.5$ the admissible risk is 0.84; revealing the future variable lowers it to 0.36. The drop of 0.48 is produced by changing what the evaluator exposes.

## 5. Enumerate an exact synthetic audit

```python
from itertools import product
from math import isclose, sqrt

rho = .5
squared = {"available": [], "future": [], "target": []}
for x, u, n in product([-1, 1], repeat=3):
    future = rho*x+sqrt(1-rho**2)*u
    y = .8*future+.6*n
    predictions = {"available": .8*rho*x, "future": .8*future, "target": y}
    for name, prediction in predictions.items():
        squared[name].append((y-prediction)**2)
mse = {name: sum(values)/len(values) for name, values in squared.items()}
assert isclose(mse["available"], .64*(1-rho**2)+.36)
assert isclose(mse["future"], .36) and mse["target"] == 0
print(mse)
```

## 6. Preserve the evidence needed to reproduce the audit

Save feature definitions, availability rules, split identifiers, code version, preprocessing state and seed settings. Reproduce the evaluation from the recorded snapshot. A fresh run under the same leakage can repeat the same optimistic result, so reproducibility and the validity of the information set need separate checks.
