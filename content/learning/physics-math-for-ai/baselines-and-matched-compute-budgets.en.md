## 1. State the resource the comparison holds fixed

A model comparison needs the same prediction task, available information, data split, and quality metric. It also needs an explicit resource policy. Training FLOPs, elapsed time, memory, and hyperparameter trials each describe a different cost. Choose the resource relevant to the claim and report the others needed to interpret it. [Dodge et al., Show Your Work](https://aclanthology.org/D19-1224/)

## 2. Include a baseline that tests the proposed mechanism

A persistence predictor can reveal the value of recent state information. A conventional statistical model can test whether the new architecture is needed for the gain. A capacity-matched model without the physical term isolates that term. A fully tuned reference tests competitive performance. Each comparison supports a particular claim.

## 3. Construct a budget-dependent ranking

Use synthetic learning curves $L_A(b)=0.3+0.4/\sqrt b$ and $L_B(b)=0.2+0.8/\sqrt b$ for budget $b>0$. At budget 4, A has loss 0.5 and B has loss 0.6. At budget 64, A has loss 0.35 and B has loss 0.30. Solve the equality to find the crossover at budget 16. A single score would conceal this dependence.

## 4. Separate model selection and final comparison

Use development data to choose hyperparameters under the declared search budget. Freeze the chosen models and the evaluation protocol before applying the final test. Count failed or discarded searches when they consumed the declared resource. Repeat the comparison with matched information and a meaningful budget range.

Subtract the curves at the same budget,

$$
\Delta(b)=L_A(b)-L_B(b)=0.1-\frac{0.4}{\sqrt b},\qquad b>0.
$$

The equality condition is $\sqrt b=4$, hence $b=16$. Positive loss difference favors B, while a negative difference favors A. The differences at budgets 4, 16 and 64 are respectively $-0.10$, $0$ and $0.05$. Comparing A at budget 4 against B at budget 64 combines algorithm choice with resource allocation. Declare whether a budget unit means time, operations or search trials. The coefficients of these synthetic curves belong to that chosen unit and loss scale.

## 5. Verify the crossover in Python

```python
from math import isclose, sqrt

loss_a = lambda budget: .3+.4/sqrt(budget)
loss_b = lambda budget: .2+.8/sqrt(budget)
assert loss_a(4) < loss_b(4)
assert isclose(loss_a(16), loss_b(16))
assert loss_b(64) < loss_a(64)
for budget in [4, 16, 64]:
    print(budget, round(loss_a(budget), 3), round(loss_b(budget), 3))
```

These curves illustrate a possible reversal. Measure the corresponding curves for the actual implementation before making an empirical claim.

## 6. Report trade-offs at deployment

Compare inference latency, memory and energy at the same output quality or state the quality differences. Include preprocessing and transfers according to a declared boundary. A model that wins after a large search may still be expensive to develop, while a compact baseline may meet the operational requirement with less effort.
