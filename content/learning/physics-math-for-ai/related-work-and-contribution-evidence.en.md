## 1. Compare claims at the level of their assumptions

A contribution statement should identify the closest work, what it established, and what remains untested under the new conditions. Record the dataset, observation window, model class, split, metric, and uncertainty reported by each source. Read the original method and experiment. A difference in terminology can hide the same method, and the same metric name can hide a different evaluation population. [Pineau et al., Improving Reproducibility in Machine Learning Research](https://jmlr.org/papers/v22/20-303.html)

## 2. Specify a small related-work matrix

In a hypothetical restart study, two previous approaches are considered. One uses local temperature observations. Another adds a longer context window. Our proposed change adds a heat-balance term. Keep the same data, forecast horizon, parameter-search policy, and error definition while comparing both contexts with and without the physical term. Treat the published results and the controlled reproduction as separate evidence.

## 3. Quantify what each comparison isolates

Suppose controlled errors are 0.60 K and 0.50 K without the physical term, for short and long context respectively. With the physical term, they are 0.54 K and 0.40 K. The gains from the physical term are 0.06 K and 0.10 K. The difference of these gains is 0.04 K. The total gain of 0.20 K from the weakest to strongest configuration includes both interventions.

## 4. State the smallest claim supported by the evidence

The hypothetical results support an observed error reduction from adding the physical term under both matched context settings. Claiming an effect in a new population requires evaluation there. A claim of novelty also needs a search of related methods and the precise contribution, such as a new observation regime, a theorem with weaker assumptions, or independent evidence for an existing method.

## 5. Compute the matched comparisons

```python
from math import isclose

short_base, long_base = .60, .50
short_physics, long_physics = .54, .40
short_gain = short_base - short_physics
long_gain = long_base - long_physics
interaction = long_gain - short_gain
assert isclose(short_gain, .06)
assert isclose(long_gain, .10)
assert isclose(interaction, .04)
print("matched effects:", round(short_gain, 2), round(long_gain, 2))
print("difference of effects:", round(interaction, 2))
```

## 6. Make the contribution easy to challenge

Write each contribution next to a table, proof, or artifact that supports it. State which assumptions a reader can change to test its scope. Keep unsuccessful reproductions and incompatible published protocols in the record, so that later work can determine whether apparent differences come from the method or evaluation.
