## 1. Start with a failure you can observe

Suppose a forecasting system predicts temperatures well during normal operation and fails after an equipment restart. State the target population of restarts, the available inputs, the forecast horizon, and the operational decision. A statement such as “better physics improves AI” becomes testable when the intervention and outcome are defined. [Nosek et al., The Preregistration Revolution](https://doi.org/10.1073/pnas.1708274114)

## 2. Write a comparison with a useful effect size

Compare an unconstrained baseline with a predictor using the same observations and a heat-balance term. Let $L_{Bi}$ and $L_{Ci}$ be the absolute temperature errors on restart episode $i$, averaged over the next 30 minutes. Use the paired improvement

$$
d_i=L_{Bi}-L_{Ci},\qquad \Delta=\mathbb E[d_i].
$$

A concrete claim is $\Delta>0.10$ K in the target population. Set the 0.10 K threshold using a decision or measurement requirement before examining the final outcomes.

## 3. Define what evidence would change the conclusion

An estimated improvement near zero challenges the claim of a useful effect. An estimate above 0.10 K with substantial uncertainty may leave the claim unresolved. A prespecified analysis could use a one-sided confidence bound for the episode-level mean. Correlated time samples within one episode belong to the same observational unit. If restarts are correlated within a machine, plan the analysis and split around machines or an appropriate cluster.

## 4. Hold the question fixed through evaluation

Choose comparable information, a fixed training budget, an episode or machine split, and the decision rule. Record choices that were tuned using development data. Use the final data after these choices are frozen. A surprising subgroup result can motivate a new hypothesis to be tested with new observations.

## 5. Compute a descriptive paired estimate

```python
from statistics import mean
from math import isclose

baseline = [.65, .52, .44, .60, .55, .50]
candidate = [.52, .38, .36, .42, .48, .47]
differences = [a-b for a, b in zip(baseline, candidate)]
effect = mean(differences)
assert isclose(effect, .105)
print("observed mean improvement:", round(effect, 3), "K")
print("observed mean above practical threshold:", effect > .10)
```

The synthetic point estimate is just above the threshold. Reporting it without uncertainty does not establish the population claim.

## 6. Record the outcome that did not happen

If the heat-balance term improves typical errors but worsens extreme restarts, record both findings against their original roles in the plan. Distinguish the initial hypothesis, the test result, and the follow-up question. Retain enough data and analysis detail for another group to reproduce the decision.
