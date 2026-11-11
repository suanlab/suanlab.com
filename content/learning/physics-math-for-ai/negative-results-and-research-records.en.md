## 1. Give each run a stable identity

Record the question, code revision, data snapshot, split, configuration, seed, environment and artifact locations. Identify the metric definition and resource boundary. A stable identifier links the configuration, raw output, analysis and later interpretation. Organized metadata supports discovery and reuse of scientific results. [Wilkinson et al., FAIR Guiding Principles](https://doi.org/10.1038/sdata.2016.18)

## 2. Separate an unsuccessful hypothesis from an execution failure

A completed evaluation can show no benefit, a worse trade-off, or a failure in a specific regime. Those are scientific outcomes. A crashed process has no valid final score unless a partial-outcome rule was defined in advance. Record the exception, stage and partial artifacts, then diagnose the implementation before interpreting model quality.

## 3. See how selective reporting changes a summary

Suppose ten completed validation runs have losses 0.42, 0.55, 0.50, 0.60, 0.48, 0.57, 0.45, 0.65, 0.53 and 0.59. The best run has loss 0.42, while the mean across runs is 0.534. Reporting either quantity is possible if labeled and accompanied by the selection policy. Selecting the best run and evaluating it on independent final data answers a separate question.

## 4. Retain explanations that did not survive testing

For each working hypothesis, record the prediction, intervention, outcome and interpretation. A negative result under one budget or data regime constrains that claim. It can motivate a new experiment with a changed assumption. Keep the original result linked to the revised hypothesis so the analysis history remains inspectable.

## 5. Check a record and its numerical summary

```python
import json
from statistics import mean
from math import isclose

losses = [.42, .55, .50, .60, .48, .57, .45, .65, .53, .59]
record = {
    "run_id": "synthetic-run-01",
    "question": "does the intervention improve held-out episode error?",
    "split_id": "example-machine-split",
    "seed": 2026,
    "status": "completed",
    "metric": {"name": "MAE", "unit": "K", "value": losses[0]},
    "selection": "all completed validation runs retained",
}
restored = json.loads(json.dumps(record))
assert restored == record
assert isclose(mean(losses), .534)
print("best:", min(losses), "mean:", mean(losses), "count:", len(losses))
```

## 6. Write a decision another researcher can reconstruct

Summarize the evidence, competing explanations, known limitations, and next discriminating observation. Link each claim to the run or proof that supports it. Include negative controls and unfavorable regimes. Separate the retained research record from a concise paper narrative, and make both traceable to the same artifacts.
