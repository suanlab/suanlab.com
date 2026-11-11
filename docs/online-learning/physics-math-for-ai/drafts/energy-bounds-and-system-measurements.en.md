## 1. Define what the lower bound counts

For an ideal, unbiased reset with the assumptions from the preceding unit, a single bit has mean heat lower bound $k_BT\ln 2$. If a task were specified as $N$ independent, unbiased resets at the same temperature, the bound for those resets would be $Nk_BT\ln 2$. At 300 K, one billion such resets have a combined bound of about $2.87\times10^{-12}$ J. Counting FLOPs or logic transitions in a real inference run does not determine $N$.

## 2. Match units, boundaries, and tasks

An instrument may report board power while an inference benchmark measures energy at the wall. A bit-reset bound describes heat for an operation on a specified memory. A measured joule per inference includes everything inside the chosen system boundary during the run. A ratio between these quantities is interpretable only after relating the operation counts, the workload, and the boundaries. [Tschand et al., MLPerf Power benchmarking methodology](https://arxiv.org/abs/2410.12032)

## 3. Collect paired measurements

Run both candidate implementations in each block under similar ambient temperature, load, and power meter settings. Alternate or randomize which implementation runs first. Each block can include a longer run with many completed inferences to reduce timing quantization. Record latency and output quality alongside joules per completed inference. A paired difference $d_i=E_{A,i}-E_{B,i}$ removes a common additive block offset in the model $E_{j,i}=\mu_j+c_i+\eta_{j,i}$.

## 4. Separate repeatability from calibration

With independent blocks the estimated standard error of the mean paired difference is $s_d/\sqrt n$, where $s_d$ is the sample standard deviation. If the same meter applies one uncertain multiplicative gain to both runs, that scale uncertainty persists as the number of blocks increases. Report it separately from repeatability. With only a few blocks and possible temperature drift, inspect the differences in time order before relying on a confidence interval.

## 5. Calculate from an illustrative paired dataset

The values below are synthetic and use millijoules per inference at a shared measurement boundary. A positive difference favors B on energy for these blocks.

```python
from math import isclose, log, sqrt
from statistics import mean, stdev

a = [60, 62, 59, 61, 63, 60]
b = [55, 58, 56, 56, 59, 55]
differences = [x - y for x, y in zip(a, b)]
delta = mean(differences)
standard_error = stdev(differences) / sqrt(len(differences))
scale_uncertainty = 0.01 * abs(delta)
bound_joule = 1e9 * 1.380649e-23 * 300 * log(2)
assert isclose(delta, 26 / 6)
assert isclose(mean([x + 4 for x in a]) - mean([y + 4 for y in b]), delta)
print(f"{delta:.3f} mJ difference, repeatability SE {standard_error:.3f} mJ")
print(f"1% common meter scale: {scale_uncertainty:.3f} mJ")
print(f"illustrative reset bound: {bound_joule:.3e} J")
```

## 6. Make each claim falsifiable

State the operation model for the theoretical bound. For a system measurement, publish the run scripts, power trace timestamps, quality and throughput criteria, and the reduction from raw samples to joules per completed task. State which change would overturn a claimed advantage, such as including host preprocessing or comparing both systems at the same throughput.
