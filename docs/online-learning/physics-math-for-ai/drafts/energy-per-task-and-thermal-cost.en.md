## 1. Define a completed task and a measurement boundary

Compare energy for the same task, such as 100 predictions at a stated input shape and precision. Decide whether measurement covers a chip, board, host, or wall outlet. Power is energy per time, in watts. Energy per completed task is the integral of power during the run divided by the completed task count. [Horowitz, Computing’s Energy Problem](https://ieeexplore.ieee.org/document/6757323)

## 2. Account for switching and data movement

Charging a capacitance $C$ from an ideal supply at voltage $V$ draws energy $CV^2$. Half is stored on the capacitor and half is dissipated during resistive charging. Discharging dissipates the stored half, making the full ideal charge–discharge cycle cost $CV^2$. If an effective capacitance charges on average $\alpha$ times per clock cycle, its average dynamic power is $\alpha CV^2f$. Leakage, memory transfers, regulators, and cooling add costs outside this simple switching model. The effective capacitance and switching activity depend on the circuit and workload.

## 3. Integrate measured power samples

Let $P_i$ be power at time $t_i$ on a synchronized clock. Linear interpolation between samples gives

$$
E\approx\sum_i \frac{P_i+P_{i+1}}{2}(t_{i+1}-t_i).
$$

This handles uneven sampling intervals. A narrow spike between samples can still escape the calculation. If the matched idle power is $P_0$, then $E-P_0(t_{\rm end}-t_{\rm start})$ is incremental energy above that chosen baseline. Record both the gross and incremental quantities, along with measurement boundaries and the baseline procedure.

## 4. Work through two traces

A 40 W run lasting 0.20 s consumes 8 J. A 60 W run lasting 0.10 s consumes 6 J. For 100 completed predictions these give 80 and 60 mJ per prediction. With a matched 20 W idle baseline, each run adds 4 J above baseline. The gross energy and incremental energy answer different questions.

Heat removal depends on where and when power is dissipated. A faster run can consume less total energy yet impose a higher short-term thermal load. For constant power and an approximate thermal resistance $R_\theta$ in K/W, steady-state temperature rise above ambient is $P R_\theta$. Temperature needs time to approach this rise, and a transient estimate needs thermal capacitance as well. Record temperature and throttling when comparing sustained runs.

## 5. Integrate uneven samples in Python

```python
from math import isclose

def integrate(times, powers, idle=0.0):
    return sum((0.5 * (a + b) - idle) * (end - start)
               for start, end, a, b in zip(times, times[1:], powers, powers[1:]))

runs = [
    ([0, 0.03, 0.20], [40, 40, 40]),
    ([0, 0.06, 0.10], [60, 60, 60]),
]
for times, powers in runs:
    gross = integrate(times, powers)
    extra = integrate(times, powers, idle=20)
    assert isclose(extra, 4)
    print(f"{gross:.1f} J gross, {extra:.1f} J incremental, {gross / 100 * 1000:.0f} mJ/prediction")
assert isclose(integrate([0, 0.05, 0.2], [10, 20, 50]), 6)
```

## 6. Design an energy comparison for AI inference

Hold model, output quality target, precision, and task count fixed. Record whether preprocessing, data loading, and cooling fall inside the measurement boundary. Repeat runs after warm-up and log the thermal state. For short workloads, run a longer batch or use an instrument with sufficient temporal resolution. Compare energy per task alongside latency and throughput so the trade-offs remain visible.
