## 1. State which process the model omits

Write the observation as $y(t)=m(t,\theta)+\delta(t)+\epsilon(t)$, where $m$ is the simulator, $\delta$ is systematic discrepancy, and $\epsilon$ is measurement error. A repeatable gap can arise from a missing process, a biased boundary condition or an inadequate approximation. Measuring more precisely can make the gap clearer. [Brynjarsdóttir and O’Hagan, Learning About Physical Parameters](https://doi.org/10.1088/0266-5611/30/11/114007)

## 2. Fit a decay model with an omitted equilibrium

Consider exact synthetic data $y(t)=c+(1-c)e^{-kt}$, with $0\le c<1$ and $k>0$. The initial value is 1 and the equilibrium is $c$. The simplified model $m(t,\widehat k)=e^{-\widehat k t}$ fixes the equilibrium at zero. Match one positive-time calibration reading at $t_c$:

$$
\widehat k=-\frac{\ln y(t_c)}{t_c}.
$$

When $c>0$, $y(t_c)>e^{-kt_c}$, so the fitted rate is smaller than the true rate. The rate has absorbed part of the equilibrium discrepancy.

## 3. Check the fitted model beyond calibration

At $t_c=1$, $c=0.25$ and $k=1$, the reading is about 0.526 and the fitted rate is about 0.643. At $t=3$ the simplified prediction is $y(1)^3\approx0.145$, while the data-generating value is about 0.287. The gap grows even though the initial and calibration readings are matched.

## 4. Compare model changes and new observations

An independent measurement of equilibrium fixes $c$ and permits recovery of $k$ from $(y(t_c)-c)/(1-c)$. A flexible discrepancy can also fit the calibration data, but different simulator parameters may then be compensated by different discrepancy curves. Choose additional observations where these explanations predict different outcomes.

## 5. Reproduce the calibration and test

```python
from math import exp, isclose, log

c, rate, calibration_time = .25, 1., 1.
truth = lambda t: c + (1-c)*exp(-rate*t)
fit = -log(truth(calibration_time))/calibration_time
simple = lambda t: exp(-fit*t)
assert isclose(simple(0), truth(0))
assert isclose(simple(calibration_time), truth(calibration_time))
assert fit < rate and truth(3)-simple(3) > .14
corrected = -log((truth(calibration_time)-c)/(1-c))/calibration_time
assert isclose(corrected, rate)
print(fit, truth(3)-simple(3), corrected)
```

## 6. Record the regime where the model is useful

Plot held-out errors by time or physical regime. Compare noise, numerical discretization and structural discrepancy with separate experiments where possible. Report the assumptions that identify the parameter and distinguish estimated simulator settings from independently measured physical quantities.
