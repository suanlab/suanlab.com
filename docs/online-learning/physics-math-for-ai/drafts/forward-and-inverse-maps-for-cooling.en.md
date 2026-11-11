## 1. Predict what a thermometer will see

Let $T(t)$ be the temperature of an object, $T_\infty$ the measured ambient temperature, and $k>0$ a cooling rate. If the object is approximately uniform in temperature, the environment is fixed, and heat loss is proportional to the temperature difference, the excess temperature satisfies

$$
\dot y=-ky,\qquad y=T-T_\infty,\qquad y(t)=Ae^{-kt}.
$$

The initial excess is $A>0$. For observation times $t_1,\ldots,t_m$, the forward map takes $(A,k)$ to the vector of predicted excess temperatures. The choice of what to measure and when to measure it is part of the map. [Tarantola, The General Discrete Inverse Problem](https://epubs.siam.org/doi/10.1137/1.9780898717921.ch1)

## 2. Reverse the direction of the question

The inverse problem starts with measured values and asks which parameters could have produced them. A single positive observation $y_1$ at time $t_1$ gives

$$
A=y_1 e^{kt_1}.
$$

Each admissible $k$ has a corresponding $A$. The curves cross at the one measured point, and their predictions at other times can differ. A known initial temperature would add a constraint and change the inverse problem.

## 3. Solve with two noiseless observations

For $t_2>t_1$ and $y_1>y_2>0$, take the ratio of observations to eliminate $A$:

$$
k=\frac{\ln(y_1/y_2)}{t_2-t_1},\qquad A=y_1 e^{kt_1}.
$$

If $A=40$ K and $k=0.2\,\mathrm{min}^{-1}$, the excess temperatures at 2 and 6 minutes are about 26.81 K and 12.05 K. Substituting the unrounded values recovers the parameters. Celsius temperature differences and Kelvin temperature differences have the same size.

## 4. Diagnose a poor observation design

Two nearly coincident observation times make the denominator small. Late readings can also be near zero, making their relative error large before taking a logarithm. At fixed $t_1,t_2$, the first-order perturbation is

$$
\delta k\approx\frac{\delta y_1/y_1-\delta y_2/y_2}{t_2-t_1}.
$$

Noise can even produce a nonpositive reading or apparent warming, placing the observations outside the exact positive-cooling model. With many noisy readings, fit the forward model under a stated noise model and use separate measurements to check it.

## 5. Reconstruct and verify in Python

```python
from math import exp, isclose, log

def forward(amplitude, rate, time):
    return amplitude * exp(-rate * time)

def inverse(y1, y2, t1, t2):
    rate = (log(y1) - log(y2)) / (t2 - t1)
    return y1 * exp(rate * t1), rate

t1, t2 = 2, 6
y1, y2 = [forward(40, .2, time) for time in (t1, t2)]
for rate in (.05, .2, .4):
    assert isclose(forward(y1 * exp(rate * t1), rate, t1), y1)
amplitude, rate = inverse(y1, y2, t1, t2)
assert isclose(amplitude, 40)
assert isclose(rate, .2)
print(f"A={amplitude:.2f} K, k={rate:.3f}/min")
print("perturbed late observation:", inverse(y1, y2 + .5, t1, t2))
```

## 6. Use a new observation to test the model

Use the estimated parameters to predict a reading at a third time. Compare that prediction with a reading withheld during fitting. A large difference can arise from sensor noise, an incorrect ambient temperature, or a cooling model that omits changing heat transfer. Record the assumed physics, observation operator, and estimation step separately so each explanation can be tested.
