## 1. Give each residual a reference

An observation residual compares a prediction with a measurement. For samples $y_i$ at times $t_i$, use $r_i^{\rm obs}=x(t_i)-y_i$. An equation residual asks how well a candidate satisfies a proposed differential law. For $\dot x=-kx$, use $r^{\rm eq}(t)=\dot x(t)+kx(t)$. The former has the unit of $x$, and the latter has the unit of $x$ per time. [Raissi, Perdikaris and Karniadakis, Physics-Informed Neural Networks](https://doi.org/10.1016/j.jcp.2018.10.045)

## 2. Construct several fits through the same observations

Suppose $k=1$ and the exact observations at $t=0,1,2$ are $y_i=e^{-t_i}$. Consider

$$
x_b(t)=e^{-t}+b\sin(\pi t).
$$

Every real $b$ gives zero observation residual at these three times. The sine term vanishes at integer times. Measurements between those times can distinguish the candidates.

## 3. Differentiate the candidate

The equation residual is

$$
r^{\rm eq}(t)=b[\pi\cos(\pi t)+\sin(\pi t)].
$$

At $t=0$, it is $\pi b$. For $b=0.2$ the equation residual there is about 0.628 per unit time even though the observation is matched. Over $[0,2]$, the equation residual RMS is $|b|\sqrt{(\pi^2+1)/2}$. The cross term integrates to zero over a full period.

## 4. Separate fitting, physics checks and prediction

Fit parameters using training observations. Evaluate equation residuals at additional points with documented sampling. Use held-out observations to test predictions and check whether the governing law describes the apparatus. An exactly zero equation residual for a wrong initial condition or a wrong physical law can still give poor predictions.

## 5. Verify both residuals in Python

```python
import numpy as np

b = .2
t_obs = np.array([0., 1., 2.])
x = lambda t: np.exp(-t) + b*np.sin(np.pi*t)
assert np.allclose(x(t_obs) - np.exp(-t_obs), 0.)
t = np.linspace(0, 2, 4096, endpoint=False)
equation_residual = b*(np.pi*np.cos(np.pi*t) + np.sin(np.pi*t))
rms = np.sqrt(np.mean(equation_residual**2))
assert np.isclose(rms, abs(b)*np.sqrt((np.pi**2 + 1)/2))
print("equation RMS:", rms, "held-out error at t=0.5:", x(.5)-np.exp(-.5))
```

## 6. Put residuals on interpretable scales

When combining terms into a loss, nondimensionalize or use scales with physical or statistical meaning. For example, divide observation residuals by sensor standard deviations and equation residuals by a characteristic rate. Report the two components, the residual sampling points, and results on independent data. Different weights can select different compromises.
