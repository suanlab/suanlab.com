## 1. Ask whether different parameters predict the same data

Parameters are structurally identifiable for a specified experiment when distinct parameter values give distinct ideal observations. Practical identifiability concerns how tightly finite, noisy observations constrain them. These properties depend on the physical model, the observation operator, and the experimental conditions. [Raue et al., structural and practical identifiability](https://doi.org/10.1093/bioinformatics/btp358)

## 2. Separate two mixed signals

Suppose two unknown amplitudes $a,b$ contribute to measurements $y_1=a+b$ and $y_2=a+rb$. The sensor coefficient $r$ is known. If $r=1$, both instruments give the same equation. Every pair on the line $a+b=y_1$ is compatible with that ideal measurement. Repeating it can reduce noise in the measured sum while leaving the individual amplitudes undetermined.

## 3. Choose a complementary measurement

If $r\ne1$, subtraction gives

$$
b=\frac{y_1-y_2}{1-r},\qquad a=y_1-b.
$$

For $(a,b)=(2,3)$ and $r=-1$, the readings are 5 and −1. Their sum gives twice $a$, and their difference gives twice $b$. At $r=0.9$ the second reading is 4.7. The same algebra still gives a unique answer, but dividing by $1-r=0.1$ magnifies errors in the measured difference.

## 4. Compute uncertainty from the sensor design

With independent zero-mean sensor errors of common variance $\sigma^2$, the difference has variance $2\sigma^2$, giving

$$
\operatorname{Var}(\widehat b)=\frac{2\sigma^2}{(1-r)^2}.
$$

More generally, with full-rank sensitivity matrix $S$ and observation noise covariance $\Sigma$, generalized least squares has parameter covariance $(S^T\Sigma^{-1}S)^{-1}$ when $\Sigma$ is positive definite. The directions of high uncertainty depend on both the sensor responses and the noise. For nonlinear models the Jacobian gives a local approximation, and a global search may still reveal different parameter values with the same outputs.

## 5. Verify the measurement directions with NumPy

```python
import numpy as np

theta = np.array([2.0, 3.0])
sigma = 0.1
for r in (-1, 0, .9):
    sensors = np.array([[1, 1], [1, r]], dtype=float)
    observations = sensors @ theta
    estimate = np.linalg.solve(sensors, observations)
    covariance = sigma**2 * np.linalg.inv(sensors.T @ sensors)
    assert np.allclose(estimate, theta)
    assert np.isclose(covariance[1, 1], 2 * sigma**2 / (1 - r)**2)
    print(r, observations, np.sqrt(covariance[1, 1]))

repeated = np.ones((2, 2))
assert np.linalg.matrix_rank(repeated) == 1
assert np.allclose(repeated @ [2, 3], repeated @ [1, 4])
```

## 6. Design an informative follow-up

Choose a new sensor or excitation that changes how the weak parameter combination enters the data. Estimate the expected noise before buying precision or increasing sample count. Simulate competing parameter values, then pick observations where their predictions separate relative to noise. Reserve an additional measurement to test the fitted model rather than using every reading for estimation.
