## 1. Add a preference to a noisy fit

For data $y$ and forward matrix $A$, zeroth-order Tikhonov regularization minimizes

$$
\|Ax-y\|_2^2+\lambda\|x\|_2^2,\qquad \lambda\ge0.
$$

The first term measures data mismatch. The second penalizes large coefficients in the chosen coordinates. Parameter scaling changes the meaning of this penalty, so declare units or nondimensionalize first. [Hansen, Discrete Inverse Problems, Chapter 4](https://epubs.siam.org/doi/10.1137/1.9780898718836.ch4)

## 2. Derive the scalar mode filter

A singular mode of strength $s>0$ and observed coordinate $y$ minimizes $(sx-y)^2+\lambda x^2$. Setting the derivative to zero yields

$$
\widehat x_\lambda=\frac{s}{s^2+\lambda}y.
$$

The noise gain is $g_\lambda=s/(s^2+\lambda)$. If $y=sx_*+\eta$, the signal is multiplied by $s^2/(s^2+\lambda)$ and the noise by $g_\lambda$. At $s=0$ and $\lambda>0$ the formula chooses zero. That value comes from the penalty and the zero-centered convention.

## 3. See bias and variance in the same equation

For zero-mean noise of variance $\sigma^2$, the bias is $-\lambda x_* /(s^2+\lambda)$ and the variance is $g_\lambda^2\sigma^2$. The mean squared error is bias squared plus variance. For $s=0.05$, $x_*=1$, and $\sigma=0.05$, the unregularized estimate has noise standard deviation 1. At $\lambda=s^2=0.0025$, both the signal retention and noise standard deviation are 0.5. The mean squared error drops from 1 to 0.5 in this example.

## 4. Extend the filter to a matrix

Write $A=U\operatorname{diag}(s_i)V^T$. Project $y$ onto the left singular vectors, multiply each coordinate by $s_i/(s_i^2+\lambda)$, and reconstruct with the right singular vectors. Small singular modes are shrunk more strongly for the same positive $\lambda$. If $A$ and $x$ have physical units, use an explicitly scaled penalty. For the stated objective $\lambda$ is the coefficient itself.

## 5. Compare the direct and filtered solutions in Python

```python
import numpy as np

strengths = np.array([1.0, .05])
A = np.diag(strengths)
truth = np.ones(2)
y = A @ truth + np.array([0.0, .05])
for penalty in [0.0, .0025, .025]:
    x = np.linalg.solve(A.T @ A + penalty * np.eye(2), A.T @ y)
    filters = strengths / (strengths**2 + penalty)
    assert np.allclose(x, filters * y)
    print(penalty, x)
assert np.isclose((strengths[1] / (strengths[1]**2 + .0025)) * .05, .5)
```

## 6. Choose the penalty using the intended inference task

The true parameters are usually unavailable. Use held-out measurements, an independently estimated noise scale, or a justified prior to choose a penalty. Check whether the directions most important to the scientific question survive shrinkage. Report the penalty definition and assess sensitivity to its choice. A small residual can be obtained by fitting noise, and a large penalty can suppress genuine signal.
