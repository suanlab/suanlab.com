## 1. Test existence, uniqueness, and stability

An inverse problem is well posed in the Hadamard sense when a solution exists for the allowed data, is unique, and depends continuously on the data in the chosen norms. If any of these conditions fails, it is ill posed. A finite-dimensional invertible system can still be ill conditioned when a small data error produces a large change in the solution. [Hansen, Discrete Inverse Problems](https://doi.org/10.1137/1.9780898718836)

## 2. Observe two directions with different strengths

Consider the two-mode forward map $A_s=\operatorname{diag}(1,s)$ with $0\le s\le1$. The first parameter appears directly in the measurements, and the second appears with strength $s$. If $s>0$, exact noiseless inversion divides the second measurement by $s$. The singular values are 1 and $s$, so the Euclidean inverse operator norm is $1/s$. For this normalization, the condition number is also $1/s$.

## 3. Amplify a controlled perturbation

For true parameters $(1,1)$, the noiseless data are $(1,s)$. Adding an error $\delta$ to the second component changes the recovered parameter to

$$
\widehat x_2=1+\frac{\delta}{s}.
$$

A perturbation of $0.01$ produces an error of $0.02$ at $s=0.5$ and an error of 1 at $s=0.01$. The data perturbation is the same size in both cases. A perturbation of the same size in the first component changes only the first parameter by $0.01$.

## 4. Inspect the rank-deficient limit

At $s=0$, all exact forward data have second component zero. If the observed second component is nonzero, there is no exact solution. If it is zero, every value of $x_2$ gives the same observation. A least-squares problem can still be formulated with incompatible data. Setting $x_2=0$ in the minimum-norm solution is a choice among equally good data fits.

## 5. Reproduce directional error amplification

```python
import numpy as np

truth = np.array([1.0, 1.0])
for strength in (.5, .1, .01):
    forward = np.diag([1.0, strength])
    clean = forward @ truth
    weak_error = np.linalg.solve(forward, clean + [0, .01]) - truth
    strong_error = np.linalg.solve(forward, clean + [.01, 0]) - truth
    assert np.allclose(weak_error, [0, .01 / strength])
    assert np.allclose(strong_error, [.01, 0])
    print(strength, np.linalg.cond(forward), weak_error[1])

singular = np.diag([1.0, 0.0])
assert np.allclose(singular @ [1, -5], singular @ [1, 5])
```

## 6. Connect modes to experimental design

In a general linear system, singular vectors identify input and observation combinations with different strengths. A small singular value points to a direction that needs stronger measurements or additional assumptions. Compare candidate sensors by how they change the weak direction, and use data perturbations to distinguish numerical roundoff from measurement sensitivity. Record the noise level before deciding how much to trust a reconstructed mode.
