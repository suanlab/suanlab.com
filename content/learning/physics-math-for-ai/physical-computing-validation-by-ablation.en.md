## 1. Specify the expected behavior and how to read it

Suppose a two-port analog accelerator is programmed to produce a field or voltage vector $y=Mx$ with

$$
M=\begin{pmatrix}0.75&0.25\\0.25&0.75\end{pmatrix}.
$$

State the input range, the signed readout convention, and a tolerance in the same units as $y$. A coherent optical system requires a phase reference to reconstruct signed real amplitudes. Use a digital reference for each input. Log preprocessing, conversions, and synchronization separately when comparing latency or energy. [Shen et al., coherent nanophotonic circuit experiment](https://arxiv.org/abs/1610.02365)

## 2. Calibrate on the basis

The outputs for $e_1=(1,0)^T$ and $e_2=(0,1)^T$ are the columns of $M$. If the device scales both channels by a constant gain $g$, basis calibration recovers the effective matrix $gM$. It predicts subsequent outputs when the device is linear. Keep the programmed matrix and the fitted matrix so the absolute error and the residual after calibration can both be reported.

## 3. Hold out constructive and canceling inputs

Now suppose the readout clips each channel above magnitude $L=0.8$. With $g=1$, basis outputs have magnitudes at most 0.75, so basis calibration still returns $M$. The held-out input $e_1+e_2$ gives target $(1,1)^T$ and observed $(0.8,0.8)^T$. The held-out input $e_1-e_2$ gives $(0.5,-0.5)^T$ and stays within the readout range.

Compute the superposition residual

$$
r=\widetilde y(x_1+x_2)-\widetilde y(x_1)-\widetilde y(x_2).
$$

For a linear device it is zero up to measurement error. For the clipping example on the basis sum it is $(-0.2,-0.2)^T$.

## 4. Change one condition to test an explanation

Repeat the superposition at half amplitude. The clipping model predicts no residual because each output is within range. A fixed linear gain error also gives a zero superposition residual, but the deviation from the programmed matrix scales with input. Changing the readout range at the same input provides an additional test for clipping. Repeat under the same temperature and timing conditions to estimate measurement variability.

## 5. Verify calibration and held-out tests in Python

```python
import numpy as np

target = np.array([[.75, .25], [.25, .75]])
basis = np.eye(2)

def measure(x, gain=1.0, limit=0.8):
    return np.clip(gain * (target @ x), -limit, limit)

fitted = np.column_stack([measure(x) for x in basis])
assert np.allclose(fitted, target)
for x in (np.array([1, 1]), np.array([1, -1]), np.array([.5, .5])):
    observed = measure(x)
    residual = observed - fitted @ x
    print(x, observed, residual)
assert np.allclose(measure([1, 1]) - fitted @ [1, 1], [-.2, -.2])
assert np.allclose(measure([1, -1]), fitted @ [1, -1])
assert np.allclose(measure([.5, .5]), fitted @ [.5, .5])
```

## 6. Report validation at the task level

Choose held-out vectors that represent actual inputs and edge cases. Report both residuals and downstream task quality, along with the required calibration frequency and the energy and timing boundaries. Keep the intended operation count, measured data movement, and physical measurement linked to the same experiment. These records let another group check which part of a result comes from the algorithm, the device, or the readout.
