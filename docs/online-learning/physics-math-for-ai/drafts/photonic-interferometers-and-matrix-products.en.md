## 1. Encode data in coherent optical fields

Two waveguides carry complex field amplitudes $x=(x_1,x_2)^T$. Relative phase determines whether fields reinforce or cancel when they mix. For a common power normalization, the input power is proportional to $|x_1|^2+|x_2|^2$. A negative real amplitude represents a phase offset of $\pi$ from a positive amplitude. An ordinary detector measures intensity $|x_j|^2$ and loses that sign information. [Shen et al., Deep Learning with Coherent Nanophotonic Circuits](https://arxiv.org/abs/1610.02365)

## 2. Turn phase control into a two-channel rotation

Use ideal 50:50 couplers in the convention

$$
B=\frac{1}{\sqrt{2}}\begin{pmatrix}1&i\\i&1\end{pmatrix},
\qquad P(\theta)=\operatorname{diag}(e^{i\theta},e^{-i\theta}).
$$

A coupler, differential phase shift and inverse coupler implement

$$
B^\dagger P(\theta)B
=R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}.
$$

The phase difference between the two internal paths is $2\theta$. With input $(1,0)^T$, the output powers are $\cos^2\theta$ and $\sin^2\theta$. Increasing $\theta$ from 0 to $\pi/2$ transfers power between the ports while preserving their total in the lossless model.

## 3. Add a diagonal attenuation stage

A real matrix has singular value decomposition $M=U\Sigma V^T$. Orthogonal matrices in two dimensions can be built from rotations and, when needed, a sign flip implemented as a phase shift. A passive attenuator can realize singular values up to one as field-amplitude transmission coefficients. If the largest singular value exceeds one, program a scaled matrix and track the scale in the surrounding system. For $D=\operatorname{diag}(1,1/2)$,

$$
R(\pi/4)D R(-\pi/4)=\begin{pmatrix}3/4&1/4\\1/4&3/4\end{pmatrix}.
$$

An amplitude coefficient of $1/2$ transmits one-quarter of the power in that channel.

## 4. Compute a signed example by hand

The example matrix sends $x=(1,-1)^T$ to $y=(1/2,-1/2)^T$. The input power in the chosen normalization is 2, and the output power is $1/2$. Direct detection returns $(1/4,1/4)$ for both this input and its global negative. A coherent reference can distinguish the signs by turning phase differences into a measurable intensity difference.

## 5. Verify the circuit with NumPy

```python
import numpy as np

B = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)

def rotation(theta):
    phases = np.diag(np.exp(1j * np.array([theta, -theta])))
    return B.conj().T @ phases @ B

theta = np.pi / 4
R = rotation(theta)
D = np.diag([1, 0.5])
M = R @ D @ rotation(-theta)
x = np.array([1, -1])
y = M @ x
assert np.allclose(R.conj().T @ R, np.eye(2))
assert np.allclose(M, [[.75, .25], [.25, .75]])
assert np.allclose(y, [.5, -.5])
assert np.allclose(np.abs(y)**2, np.abs(M @ (-x))**2)
print(np.real_if_close(y), np.abs(y)**2)
```

## 6. Design an experiment that tests a programmed matrix

Use calibrated coherent inputs, measure amplitudes with a phase reference, and compare the reconstructed matrix with its programmed target. A power-only measurement of basis inputs can confirm transmission magnitudes. Superpositions expose relative phase errors that basis-input powers miss. Record the phase settings, insertion loss, and readout convention so an accuracy claim can be reproduced. Report optical core behavior and system costs for conversion, control, and readout when comparing implementations.
