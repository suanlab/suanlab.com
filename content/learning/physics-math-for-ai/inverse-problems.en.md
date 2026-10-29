## Observations do not always determine their causes

A forward model maps parameters to observations. An inverse problem asks which parameters could have produced the observations. A perfect fit can be nonunique, and a unique fit can still be extremely sensitive to noise. Distinguish existence, uniqueness and stability before interpreting a recovered model.

Use synthetic truth $\theta=(1,2)^T$ and

$$
X=\begin{pmatrix}1&1\\\varepsilon&-\varepsilon\end{pmatrix},\qquad
 y=X\theta+\begin{pmatrix}0\\\nu\end{pmatrix}
=\begin{pmatrix}3\\-\varepsilon+\nu\end{pmatrix}.
$$

Here $\varepsilon\ge0$ controls contrast sensitivity and $\nu$ is known synthetic noise in the second observation. It is not the finite-difference epsilon used in earlier units.

## Separate sum and contrast

Define $s=\theta_1+\theta_2$ and $d=\theta_1-\theta_2$. Observations measure $s$ directly and $\varepsilon d$ weakly. Without regularization and for $\varepsilon>0$, the reconstruction is $s=3$, $d=-1+\nu/\varepsilon$.

Thus parameter errors are $\nu/(2\varepsilon)$ and $-\nu/(2\varepsilon)$. The Euclidean error is $|\nu|/(\sqrt2\varepsilon)$. A smaller observation scale can greatly amplify the same noise.

At $\varepsilon=0.1,\nu=0.1$, observed contrast is zero and the reconstruction is $(1.5,1.5)$, with error $1/\sqrt2$. At scale 1 with the same noise, the estimate is $(1.05,1.95)$ and the error is ten times smaller.

## Analyze rank and conditioning

The two row directions are orthogonal, with singular values $\sqrt2$ and $\sqrt2\varepsilon$ within the lab’s range. The condition number is $1/\varepsilon$ when positive. When $\varepsilon=0$, the sum is identifiable but the contrast is not. Infinitely many parameter pairs yield the same noiseless observations.

The lab displays a minimum-norm solution $(1.5,1.5)$ in the rank-deficient unregularized case. This is an algorithmic selection, not evidence that the true parameters are equal. Reporting one numerical answer without its identifiability conditions hides an important limitation.

## Derive ridge in the informative coordinates

Minimize $\lVert X\theta-y\rVert^2/2+\lambda\lVert\theta\rVert^2/2$. Since $\lVert\theta\rVert^2=(s^2+d^2)/2$, the two scalar problems separate:

$$
 s_\lambda=\frac6{2+\lambda},\qquad
 d_\lambda=\frac{2\varepsilon(-\varepsilon+\nu)}{2\varepsilon^2+\lambda}.
$$

Recover $\theta_1=(s_\lambda+d_\lambda)/2$ and $\theta_2=(s_\lambda-d_\lambda)/2$. Ridge shrinks both coordinates, so it introduces bias as well as reducing sensitivity. If both contrast scale and penalty vanish, choose the minimum-norm contrast zero explicitly rather than dividing by zero.

## Observation design is a different intervention

Increasing contrast sensitivity changes the actual information in observations. Increasing regularization changes a preference in the objective. Both can stabilize a numerical answer but they are scientifically different actions. A new independent sensor may resolve ambiguity that no amount of optimization can resolve from the old data.

The chart varies the noise while holding the current observation scale and penalty fixed. Blue shows the resulting estimate of the first parameter; orange marks its true value 1. The selected noise affects the readout. The graph is a sensitivity sweep, not a probability distribution over noise or parameters.

## From this example to scientific AI

Inverse imaging, parameter identification and differentiable simulation share these issues. A nonlinear inverse map may have local weak directions in its Jacobian and multiple distant solutions. A small residual alone neither proves a correct physical law nor supplies calibrated uncertainty.

A complete study should test synthetic recovery with known truth, perturb observations, examine singular directions, compare prior or penalty sensitivity and evaluate new held-out measurements. Real model discrepancy, correlated noise and misspecified boundary conditions need additional terms; they are intentionally absent here.

## Exercises and research checks

1. At scale 0.5 and noise 0.1 without ridge, find the estimate. **Answer:** $(1.1,1.9)$.
2. At zero contrast, list two distinct solutions with sum 3.
3. Explain why reducing training residual cannot fix structural nonidentifiability.
4. Propose a new observation that independently measures the missing contrast, then predict how its noise scale affects recovery.

The algebra follows directly from the SVD and ridge units. For statistical regression and estimation context, see the authors’ [Introduction to Statistical Learning](https://www.statlearning.com/). This synthetic linear problem is not a reproduced inverse-physics benchmark.
