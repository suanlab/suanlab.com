## Estimate a moving state, not only the latest sensor value

A noisy position sensor does not directly measure velocity. A dynamical model connects observations over time, while uncertainty describes what the filter does not know. Kalman filtering propagates a Gaussian state distribution and conditions it on each available measurement. Its accuracy statements depend on the model, prior and noise assumptions; the visible simulator truth is not supplied to the filter.

The lab starts at mean zero with covariance p0 I and draws the simulator's initial state from that prior. There is no measurement update at time zero. At steps one through N, it predicts, generates a position observation and updates if that observation is available.

## A discrete random-acceleration model

Use position and velocity state with interval dt:

$$
A=\begin{pmatrix}1&dt\\0&1\end{pmatrix},\quad
G=\begin{pmatrix}dt^2/2\\dt\end{pmatrix},\quad
x_t=Ax_{t-1}+G a_t,\quad a_t\sim N(0,q).
$$

Observations are y_t=(1,0)x_t+v_t with independent v_t~N(0,R). Thus Q=qGGᵀ. Acceleration is drawn once per interval in this discrete model. It is not continuous Brownian acceleration, whose integrated covariance has different powers and coefficients. The controls use the same q,R in the simulator and filter; mismatched-model estimation is a separate problem.

## Prediction and Gaussian conditioning

For previous mean m and covariance P,

$$
m^-=Am,\qquad P^-=APA^T+Q,
$$

$$
S=CP^-C^T+R,\quad K=P^-C^T/S,\quad
m^+=m^-+K(y-Cm^-).
$$

C=(1,0). The covariance update uses Joseph form:

$$
P^+=(I-KC)P^-(I-KC)^T+KRK^T.
$$

With this Kalman gain, it equals the Gaussian conditional covariance in exact arithmetic. The formulation expresses the update as a sum of covariance terms rather than relying on cancellation alone. A velocity correction can occur even without a velocity sensor because predicted position and velocity errors are correlated.

## Missing measurements preserve uncertainty

On a dropped step, m+=m- and P+=P-. The process and potential sensor noise draws are still generated in the same order. Changing the drop interval therefore changes what the filter sees without changing the simulated physical trajectory or its potential measurements.

Every available update reduces uncertainty relative to that step's predicted covariance. Posterior uncertainty across different times need not decrease because dynamics and process noise intervene. Realized squared error need not decrease either; it is one random path, not the covariance itself.

## Research extension: check calibration across a whole trajectory

For the correctly specified linear Gaussian model, innovation $\nu_t=y_t-H\hat x_t^-$ has conditional variance $S_t$. Standardized innovations $\nu_t/\sqrt{S_t}$ are independent standard normals under a fixed observation schedule and known parameters. Thus normalized innovation squared has expectation one; an individual trajectory need not average to one. Persistent correlation suggests a mismatched motion/noise model, not merely a poor random seed.

The innovation log likelihood is $-\tfrac12\sum_t[\log(2\pi S_t)+\nu_t^2/S_t]$. Fitting parameters on the same innovations changes the reference distribution. Data-dependent missingness also needs its own analysis.

A 95% ellipse at each time is not a 95% simultaneous trajectory statement. For $M$ two-dimensional Gaussian errors, the union bound gives simultaneous coverage at least $1-\delta$ using squared Mahalanobis threshold $h=-2\log(\delta/M)$, without assuming errors at different times are independent.

| $M$ | Threshold for $\delta=0.05$ |
|---|---:|
| 1 | $2\log20$ |
| 4 | $2\log80$ |
| 24 | $2\log480$ |

```python
from math import log,exp,isclose
delta=.05
for M in [1,4,24]:
    h=-2*log(delta/M)
    marginal_tail=exp(-h/2) # Exact survival function for chi-square(2).
    assert isclose(M*marginal_tail,delta,rel_tol=1e-14)
    print(M,h,1-M*marginal_tail)
```

The code checks the exact chi-square-two tail and its union bound. This auxiliary certificate does not relabel the native pointwise ellipse as simultaneous.

## Worked exercise and solution

Choose q=0,R=1,dt=1,p0=1 and one step. Compute covariance without using the realized measurement.

**Solution.** P-=[[2,1],[1,1]], S=3 and K=(2/3,1/3)ᵀ. The posterior covariance is [[2/3,1/3],[1/3,2/3]]. The position variance becomes 2/3, independently of the particular observed y. The posterior mean does depend on y. With no observations for four steps instead, P=A⁴I(A⁴)ᵀ=[[17,4],[4,1]]. Without new observations, this model therefore gains position uncertainty even when no process noise is added.

## Read joint coverage correctly

The native diagram uses estimation-error coordinates. Its orange contour satisfies

$$
e^TP^{-1}e=-2\log(0.05)\approx5.991.
$$

The whitened squared radius has distribution P(radius²≤h)=1-exp(-h/2). For a two-dimensional Gaussian this is therefore a pointwise 95% joint contour, with radius about 2.448 in whitened coordinates. Radius two would give 1-exp(-2), about 86.5%, rather than 95%. A one-dimensional two-sigma interval is another quantity. The blue realized error may lie outside the contour; no single realization, or simultaneous inclusion over all times, is guaranteed.

## Connection to AI and robotics

State estimators separate a sensor stream from uncertainty about an unobserved dynamical state. Learned filters need assumptions about transition and sensor distributions, and calibration should be checked separately from mean prediction error. This lab computes matched linear-Gaussian filtering; nonlinear filters, outliers, unknown noise models and data association require additional methods. Its covariance recursion is a model-conditioned calculation, not a universal error bound.

## Primary reference

MIT, [Dynamic Systems and Control, 6.241J open textbook](https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/resources/mit6_24js11_coursetextbook_pdf/): state estimation and least-squares foundations. The random-acceleration example, matched simulator, covariance calculations and joint-coverage exercise here are original constructions derived from Gaussian conditioning.
