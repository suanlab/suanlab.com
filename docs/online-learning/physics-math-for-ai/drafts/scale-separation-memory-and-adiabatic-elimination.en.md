## 1. Separate relaxation times

Two variables can evolve on very different time scales. A reduced model may replace the fast variable by a function of the slow one. Test the reduction by comparing trajectories, initial conditions and decay rates. The ratio of time scales is part of the model, and the observation window determines which discrepancies matter.

## 2. Remove a fast variable without approximation

Take $\dot x=-x+y$ and $\dot y=D(\gamma x-y)$ with $D>1$ and $0\leq\gamma<1$. The trace is negative and the determinant is $D(1-\gamma)>0$, so both modes decay. Solve the second equation by an integrating factor:
$$
y(t)=e^{-Dt}y_0+\gamma D\int_0^t e^{-D(t-s)}x(s)\,ds .
$$
Substituting into the first equation gives a slow-variable equation with memory of earlier $x$ and a term from the initial hidden state. Two systems with the same $x_0$ and different $y_0$ have different $\dot x(0)$. This linear example illustrates how projecting dynamics can generate memory. [Mori, Transport, Collective Motion, and Brownian Motion](https://doi.org/10.1143/PTP.33.423)

## 3. Replace the memory integral by an instantaneous term

For large $D$, the exponential kernel has width $1/D$. After the initial layer, if $x$ changes little over that width, set $y\approx\gamma x$ and $\dot x\approx-(1-\gamma)x$. A quantitative check follows by differentiating $y-\gamma x$:
$$
|y(t)-\gamma x(t)|
\leq e^{-Dt}|y_0-\gamma x_0|
 +\frac{\gamma L}{D}(1-e^{-Dt}),\qquad L=\sup_{0\le s\le t}|\dot x(s)|.
$$
The first term measures the initial mismatch. The second measures how much the slow variable changes during the fast response. Even if $y_0=\gamma x_0$, lag develops when $x$ changes.

## 4. Compare time scales of the full system

The decay eigenvalues are
$$
\lambda_\pm=-\frac{1+D}{2}\pm\frac12\sqrt{(D-1)^2+4\gamma D}.
$$
Use $\lambda_+=-2D(1-\gamma)/(1+D+\sqrt{(D-1)^2+4\gamma D})$ to compute the slow rate without subtracting two nearly equal numbers. Compare $\lambda_+$ with the reduced rate $-(1-\gamma)$ and inspect the trajectory after several fast time constants. When the rates are close to one another, neither mode is quickly forgotten.

## 5. Test the approximation on trajectories

```python
from math import exp, sqrt

def slow_state(t, D, gamma, y0):
    gap = sqrt((D-1)**2 + 4*gamma*D)
    fast = -(1+D+gap)/2
    slow = D*(1-gamma)/fast
    weight_slow = (y0-1-fast)/gap
    return weight_slow*exp(slow*t) + (1-weight_slow)*exp(fast*t)

gamma = 0.6
times = [i/100 for i in range(501)]
def error(D, y0):
    return max(abs(slow_state(t,D,gamma,y0) - exp(-(1-gamma)*t)) for t in times)

assert slow_state(0, 4, gamma, 0) == 1
assert error(32,gamma) < error(4,gamma)
for D in [2,4,8,32]:
    print(D, error(D,gamma), error(D,0))
```

## 6. Choose what to validate

- Start two full trajectories at the same $x_0$ and different $y_0$. Compare initial slopes and late-time behavior.
- Increase $D$ and test whether the approximation improves at the times used by the application.
- Report a trajectory error as well as a decay-rate error. Agreement between rates does not fix amplitudes.
- In a learned time-step model, test whether recent observations improve prediction when hidden degrees of freedom have yet to relax.
