## 1. Use coordinates to label events

Coordinates are labels, while an observable specifies a physical measurement. A coordinate time difference becomes a clock reading only after specifying a worldline and the metric along it. A vector component can also change with the basis while its contraction with an observer's frame describes a measurement. Accelerated coordinates in flat spacetime let us check these distinctions explicitly. [Tong, Geodesics in Spacetime §1](https://davidtong.org/teaching/general-relativity/grhtml/S1)

## 2. Construct an accelerated chart

Set $c=1$ and use the right wedge $X>|T|$ of Minkowski spacetime. The transformation
$$
T=\rho\sinh\eta,\qquad X=\rho\cosh\eta,\qquad \rho>0
$$
assigns constant $\rho$ to hyperbolic worldlines. The inverse is $\rho=\sqrt{X^2-T^2}$ and $\eta=\operatorname{artanh}(T/X)$. Differentiating the transformation gives
$$
-dT^2+dX^2=-\rho^2d\eta^2+d\rho^2.
$$
The metric component $g_{\eta\eta}=-\rho^2$ varies even though the curvature is zero. The right-wedge chart does not cover all Minkowski events.

## 3. Read an accelerated clock and accelerometer

Along a constant-$\rho$ worldline, the proper time between readings $\eta_1,\eta_2$ is $\rho(\eta_2-\eta_1)$. The inertial components of four-velocity are $(\cosh\eta,\sinh\eta)$. Differentiating with respect to proper time and taking the acceleration norm gives the accelerometer reading $a=1/\rho$.

For example, clocks at $\rho=1$ and $\rho=3$ accumulate one and three time units over the same unit change in $\eta$. Their respective proper accelerations are one and one third. The comparison uses different worldlines in the same flat spacetime.

## 4. Compare coordinate and local speeds

Along a light ray, $d\rho/d\eta=\pm\rho$. A static observer's local radial length increment is $d\rho$ and its local time increment is $\rho\,d\eta$. Their ratio gives the local light speed $\pm1$ everywhere in the wedge. A changing coordinate speed does not alter that local measurement. The chart boundary at $\rho=0$ is not a curvature singularity.

## 5. Verify an interval in both charts

For two events on the same accelerated worldline, at $\eta=0$ and $\eta=\Delta\eta>0$, the inertial straight path between them has proper time $2\rho\sinh(\Delta\eta/2)$. The accelerated clock instead reads $\rho\Delta\eta$. The two calculations describe different paths. The lab shows them alongside the coordinate labels.

```python
from math import atanh, cosh, isclose, sinh, sqrt
rho, eta = 3, 0.8
T, X = rho*sinh(eta), rho*cosh(eta)
assert isclose(sqrt((X-T)*(X+T)), rho)
assert isclose(atanh(T/X), eta)
accelerated = rho*eta
inertial = sqrt((T-X+rho)*(T+X-rho))
assert isclose(inertial, 2*rho*sinh(eta/2))
assert inertial > accelerated
assert rho/rho == 1
print(accelerated,inertial,1/rho)
```

## 6. Audit a reported quantity

- Record the coordinate chart, its domain and the worldline used by the measurement.
- Change $\rho$ at fixed $\Delta\eta$ and compare proper time, proper acceleration and coordinate light speed.
- Transform an event back and forth to verify its labels. Compute a physical reading from the metric and the chosen observer.
