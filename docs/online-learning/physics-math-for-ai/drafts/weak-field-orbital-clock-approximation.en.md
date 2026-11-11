## 1. Identify the small parameters

A weak-field slow-motion approximation expands in $|\Phi|/c^2$ and $v^2/c^2$, where $\Phi$ is the Newtonian potential and $v$ is the coordinate speed in a compatible weak-field chart. To first order, the clock rate is
$$
\frac{d\tau}{dt}\simeq 1+\frac{\Phi}{c^2}-\frac{v^2}{2c^2}.
$$
The omitted terms are second order in the small quantities. Their size matters when a model must meet a numerical tolerance.

## 2. Use an exact circular orbit as a benchmark

In the Schwarzschild exterior use $G=c=1$, and put $u=M/r$. For an equatorial circular geodesic, the radial equation gives $\Omega^2=(d\phi/dt)^2=M/r^3$. Substituting $dr=0$ into the metric gives
$$
\left(\frac{d\tau}{dt}\right)^2=1-2u-r^2\Omega^2=1-3u.
$$
Here $t$ is the Schwarzschild time normalized at infinity. The circular orbit is stable for $r>6M$ and marginal at $6M$, the lower limit of this experiment. [Carroll, The Schwarzschild Solution](https://ned.ipac.caltech.edu/level5/March01/Carroll3/Carroll7.html)

## 3. Separate local motion from the gravitational clock rate

At the same radius a static clock has rate $\sqrt{1-2u}$. It measures the circular particle's local speed squared as $u/(1-2u)$. Multiplying the static clock rate by the corresponding special-relativistic time-dilation factor gives $\sqrt{1-3u}$. In the weak field the gravitational and kinematic corrections have leading terms $-u$ and $-u/2$. Their sum gives $1-3u/2$.

## 4. Quantify the truncation

Set $a=3u$. Rationalizing the difference yields
$$
1-\frac a2-\sqrt{1-a}
 =\frac{a^2}{4\big(1-a/2+\sqrt{1-a}\big)}.
$$
The first-order formula overestimates the exact clock rate, and the error is quadratic for small $u$. At $r=12M$, the rates are about $0.875$ and $0.8660$. Decide whether the difference of about $0.009$ is acceptable for the intended comparison. A tolerance on a rate is dimensionless. To bound an accumulated clock error at fixed radius, multiply the rate error by the elapsed coordinate time.

```python
from math import isclose, sqrt

def rates(radius):
    u = 1/radius
    a = 3*u
    exact = sqrt(1-a)
    linear = 1-a/2
    error = a*a/(4*(linear+exact))
    static = sqrt(1-2*u)
    local_v_squared = u/(1-2*u)
    assert isclose(static*sqrt(1-local_v_squared),exact)
    assert isclose(linear-exact,error,rel_tol=1e-11)
    return exact, linear, error

for r in (6,12,48):
    print(r,rates(r))
assert rates(6)[2] > rates(12)[2] > rates(48)[2]
```

## 5. Decide where the approximation is useful

- Compare the error at $r=6M$, $12M$ and $48M$.
- Keep the required accuracy fixed and find a radius that satisfies it.
- Compare the static and orbiting clock rates at one radius before combining gravity and motion.
- State the assumptions about field strength, speed and orbit. Recheck those assumptions before applying the same expansion to a different system.
