## 1. Specify the measurement

A clock records its proper time. A receiver counts the light wave crests arriving per unit of that proper time. For a future photon wave vector $k$ and an observer with four-velocity $u$, use signature $(-,+,+,+)$ and measure angular frequency $\omega=-g(u,k)$. This contraction is a scalar. Changing coordinates does not change the reading of the same observer, while changing the observer can change it.

## 2. Compare static clocks outside a spherical mass

Use $G=c=1$, write the central mass as $M$, and let $f(r)=1-2M/r$ in the Schwarzschild exterior $r>2M$. A static clock has $d\tau=\sqrt{f(r)}\,dt$. The time-translation symmetry conserves $-k_t$ along a photon ray, so a static observer measures $\omega=-k_t/\sqrt{f(r)}$. For static emission at $r_e$ and static reception at $r_o$,
$$
\frac{\omega_o}{\omega_e}=\sqrt{\frac{f(r_e)}{f(r_o)}}.
$$
If $r_o>r_e$, the ratio is smaller than one. This comparison refers to worldlines that hold their radii fixed. [Philipp et al., Redshift and frequency comparison in Schwarzschild spacetime](https://arxiv.org/abs/1711.01237)

## 3. Separate the emitter’s local motion

Take an outward radial photon and let the emitter have local speed $\beta$ in that same direction, measured by the static observer at $r_e$. The emitted frequency in the emitter's rest frame is $\gamma(1-\beta)$ times the local static frequency. The received-to-emitted ratio is therefore
$$
\frac{\omega_o}{\omega_e}=
\sqrt{\frac{f(r_e)}{f(r_o)}}\sqrt{\frac{1+\beta}{1-\beta}} .
$$
The first factor comes from comparing static observers at the two radii. The second is the local Doppler factor at emission. A sufficiently large outward speed can offset the gravitational redshift. Define $z=\omega_e/\omega_o-1$, which is positive for redshift.

## 4. Calculate a cancellation case

Set $r_e=4M$, $r_o=12M$. The static frequency ratio is $\sqrt{3/5}$. The ratio becomes one at $\beta=1/4$, since the squared Doppler factor is then $5/3$. At larger positive speeds it exceeds one. This calculation uses a radial ray reaching the receiver, with a static receiver and an emitter momentarily moving at the prescribed speed.

```python
from math import isclose, sqrt

def observed_ratio(emitter, receiver, beta):
    f = lambda r: 1 - 2/r
    return sqrt(f(emitter)/f(receiver) * (1+beta)/(1-beta))

assert observed_ratio(4,12,0) < 1
assert isclose(observed_ratio(4,12,.25), 1)
assert observed_ratio(4,12,.5) > 1
assert observed_ratio(4,12,-.25) < observed_ratio(4,12,0)
print(observed_ratio(4,12,0), observed_ratio(4,12,.25))
```

## 5. Inspect a frequency comparison

- Increase the receiver radius while keeping the emitter static. Inspect the ratio and the sign of $z$.
- At the same two radii, adjust the emitter speed until the factors cancel.
- For an inward-moving emitter, keep the photon outward and check the sign of the Doppler correction.
- State which observer measures each frequency before comparing your result to a redshift convention.
