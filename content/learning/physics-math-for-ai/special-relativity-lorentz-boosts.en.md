## 1. Compare coordinates for the same event

An event is a point in space and time, such as a flash at the front of a train. An inertial frame assigns it a spatial coordinate and a clock reading from synchronized clocks at rest in that frame. Two frames can assign different readings to the same event. The transformations below relate those coordinates for frames in uniform relative motion. [Einstein, On the Electrodynamics of Moving Bodies, §§1–3](https://courses.physics.ucsd.edu/2016/Winter/physics2d/einstein1905.pdf)

## 2. Fix the sign and units of a boost

Use coordinates $(ct,x)$ so both entries have length units. The frame $S'$ moves in the $+x$ direction at speed $v$ relative to $S$, and origins coincide at $t=0$. With $\beta=v/c$ and $\gamma=(1-\beta^2)^{-1/2}$,
$$
ct'=\gamma(ct-\beta x),\qquad x'=\gamma(x-\beta ct).
$$
The line $x=\beta ct$ becomes $x'=0$, as expected for the moving origin. Apply a boost with $-\beta$ to recover the original coordinates. Directions perpendicular to the boost remain unchanged.

## 3. Check light and the invariant interval

Substituting $x=ct$ gives $x'=ct'$, and substituting $x=-ct$ gives $x'=-ct'$. The coordinate speed of light has magnitude $c$ in each frame. For any two events, direct expansion gives
$$
(\Delta ct')^2-(\Delta x')^2=(\Delta ct)^2-(\Delta x)^2.
$$
The mixed terms cancel. The preserved difference distinguishes a boost from a rotation with an ordinary Euclidean dot product.

## 4. Compare simultaneity, clocks and rods

For two events with $\Delta ct=0$ and $\Delta x=3$ light-seconds at $\beta=0.6$, the time difference is $\Delta t'=-2.25$ seconds. Their coordinate ordering changes between frames even when both use Einstein synchronization. For a clock at rest in $S'$, two ticks satisfy $\Delta x'=0$, and the inverse transformation gives $\Delta t=\gamma\Delta t'$.

To measure a moving rod in $S$, record both endpoint positions at the same $t$. Their position difference in the rest frame is $\Delta x'=\gamma\Delta x$. The rod’s rest length is that fixed rest-frame separation, and the moving-frame measurement gives the rest length divided by $\gamma$. The two endpoint events used in this measurement have different times in the rod frame.

## 5. Add collinear boosts with rapidity

Rapidity $\eta$ is defined by $\beta=\tanh\eta$. Then $\gamma=\cosh\eta$ and $\gamma\beta=\sinh\eta$. Writing light-cone coordinates $u=ct+x$ and $w=ct-x$ gives
$$
u'=e^{-\eta}u,\qquad w'=e^\eta w.
$$
The product $uw$ is invariant. Two successive collinear boosts add their rapidities. Their speed parameters combine as $(\beta_1+\beta_2)/(1+\beta_1\beta_2)$.

## 6. Check the calculations

```python
from math import isclose, sqrt

def boost(ct, x, beta):
    gamma = 1/sqrt((1-beta)*(1+beta))
    return gamma*(ct-beta*x), gamma*(x-beta*ct)

beta = 0.6
assert isclose(boost(5, 3, beta)[1], 0, abs_tol=1e-14)
assert isclose(boost(0, 3, beta)[0], -2.25)
assert isclose(boost(1, 1, beta)[0], boost(1, 1, beta)[1])

for ct, x in [(2, 1), (1, 2), (2, -2)]:
    transformed = boost(ct, x, beta)
    ctp, xp = transformed
    assert isclose(ctp**2-xp**2, ct**2-x**2, abs_tol=1e-14)
    assert all(isclose(p,q,abs_tol=1e-14) for p,q in zip(boost(*transformed,-beta),(ct,x)))
print(boost(0, 3, beta))
```

## 7. Inspect a measurement

- Move along the line $x=\beta ct$ and verify that every event remains at $x'=0$.
- Compare $\Delta t'$ for simultaneous separated events at positive and negative $\beta$.
- Use two events on the same clock to measure elapsed proper time. Use a pair simultaneous in the measuring frame to measure a rod.
- Apply a boost and its inverse to the same dataset and compare the invariant before rounding any displayed values.
