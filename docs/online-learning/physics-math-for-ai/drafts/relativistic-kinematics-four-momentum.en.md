## 1. Package energy and momentum into one vector

Use $c=1$ and the metric with positive time sign. For a massive particle of rest mass $m$ moving with velocity $\mathbf v$, the four-momentum is $p=(E,\mathbf p)=(\gamma m,\gamma m\mathbf v)$. This is mass times four-velocity. Its squared Minkowski norm gives the mass-shell condition
$$
E^2-|\mathbf p|^2=m^2.
$$
Take the positive-energy branch. A photon has $m=0$ and $E=|\mathbf p|$ even though it has no rest frame. At low speed the massive particle's kinetic energy approaches $m|\mathbf v|^2/2$.

## 2. Transform a collision as a whole

Energy and momentum add component by component. If a process conserves both, its initial and final total four-momenta are equal. Write the total as $P=(E_{\rm tot},\mathbf p_{\rm tot})$. For a timelike total momentum there is a center-of-momentum frame, with velocity $\mathbf p_{\rm tot}/E_{\rm tot}$ relative to the lab. In that frame the spatial total vanishes. The energy available there is
$$
W=\sqrt{E_{\rm tot}^2-|\mathbf p_{\rm tot}|^2}.
$$
Forces and interactions affect which outcomes occur. Four-momentum conservation limits which outcomes are kinematically possible. [Particle Data Group, Kinematics §49.2](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf)

## 3. Compare a fixed target with opposing beams

For a projectile of energy $E_1$ and mass $m_1$ striking a stationary target of mass $m_2$, use the mass-shell equation to simplify
$$
W^2=m_1^2+m_2^2+2m_2 E_1.
$$
Some lab energy goes into the collective motion of the outgoing system. With opposing beams whose momenta cancel, $W=E_{\rm tot}$. At equal total lab energy, the two arrangements can therefore give different available energy.

## 4. Find a kinematic threshold

If a proposed final state contains particles with masses $m_{f,i}$, its center-of-momentum energy must be at least $\sum_i m_{f,i}$. At threshold all final particles can be at rest in that frame. This criterion tests energy and momentum only. Other conservation laws and the interaction can still exclude the process.

In a toy example $m_1=m_2=1$, and the final mass sum is 3. A fixed target requires $E_1\geq 3.5$, or projectile kinetic energy at least 2.5. With two opposing beams of equal mass and total kinetic energy 2.5, total momentum is zero and $W=4.5$. The difference comes from the initial total momentum.

## 5. Check the transformation and threshold numerically

```python
from math import isclose, sqrt

m1 = m2 = 1
target_mass = 3
threshold_E = (target_mass**2 - m1**2 - m2**2)/(2*m2)
p1 = sqrt((threshold_E-m1)*(threshold_E+m1))
total = threshold_E + m2
beta = p1/total
gamma = 1/sqrt((1-beta)*(1+beta))
W = sqrt((total-p1)*(total+p1))
assert isclose(threshold_E-m1,2.5)
assert isclose(W,3)
assert isclose(gamma*(total-beta*p1), W)
assert isclose(gamma*(p1-beta*total),0,abs_tol=1e-14)
print(W, total)
```

## 6. Inspect a new collision

- Record masses, lab energies, and signed momenta separately.
- Calculate $W$ before using lab energy to judge a threshold.
- Boost to the center-of-momentum frame and check that total momentum is zero.
- Compare two arrangements at the same total kinetic energy and explain which has more energy available in the center-of-momentum frame.
