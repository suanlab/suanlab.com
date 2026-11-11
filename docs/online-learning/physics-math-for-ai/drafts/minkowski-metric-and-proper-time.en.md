## 1. A bilinear form with a time direction

In inertial coordinates $X=(ct,x,y,z)$ use $\eta=\operatorname{diag}(1,-1,-1,-1)$. The Minkowski product of two displacements is $U^\mathsf T\eta V$. It is symmetric, bilinear and nondegenerate. Its squared value on a displacement can be positive, zero or negative. A Lorentz transformation $\Lambda$ satisfies $\Lambda^\mathsf T\eta\Lambda=\eta$, so it preserves this product. Some references use the opposite overall sign. [Minkowski, Space and Time](https://www.gutenberg.org/files/66944/66944-h/66944-h.htm)

## 2. Read the sign of a separation

A nonzero displacement with squared interval $s^2>0$ is timelike, $s^2=0$ is null, and $s^2<0$ is spacelike. For example, in units where $c=1$, the displacements $(3,1)$, $(2,2)$ and $(1,3)$ have squared intervals $8$, $0$ and $-8$. A null displacement can be nonzero. The squared interval is neither positive definite nor a Euclidean distance.

## 3. Add the elapsed time along a path

For a smooth timelike trajectory with speed $v(t)$ in an inertial frame, the reading of an ideal clock increases according to
$$
d\tau=\sqrt{dt^2-d\mathbf{x}^2/c^2}
      =dt\sqrt{1-v(t)^2/c^2}.
$$
For each straight future-directed segment, multiply its coordinate duration by the factor under the square root. Add the resulting proper times across the segments. Null segments contribute zero in the limiting path-length formula. A spacelike segment cannot represent a clock trajectory. [Tong, Geodesics in Spacetime §1](https://davidtong.org/teaching/general-relativity/grhtml/S1)

## 4. Compare two routes between fixed events

Use $c=1$ and fix departure at $(0,0)$ and reunion at $(6,0)$. A stationary clock measures 6. A traveler passing through $(3,2)$ on two straight legs measures $2\sqrt{9-4}=2\sqrt5$. The endpoint interval is the same for both paths, while the elapsed proper time depends on the route. The idealized change of velocity at the turn can be approximated by a short smooth turnaround.

## 5. Why the straight route maximizes proper time

Take the rest frame of the timelike displacement from departure to reunion. Its endpoints have the same spatial coordinates. For every piecewise smooth timelike route the integrand $\sqrt{1-v^2/c^2}$ is at most one. Hence the proper time cannot exceed the endpoint coordinate duration in that frame. Equality requires zero speed except on sets of zero duration. This is the reverse triangle inequality for future-directed timelike segments.

In $1+1$ dimensions the future unit timelike hyperbola is $(\cosh\eta,\sinh\eta)$. The Minkowski product of two such vectors is $\cosh(\eta_1-\eta_2)\geq 1$. Their relative rapidity sets the hyperbolic angle. This explains the sign of the reverse inequality directly from the bilinear form.

## 6. Check the numbers

```python
from math import isclose, sqrt

def proper_time(*events):
    elapsed = 0
    for (ct0,x0), (ct1,x1) in zip(events,events[1:]):
        dt, dx = ct1-ct0, x1-x0
        if dt < abs(dx):
            raise ValueError('The leg is spacelike or past-directed')
        elapsed += sqrt((dt-dx)*(dt+dx))
    return elapsed

A, B = (0,0), (6,0)
assert proper_time(A, B) == 6
assert isclose(proper_time(A,(3,2),B), 2*sqrt(5))
assert proper_time(A,(3,3),B) == 0
for t in range(1,6):
    for x in range(-min(t,6-t),min(t,6-t)+1):
        assert proper_time(A,(t,x),B) <= proper_time(A,B)
print(proper_time(A,(3,2),B))
```

## 7. Inspect the experiment

- Keep departure and reunion fixed and move the turnaround. Compare elapsed time for the two routes.
- At a turnaround time of 3, move the turnaround position from 0 through 3. Inspect what happens as both legs approach the light cone.
- Move the turnaround outside the feasible region. Identify which segment becomes spacelike before interpreting a clock reading.
