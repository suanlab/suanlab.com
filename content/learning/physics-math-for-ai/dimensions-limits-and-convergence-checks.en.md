## 1. Choose an exactly solvable reference

A unit-mass harmonic oscillator satisfies $\ddot x=-\omega^2x$. Use $x(0)=1$ in a chosen length unit and $\dot x(0)=0$. Its exact solution is $x(t)=\cos(\omega t)$ and $v(t)=-\omega\sin(\omega t)$. This gives a trajectory, a conserved energy and a controlled time scale for checking a simulator. [Hairer, Important Aspects of Geometric Numerical Integration](https://access.archive-ouverte.unige.ch/access/metadata/b8e62cf8-dac5-4a29-880a-7e34961ecffb/download)

## 2. Check dimensions before comparing numbers

Introduce $\tau=\omega t$ and $p=v/\omega$, so $x$ and $p$ have the same unit. The equations become $dx/d\tau=p$ and $dp/d\tau=-x$. A physical time step $\Delta t$ becomes dimensionless $h=\omega\Delta t$. When changing the time unit, transform both the numeric frequency and time step so their product stays fixed.

## 3. Derive the numerical energy change

Explicit Euler updates both variables from the old state:

$$
x_{n+1}=x_n+hp_n,\qquad p_{n+1}=p_n-hx_n.
$$

Let $E_n=(x_n^2+p_n^2)/2$. Expanding the squares cancels the cross terms and gives $E_{n+1}=(1+h^2)E_n$. The energy grows for every nonzero step size. After $N$ steps its ratio is $(1+h^2)^N$.

## 4. Check convergence at a fixed physical horizon

Set the final dimensionless time to $T=4$ and use $h=T/N$. Then $E_N/E_0=(1+T^2/N^2)^N\to1$ as $N\to\infty$. Keeping $N$ fixed while changing $h$ also changes the final time and answers a different question. Compare both energy and trajectory errors at the same horizon.

## 5. Verify the identity independently

```python
from math import cos, hypot, isclose, sin

T = 4.
previous_error = None
for steps in (20, 40, 80, 160):
    h = T/steps
    x, p = 1., 0.
    for _ in range(steps):
        x, p = x+h*p, p-h*x
    energy_ratio = x*x+p*p
    assert isclose(energy_ratio, (1+h*h)**steps, rel_tol=1e-12)
    error = hypot(x-cos(T), p+sin(T))
    if previous_error is not None:
        assert error < previous_error
    previous_error = error
    print(steps, energy_ratio, error)
```

## 6. Build a validation matrix

Record the initial conditions, units, conserved quantities, exactly solvable limits, and refinement schedule. Check that unexpected drift is a property of the chosen method rather than a unit or update-order bug. For a learned simulator, compare long-horizon invariants, state errors and out-of-training step sizes under the same physical conditions.
