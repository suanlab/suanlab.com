## 1. From finite fluctuations to criticality

A finite Ising lattice has a partition function that is a finite sum of positive weights at any finite positive temperature. Its free energy is smooth there. As the system grows, fluctuations can extend across more sites and a smooth peak can sharpen. A critical singularity concerns the thermodynamic limit. One tall peak in a finite system provides a size-dependent observation.

## 2. The reference point and the correlation length

Use the ferromagnetic square-lattice Ising model with horizontal coupling $K_x=J_x/T$, vertical coupling $K_y=J_y/T$, and zero field. The infinite lattice has its transition along

$$
\sinh(2K_x)\sinh(2K_y)=1.
$$

In the isotropic case this gives $K_c=\tfrac12\log(1+\sqrt2)\approx 0.440687$. Use this infinite-system result as an external reference when studying finite samples. In the one-dimensional nearest-neighbor chain the correlation length grows as the temperature tends to zero, and stays finite for every finite positive temperature. The spin symmetry is shared while the spatial dimension differs. [Onsager, Crystal Statistics I](https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117) · [Tong, Statistical Physics §5](https://davidtong.org/teaching/statistical-physics/statmechhtml/S5)

## 3. What universality compares

Write $t=(T-T_c)/T_c$. The critical correlation length scales as $\xi\sim |t|^{-\nu}$. The spontaneous magnetization on the ordered side behaves as $m_{\rm sp}\sim(-t)^\beta$, and the connected susceptibility as $\chi\sim|t|^{-\gamma}$. At criticality the connected correlation typically decays as $r^{-(d-2+\eta)}$.

Universality compares exponents and scaling functions after accounting for geometry and normalization conventions. The location of $T_c$ and the amplitudes may depend on microscopic couplings. The square-lattice Ising model has $\beta=1/8$, $\gamma=7/4$, $\nu=1$ and $\eta=1/4$. Mean-field theory instead gives $\beta=1/2$, $\gamma=1$, $\nu=1/2$ and $\eta=0$. Deriving a mean-field result shows what the approximation predicts. Comparing it with the exact two-dimensional values measures the importance of fluctuations. [Tong, Statistical Field Theory §§1.2 and 2.2](https://davidtong.org/teaching/statistical-field-theory/)

## 4. A dimensionless observable from the shape of the distribution

Use $M=\sum_i s_i$ and $N=L^2$. At zero field the finite ensemble has $\langle M\rangle=0$ by sign symmetry. Its distribution can still become concentrated near two opposite ordered values. The Binder ratio measures shape through the second and fourth moments.

$$
U_L=1-\frac{\langle M^4\rangle}{3\langle M^2\rangle^2}.
$$

For independent unbiased spins, $\langle M^2\rangle=N$ and $\langle M^4\rangle=3N^2-2N$, so $U_L=2/(3N)$. For a two-point distribution at $M=\pm N$, $U_L=2/3$. This explains why the ratio grows when the finite distribution becomes strongly bimodal. [Binder, Finite Size Scaling Analysis of Ising Model Block Distribution Functions](https://doi.org/10.1007/BF01293604)

## 5. How a crossing can drift

For a fixed aspect ratio and boundary condition, finite-size scaling near criticality predicts a leading curve $U_L(t)\approx f(tL^{1/\nu})$. At $t=0$, the leading term is independent of $L$. Corrections can shift the crossing of curves for two finite sizes away from the thermodynamic critical point. Compare several size pairs and state the geometry before interpreting a crossing.

The exercise uses a cylinder, with periodic bonds around a row and open boundaries in the other direction. It compares fixed square aspect ratios as size changes. The reported ratio is for the magnetization of the entire finite cylinder.

## 6. Compute the finite moments by adding rows

The row state $b$ contains $L$ spins and has magnetization $m_b$. Let $S_k(a)$ be the accumulated unnormalized $k$th moment for a lattice whose last row is $a$. Adding $b$ increases the magnetization by $m_b$. Expanding the resulting power gives

$$
S_k^{\rm new}(b)=\sum_a w_{ab}\sum_{j=0}^k {k\choose j} S_j(a)m_b^{k-j}.
$$

The transfer weight $w_{ab}$ contains the new horizontal bonds in $b$ and vertical bonds between $a$ and $b$. After summing over the last row, divide moments by $S_0$. Subtracting the maximum bond sum from each exponent introduces a common factor that cancels in normalized moments.

```python
from math import comb
import numpy as np

def binder(length, kx, ky):
    bits = np.arange(1 << length)[:, None] >> np.arange(length)
    spins = 2 * (bits & 1) - 1
    horizontal = (spins * np.roll(spins, -1, axis=1)).sum(axis=1)
    m = spins.sum(axis=1)
    row = np.exp(kx * (horizontal - length))
    transfer = np.exp(ky * (spins @ spins.T - length)) * row[None, :]
    moments = np.array([row * m**k for k in range(5)])
    for _ in range(length - 1):
        incoming = moments @ transfer
        moments = np.array([
            sum(comb(k, j) * incoming[j] * m**(k-j) for j in range(k+1))
            for k in range(5)
        ])
    z, _, m2, _, m4 = moments.sum(axis=1)
    return 1 - m4 * z / (3 * m2**2)

def crossing(left, right):
    lo, hi = 0.4, 0.6
    assert binder(left, lo, lo) > binder(right, lo, lo)
    assert binder(left, hi, hi) < binder(right, hi, hi)
    for _ in range(25):
        mid = (lo + hi) / 2
        if binder(left, mid, mid) > binder(right, mid, mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

for length in [3, 4, 6, 8]:
    np.testing.assert_allclose(binder(length, 0, 0), 2/(3*length**2), atol=1e-14, rtol=0)
print('infinite Kc', np.arcsinh(1)/2)
print('finite crossings', crossing(3, 4), crossing(6, 8))
```

## 7. Check a finite-size inference

- At $K=0$, reproduce $U_L=2/(3L^2)$ for every tested size. This checks normalization and the moment update.
- Compare the crossing for sizes $(3,4)$ with the crossing for $(6,8)$. Report the departure of both from the known isotropic $K_c$.
- Change the horizontal and vertical couplings together and compare against the exact critical line. A finite crossing also depends on aspect ratio, boundary conditions, and size corrections.
- If you use sampled data, estimate uncertainty with blocks long enough to account for time correlations. The row calculation avoids sampling error for the stated finite lattice.
