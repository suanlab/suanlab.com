## 1. From orbital levels to bands

A repeated chain gives each cell the same pair of local orbitals, A and B. A particle can move between orbitals through a hopping term in the Hamiltonian. In a finite ring, periodic boundaries select discrete Bloch wave numbers. As the ring grows, these wave numbers fill an interval and the allowed energies trace continuous bands. This model keeps one orbital per site, nearest-neighbor hopping, and independent spinless particles. [MIT OpenCourseWare, electronic band structure and tight binding](https://ocw.mit.edu/courses/6-730-physics-for-solid-state-applications-spring-2003/resources/lecture_12c/)

## 2. Compress the chain into a Bloch Hamiltonian

Let $a$ be the two-site cell length, $q=ka$, $t_1\ge0$ the A–B hopping inside a cell, and $t_2\ge0$ the hopping between cells. Choose on-site energies $+\Delta$ on A and $-\Delta$ on B after subtracting their mean. Factoring out the Bloch phase $e^{iqn}$ in cell $n$ leaves

$$
H(q)=\begin{pmatrix}
\Delta & t_1+t_2 e^{-iq}\\
t_1+t_2 e^{iq} & -\Delta
\end{pmatrix}.
$$

The intercell term receives a phase because it connects adjacent cells. The conjugate off-diagonal entries make $H$ Hermitian. When $\Delta=0$ this is the bulk Su–Schrieffer–Heeger Hamiltonian in the chosen cell convention. [Asbóth, Oroszlány and Pályi, Chapter 1](https://arxiv.org/abs/1509.02295)

## 3. Find the band edges and the gap

The trace vanishes and the determinant is $-\Delta^2-|t_1+t_2e^{-iq}|^2$. The two eigenvalues are

$$
E_\pm(q)=\pm\sqrt{\Delta^2+(t_1-t_2)^2+4t_1t_2\cos^2(q/2)}.
$$

At $q=0$, hopping contributions add. At $q=\pi$, the contributions differ. For positive hoppings, the upper band reaches its minimum and the lower band reaches its maximum at the zone boundary. The bulk energy gap is $E_g=2\sqrt{\Delta^2+(t_1-t_2)^2}$. If a hopping vanishes, both bands are flat and the same gap holds at every $q$.

For $t_1=1$, $t_2=0.6$ and $\Delta=0$, the zone-center levels are $\pm1.6$, and the boundary levels are $\pm0.4$. The gap is $0.8$. A second model with $t_1=t_2=1$ and $\Delta=0.4$ gives the same gap, while its zone-center energies differ. One gap measurement leaves both the hopping contrast and on-site difference undetermined.

## 4. Distinguish band folding and a gap opening

With $t_1=t_2=t$ and $\Delta=0$, translating the chain by $a/2$ preserves the Hamiltonian. A one-site cell gives the dispersion $2t\cos(ka/2)$. In the smaller zone for the two-site cell, the shifted portion becomes the second branch. They touch at the boundary. Alternating hoppings or unequal on-site energies remove the half-cell translation symmetry and separate the two bands.

Away from degeneracy the wave-packet group velocity is $v_\pm=(a/\hbar)dE_\pm/dq=\mp at_1t_2\sin(q)/(\hbar|E_\pm|)$. At a crossing, each energy-sorted branch has a cusp in the periodic zone, so report slopes on the chosen smooth continuation. A band gap separates occupied and empty states in this model when the lower band is filled and the upper band is empty. That interpretation depends on filling.

## 5. Verify against a Hermitian eigensolver

Run the following in Python with NumPy. Sweep the zone and compare the independent eigensolver to the analytical gap.

```python
import numpy as np

q = np.linspace(-np.pi, np.pi, 1001)

def bands(t1, t2, delta):
    coupling = t1 + t2 * np.exp(-1j * q)
    h = np.zeros((q.size, 2, 2), dtype=complex)
    h[:, 0, 0], h[:, 1, 1] = delta, -delta
    h[:, 0, 1], h[:, 1, 0] = coupling, coupling.conj()
    return np.linalg.eigvalsh(h)

for t1, t2, delta in [(1, .6, 0), (1, 1, .4), (1, 1, 0), (1, 0, .4)]:
    e = bands(t1, t2, delta)
    gap = e[:, 1].min() - e[:, 0].max()
    expected = 2 * np.hypot(delta, t1 - t2)
    assert np.isclose(gap, expected, atol=1e-12)
    assert np.allclose(e[:, 0], -e[:, 1])
    print(f"gap={gap:.3f}, center={e[q.size // 2, 1]:.3f}")

assert np.allclose(bands(1, .6, 0), bands(.6, 1, 0))
```

## 6. Design a measurement for an inverse problem

Predict which pairs of parameters can give the same dispersion. Exchanging $t_1$ and $t_2$ preserves every energy in this periodic model. The sublattice content of an eigenstate can carry additional information. For instance, at the zone boundary the upper-band A weight is $(1+\Delta/\sqrt{\Delta^2+(t_1-t_2)^2})/2$ when the eigenvalue is simple. Measuring energy and site-resolved weight is therefore more informative than fitting a single gap. When planning a data-driven fit, record the cell convention, measured momenta, and which observables distinguish competing Hamiltonians.
