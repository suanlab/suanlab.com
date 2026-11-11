## 1. Repeat a cell and perturb its atoms

A lattice describes periodic translations. A basis specifies the atoms in one cell. Here each cell of length $a$ contains masses $m_A,m_B$ at positions $na$ and $na+a/2$. Equal springs of stiffness $\kappa$ connect neighboring atoms. Small displacements give quadratic elastic energy $\kappa(\Delta u)^2/2$ per bond, so normal modes can be superposed. [MIT OCW, Lattice Waves in 1D with Diatomic Basis](https://ocw.mit.edu/courses/6-730-physics-for-solid-state-applications-spring-2003/resources/lecture_9b/)

## 2. Turn the repeated chain into a two-dimensional eigenproblem

Use a wave with wave number $k$ and phase $e^{ikx}$ at the equilibrium position of each atom. Let $q=ka$ and multiply the displacement amplitudes by the square roots of the masses. Newton's equations become a symmetric eigenproblem $D(q)v=\omega^2v$ with
$$
D(q)=\begin{pmatrix}
2\kappa/m_A & -2\kappa\cos(q/2)/\sqrt{m_Am_B}\\
-2\kappa\cos(q/2)/\sqrt{m_Am_B} & 2\kappa/m_B
\end{pmatrix}.
$$
The off-diagonal term is the sum of the phase factors for the two neighboring bonds. The mass weighting lets us use ordinary orthogonal eigenvectors.

## 3. Read acoustic and optical branches

At $q=0$ equal physical displacements stretch no spring. The acoustic eigenvalue is zero, a translation of the chain. The optical mode has opposite physical displacements with $m_Au_A+m_Bu_B=0$, and squared frequency $2\kappa(1/m_A+1/m_B)$. Close to zero the acoustic mode has $\omega\approx c_s|k|$ with $c_s=a\sqrt{\kappa/[2(m_A+m_B)]}$. Its slope gives the speed of long-wavelength sound in this model.

## 4. Check the zone edge and the equal-mass limit

At $q=\pi$, the neighbor phases cancel and the two squared frequencies are $2\kappa/m_A$ and $2\kappa/m_B$. Unequal masses therefore give separated branch edges. With equal masses, the two branches touch there. In that limit the true period becomes $a/2$, and representing it with a two-atom cell folds a single branch into the smaller Brillouin zone.

## 5. Quantize a nonzero normal mode

A nonzero-frequency harmonic mode has oscillator levels $\hbar\omega(n+\tfrac12)$. Adding one phonon raises its energy by $\hbar\omega$. A phonon labels collective motion; it does not belong to one atom. Treat the exactly uniform translation separately as center-of-mass motion. A finite periodic chain of $N$ cells has $N$ allowed wave numbers and two modes per wave number.

```python
from math import cos, isclose, pi, sqrt
import numpy as np

def squared_modes(q, a, b):
    coupling = -2*cos(q/2)/sqrt(a*b)
    return np.linalg.eigvalsh([[2/a,coupling],[coupling,2/b]])

assert np.allclose(squared_modes(0,1,4),[0,2.5])
assert np.allclose(squared_modes(pi,1,4),[0.5,2])
assert np.allclose(squared_modes(pi,1,1),[2,2])
q = .001
assert isclose(sqrt(squared_modes(q,1,4)[0])/q,sqrt(1/10),rel_tol=1e-6)
print(squared_modes(pi,1,4))
```

## 6. Inspect the dispersion

- Vary the mass ratio and compare the two frequencies at the zone edge.
- Compare the displacement directions of the two long-wavelength modes.
- Use the slope near $q=0$ to compare sound speeds for different masses.
- Explain why a doubled cell changes how branches are displayed without adding physical atoms.
