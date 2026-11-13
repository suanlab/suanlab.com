## 1. A probability chart can hide a different state

Imagine two preparations that each give outcomes0 and1 with equal frequency. A single chart makes them look identical. Quantum amplitudes contain additional relative-phase information: another orthonormal measurement basis can distinguish the preparations perfectly. We will derive that statement with two complex coordinates, without treating amplitudes themselves as probabilities.

Keep three objects separate throughout: the abstract state, its coordinate vector in a named basis, and the probabilities associated with a specified measurement. A coordinate change is a change of description. Changing which physical measurement is performed is a different experimental choice.

## 2. Complex inner products provide the geometry

A Hilbert space is a complex inner-product space complete in its induced norm. Completeness means a norm-Cauchy sequence has a limit inside the space. Finite-dimensional complex inner-product spaces are already complete; function spaces require additional care. The present computation uses C², while the next chapter examines an infinite-dimensional example.

We use the physics convention: the first slot is conjugate-linear and the second is linear. Thus a bra is a conjugate transpose, not just a transpose.

$$
\langle z,w\rangle=\sum_j\overline{z_j}w_j,\qquad \langle az,w\rangle=\overline a\langle z,w\rangle,\qquad \langle z,aw\rangle=a\langle z,w\rangle.
$$

For z=(1,i), the bra is (1,−i), so its squared norm is2. Omitting the conjugate would produce1+i²=0 for a nonzero vector and would destroy positive definiteness. This is an algebraic error with observable consequences.

## 3. Normalize a nonzero representative

Raw coordinates specify a direction before they specify a unit vector. Dividing by the positive norm yields a normalized representative. The zero vector cannot be normalized and is rejected; adding a small number would create an unrequested preparation.

$$
N=z^\dagger z>0,\qquad |\psi\rangle=\frac{z}{\sqrt N},\qquad \langle\psi|\psi\rangle=1.
$$

In a named orthonormal basis, the squared modulus of each normalized coordinate gives its projective-outcome probability. The sum is one because of orthonormality. The coordinate can be negative or imaginary; the probability cannot. General measurements and mixed states will be treated separately.

## 4. A pure physical state is a ray

A ray identifies nonzero vectors differing by a nonzero complex factor. Normalization removes its magnitude while leaving an overall unit phase. This section concerns pure states; not every quantum preparation or subsystem has a pure-state description.

$$
z\sim cz\quad(c\ne0),\qquad |\psi\rangle\sim e^{i\gamma}|\psi\rangle.
$$

Multiplying every amplitude by i gives the same ray. Multiplying only one amplitude by i generally changes the ray. The distinction is between overall and relative phase, not between real and imaginary numbers. The basic definitions follow the [Preskill foundations notes](https://www.preskill.caltech.edu/ph219/chap2_15.pdf); the examples below are explicit calculations for this preparation family.

## 5. Relative phase enters an interference probability

Let |+⟩=(|0⟩+|1⟩)/√2. Projecting onto this normalized vector adds complex amplitudes before taking the squared modulus. The cross term retains relative-phase information.

$$
p_+=\frac{|z_0+z_1|^2}{2N}=\frac12+\frac{\operatorname{Re}(\overline{z_0}z_1)}N.
$$

For z=(1,1), p+=1; for z=(1,−1), p+=0. Both preparations have computational-basis probabilities(1/2,1/2). A global phase multiplies the numerator amplitude by that same phase and leaves its squared modulus unchanged. Averaging the two probabilities first would miss interference.

## 6. Project onto a complete orthonormal basis

Choose a nonzero pair(u,v), with B=|u|²+|v|². The following columns form an orthonormal basis. Compute its coordinates by applying U†, not U, to the normalized state.

$$
U=\frac1{\sqrt B}\begin{pmatrix}u&-\overline v\\v&\overline u\end{pmatrix},\qquad U^\dagger U=I,\qquad \beta=U^\dagger\frac z{\sqrt N}.
$$

Orthogonality follows because the cross inner product cancels exactly. Reconstruction Uβ returns the original representative. If this is a passive change of coordinates, all other objects describing the same experiment must transform consistently. If the columns instead specify a different measurement performed on the same prepared state, their outcome probabilities can differ from those in the computational basis.

## 7. Compare amplitude overlap with fidelity

The inner product of two normalized states is a complex amplitude. Its phase depends on the representatives. We use the squared convention for pure-state fidelity:

$$
F(\psi,\phi)=|\langle\psi|\phi\rangle|^2=\frac{|z^\dagger w|^2}{(z^\dagger z)(w^\dagger w)},\qquad 0\le F\le1.
$$

Cauchy–Schwarz supplies the bounds. F=1 means the same ray, not necessarily the same coordinate vector. F=0 means orthogonal rays. Matching the probabilities of one measurement alone establishes neither conclusion. Some literature calls the unsquared overlap magnitude a fidelity; always inspect the convention before comparing losses or error bounds.

## 8. Use the Bloch sphere only for this dimension

A pure C² ray has a useful three-coordinate representation on a unit sphere. The coordinates are invariant under global phase.

$$
r=\frac1N\left(2\operatorname{Re}(\overline z_0z_1),\;2\operatorname{Im}(\overline z_0z_1),\;|z_0|^2-|z_1|^2\right),\qquad r\cdot r=1.
$$

The positive-y state is(1,i)/√2 under our sign convention. The sphere is not a drawing of a particle's spatial location. It also does not represent arbitrary-dimensional pure states by the same three numbers. At a pole, one amplitude vanishes and its phase is undefined; the program records this condition instead of assigning a fictitious angle.

| State representative | Probability0 | Probability +x | Probability +y |
| --- | --- | --- | --- |
| (1,1)/√2 | 1/2 | 1 | 1/2 |
| (1,−1)/√2 | 1/2 | 0 | 1/2 |
| (1,i)/√2 | 1/2 | 1/2 | 1 |
| (1,−i)/√2 | 1/2 | 1/2 | 0 |

## 9. A nonorthogonal basis needs a Gram matrix

Consider b₀=e₀ and b₁=e₀+e₁. A vector z has coefficients d=(z₀−z₁,z₁) in this basis. The physical norm is not the sum of squared coefficient magnitudes.

$$
G_{ij}=\langle b_i,b_j\rangle,\qquad G=\begin{pmatrix}1&1\\1&2\end{pmatrix},\qquad \|z\|^2=d^\dagger Gd.
$$

For the normalized state e₁, d=(−1,1). Its coefficient squares sum to2, while its Gram norm is1. Those two coefficients cannot be a two-outcome probability law. This issue also appears in nonorthogonal feature dictionaries, finite elements and generalized eigenvalue problems.

| Normalized physical vector | Coefficients in (b₀,b₁) | Naive coefficient-square sum | Gram norm |
| --- | --- | --- | --- |
| e₀ | (1,0) | 1 | 1 |
| e₁ | (−1,1) | 2 | 1 |
| (e₀+e₁)/√2 | (0,1/√2) | 1/2 | 1 |

## 10. Vector distance and ray distance answer different questions

Two unit representatives can be far apart even when they describe the same ray. A phase-aligned distance minimizes over the redundant overall phase.

$$
\|\psi-\phi\|^2=2-2\operatorname{Re}\langle\psi|\phi\rangle,\qquad \min_\gamma\|\psi-e^{i\gamma}\phi\|^2=2-2|\langle\psi|\phi\rangle|.
$$

For φ=iψ, the unaligned squared distance is2 and the aligned one is0. For φ=−ψ they are4 and0. A state-comparison objective must specify whether overall phase is meaningful for the problem being represented. A loss on raw coordinates can penalize a physically redundant change.

## 11. Separate algebraic certificates from drawing coordinates

The preparation uses sixteen owned integers of magnitude at most10⁶. State, reference and basis pairs must each be nonzero; the phase pair and real scale must also be nonzero, and the scale denominator positive. These restrictions define a finite teaching family, not all possible quantum states.

Probabilities, squared overlaps, ray classifications, Gram norms and reconstruction identities use exact complex rational arithmetic. Normalized real and imaginary coordinates can require square roots; directed512-bit endpoints enclose them. Rounded coordinate squares need not add to exactly one in ordinary floating arithmetic. The exact normalization identity is the certificate, not a tolerance or clipping rule.

## 12. Connect the geometry to AI representations

A normalized complex feature vector offers a controlled way to study phase-invariant similarity. One possible similarity is

$$
k(x,y)=|\langle\psi(x)|\psi(y)\rangle|^2.
$$

The same expression also illustrates an information loss: it deliberately removes global phase, while amplitudes in one fixed basis can discard additional relative-phase information. Decide which invariances an application needs before choosing a feature map or loss. This finite state-vector calculation alone establishes no computational advantage for a quantum device, no data-loading cost model and no learning-theoretic guarantee.

## Research extension: local ray distance and a phase around three states

### Remove the motion that only changes a representative

Suppose a twice differentiable normalized state depends on a real parameter s. A large coordinate derivative can come entirely from multiplying the same state by a rapidly changing global phase. Project the derivative perpendicular to the state before measuring its size:

$$
v=(I-|\psi\rangle\langle\psi|)\partial_s\psi,\qquad
 g_s=\langle\partial_s\psi|\partial_s\psi\rangle-|\langle\psi|\partial_s\psi\rangle|^2=\|v\|^2.
$$

Under ψ(s)→exp(iχ(s))ψ(s), the extra derivative is parallel to ψ and disappears under this projection. Thus g_s measures change of the ray. Expanding the neighboring overlap and differentiating the normalization twice gives the first formula below. For ψ=(cos(θ/2),exp(iφ)sin(θ/2)), substitution gives the second:

$$
F(\psi(s),\psi(s+ds))=1-g_s\,ds^2+o(ds^2),\qquad
 d\ell^2=\tfrac14(d\theta^2+\sin^2\theta\,d\varphi^2).
$$

Here dℓ² is g_s ds² along the chosen parameter curve. At a pole, changing φ has no effect on the ray; at the equator it does. In an optimization problem this metric distinguishes a parameter update that changes predicted quantum probabilities from one that only changes the representative. It does not specify an optimizer or a computational speedup.

### A phase that survives independent choices of representatives

Pairwise fidelity removes overlap phase. With three normalized states, a cyclic product can retain information while still removing each state's arbitrary global phase:

$$
B=\langle a|b\rangle\langle b|c\rangle\langle c|a\rangle,\qquad \Phi=\operatorname{Arg}B,\quad B\ne0.
$$

Use the chapter's convention that the first inner-product slot is conjugate-linear. If any adjacent overlap vanishes, B=0 and its phase is undefined. Rephase a,b,c independently; the three factors cancel. A common unitary also preserves each overlap. Complex conjugation behaves differently:

$$
B\mapsto e^{i(\chi_b-\chi_a+\chi_c-\chi_b+\chi_a-\chi_c)}B=B,
 \qquad B(a^*,b^*,c^*)=B(a,b,c)^*.
$$

The star here conjugates every coordinate in the same fixed basis. For a qubit, let r_a,r_b,r_c be the unit Bloch vectors already defined in this chapter. Multiplying the rank-one projectors, using (r·σ)(s·σ)=(r·s)I+i(r×s)·σ and taking the trace, gives:

$$
B=\frac{1+\mathbf r_a\cdot\mathbf r_b+\mathbf r_b\cdot\mathbf r_c+\mathbf r_c\cdot\mathbf r_a
 +i\mathbf r_a\cdot(\mathbf r_b\times\mathbf r_c)}4.
$$

The imaginary part records an oriented volume. Three vectors in C² have a singular Gram matrix. Expanding its determinant gives a useful check, with F_ab=|⟨a|b⟩|² and cyclic definitions:

$$
1-F_{ab}-F_{bc}-F_{ca}+2\operatorname{Re}B=0,\qquad
 |B|^2=F_{ab}F_{bc}F_{ca}.
$$

Consequently pair fidelities determine Re B and the magnitude of Im B for this qubit triangle, but can lose its orientation sign. Do not claim they lose every geometric feature. In larger Hilbert spaces the three-vector Gram determinant need not vanish.

One can also align representatives successively so each next overlap is positive real:

$$
b'=e^{-i\operatorname{Arg}\langle a|b\rangle}b,\quad
 c'=e^{-i\operatorname{Arg}\langle b'|c\rangle}c,\quad
 a'=e^{-i\operatorname{Arg}\langle c'|a\rangle}a=e^{-i\Phi}a.
$$

The final representative differs from the initial one by exp(−iΦ). This sign follows from the stated alignment rule; Φ itself names the cyclic overlap phase. No Hamiltonian, elapsed time or adiabatic evolution has been chosen. [Rabei, Arvind, Mukunda and Simon](https://arxiv.org/pdf/quant-ph/9907080) develop the kinematic connection between cyclic invariants and geometric phase; the elementary identities and example here can be checked directly.

### Exact example: same fidelities, opposite orientation

Take a north-pole state and two equatorial states:

$$
a=(1,0),\quad b=\tfrac1{\sqrt2}(1,1),\quad c_\varphi=\tfrac1{\sqrt2}(1,e^{i\varphi}),\quad
 B=\tfrac14(1+e^{i\varphi}),\quad F_{bc}=\tfrac12(1+\cos\varphi),\quad
 \Phi=\varphi/2\quad(-\pi<\varphi<\pi).
$$

At φ=±π the second overlap vanishes: the phase is undefined, rather than zero by convention. Away from that endpoint use t=tan(φ/2), so cosφ=(1−t²)/(1+t²) and sinφ=2t/(1+t²). The real and imaginary parts below are exact rational numbers. Run this standalone Python example; it adds no three-state-loop control to the existing two-state reader.

```python
from fractions import Fraction as Q
from math import atan
for t in [Q(-1), Q(-1, 2), Q(1, 2), Q(1)]:
    fidelity = 1 / (1 + t*t)
    real, imag = fidelity / 2, t*fidelity / 2
    assert real*real + imag*imag == fidelity / 4
    print(t, fidelity, real, imag, f"{atan(float(t)):.6f}")
```

| t | F_bc | Re B | Im B | Phi (rad) |
|---|---|---|---|---|
| -1 | 1/2 | 1/4 | -1/4 | -0.785398 |
| -1/2 | 4/5 | 2/5 | -1/5 | -0.463648 |
| 1/2 | 4/5 | 2/5 | 1/5 | 0.463648 |
| 1 | 1/2 | 1/4 | 1/4 | 0.785398 |

![A complex-plane circle shows conjugate cyclic overlaps with the same real part; paired curves compare even fidelity and the sign-changing phase.](/assets/learning/quantum-state-research-cyclic-ray-phase.en.svg)

The blue and orange marked points have identical pair fidelities but opposite imaginary parts. The phase curve excludes the zero-overlap endpoints. The plotted phase is an angle of B, not a probability.

#### Research check 1: a moving representative of one ray

For ψ(s)=exp(7is)a with fixed normalized a, the raw derivative norm squared is49. Its component perpendicular to ψ vanishes, so g_s=0 and neighboring fidelity stays1. Coordinate speed alone is misleading.

#### Research check 2: fix the orientation sign

For a=(1,0), b=(1,1)/√2, c=(1,i)/√2, the three overlaps are 1/√2, (1+i)/2 and 1/√2. Hence B=(1+i)/4 and Φ=π/4. Conjugating all three states gives −π/4 while preserving each fidelity. Reversing the loop also conjugates B.

#### Research check 3: an orthogonal edge

Set c=(1,−1)/√2. Then ⟨b|c⟩=0 and B=0. Neither an argument of B nor positive-overlap phase alignment on this edge is defined. A numerical routine should report this boundary explicitly, rather than silently substitute a zero angle.

## 13. Six worked checks

**1. Normalize (3+4i,0).** Its norm is5. The normalized representative is((3+4i)/5,0), with computational probabilities(1,0). It is the same ray as e₀ despite its complex first coordinate.

**2. Compare plus and minus.** Their computational probabilities are both(1/2,1/2), their inner product is0, and their +x probabilities are1 and0. One probability chart did not identify the ray.

**3. Check the conjugate sign.** For ψ=(1,i)/√2, the bra is(1,−i)/√2 and the +y probability is1. Multiplying the first argument of an inner product by i multiplies the result by−i, whereas multiplying the second argument by i multiplies it by i.

**4. Apply a global phase.** Replacing ψ by iψ preserves every chosen-basis probability and gives fidelity1. Its representative squared distance is2. With−ψ the distance is4; phase alignment gives0 in both cases.

**5. Use the skew basis for e₁.** Solving e₁=d₀e₀+d₁(e₀+e₁) gives d=(−1,1). The naive square sum is2. Multiplication by G gives d†Gd=1, the correct norm.

**6. Choose an imaginary basis.** Let(u,v)=(1,i) and ψ=(1,1)/√2. Applying U† gives β=((1−i)/2,(1−i)/2). Each squared modulus is1/2 and reconstruction returns ψ. Taking a transpose without conjugation would be a different, incorrect calculation.

## 14. Continue from finite coordinates to functions

A finite vector experiment makes phase and basis choices visible, but completeness becomes substantive for infinite-dimensional spaces. The next chapter uses L² wavefunctions, probability integrals and finite projections to show why a sample grid or a coefficient cutoff is not the entire state space.

The [TU Delft inner-product and basis treatment](https://interactivetextbooks.tudelft.nl/mqp/hilbert-spaces-and-operators.html) provides supporting notation. General observables, self-adjoint domains, mixed states, tensor products and dynamics remain subsequent topics. This finite experiment does not complete the whole Hilbert-space research scope.
