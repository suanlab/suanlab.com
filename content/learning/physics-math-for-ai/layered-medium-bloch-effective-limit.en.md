## When can average material properties replace structure?

Imagine alternating layers with relative permittivities one and nine. If each layer is sufficiently thin compared with wavelength, replacing them by an average permittivity looks attractive. At higher frequency, reflections from repeated boundaries interfere and can create propagation gaps. A single positive constant average cannot reproduce all of that behavior. We will compare a periodic structure, its long-wave effective model, and measurements of a finite stack. Start by identifying the length scale and observation regime in which averaging is intended to work.

## Fixing assumptions, polarization, and units

Assume linear isotropic lossless layers with relative permeability one and positive relative permittivity. Incidence is normal to the layers along x, electric field points along y, and magnetic field along z. Use exp(−iωt), with reference vacuum on both sides. Negative, complex, or dispersive permittivity, oblique incidence, and surface currents are outside this family. An arithmetic average derived for this tangential electric field is not a universal isotropic effective material law. Other field directions can require different averages.

## Period, layer fraction, and reduced frequency

Let period a contain thickness fa of material one and (1−f)a of material two. Reduced frequency Ω measures phase relative to a traversal of one period at the reference speed. Physical angular frequency is Ωc_ref/a; it is not the same number as Ω. If f is zero or one, only one material remains. Equal permittivities also give a homogeneous structure. An empty stack with N=0 transmits through vacuum, while the separately defined unit cell still has its own Bloch properties.

$$
\xi=x/a,\quad \Omega=\omega a/c_{ref},\quad d_1=fa,\quad d_2=(1-f)a,\quad n_j=\sqrt{\epsilon_j},\quad \delta_1=\Omega n_1 f,\quad \delta_2=\Omega n_2(1-f).
$$

## Two state variables that connect across boundaries

Tangential E and H are continuous across these interfaces without surface currents. Use the state (E,Z_ref H). Its second component is magnetic field multiplied by the reference impedance, not H itself. In vacuum a right-going wave has components of the same sign; a left-going wave has opposite signs. The state remains continuous at an interface, but energy density can change because its material coefficient changes. Continuity of the selected state does not imply continuity of every physical quantity.

$$
\psi=\begin{pmatrix}E_y\\Z_{ref}H_z\end{pmatrix},\qquad \psi(\xi_j^-)=\psi(\xi_j^+),\qquad \partial_\xi\psi=i\Omega\begin{pmatrix}0&1\\\epsilon_j&0\end{pmatrix}\psi.
$$

## The transfer matrix of one layer

Integrating the constant-coefficient first-order equations across one layer gives a sine-and-cosine matrix. It maps the left state to the right. The upper off-diagonal coefficient contains 1/n and the lower one n, so they are not interchangeable. Reversing the propagation or time convention changes imaginary signs. Its determinant is one, a structural property of this lossless reciprocal propagation. A determinant residual computed from the same formulas is useful diagnostically but is not an independent verification of the implementation.

$$
M_j=\begin{pmatrix}\cos\delta_j&i\sin\delta_j/n_j\\i n_j\sin\delta_j&\cos\delta_j\end{pmatrix},\qquad \det M_j=1.
$$

## Layer order and the unit-cell matrix

The wave traverses layer one and then layer two, so the matrix is M2 M1. Matrix multiplication generally does not commute. The eigenvalues of a repeating interior and the reflection phase of a finite device are different observations. Changing the unit-cell origin can preserve its trace while changing the entrance boundary of a finite stack. Keep the leftmost material explicit when drawing the geometry and checking continuity of the internal field.

$$
M=M_2M_1,\qquad \det M=1,\qquad D=\tfrac12\operatorname{tr}M.
$$

## Bloch propagation is an eigenvalue problem

In an infinitely repeated structure, seek a state multiplied by a complex number λ after one period. This is the Bloch condition. Determinant one makes the two eigenvalues reciprocal and reduces the characteristic equation to a relation involving D. Propagating modes have unit-magnitude λ. In a gap one solution decays and the other grows. The actual combination inside a finite device is selected by both boundary conditions. Deleting the growing solution indiscriminately can prevent those boundaries from being satisfied.

$$
M\psi=\lambda\psi,\qquad \lambda^2-2D\lambda+1=0,\qquad \lambda_+\lambda_-=1.
$$

## Reading pass bands and gaps from half trace

With positive real lossless coefficients, D is real. If |D|<1, Bloch phase φ is real. If |D|>1, a positive attenuation κ appears. Gaps with D>1 and D<−1 have real phase parts zero and π respectively. Exact |D|=1 marks an edge, but a rounded value of one does not establish exact equality. The model distinguishes the analytically known zero-frequency edge from a numerical neighborhood where the edge classification is unresolved.

$$
|D|<1:\ D=\cos\phi;\qquad |D|>1:\ |D|=\cosh\kappa,\quad \kappa>0.
$$

## The bilayer dispersion relation

Multiply the layer matrices and take the trace to obtain the relation below. Phase accumulation appears through δ1 and δ2, while the index ratio captures interface mismatch. Equal indices reduce it to the cosine addition formula for a homogeneous medium. A repeated geometric partition alone therefore does not create a gap. Material contrast and electrical thickness must work together. Compare this exact unit-cell relation with the effective-medium relation on the same reduced-frequency axis.

$$
D=\cos\delta_1\cos\delta_2-\frac12\left(\frac{n_1}{n_2}+\frac{n_2}{n_1}\right)\sin\delta_1\sin\delta_2.
$$

## Why an average emerges at long wavelength

Expand sine and cosine at small Ω and collect the quadratic term. The effective permittivity becomes a thickness-weighted arithmetic average. Intuitively the tangential E changes little within one electrically short cell. With ε1=1, ε2=9, and f=1/2, εeff=5. This positive constant model has real phase Ω√5 at every frequency, so it cannot represent all Bragg gaps of the structured medium. Success in a long-wave regime and success across a whole spectrum are separate validation claims.

$$
\epsilon_{eff}=f\epsilon_1+(1-f)\epsilon_2,\qquad D=1-\tfrac12\epsilon_{eff}\Omega^2+O(\Omega^4),\qquad D_{eff}=\cos(\Omega\sqrt{\epsilon_{eff}}).
$$

## The first term missed by the constant average

Expanding to fourth order reveals a squared material contrast and the factor f²(1−f)². The leading difference vanishes for equal materials or a missing layer. At a finite frequency, however, one leading term is not the complete error. The relation below is a long-wave asymptotic expansion, not a rigorous global error bound. If a learned effective property is made frequency-dependent, specify which observation and asymptotic order it is intended to reproduce.

$$
D-D_{eff}=-\frac{(\epsilon_1-\epsilon_2)^2f^2(1-f)^2}{24}\Omega^4+O(\Omega^6).
$$

## Small phases and band-edge arithmetic

At low frequency, direct evaluation of D can round to one. Rewriting D−1 as a sum of small quantities preserves a nonzero phase over a wider range. Near an edge, cancellation and phase-evaluation errors still matter. The model uses a floating-point decision margin, not rigorous interval certification, and reports unresolved cases accordingly. If a nonzero term falls below representable range, it must not turn into a claim of vacuum propagation, an exact edge, or an exact zero.

$$
D-1=-2\sin^2\frac{\delta_1+\delta_2}{2}-\eta\sin\delta_1\sin\delta_2,\qquad \eta=\frac{(\epsilon_1-\epsilon_2)^2}{2n_1n_2(n_1+n_2)^2}.
$$

## Reflection, transmission, and fields of a finite stack

After N periods the material returns to vacuum. Set incident E amplitude to one and solve for reflection r and transmission t from both boundaries. Equal exterior media make reflectance and transmittance the squared amplitude magnitudes. Internal fields follow by propagating the entrance state through the layers. The implementation composes power-normalized scattering matrices to reduce numerical problems from repeated large transfer products. Independent checks instead use matrix exponentials and a boundary linear solve.

$$
\psi_L=\begin{pmatrix}1+r\\1-r\end{pmatrix},\qquad \psi_R=\begin{pmatrix}t\\t\end{pmatrix}=M^N\psi_L,\qquad R=|r|^2,\quad T=|t|^2.
$$

## Gap attenuation is different from absorption

This lossless finite stack obeys R+T=1. Decay of a Bloch mode in a gap does not mean energy has become heat. Much of the incident energy is reflected, and a finite thickness can still transmit. Inferring T directly as exp(−2Nκ) from an infinite-medium attenuation misses port coupling and interference. Check internal net flux using the normalized real part of E times conjugate magnetic state. Subtracting large opposing wave contributions can lose precision, so its residual is a diagnostic rather than a certificate.

$$
R+T=1,\qquad S_{norm}=\operatorname{Re}[E_y(Z_{ref}H_z)^*],\qquad u_{norm}=\tfrac14[\epsilon_j|E_y|^2+|Z_{ref}H_z|^2].
$$

At exactly zero reduced frequency, the transfer matrix has its static limit. A cycle average is undefined there. For the selected limiting boundary state E=Z_ref H=1, static normalized energy density is (epsilon+1)/2; the positive-frequency cycle-average limit is (epsilon+1)/4. These two quantities are reported separately.

## Principal Bloch phase is not unwrapped phase

Taking arccos D returns a principal value between zero and π. Even a homogeneous medium with true cell phase four radians gives arccos(cos4)≈2.283 radians. This is Brillouin-zone folding, not discovery of a different wave speed. A single trace cannot recover the missing multiples of 2π. One needs branch tracking across frequency or additional field information. The model reports principal Bloch phase and long-wave effective phase separately, rather than forcing them to agree.

$$
\phi_{principal}=\arccos D\in[0,\pi],\qquad \arccos(\cos4)=2\pi-4\ne4.
$$

## What to validate in a research model

When learning an effective medium, choose the target: low-frequency dispersion, finite-stack R/T, internal fields, or energy flow. Structures with the same average epsilon can differ at higher frequency and at boundaries. Split validation data by frequency, material contrast, thickness fraction, and period count. [MIT OCW Lecture 23](https://ocw.mit.edu/courses/3-024-electronic-optical-and-magnetic-properties-of-materials-spring-2013/resources/mit3_024s13_2012lec23/) supplies background on transfer matrices and periodic structures. Extending beyond this positive lossless normal-incidence family requires revisiting constitutive laws, polarization, boundaries, and independent reference calculations.

## Research extension: bulk similarity and boundary phase

### Fix the state before comparing unit cells

Retain positive lossless dielectric layers, normal incidence, vacuum exterior ports and exp(−i omega t). The tangential state is E and Zref H, not separate forward-wave amplitudes. Its layer generator below follows directly from Maxwell equations. A matrix maps the left state to the right, so the first material encountered acts on the right of the product. These conventions determine the complex reflection phase. The research calculation uses the same boundary state as the native lab; no new material-coupling control is implied.

$$
A_j=\begin{pmatrix}0&1\\\epsilon_j&0\end{pmatrix},\quad M_1=e^{i\Omega f A_1},\quad M_2=e^{i\Omega(1-f)A_2},\quad M=M_2M_1,\qquad\psi=(E_y,Z_{ref}H_z)^T.
$$

### Reversal preserves the bulk eigenvalues exactly

Reversing the two layers changes M2 M1 into M1 M2. The displayed similarity transformation proves equal trace, determinant and Bloch eigenvalues at every frequency in this family. It is stronger than comparing only a long-wave average. Nevertheless similarity transforms the state at a different cut. A finite device has specified physical entrance and exit interfaces, and its boundary vectors do not automatically transform when a material ordering is changed. Bulk equivalence therefore does not settle every finite-device observation.

$$
M_{rev}=M_1M_2=M_1MM_1^{-1},\qquad\operatorname{tr}M_{rev}=\operatorname{tr}M,\qquad\det M_{rev}=\det M=1.
$$

### Track the diagonal swap through a finite stack

Each individual layer has equal diagonal entries. For matrices of this real lossless structure, swapping the diagonals reverses the multiplication order while keeping the off-diagonal entries. Applying that operation to all layers of N cells gives the matrices below. This is reversal of the entire material sequence with equal vacuum exteriors. Do not replace it by arbitrary permutations of layers: those can change the trace and power spectrum. The case N=0 is the common identity matrix and empty vacuum stack.

$$
M^N=\begin{pmatrix}a&i b\\i c&d\end{pmatrix},\qquad M_{rev}^N=\begin{pmatrix}d&i b\\i c&a\end{pmatrix},\qquad a,b,c,d\in\mathbb R,\quad ad+bc=1.
$$

### Solve the two vacuum boundaries

Use the entrance vector (1+r,1−r) and exit vector (t,t). Eliminating the two unknowns gives the shared denominator and the two reflection numerators. Here b and c are real matrix coefficients, not magnetic flux or wave amplitudes. The reference planes are the corresponding physical entrance and exit surfaces. A translated reference plane can add another reflection phase; record that choice before comparing data. The same calculation can be performed with an independent two-by-two boundary solve.

$$
\mathcal D=a+d-i(b+c),\qquad r=\frac{d-a+i(c-b)}{\mathcal D},\qquad t=\frac2{\mathcal D},\qquad r_{rev}=\frac{a-d+i(c-b)}{\mathcal D}.
$$

### Equal powers can hide different phases

Reversal changes the sign of the real part of the reflection numerator and preserves its imaginary part. Its squared magnitude and the transmission amplitude are therefore unchanged. The complex reflection amplitude generally changes, so a phase-sensitive measurement can distinguish the ordering even when R and T cannot. Where r=0 its phase is undefined; do not assign a measured phase to an absent reflected field. The figure follows a continuous reflection-phase branch anchored at the smallest plotted positive frequency; this is separate from unwrapping Bloch eigenvalues. These equal-power statements require this reciprocal lossless family and identical exterior ports. They are not claims about arbitrary lossy or unequal-port structures.

$$
t_{rev}=t,\qquad|r_{rev}|^2=|r|^2,\qquad R+T=1,\qquad r_{rev}-r=\frac{2(a-d)}{\mathcal D}.
$$

### Find the first term absent from the average generator

Near Omega=0, expand the two exponentials and then log(I+X). The first term is the averaged generator. Multiplication order first appears through the commutator at second order. The commutator is diagonal and proportional to material contrast. Reversal flips its sign. Since it changes under a cell-origin transformation, this coefficient should not be labelled a unique isotropic material property. The logarithm used here is the analytic branch near identity; this expansion does not solve the original global Bloch-phase unwrapping problem.

$$
\overline A=fA_1+(1-f)A_2,\quad[A_2,A_1]=(\epsilon_1-\epsilon_2)\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad\log M=i\Omega\overline A-\frac{\Omega^2f(1-f)}2[A_2,A_1]+O(\Omega^3).
$$

### Connect the commutator to an observable contrast

For one cell, direct multiplication gives the exact difference of diagonal entries below. Its leading term is quadratic in frequency. Insert that difference into r_rev−r to see how order can first affect complex reflection while leaving the Bloch trace fixed. Equal materials or a missing layer remove this contribution exactly. A small quadratic contrast is not a universal finite-stack error bound: at fixed N its low-frequency coefficient and remainder must still be interpreted in their stated domain, and sensitivity can change near spectral features.

$$
d-a=\left(\frac{n_1}{n_2}-\frac{n_2}{n_1}\right)\sin\delta_1\sin\delta_2=(\epsilon_1-\epsilon_2)f(1-f)\Omega^2+O(\Omega^4)\quad(N=1).
$$

### A symmetric cut removes the quadratic logarithm term

Split material one into two half layers and place material two between them. The symmetric cell is similar to the asymmetric cell and has equal diagonal entries. Its logarithm has no second-order commutator term. This is a useful representation for a repeating bulk, but N symmetric cells placed between vacuum interfaces form a different finite termination from N asymmetric cells. The table therefore distinguishes reversal, which preserves powers here, from the symmetric physical termination, which can change transmitted power. Do not attribute that change to different bulk eigenvalues.

$$
P=e^{i\Omega f A_1/2},\qquad M_{sym}=P M_2 P=PMP^{-1},\qquad\operatorname{tr}M_{sym}=\operatorname{tr}M,\qquad(M_{sym})_{11}=(M_{sym})_{22}.
$$

$$
\log M_{sym}=i\Omega\overline A+O(\Omega^3),\qquad\log M_{rev}=i\Omega\overline A+\frac{\Omega^2f(1-f)}2[A_2,A_1]+O(\Omega^3).
$$

### Verify the local logarithm without crossing a branch

When a subordinate matrix norm of X=M−I is less than one, the convergent power series supplies a separate computation of log M and an exact-arithmetic truncation-tail bound. That bound does not include floating-point summation error or certify a physical effective medium at all frequencies. Compare successively halved Omega: retaining the commutator leaves a cubic remainder, while the symmetric averaged generator also has a cubic remainder. Do not extend this near-identity construction through a band edge merely because a numerical matrix logarithm returns a value.

$$
X=M-I,\quad\|X\|<1:\qquad\log M=\sum_{m=1}^{K}\frac{(-1)^{m+1}}mX^m+\mathcal R_K,\qquad\|\mathcal R_K\|\le\frac{\|X\|^{K+1}}{(K+1)(1-\|X\|)}.
$$

### Execute a boundary comparison

The code chooses epsilon1=2, epsilon2=8, fraction=1/3 and one cell. It records Omega, the common half trace, the complex reflection contrast magnitude, the transmission-amplitude contrast magnitude, and the power contrast between symmetric and asymmetric physical terminations. The reversed transmission contrast is zero to printed precision; the independent calculation also checks its complex value before formatting. The native lab remains the original two-layer cell and finite stack, with reversal represented by swapping materials and complementary thickness fraction. The symmetric termination and local matrix-log calculation are supplementary research models.

| Omega | common half trace | reflection amplitude contrast | reversed transmission contrast | termination power contrast |
| --- | --- | --- | --- | --- |
| 0.05000000 | 0.99250891 | 0.00330248 | 0.00000000 | 0.00000235 |
| 0.10000000 | 0.97014233 | 0.01285954 | 0.00000000 | 0.00003400 |
| 0.20000000 | 0.88226449 | 0.04678289 | 0.00000000 | 0.00038409 |
| 0.40000000 | 0.55542921 | 0.14411470 | 0.00000000 | 0.00244776 |

```python
from math import sqrt, sin, cos

def product(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def layer(epsilon, width, omega):
    n = sqrt(epsilon)
    phase = omega*n*width
    return [[cos(phase), 1j*sin(phase)/n], [1j*n*sin(phase), cos(phase)]]
def scattering(m):
    a, b = m[0]
    c, d = m[1]
    denominator = a+d-b-c
    return (d-a+c-b)/denominator, 2/denominator
for omega in [0.05, 0.1, 0.2, 0.4]:
    first, second = layer(2, 1/3, omega), layer(8, 2/3, omega)
    half = layer(2, 1/6, omega)
    forward, reverse = product(second, first), product(first, second)
    symmetric = product(product(half, second), half)
    r, t = scattering(forward)
    rr, tr = scattering(reverse)
    rs, ts = scattering(symmetric)
    trace = (forward[0][0]+forward[1][1]).real/2
    values = [omega, trace, abs(rr-r), abs(tr-t), abs(abs(ts)**2-abs(t)**2)]
    print(" ".join(f"{value:.8f}" for value in values))
```

[MIT OCW Lecture 23](https://ocw.mit.edu/courses/3-024-electronic-optical-and-magnetic-properties-of-materials-spring-2013/resources/mit3_024s13_2012lec23/) supplies layered-media transfer and Bloch context. The reversal identities, finite-port phase comparison, local logarithm and symmetric-termination distinction are derived explicitly here with the displayed E/H convention.

![Equal bulk trace, reversed reflection phases and different physical terminations](/assets/learning/electromagnetism-bilayer-origin-boundary-phase.en.svg)

### Research questions with explanations

**Can reversed stacks be distinguished by power-only measurements here?** **Answer:** Not by R and T under the declared identical lossless reciprocal ports. Complex reflection can distinguish their phases.

**Does a symmetric-cell termination change the Bloch eigenvalues?** **Answer:** No. Its cell matrix is similar. Placing the newly cut finite structure between unchanged vacuum ports changes the physical boundary problem.

**Is the second-order commutator a unique isotropic material constant?** **Answer:** No. It changes with ordering and the cell-origin representation. The full state and boundary transformation must accompany any effective interpretation.

## Check your understanding with worked exercises

### Exercise 1

For ε1=1, ε2=9, f=1/2, find effective epsilon and its phase at Ω=0.1.

**Answer:** The average is five, and effective phase is 0.1√5≈0.223607 rad. This is a long-wave approximation, not an assertion of exact Bloch phase at all frequencies.

### Exercise 2

Why is the unit-cell matrix M2 M1 rather than M1 M2?

**Answer:** The first layer gives M1ψ; the second gives M2(M1ψ). The matrix acting first sits on the right. Check both traversal direction and the left-to-right material order.

### Exercise 3

For this bilayer at Ω=1, evaluate D and classify the regime.

**Answer:** The layer phases are 0.5 and 1.5; the mismatch multiplier is 5/3. D≈−0.734963 gives a pass band and principal phase about 2.396409 rad. The effective phase √5≈2.236068 differs.

### Exercise 4

A lossless stack has R=0.98 and a decaying internal Bloch mode. Is absorption 0.98?

**Answer:** No. The reflected fraction is 0.98 and transmitted fraction 0.02. Absorption is zero in this model. Bloch decay and thermal dissipation are different. Unequal exterior media would also require an impedance factor in power transmittance.

### Exercise 5

For ε1=ε2=1 and Ω=4, why is principal Bloch phase not four?

**Answer:** D=cos4 yields principal phase 2π−4≈2.283185 rad. Eigenvalues do not distinguish unwrapped phases differing by multiples of 2π. Folding exists even in a uniform material, so this difference alone is not evidence that homogenization failed.

### Exercise 6

If D prints as one at small Ω, is it an exact band edge?

**Answer:** Rounding does not establish that. A stable D−1 can retain the nonzero term approximately −εeff Ω²/2. Report a resolved phase, representation loss, or an unresolved edge decision as appropriate. Keep this separate from explicitly exact Ω=0.
