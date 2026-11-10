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
