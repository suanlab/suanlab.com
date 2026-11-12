## Follow a real electric vector, not a drawing convention

At a fixed point in a monochromatic light beam, the transverse electric vector can move along a line, circle or ellipse. Polarization describes that motion and its relation to the propagation direction. It adds a second physical transverse component to the scalar waves studied earlier; it does not turn the imaginary part of one phasor into another material displacement.

We will construct the two real components from a Jones vector, pass them through ideal optical elements and predict measured intensity. The distinction between an actual time-dependent vector, a complex representation and a quadratic detector observation remains essential. Random mixtures and partial polarization require the separate coherency description in the following lesson.

## Coordinates, units and assumptions

Choose propagation along $+z$ and a right-handed basis $(x,y,z)$. Fix the convention $\exp[i(kz-\omega t)]$. In a homogeneous isotropic medium, a transverse plane wave has two electric components perpendicular to its wave vector. Strong focusing, multiple propagation directions and anisotropic electromagnetic boundary problems can require more than this fixed transverse model.

The amplitudes here are dimensionless calibrated field units. Define normalized intensity $I=|E_x|^2+|E_y|^2$. Actual irradiance in W/m² requires the field calibration, a specified medium impedance and the real-field cycle-average factor. This normalized intensity is not the mechanical spring energy from the previous lesson.

Assume ideal monochromatic, linear, deterministic elements. A retarder introduces relative phase without absorption or reflection; a polarizer transmits one direction and rejects the orthogonal direction. Their ideal properties do not certify a real component over every wavelength or angle.

## Construct the Jones vector

Write nonnegative amplitudes $a_x,a_y$ and phases $\phi_x,\phi_y$:

$$
\mathbf E=\begin{pmatrix}a_xe^{i\phi_x}\\a_ye^{i\phi_y}\end{pmatrix},
\qquad
\boldsymbol{\mathcal E}(t)=\operatorname{Re}\{\mathbf E e^{-i\omega t}\}.
$$

The entries label physical $x$ and $y$ directions. Each entry is complex because it encodes an amplitude and temporal phase. At a chosen temporal phase $\tau=\omega t$, a component $u+iv$ contributes the real value $u\cos\tau+v\sin\tau$.

This formula provides a direct check of every ellipse animation. A complex vector plot alone is not the trajectory of the real electric vector. Nor is an optical oscillation rendered slowly on screen the actual visible-light frequency; the display parameter is temporal phase, not an unspecified number of physical seconds.

## Normalization and the role of global phase

Multiplying $\mathbf E$ by a nonzero complex scalar preserves its polarization shape. Its magnitude changes intensity, while a common phase changes the time origin. A normalized Jones state divides by $\sqrt I$, but the original intensity must be retained if transmitted power is being compared.

For example, $(1,i)$ and $(1,i)/\sqrt2$ have the same circular polarization but normalized intensities2 and1. Treating normalization as a physical optical operation would incorrectly report a loss of half the power.

A global phase is invisible to polarimetry without an external phase reference. It still changes the real vector at a specified clock phase and can matter in interference against another beam. When the field is zero, division by its norm does not define a polarization state. Zero intensity is not unpolarized light of positive intensity.

## Derive the ellipse from real fields

Absorb the first phase into the time origin and let $\delta=\phi_y-\phi_x$. Then

$$
\mathcal E_x=a_x\cos\tau,\qquad
\mathcal E_y=a_y\cos(\tau-\delta).
$$

If both amplitudes are nonzero, eliminate $\tau$ using $X=\mathcal E_x/a_x$ and $Y=\mathcal E_y/a_y$:

$$
X^2+Y^2-2XY\cos\delta=\sin^2\delta.
$$

This is an ellipse, degenerating to a line for appropriate relative phases. The derivation also explains why a relative phase is unavailable when either component is absent: there is no second oscillation against which to measure it. Do not divide by a vanishing component to force the same algebra to apply.

## Linear and circular states with an explicit viewpoint

Relative phase0 or $\pi$ gives a line. Equal nonzero amplitudes with relative phase $\pm\pi/2$ give a circle. Unequal amplitudes or other phase differences generally give an ellipse whose principal axes need not align with the chosen basis.

For $(1,i)/\sqrt2$ at $z=0$,

$$
\boldsymbol{\mathcal E}(\tau)=\frac1{\sqrt2}
\begin{pmatrix}\cos\tau\\\sin\tau\end{pmatrix}.
$$

With the stated basis, this rotates counterclockwise for an observer looking toward the source and is called left circular under the convention used here. Changing observation direction or harmonic convention can reverse handedness labels. The explicit vector and sign convention are more reliable than a bare word such as “left.”

## Principal axes and quantities that become undefined

The real trajectory is the image of a unit circle under

$$
F=\begin{pmatrix}\operatorname{Re}E_x&\operatorname{Im}E_x\\
\operatorname{Re}E_y&\operatorname{Im}E_y\end{pmatrix}.
$$

Its semiaxes are the singular values of $F$. Define $S_1=|E_x|^2-|E_y|^2$, $S_2=2\operatorname{Re}(E_xE_y^*)$ and $S_3=-2\operatorname{Im}(E_xE_y^*)$. Then

$$
a^2=\frac{I+\sqrt{S_1^2+S_2^2}}2,\qquad
b=\frac{|S_3|}{2a},\qquad
\alpha=\tfrac12\operatorname{atan2}(S_2,S_1).
$$

The stable formula for $b$ avoids subtracting nearly equal terms for an almost linear state. A circle has no unique major-axis direction, so $\alpha$ is undefined there. A zero field has no normalized ellipticity. An angle near such a degeneracy can be numerically ill-conditioned even when the underlying field is well computed.

## A retarder changes relative phase

In its principal axes an ideal retarder has matrix

$$
D(\Delta)=\begin{pmatrix}1&0\\0&e^{i\Delta}\end{pmatrix}.
$$

It preserves $\mathbf E^\dagger\mathbf E$ because $D^\dagger D=I_2$. Here the dagger denotes conjugate transpose. This is a unitary complex transformation, not generally a real orthogonal matrix. A quarter-wave plate has $\Delta=\pi/2$, a half-wave plate $\Delta=\pi$ and a full-wave plate $\Delta=2\pi$ at its design wavelength.

If the principal refractive indices are specified and reflections are neglected, $\Delta=2\pi(n_y-n_x)\ell/\lambda_0$ for thickness $\ell$ and vacuum wavelength $\lambda_0$. The lesson directly controls $\Delta$; it does not claim that an arbitrary physical plate retains that value across a broad spectrum. A common propagation phase is omitted only because the present polarization observations do not use an external phase reference.

## Rotate the axes and respect element order

Let

$$
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix},
\qquad J_R=R(\theta)D(\Delta)R(\theta)^T.
$$

The first column of $R$ specifies the first physical principal direction. Rotate the input into those axes, apply the phase delays and rotate back. This fixes the sign convention rather than borrowing a rotation formula with unspecified active or passive meaning.

If a wave passes through $J_1$ and then $J_2$, its output is $J_2J_1\mathbf E$. The rightmost matrix acts first. Optical elements need not commute: changing order can change both polarization and transmitted intensity. The implementation exposes the intermediate field so this difference can be checked rather than hidden inside a combined matrix.

## A linear analyzer is a projection

For transmission direction $\mathbf n=(\cos\gamma,\sin\gamma)^T$, use

$$
P_\gamma=\mathbf n\mathbf n^T,\qquad
\mathbf E_{\rm out}=\mathbf n(\mathbf n^T\mathbf E).
$$

The unanalysed squared norm has no cross term between orthogonal components. Projection combines their contributions on one transmission axis, so phase-sensitive interference enters the transmitted amplitude before squaring.

The projection is Hermitian and idempotent, not unitary. A unit-intensity linear input at angle $\alpha$ therefore transmits $I_{\rm out}=\cos^2(\alpha-\gamma)$, Malus' law. For any deterministic field, the projected norm cannot exceed the incident norm under this ideal passive model.

The real-vector squared magnitude averages to $I_{\rm out}/2$ over a complete optical cycle. That half factor belongs to converting the complex-amplitude normalization to the real-field mean; it is not an extra arbitrary polarizer loss. A completely blocked output has no observable polarization direction or normalized transmitted state.

## Quarter plates, half plates and a noncommuting example

A unit-intensity $45^\circ$ linear input is $(1,1)/\sqrt2$. A quarter plate along the coordinate axes produces $(1,i)/\sqrt2$, circular polarization with the convention above. A circular input of the appropriate handedness is converted back to a linear state by the inverse operation.

A half plate with axis $\theta$ maps horizontal linear polarization to angle $2\theta$. With $\theta=\pi/4$, its Jones matrix exchanges horizontal and vertical components. This gives a concrete order test: horizontal input, then that half plate, then a horizontal analyzer is completely blocked. Reversing the elements first transmits the horizontal input and then rotates it to vertical, retaining unit intensity.

These are ideal single-wavelength statements. A real device can introduce attenuation, wavelength-dependent retardance and interface effects that require a more detailed model.

## Work through an elliptical preparation

Take

$$
\mathbf E=\begin{pmatrix}\sqrt3/2\\e^{i\pi/3}/2\end{pmatrix}.
$$

The normalized intensity is1 and the polarization coordinates are $S_1=1/2$, $S_2=\sqrt3/4$ and $S_3=3/4$. Their squared sum is1, as expected for a nonzero deterministic Jones field. The ellipse orientation is $\tfrac12\operatorname{atan2}(\sqrt3/4,1/2)$, approximately20.45 degrees.

The major and minor semiaxes are approximately0.9114 and0.4114, with positive signed ellipticity. Their squared sum is1 and twice their product is $S_3=0.75$. A horizontal analyzer retains only the first component: intensity becomes3/4, the output is horizontal linear polarization and its real-field squared magnitude averages to3/8.

This example separates a normalized shape description from actual transmission. Reporting only the final unit Jones direction would hide the lost quarter of incident normalized power.

## What an analyzer scan can identify

A rotating linear analyzer measures a quadratic function of its angle. It identifies linear polarization preferences but does not directly observe an absolute optical phase. Opposite circular Jones states both transmit half their intensity at every linear-analyzer angle, despite opposite real-vector rotations.

Even more importantly, a completely unpolarized ensemble can produce the same flat linear-analyzer scan as a fully circular deterministic state. The following lesson introduces coherency matrices and circular-sensitive analysis to resolve that ambiguity. A single deterministic Jones vector cannot describe all those ensembles.

Before fitting an inverse model, list what the actual measurements contain. A visually convincing ellipse reconstructed from insufficient intensity data may be one compatible explanation rather than a uniquely identified physical state.

## AI experiments and independent negative controls

An AI model can learn optical-element transformations or infer a preparation from analyzer intensities. Evaluate complex-field error, intensity error and polarization-shape error separately. A global phase changes one error while leaving others unchanged; a normalized output can hide an incorrect transmitted intensity.

Use a lossless retarder as a norm-preservation control, a crossed analyzer as a blocked-output control and a reversed device sequence as a noncommutation control. Do not split adjacent temporal samples of the same ellipse between training and evaluation and call that evidence of new optical-device generalization.

Independent vector projections test matrix ordering, singular-value decomposition tests ellipse axes and direct real-time integration tests cycle factors. These checks are stronger than generating reference curves by calling the same product routine. Mathematical validation and actual reader-interface verification remain separate stages.

## Research extension: unresolved color can hide pure polarization

The monochromatic retarder above preserves the degree of every nonzero Jones state. Why can a broadband measurement nevertheless become partially polarized? The detector may discard frequency information before reporting a polarization matrix. We derive that averaging explicitly and show why frequency-dependent compensation must act before the information is discarded. This is a supplementary spectral model, not a new native wavelength or delay control.

### Define what the detector averages

At each frequency prepare the same unit Jones direction (1,1)/sqrt(2), with total normalized spectral intensity I>0. Let Omega be angular-frequency detuning from the carrier, sigmaOmega its standard deviation and tau the differential group delay. Assume a fixed-axis, linear-retardance model with no absorption or reflection:


$$
\rho(\Omega)=\frac{e^{-\Omega^2/(2\sigma_\Omega^2)}}{\sqrt{2\pi}\sigma_\Omega},\quad J(\Omega)=\begin{pmatrix}1&0\\0&e^{i(\delta_0+\tau\Omega)}\end{pmatrix},\qquad s=\sigma_\Omega|\tau|.
$$

Omega and sigmaOmega have units rad/s, tau has units seconds and delta0 and s are dimensionless. The Gaussian is a detuning-envelope idealization; interpreting it as an optical spectrum requires a carrier much larger than the bandwidth. The detector is assumed to sum spectral intensities without cross-frequency interference, for example under a stationary incoherent-frequency model or an appropriate long-time average. A finite time gate that retains cross-frequency terms needs a different observation model.

Every J(Omega) is unitary. At a resolved frequency the output vector is (1, exp(i(delta0+tau Omega)))/sqrt(2), whose outer product has rank one. However, the unresolved coherency is the average of outer products, not the outer product of an averaged Jones vector.


$$
\chi(\tau)=\int\rho(\Omega)e^{i\tau\Omega}d\Omega,\qquad\overline C=\frac I2\begin{pmatrix}1&e^{-i\delta_0}\overline{\chi(\tau)}\\e^{i\delta_0}\chi(\tau)&1\end{pmatrix}.
$$

The upper-right sign follows from Ex times the conjugate of Ey and agrees with the existing S3 convention. Chi is the characteristic function of the normalized spectrum. Its magnitude is at most one by the triangle inequality. The trace stays I, even when the off-diagonal terms shrink: no power has been absorbed.

### Evaluate the Gaussian and the actual analyzer readings


$$
\chi(\tau)=e^{-s^2/2},\qquad \overline S=I(1,0,e^{-s^2/2}\cos\delta_0,e^{-s^2/2}\sin\delta_0),\qquad P=e^{-s^2/2}.
$$

The Gaussian Fourier integral gives the exponential; equivalently its characteristic function obeys chi-prime(u)=−u chi(u) after integration by parts, with chi(0)=1. The coherency eigenvalues are I(1±P)/2. Mixedness arises because the spectral components have different relative phases and their label is unobserved. It does not contradict purity at each frequency.


$$
I_\gamma=\frac I2[1+P\cos\delta_0\sin(2\gamma)],\qquad I_{C\pm}=\frac I2[1\pm P\sin\delta_0].
$$

These are integrated positive analyzer intensities. With delta0=pi/2 every linear analyzer reads I/2 regardless of delay; the circular-sensitive pair reveals the changing S3. This connects the spectral mechanism to the existing observation ambiguity without inventing a deterministic ellipse for the averaged beam. The table uses I=1, delta0=pi/2 and nine significant digits. The Python cell integrates the spectrum rather than calling the product calculator.

| s | degree P | I_C+ | I_C- |
|---|---|---|---|
| 0.25 | 0.969233234 | 0.984616617 | 0.0153833828 |
| 0.5 | 0.882496903 | 0.941248451 | 0.0587515487 |
| 1 | 0.60653066 | 0.80326533 | 0.19673467 |
| 2 | 0.135335283 | 0.567667642 | 0.432332358 |

```python
from math import exp, cos, sqrt, pi

def characteristic(s, n=16000):
    step = 18/n
    def f(q):
        return exp(-q*q/2)/sqrt(2*pi)*cos(s*q)
    return step/3*(f(-9)+f(9)+sum(
        (4 if j % 2 else 2)*f(-9+j*step) for j in range(1, n)))

# Unit total intensity; mean retardance pi/2; unresolved Gaussian spectrum.
for s in [.25, .5, 1, 2]:
    degree = characteristic(s)
    assert abs(degree-exp(-s*s/2)) < 2e-13
    print(s, *(format(value, '.9g') for value in
          [degree, (1+degree)/2, (1-degree)/2]))
```

The omitted Gaussian probability outside q=Omega/sigmaOmega in [−9,9] is erfc(9/sqrt(2)); it bounds the absolute missing characteristic integral. Quadrature discretization remains separate and is checked by doubling the partition. Numerical integration is evidence for the explicit derivation, not an all-wavelength material certificate.

### Separate a bandwidth bound from a Gaussian assumption


$$
\mathbb E\Omega=0,\quad\mathbb E\Omega^2=\sigma_\Omega^2:\qquad P=|\chi(\tau)|\ge\max(0,1-s^2/2),\qquad 1-P\le\min(1,s^2/2).
$$

Use cos(z)≥1−z²/2 and abs(chi)≥Re chi. This finite-second-moment lower bound does not require Gaussianity or a symmetric spectrum. It does not make the full degree a function of bandwidth alone. For an equal two-line spectrum at detunings ±sigmaOmega,


$$
\rho_{\rm two}=\tfrac12[\delta(\Omega-\sigma_\Omega)+\delta(\Omega+\sigma_\Omega)]:\qquad\chi(\tau)=\cos(\sigma_\Omega\tau),\quad P=|\cos(\sigma_\Omega\tau)|.
$$

The variance matches the Gaussian, but the degree revives. When cos is negative, the signed Stokes preference reverses even though degree is nonnegative. Thus fitting delay from degree alone requires the spectral model and can be nonunique. A monotonic Gaussian decay is not a universal law of lossless retarders.

### Compensate before discarding the spectral label


$$
J_c(\Omega)=\operatorname{diag}(1,e^{-i(\delta_0+\tau\Omega)}):\quad J_c(\Omega)J(\Omega)=I_2,\quad P_{\rm compensated}=1,\qquad P(U\overline C U^\dagger)=P(\overline C)\quad\text{for fixed unitary }U.
$$

An ideal inverse phase response acting on the frequency-resolved field restores the original polarization before detection. In contrast, a frequency-independent unitary acting on the already averaged matrix preserves its eigenvalues and cannot restore degree. This does not prohibit physical compensation of a broadband beam: it says that the reduced 2×2 matrix alone has discarded the spectral correlations that such a compensator uses. The exact inverse is a declared ideal response, not a promise about a realizable plate at every frequency.

![Equal spectral variances give different unresolved polarization degrees; a two-line spectrum revives and can reverse its signed circular preference.](/assets/learning/waves-polarization-spectral-delay.en.svg)

The [primary PMD study abstract](https://opg.optica.org/jlt/abstract.cfm?uri=jlt-22-3-763) motivates retaining the spectral model when interpreting polarization degree. The Gaussian, two-line and ideal compensation calculations here are derived for the explicitly stated first-order fixed-axis model; they are not a reproduction of the full all-order study.

### Research questions with explanations

**Does lower observed degree prove absorption?** **Answer:** No. Every frequency transformation is unitary and the integrated trace stays I. The missing frequency label makes a mixture of distinct pure directions.

**Does the same bandwidth imply the same delay-to-degree curve?** **Answer:** No. The Gaussian and two-line distributions have the same variance but different characteristic functions. Signed Stokes data and spectral information retain distinctions lost by degree alone.

**Can compensation and spectral averaging always be interchanged?** **Answer:** No. The exact frequency-dependent inverse acts on correlations before detection. A fixed unitary on the reduced matrix preserves degree and is a different operation.

## Explained exercises

### Exercise 1 — Normalize a state without inventing a physical loss

Compare $(1,i)$ and $(1,i)/\sqrt2$. What changes?

**Solution:** Both describe the same circular shape and handedness. Their normalized intensities are2 and1. Dividing a stored vector to represent a unit polarization state does not mean a real element absorbed half the power; retain the original intensity separately.

### Exercise 2 — Recover the actual vector at a quarter cycle

For $\mathbf E=(1,i)$ and $\tau=\pi/2$, find the real electric vector.

**Solution:** With $e^{-i\tau}$, the real components are $(\cos\tau,\sin\tau)=(0,1)$. Reversing the harmonic sign reverses the temporal sense of rotation. State the convention before assigning a circular-handedness name.

### Exercise 3 — Predict a Malus measurement

A unit-intensity horizontal input passes through an analyzer at60 degrees. Find normalized output intensity and real-square cycle mean.

**Solution:** The intensity is $\cos^2(60^\circ)=1/4$. The cycle mean of the real-vector squared magnitude is1/8 under this complex-amplitude normalization. These are distinct observables with a known factor, not two inconsistent power predictions.

### Exercise 4 — Reverse two elements

Use the half plate that exchanges horizontal and vertical components and a horizontal analyzer. Compare both orders for horizontal input.

**Solution:** Plate first gives vertical input to the analyzer and zero output. Analyzer first transmits horizontal intensity1, then the plate rotates it to vertical. The matrices do not commute, and the intermediate state explains the difference.

### Exercise 5 — Can polarimetry recover a global phase?

Multiply a Jones vector by $e^{i0.7}$. Which quantities change?

**Solution:** Its real vector at a fixed temporal phase changes. Intensity, ellipse axes and all analyzer intensities remain the same. An external interferometric phase reference could reveal the change, so the invisibility claim applies to the specified polarimetric observations.

### Exercise 6 — A blocked beam has no preferred ellipse

What normalized polarization should be reported after complete extinction?

**Solution:** None is defined because the field norm is zero. Intensity and transmitted fraction can be zero, while normalized direction, degree and ellipticity are unavailable. This differs from positive-intensity unpolarized light, which has observable transmission statistics.

## Sources and reproducible checks

[TU Delft Interactive Optics](https://interactivetextbooks.tudelft.nl/interactive-optics/content/Chap4_Polarisation/Polarization_2022_01Clean.html) motivates Jones states and ideal elements. [NIST SCATMECH conventions](https://pages.nist.gov/SCATMECH/docs/convent.htm) make the harmonic and circular-polarization conventions explicit. The equations, numerical preparation, ordering examples and checks above are derived for the stated model.

`jonesPolarizationExperiment` in `src/lib/learning-polarization.ts` returns intermediate and final Jones fields, ellipse diagnostics and actual real-vector trajectories. The independent Python generator uses geometric projections, NumPy SVD and QUADPACK cycle integrals. The TypeScript tests compare those references and physical limits. This evidence supports the calculation and manuscript; homepage registration and visual interaction checks require a separate integration step.
