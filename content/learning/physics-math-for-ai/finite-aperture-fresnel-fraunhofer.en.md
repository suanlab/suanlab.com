## An aperture contains many contributing positions

A double-slit drawing often shows two points and a cosine fringe pattern. A real slit has width. Different positions inside it contribute different phases at a detector, creating an envelope in addition to interference between the slit centers. Nearer the aperture, the internal quadratic phase also changes the pattern.

We calculate the complex field first and then its normalized intensity. Fresnel propagation and its Fraunhofer approximation use the same aperture and normalization. A discrepancy is a model comparison, not automatically a numerical integration failure.

## Coordinates and physical units

Let $w$ be the full width of each slit in m, $\lambda$ the wavelength in m and $z>0$ the propagation distance in m. Define

$$
F=\frac{w^2}{\lambda z},\qquad
\eta=\frac{xw}{\lambda z},\qquad
z=\frac{w^2}{\lambda F},\quad x=\frac{\eta w}{F}.
$$

Here $F$ uses the full slit width, not its half-width. Other books may define a Fresnel number using a radius or the total aperture span, introducing factors of four or more. Always state the length used. Holding $\eta$ fixed holds a scaled observation coordinate, not a fixed physical screen position when $z$ changes.

## Define one or two rectangular slits

Use normalized aperture coordinate $u=s/w$. A single slit occupies $[-1/2,1/2]$. Two slits have centers $u=\pm d/2$, each of width1, where $d$ is center separation divided by $w$. Require $d\ge1$ so they do not overlap.

Inside each slit the complex transmission is constant: amplitudes $a_1, a_2\ge0$ and phases0,$\phi$. A zero amplitude blocks that branch. For one slit only the first amplitude is used. At $d=1$ two equal-phase slits touch and reproduce one continuous aperture of full width $2w$; this is a useful consistency test.

## The one-dimensional Fresnel kernel

For a scalar field invariant in the second transverse direction, the paraxial propagation kernel is

$$
U_F(x, z)=\frac{e^{-i\pi/4}}{\sqrt{\lambda z}}
\int A(s)\exp\!\left[\frac{i\pi(x-s)^2}{\lambda z}\right]ds.
$$

The common longitudinal phase $e^{ikz}$ is omitted consistently. The prefactor is one-dimensional; the familiar two-dimensional aperture integral instead has a prefactor proportional to $1/(i\lambda z)$. Mixing the two normalizations changes dimensions and propagation scaling.

For dimensionless transmission $A$, the resulting normalized field is dimensionless. This scalar paraxial model is not an exact Maxwell solution for every material edge, polarization or subwavelength opening.

## Aperture phasors and numerical coordinates

Substituting $s=wu$ gives

$$
U_F=\sqrt F\, e^{i\pi\eta^2/F-i\pi/4}
\int A(wu)\, e^{i\pi Fu^2-i2\pi\eta u}\, du.
$$

The integral adds complex vectors. Contributions with different phases can cancel even when the aperture is transparent. The factorization keeps large observation-plane phase out of the numerical aperture integrand, improving its numerical behavior without discarding that phase.

A plotted quadrature point represents a contribution to an integral, not a separate physical photon or an independent point source with arbitrary strength. Changing the quadrature count should not change total slit transmission.

## Derive the Fraunhofer approximation

Fraunhofer propagation additionally drops the aperture's quadratic phase $\exp(i\pi Fu^2)$:

$$
U_{FF}=\sqrt F\, e^{i\pi\eta^2/F-i\pi/4}
\int A(wu)e^{-i2\pi\eta u}\, du.
$$

The remaining integral is a Fourier transform of the aperture. The observation-plane phase is retained in both models. It can be omitted for a single intensity image, but doing so in only one field would corrupt a complex-field error comparison or a later interference calculation.

The appropriate small quantity depends on the entire illuminated aperture, including separated slit centers, not just the width of one slit.

## The single-slit envelope

For a unit-amplitude slit centered at $u=0$,

$$
\int_{-1/2}^{1/2}e^{-i2\pi\eta u}du
=\frac{\sin(\pi\eta)}{\pi\eta},
$$

with value1 at $\eta=0$. Therefore

$$
I_{FF}=|U_{FF}|^2=F\operatorname{sinc}^2(\pi\eta).
$$

The first zeros occur at $\eta=\pm1$, corresponding to $x=\pm\lambda z/w$ in the paraxial coordinate. Increasing slit width narrows the angular envelope. In this formula $\operatorname{sinc}q=\sin q/q$; libraries using $\sin(\pi q)/(\pi q)$ require a different argument.

## Finite double-slit interference

For the two equal-width slits,

$$
U_{FF}=\sqrt F\, e^{i\pi\eta^2/F-i\pi/4}
\operatorname{sinc}(\pi\eta)
\left[a_1e^{i\pi d\eta}+a_2e^{i\phi-i\pi d\eta}\right].
$$

The bracket produces center-to-center interference while the sinc factor supplies the finite-width envelope. For equal amplitudes and zero relative phase, intensity is $4F\operatorname{sinc}^2(\pi\eta)\cos^2(\pi d\eta)$. A fringe maximum can coincide with an envelope zero and disappear.

Changing $d$ changes fringe spacing without changing the slit widths or their transmitted norm. A phase $\pi$ creates an antisymmetric aperture and an on-axis dark field in both Fresnel and Fraunhofer propagation for equal illumination.

## Transmission and observation normalization

For the nonoverlapping rectangular slits, the expected transmitted squared-field norm is

$$
P_A=\int|A(s)|^2ds=w\sum_j a_j^2.
$$

It has units m under our dimensionless-field convention. It is not a power in watts without the remaining physical calibration and transverse extent. The one-dimensional paraxial propagator preserves this full-line norm mathematically; a finite screen crop generally does not capture all of it.

The calculator returns the aperture norm as an expected quantity, not as an independently measured infinite-screen integral. It also returns $|U|^2/F$ to compare shapes with the known distance scale removed. This is not independent peak normalization: doubling an aperture amplitude still multiplies both raw and scaled intensities by four.

## Complex-field error and intensity error

Compare both

$$
|U_F-U_{FF}|,\qquad |U_F|^2-|U_{FF}|^2.
$$

A global phase change can produce large field error while leaving an isolated intensity unchanged. Conversely, phase errors become observable when this field interferes with another reference. A relative error at an exact dark minimum is undefined or extremely unstable; an absolute error with a stated normalization is more informative.

The two model fields here retain the same omitted longitudinal phase and the same retained observation phase. This makes their complex difference meaningful. Do not rotate one field separately just to make the difference smaller.

## Approximation diagnostics are not certificates

Let $h$ be the maximum absolute coordinate of an illuminated slit edge. Useful diagnostics include

$$
\frac{|x|+h}{z},\qquad
\Phi_A=\frac{\pi h^2}{\lambda z},\qquad
\Phi_4\approx\frac{2\pi}{\lambda}\frac{(|x|+h)^4}{8z^3}.
$$

The first measures geometry, the second the aperture phase discarded by Fraunhofer, and the third estimates the next distance-expansion phase term. They answer different questions. A small $F$ based on one slit does not guarantee a small $\Phi_A$ for a widely separated double aperture.

These are interpretation aids, not universal error bounds or automatic proof of physical accuracy. The product still evaluates the specified paraxial model when its physical approximation deserves caution; it does not relabel that calculation as an exact full-wave solution.

## Numerical integration and independent references

The Fresnel integrand is integrated separately on each transparent interval. Composite eight-point Gaussian quadrature uses a subdivision count based on the internal phase variation; a doubled subdivision count provides a diagnostic difference. A small difference between these two calculations is not by itself independent validation.

Reference fixtures therefore use actual unfactored complex kernel integration with SciPy and a separate Fresnel special-function expression. The far-field aperture integral is independently integrated as well. Tests cover touching slits, blocked branches, translation, phase inversion, known sinc zeros and a far-field sequence. Plot samples must remain distinct from quadrature nodes.

## Worked propagation and aperture comparisons

Take $w=50$ μm,$\lambda=500$ nm and $F=1$. Then $z=5$ mm and $\eta=1$ corresponds to $x=50$ μm. A single slit's Fraunhofer intensity is1 at the center and zero at this first minimum. Fresnel propagation need not share those exact zeros when the discarded aperture phase matters.

For the single slit,$\Phi_A=\pi F/4$. With two slits at $d=2$, the outer illuminated edge is $h=1.5w$, so $\Phi_A=2.25\pi F$, nine times the single-slit value. Calling both arrangements “far field” from the single-slit number alone would ignore most of the aperture geometry.

For a second distance comparison,reduce $F$ to0.1 without changing width or wavelength. Then $z=50$ mm and the first Fraunhofer minimum is at $x=0.5$ mm. The raw on-axis Fraunhofer intensity decreases from1 to0.1,while its scaled value $I/F$ stays1. This distinguishes geometric spreading from a loss of aperture transmission. Normalizing both displayed peaks to1 would erase the raw change. A fixed physical observation point also moves to a different $\eta$,so it is a different comparison from keeping the scaled coordinate fixed.

## Guided reasoning and AI relevance

First block one slit and compare the far-field intensity with a centered single slit. The phase changes under translation while intensity remains the same. Next let two equal slits touch and verify the wider-aperture limit. Then change their relative phase to $\pi$ and inspect the on-axis cancellation without changing transmitted norm.

Finally decrease $F$ at fixed $\eta$ and compare complex and intensity errors using the same distance normalization. A learned diffraction surrogate should distinguish physical parameter changes from sampling changes, and should retain phase when downstream interference matters. These examples do not complete polarization, general dispersion or the later Fourier-optics curriculum.

## Research extension: bound the far-field error and keep its phase meaning

A phase-variation diagnostic can be turned into an actual absolute bound between the exact Fresnel and Fraunhofer integrals of this paraxial model. The bound also explains when an isolated intensity image can agree even while a phase-referenced field is wrong. The result concerns these integrals, not a full-wave Maxwell error or a certified floating-point quadrature.

### Remove only the common known distance factor



$$
C_F(\eta)=e^{i\pi\eta^2/F-i\pi/4},\quad \mathcal A_F=\frac{U_F}{\sqrt F C_F}=\int A(wu)e^{i\pi Fu^2-i2\pi\eta u}du,\qquad
\mathcal A_0=\frac{U_{FF}}{\sqrt F C_F}=\int A(wu)e^{-i2\pi\eta u}du,\quad F>0.
$$

Both fields use exactly the same known phase and spreading factor. This does not rotate one result independently. The reduced intensities are the existing shared shape quantities I/F, not separate peak-normalized images.



$$
M_0=\int|A(wu)|du,\quad M_2=\int|A(wu)|u^2du,\qquad
M_0=\sum_j a_j,\quad M_2=\sum_j a_j(c_j^2+1/12)\quad\text{for unit-width rectangular slits centered at }c_j.
$$

These are absolute-amplitude moments, not the transmitted squared-field norm. A blocked branch contributes zero. The moments are dimensionless in the normalized aperture coordinate.

### Prove a uniform absolute model bound



$$
|\mathcal A_F-\mathcal A_0|\le\int|A(wu)|\,|e^{i\pi Fu^2}-1|du\le\min\{2M_0,\pi FM_2\}.
$$

Apply the triangle inequality and abs(exp(ix)−1)≤min(2,abs(x)). No cancellation assumption is needed, and the bound is uniform in eta for the stated integrals. It does not prove paraxial physical validity at arbitrary screen angles. A computed quadrature still requires its own numerical error estimate; the native refinement diagnostic is unchanged.



$$
\big||\mathcal A_F|^2-|\mathcal A_0|^2\big|\le2M_0|\mathcal A_F-\mathcal A_0|\le2M_0\min\{2M_0,\pi FM_2\},\qquad
|U_F-U_{FF}|=\sqrt F|\mathcal A_F-\mathcal A_0|,\quad \frac{|I_F-I_{FF}|}{F}=\big||\mathcal A_F|^2-|\mathcal A_0|^2\big|.
$$

Use abs(AF),abs(A0)≤M0 and factor a difference of squares. These are absolute errors, still meaningful at an exact dark zero where relative error is unavailable. For a fixed aperture, the reduced field error is O(F) and the raw intensity-error bound is O(F²).

### Define a different geometry-based phase approximation explicitly



$$
U_{FF,q}=e^{i\pi Fq}U_{FF},\quad q\in\mathbb R\text{ chosen from the aperture before comparison},\qquad
|\mathcal A_F-e^{i\pi Fq}\mathcal A_0|\le\pi F\int|A(wu)|\,|u^2-q|du.
$$

Factor exp(i pi F q) from the Fresnel integrand, then apply the same inequality to the remaining phase. This defines a separate, phase-corrected approximation. It does not change the native Fraunhofer curve or license fitting a phase after seeing the exact output. Keep the added phase whenever this predicted field interferes with a reference.



$$
c_1=-d/2,\ c_2=d/2,\ d\ge1,\ q=d^2/4:\qquad
\int|A(wu)|\,|u^2-q|du=\frac d4(a_1+a_2),\quad M_2=(a_1+a_2)(d^2/4+1/12).
$$

For the positive-center interval put u=d/2+t, with t between −1/2 and 1/2. Then abs(u²−q)=abs(t)(d+t); its integral is d/4 because the odd term cancels. The negative-center interval is its mirror. Constant nonnegative amplitudes supply the displayed weights, regardless of their relative optical phase. The original bound grows as F d², whereas this phase-corrected bound grows as F d.

### Build an intensity agreement with a persistent field mismatch

Use two equal unit-amplitude slits with zero relative phase, eta=0, and change their separation together with F. This deliberately changes aperture geometry; it is not the original fixed-aperture distance refinement.



$$
d=\frac2{\sqrt F},\quad q=d^2/4=1/F,\quad e^{i\pi Fq}=-1,\quad \mathcal A_0=2,\qquad
|\mathcal A_F+2|\le\pi\sqrt F\longrightarrow0,\quad |\mathcal A_F-2|\longrightarrow4,\quad \big||\mathcal A_F|^2-4\big|\longrightarrow0.
$$

The reduced Fresnel field tends to −2 while the unmodified reduced Fraunhofer field stays +2. Their isolated intensities approach the same value, yet their relative phase is wrong. Raw complex error also shrinks with sqrt(F) because of geometric spreading; that alone would conceal the persistent reduced-field mismatch. The table is a supplementary geometry sequence: all but its first separation exceed the native d≤4 range, and the last F is below its 0.01 bound.

| F | d | unmodified reduced-field error | known-phase corrected error | reduced-intensity error |
|---|---|---|---|---|
| 0.25 | 4 | 3.27165135 | 0.730223230 | 2.38153573 |
| 0.0625 | 8 | 3.80034784 | 0.201548222 | 0.758367305 |
| 0.015625 | 16 | 3.94896996 | 0.0516312734 | 0.201485253 |
| 0.00390625 | 32 | 3.98717236 | 0.0129864866 | 0.0511439685 |

```python
from cmath import exp
from math import pi, sqrt

def integrate(f, n=20000):
    h = 1/n
    return h/3*(f(-.5)+f(.5)+sum(
        (4 if j % 2 else 2)*f(-.5+j*h) for j in range(1, n)))

# Two unit slits; eta=0; geometry gives exp(i*pi*F*d^2/4)=-1.
for fresnel in [.25, .0625, .015625, .00390625]:
    d = 2/sqrt(fresnel)
    field = -2*integrate(lambda t: exp(1j*pi*fresnel*(t*t+d*t)))
    uncorrected = abs(field-2)
    corrected = abs(field+2)
    intensity_error = abs(abs(field)**2-4)
    print(fresnel, d, *(format(x, '.9g') for x in
          [uncorrected, corrected, intensity_error]))
    assert corrected <= pi*sqrt(fresnel)+1e-12
```



$$
U_{\rm ref}=\sqrt F C_F\mathcal A_0:\qquad \frac{|U_F+U_{\rm ref}|^2}{F}=|\mathcal A_F+2|^2\longrightarrow0,\qquad
\frac{|U_{FF}+U_{\rm ref}|^2}{F}=|2+2|^2=16.
$$

The reference is chosen from the known geometry, not fitted to the exact solution. An isolated-intensity loss can approach zero while a downstream coherent-reference prediction fails. A model comparison must state whether absolute phase, a prescribed common phase correction, or only isolated intensity is relevant.

![A separated-aperture sequence has decreasing isolated-intensity error but a persistent unmodified reduced-field error; the prescribed geometry phase restores the field comparison.](/assets/learning/waves-interference-aperture-bounds.en.svg)

### Research questions with explanations

**Does the moment bound certify full electromagnetic accuracy?** **Answer:** No. It bounds two exact paraxial integrals. Physical-model validity and numerical quadrature error remain separate questions.

**May an arbitrary fitted phase be removed before every comparison?** **Answer:** No. The corrected approximation must be declared in advance from geometry, and its phase retained in reference interference. The native unmodified field remains a different prediction.

**Can an excellent isolated-intensity prediction fail coherently?** **Answer:** Yes. In the sequence above the true reduced field approaches −2 while the unmodified approximation gives +2. Adding the same known reference gives limiting reduced intensities 0 and 16.

## Exercises with explanations

### Exercise 1: Width convention

A source defines Fresnel number with half-width $a=w/2$. How does its value compare with $F$ here?

**Solution:** Its number $a^2/(\lambda z)$ is $F/4$. Confusing the two changes any numerical threshold by a factor of four.

### Exercise 2: First minimum

For $w=50$ μm,$\lambda=500$ nm and $z=5$ mm, locate the first Fraunhofer minimum.

**Solution:** $x=\lambda z/w=50$ μm on either side of the center. This is the paraxial screen coordinate, not an assertion that Fresnel propagation has an identical zero.

### Exercise 3: Slits just touching

What happens at $d=1$ with equal amplitudes and equal phase?

**Solution:** The two intervals join to form one slit of width $2w$. To compare the same physical distance with the single-slit parameterization, use $F_{wide}=4F$ and $\eta_{wide}=2\eta$.

### Exercise 4: A blocked translated branch

Why can far-field intensity remain unchanged after the remaining slit is displaced?

**Solution:** Translation multiplies its Fourier amplitude by a unit-magnitude phase ramp. Intensity ignores that phase, although an interference measurement with another reference can detect it.

### Exercise 5: Peak normalization

Why not scale each computed pattern by its own maximum before comparing apertures?

**Solution:** It hides amplitude and transmission changes. Use the shared physical normalization, and label any separate shape scaling explicitly.

### Exercise 6: Numerical and physical accuracy

Does agreement of two quadrature resolutions prove the Fraunhofer approximation is physically correct?

**Solution:** No. It checks numerical evaluation of the specified integral. Fresnel-versus-Fraunhofer comparison and paraxial-versus-full-wave validity are separate questions.

## Primary framework and further study

[TU Delft's scalar diffraction chapter](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html) and [MIT2.71 lecture14, opening diffraction geometry](https://ocw.mit.edu/courses/2-71-optics-spring-2014/da64653098bef719b4537bb2d149c95d_MIT2_71S14_lec14_notes.pdf) provide the propagation framework. The one-dimensional normalization, finite rectangular examples, diagnostics and independent comparisons here are derived for the stated model. Later lessons address general dispersion, polarization and broader Fourier-optics systems.
