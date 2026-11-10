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
