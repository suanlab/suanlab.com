## The same lenses can require a different linear model

Two mutually coherent point contributions add as complex fields and then interfere in a squared observation. Mutually incoherent point sources instead contribute intensities whose cross terms vanish under the specified ensemble or exposure average. Changing this preparation changes which quantity is propagated linearly, even when the lens geometry and pupil are unchanged.

This lesson builds a continuous one-dimensional intensity image from nonnegative point-source powers. We derive the point-spread function, distinguish a raw response from its conditional normalization and obtain the optical transfer function from a pupil autocorrelation. A separate sinusoidal-intensity preparation makes contrast and phase transfer visible.

We keep point-source powers and the grating's intensity density as separate preparations. Adding their numerical coefficients without matching units would be another modelling error. Pixels, finite exposure noise and aliasing belong to the following sensor topic, not to an undeclared DFT in this calculation.

## State the coordinates, coherence and units

Use the ideal scalar paraxial 4F geometry of the previous lesson. In reduced coordinates the input and output share an object-space length scale. Actual image coordinate is $x'=mx$ with $m=-f_2/f_1$. Spatial frequency $\nu$ is in cycles/m, with the transform kernel $\exp(-i2\pi\nu x)$.

A point-source weight represents an integrated relative line-intensity quantity. Multiplying it by a kernel with units m$^{-1}$ gives an intensity density. The complete image integral has the same relative quantity as the weights. These are one-dimensional normalized optical quantities, not automatically calibrated watts or two-dimensional irradiance.

The sources are mutually spatially incoherent and quasi-monochromatic for the selected wavelength. Partial coherence generally needs mutual-intensity data. It cannot always be represented by continuously interpolating between the coherent field and incoherent intensity images of two arbitrarily chosen preparations.

## Begin with the coherent impulse response

For the complex amplitude pupil $H_c(\nu)$, define

$$
h_c(x)=\int H_c(\nu)e^{i2\pi\nu x}\,d\nu.
$$

The pupil has fixed reference half-width $a_{\rm ref}$, actual aperture fraction $\alpha$, obstruction fraction $\epsilon$, amplitude transmission $\sigma$ and optional quadratic/linear phase. Its reference cutoff is $f_{\rm ref}=a_{\rm ref}/(\lambda f_1)$ and actual cutoff is $f_c=\alpha f_{\rm ref}$.

The coherent kernel can be complex or negative real. Its sign and phase matter if fields interfere. The incoherent raw kernel is $|h_c|^2\geq0$, but simply discarding phase at the pupil before propagation would not produce that kernel. A phase pupil changes the spatial impulse response and can change intensity contrast.

The numerical model evaluates continuous pupil integrals. For zero quadratic phase it uses the elementary interval integral with its signed sinc factor. Otherwise it uses composite Gaussian quadrature and reports a coarse/refined difference as a diagnostic, rather than a certified global error estimate.

## Separate pupil energy, throughput and conditional normalization

Parseval's identity gives

$$
S=\int|h_c(x)|^2\,dx
=\int|H_c(\nu)|^2\,d\nu
=2f_c(1-\epsilon)\sigma^2.
$$

The units of $S$ are inverse length in this transfer convention. Calling it a dimensionless transmission would be incorrect. Use the fixed clear reference slit, whose spectral integral is $S_{\rm ref}=2f_{\rm ref}$, to define

$$
T=\frac S{S_{\rm ref}}=\alpha(1-\epsilon)\sigma^2.
$$

For $S>0$, define the conditional unit-mass PSF $g=|h_c|^2/S$. The reference-normalized intensity response is instead $K=|h_c|^2/S_{\rm ref}=Tg$. It retains the original throughput. Normalizing every displayed PSF to unit mass and then forgetting $T$ would hide aperture and absorption losses.

This reference throughput describes point-source line power captured relative to the specified reference aperture. It is not the spectrum-weighted coherent transmission of every input field. Changing the reference itself changes the calibration and must not be hidden during a comparison.

## Add positive source contributions instead of fields

For sources at reduced coordinates $x_j$ with weights $w_j\geq0$, the image is

$$
I_{\rm red}(x)=\sum_j w_j K(x-x_j)
=T\sum_jw_jg(x-x_j).
$$

Every term is nonnegative. Two coincident sources of unit weight produce twice the intensity of one, not four times its intensity. Squaring the sum of two identical field amplitudes would implement a coherent preparation instead.

The whole-line image integral is $T\sum_jw_j$. This is the relevant normalization check, even when a displayed finite window captures only part of a broad PSF. The output should preserve each source's position, weight and individual intensity contribution so that a learner can inspect what the sum means.

## Recover actual image inversion and density scaling

The actual image is

$$
I_{\rm actual}(x')=\frac1{|m|}I_{\rm red}(x'/m).
$$

A source at $x_j$ is geometrically mapped to $m x_j$, before any pupil-induced displacement. Negative magnification reverses the physical coordinate direction. Increasing $|m|$ spreads the same integrated quantity over a longer image interval and lowers intensity density.

For $m=-2$, the peak of an otherwise unchanged point response is halved and its width doubles. Integrating with the actual differential $dx'=|m|dx$ preserves the total transmitted quantity. A comparison that rescales coordinates but not density would falsely suggest optical gain.

The normalization is one-dimensional. It must not be copied unchanged into a two-transverse-dimensional camera, where area rather than line length is transformed.

## Derive the OTF from the pupil autocorrelation

The Fourier transform of the raw intensity kernel is

$$
A(\nu)=\int H_c(u)H_c^*(u-\nu)\,du.
$$

This follows by expanding $h_c h_c^*$ in its two frequency variables and taking the Fourier transform in $x$. The resulting difference constraint identifies the two pupil frequencies separated by $\nu$.

For an open nonzero pupil, the conditional OTF is $H_I(\nu)=A(\nu)/S$, so $H_I(0)=1$. The reference transfer is $A/S_{\rm ref}=T H_I$ and has DC gain $T$. Both are useful, but they answer different questions. An MTF is $|H_I|$; it discards the OTF sign and phase.

The code computes overlap intervals of the actual pupil supports and integrates the complex phase difference. It does not apply the coherent pupil mask directly to the intensity spectrum. A valid constructed passive pupil supplies a nonnegative PSF and the bound $|H_I|\leq1$; a generic user-drawn transfer curve would need its own physicality analysis.

## Solve the clear one-dimensional rectangular pupil

For $H_c=1$ on $|\nu|<f_c$,

$$
h_c(x)=2f_c\,\operatorname{sinc}(2f_c x),\qquad
 g(x)=2f_c\,\operatorname{sinc}^2(2f_c x),
$$

where $\operatorname{sinc}(t)=\sin(\pi t)/(\pi t)$. The OTF is the normalized overlap length of two frequency intervals:

$$
H_I(\nu)=\max\left(0,1-\frac{|\nu|}{2f_c}\right).
$$

The amplitude passband ends at $f_c$, while this intensity OTF extends to $2f_c$. That is a distinction between different linear observables and illumination preparations, not a proof that one illumination always resolves every object better.

These formulas are for a one-dimensional rectangular pupil. They are not the Airy function or circular-overlap OTF of a two-dimensional disk. A plotted sinc-squared cut through a two-dimensional rectangular aperture would also require a separate dimensional normalization before it represented total image power.

## Worked normalization and aperture-loss example

Take $\lambda=500$ nm, $f_1=100$ mm and reference half-width $1$ mm, giving $f_{\rm ref}=20{,}000$ cycles/m. For a clear unit-amplitude pupil, $S=40{,}000$ m$^{-1}$, $T=1$ and $g(0)=40{,}000$ m$^{-1}$. Its first zero is $25$ µm from the centre.

Now set aperture fraction $\alpha=0.5$ and amplitude transmission $\sigma=0.5$, without obstruction. The cutoff becomes $10{,}000$ cycles/m, $S=5{,}000$ m$^{-1}$ and $T=0.125$. The conditional peak is $20{,}000$ m$^{-1}$, but the reference-normalized peak is only $2{,}500$ m$^{-1}$. The first zero moves to $50$ µm.

Two source weights $1$ and $0.4$ have total input reference weight $1.4$, producing whole-line image integral $0.175$. Normalizing the broadened PSF to unit mass alone would conceal the factor-of-eight reference loss. With $m=-2$, the actual image density is halved again while its integral remains $0.175$.

## Measure a sinusoidal intensity preparation

For a separate nonnegative object intensity

$$
I_{\rm obj}(x)=\mu[1+\eta\cos(2\pi\nu_gx+\phi)],
\qquad0\leq\eta\leq1,
$$

the reference image is

$$
I_{\rm red}(x)=T\mu\left[1+
\eta\operatorname{Re}\{H_I(\nu_g)e^{i(2\pi\nu_gx+\phi)}\}\right].
$$

For positive throughput, positive mean and nonzero spatial frequency, the observed contrast is $\eta|H_I(\nu_g)|$. The phase of $H_I$ determines a shift or contrast inversion. Absolute brightness additionally depends on $T\mu$, and actual image density also includes $1/|m|$.

At exactly zero spatial frequency, the preparation is a constant whose value includes $\eta\cos\phi$. It is not a spatial fringe with a meaningful Michelson contrast. The model computes that actual constant mean instead of incorrectly reporting $\mu$ as the spatial mean.

## Worked contrast example without changing calibration

For the clear reference pupil, use $\mu=1$, $\eta=0.8$ and $\nu_g=15{,}000$ cycles/m. The conditional OTF is $0.625$, so the output contrast is $0.5$, with extrema $1.5$ and $0.5$.

Reduce only the aperture fraction to $0.5$ and keep unit amplitude transmission. Then $T=0.5$ and $H_I(\nu_g)=0.25$. Contrast becomes $0.2$, while mean reference brightness becomes $0.5$; the extrema are $0.6$ and $0.4$. A contrast-only display and an absolute-intensity display therefore tell complementary stories.

At $30{,}000$ cycles/m the clear coherent amplitude mask is zero, but the clear incoherent OTF is $0.25$. Explain the prepared field or intensity and the measured observable before comparing these values. Coherent intensity beats can themselves contain difference frequencies, so a cutoff slogan alone is an incomplete resolution argument.

## A passive phase pupil can invert contrast

Let the open pupil acquire quadratic phase $q(\nu/f_c)^2$ and linear phase $b\nu/f_c$. For the unobstructed one-dimensional pupil, write $t=\nu/f_c$ and $L=\max(0,2-|t|)$. Its conditional OTF is

$$
H_I(\nu)=\frac L2\,
\frac{\sin(qtL)}{qtL}\,e^{ibt},
$$

with the continuous sinc limit at zero argument. The PSF remains nonnegative, yet the OTF can be negative real or complex. A Fourier transform of a nonnegative spatial function need not be nonnegative at every frequency.

For $q=6$, $b=0$, $f_c=20{,}000$ and $\nu_g=15{,}000$, the OTF is approximately $-0.0679647$. With $\eta=0.8$ and unit mean/throughput, the image at the original cosine crest is about $0.945628$, while half a period away it is $1.054372$. The contrast has inverted. An MTF plot alone would retain only the positive magnitude and hide this reversal.

A linear phase gives an OTF phase $b\nu/f_c$, corresponding to the same reduced-coordinate shift sign derived in the coherent lesson. Physical image inversion still has to be applied afterward.

## Closed pupils, zero sources and finite windows

A closed pupil has $S=0$. Raw kernel and reference transfer are zero, but division by $S$ does not define a conditional PSF, OTF, MTF or phase. The model preserves these as unavailable. A completely obstructed pupil and zero amplitude transmission have the same blocked response for different physical reasons.

Zero source weights are different: an open optical system still has its reference impulse response and OTF, but the observed point image is zero and source-normalized point transmission is undefined. A device reference phase is not an observed phase of that zero image. For a zero-frequency grating with unit modulation and phase $\pi$, even the nominally positive base parameter describes an identically dark constant preparation.

A finite displayed PSF window is not the entire real line. Sinc-squared tails carry mass outside it. The implementation does not divide by the sampled-window sum to make a false unit-mass physical kernel. Source power is checked using the declared whole-line integral, while a finite-window measurement would need its own collection fraction.

## Numerical evidence and approximation limits

Independent reference calculations integrate the continuous complex pupil and shifted-pupil product, then add positive point-source image contributions. Periodic display samples are not used to generate an undisclosed circular convolution. Coarse/refined quadrature differences are reported separately from comparison errors against the independent integrator.

The grating intensity is evaluated in a positive cosine-squared form near dark extrema. Any roundoff-scale correction required by a passive modulation bound is exposed, while the original complex OTF remains unchanged. This is not a projection of arbitrary nonphysical transfer data into a valid pupil.

All results assume one transverse dimension, scalar paraxial optics, the chosen pupil, shift invariance and mutually incoherent sources. Pupil slope and field-of-view slope require separate inspection. Real lens calibration, vector high-NA response, finite detector area, partial coherence and noise can change an experiment. A verified ideal PSF does not prove a full camera simulator correct.

Hard overlap boundaries can create corners in the OTF, and the magnitude or phase has additional nonsmooth or undefined points at a transfer zero. A phase jump between +pi and -pi is a representation branch, not an abrupt physical pupil change. For inverse problems, compare complex transfer values or specify an appropriate phase convention instead of training on raw wrapped angles as if they were globally smooth real coordinates.

The point-source view and the grating view use separate coordinate windows. A grating profile contains at most four cycles across 129 samples, so display interpolation does not create a false low-frequency pattern at the highest supported input. This is a plotting choice for a continuous model, not the physical pixel sampling operator that the next lesson must introduce.

## Explained exercises and research connections

### 1. Add two coincident sources

For the clear reference pupil, one unit source has peak $40{,}000$ m$^{-1}$. Two incoherent unit sources at the same position have peak $80{,}000$ m$^{-1}$ and total integral $2$. A peak of $160{,}000$ would come from coherently squaring two equal amplitudes and would answer a different preparation question.

### 2. Keep loss when normalizing the PSF

For aperture fraction $0.5$ and amplitude transmission $0.5$, compute $T=0.125$. The conditional peak $20{,}000$ becomes reference peak $2{,}500$ after multiplication by $T$. Explain why a unit-area kernel and a raw reference image are both useful, and why showing only the former hides a physical loss.

### 3. Distinguish MTF from phase transfer

At the negative-OTF example, the MTF is $0.0679647$ but the OTF phase is $\pi$ modulo $2\pi$. Reconstruct the two opposite cosine extrema using the signed OTF and obtain about $0.945628$ and $1.054372$. Replacing the OTF by its magnitude would reverse those assignments and erase the contrast inversion.

### 4. Interpret a constant rather than inventing contrast

Set $\nu_g=0$, $\eta=1$ and $\phi=\pi$. The object is identically zero, so the image mean is zero even when $\mu=1$. There is no spatial maximum/minimum cycle from which to report contrast. With $\phi=0$ and $\eta=0.5$, the constant object mean is $1.5\mu$, not $\mu$.

### 5. Compare a finite window with the whole PSF

The clear conditional kernel has unit mass over the whole line. A finite interval containing a central lobe omits tails and has mass below one. The independent calculation compares finite-window quadrature with a sine-integral expression rather than silently renormalizing the samples. Explain which quantity a finite camera region would actually collect.

### 6. Build a physically inspectable inverse-imaging task

Predict a positive point-source image and grating contrast using known source powers, coordinates and reference calibration. Test independent pupil integrals, parity, phase inversion, throughput loss and blocked cases. Intensity measurements may not uniquely identify the pupil phase; reflection, shift or other ambiguities need explicit analysis. A learned kernel should satisfy more than a small training error: preserve nonnegative observations, normalization conventions and the declared imaging regime before interpreting recovered parameters physically.

## Sources and further scope

[MIT 2.71 lecture 22](https://ocw.mit.edu/courses/2-71-optics-spring-2009/bebe78c6a2ae2603a0d1a3ea85e0b4ca_MIT2_71S09_lec22.pdf) develops incoherent PSFs, pupil autocorrelation and contrast transfer. [TU Delft, Scalar Diffraction Optics](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html) connects wave propagation, lens Fourier planes and filtering. The normalization and one-dimensional numerical examples here are independently derived for the declared reference slit.

The next original topic adds observation, sampling and sensors. It will have to distinguish a continuous optical intensity from an integrated pixel reading, exposure, noise and aliasing. Those are additional physical measurement operators, not consequences already verified by this continuous PSF model.
