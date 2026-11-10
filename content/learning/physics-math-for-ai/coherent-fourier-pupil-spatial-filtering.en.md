## A lens system can edit an image in frequency space

A lens does more than draw rays toward a focus. In a suitable relay, one plane contains the spatial-frequency content of an input field. A mask placed there can remove a band, change relative phase or translate the reconstructed pattern. The result depends on the complex field that illuminates the object, not just its visible input intensity.

This lesson develops coherent one-dimensional 4F imaging. We will retain the field before detection, the physical frequency-plane coordinate and the image inversion. We will then calculate an intensity observation. The next lesson changes the preparation to spatially incoherent sources and derives a different intensity-transfer law. Neither case is a universal model for partially coherent illumination.

The purpose is to make filtering a physical forward model that can be inspected, differentiated and tested. A neural network that predicts only a blur width would miss phase conversion, coherent interference and coordinate parity even if its predictions looked smooth.

## Declare the geometry and the observation

Use two ideal positive thin lenses with focal lengths $f_1,f_2$. The input lies at the front focal plane of the first lens; the common Fourier plane lies between the lenses, with the output at the back focal plane of the second. The propagation carrier convention is $\exp[i(kz-\omega t)]$.

Let $x$ denote a reduced output coordinate referenced to the input scale. The actual image coordinate is

$$
x'=m x,\qquad m=-\frac{f_2}{f_1}.
$$

Negative magnification means image inversion. Reduced coordinates absorb this inversion and scale; they do not remove it from a physical camera. A mask or object that lacks reflection symmetry makes the distinction especially visible.

Lengths are in metres and spatial frequencies are in cycles per metre. Fields use relative amplitude units. Our model has one transverse dimension and a flux-preserving one-dimensional relay normalization. It is not a circular-pupil two-dimensional imaging calculation and does not report calibrated watts.

## Fix a Fourier-transform convention

Define the transform and its inverse by

$$
\widehat E(\nu)=\int E(x)e^{-i2\pi\nu x}\,dx,
\qquad E(x)=\int\widehat E(\nu)e^{i2\pi\nu x}\,d\nu.
$$

The $2\pi$ belongs in the exponential because $\nu$ counts cycles, not radians. If an angular spatial frequency $k_x$ is used instead, then $k_x=2\pi\nu$ and every scaling relation must be converted consistently.

These are continuous transforms. Sampling a profile for display does not turn the model into a DFT. The preparation below is a finite continuous Fourier series; its value at an arbitrary coordinate is evaluated directly. A periodic boundary condition is therefore an explicit preparation, not an unnoticed image-processing artefact.

## Connect a frequency to a position in the Fourier plane

The first lens maps a spatial frequency to Fourier-plane coordinate

$$
\xi=\lambda f_1\nu.
$$

Here $\lambda$ is the wavelength in the common background medium. A fixed mask coordinate therefore corresponds to different frequencies when wavelength or focal length changes. A frequency-axis plot and a physical mask drawing should not silently reuse the same numerical coordinate.

For $\lambda=500$ nm and $f_1=100$ mm, a component at $10{,}000$ cycles/m reaches $\xi=0.5$ mm. A symmetric clear half-width of $1$ mm then corresponds to cutoff $f_c=20{,}000$ cycles/m. This is a paraxial scaling, not an exact high-NA vector-focusing result.

## Construct a passive complex pupil

Let $a_{\rm ref}$ be a fixed reference half-width, $\alpha\in[0,1]$ the actual aperture fraction and $\epsilon\in[0,1]$ the central obstruction fraction. The outer half-width is $a=\alpha a_{\rm ref}$, with $f_c=a/(\lambda f_1)$. On its transmitted intervals define

$$
H_c(\nu)=\sigma\exp\left[iq\left(\frac{\nu}{f_c}\right)^2
+ib\frac{\nu}{f_c}\right],
\qquad \epsilon f_c\leq|\nu|<f_c,
$$

and set it to zero elsewhere. The amplitude transmission satisfies $0\leq\sigma\leq1$. The dimensionless quadratic and linear phase strengths are $q$ and $b$. Closing the aperture, setting $\epsilon=1$ or setting $\sigma=0$ blocks the pupil.

The outer edge is excluded exactly. The inner obstruction boundary is included when the pupil is otherwise open. An endpoint has measure zero in a continuous impulse integral, but a discrete prepared harmonic can sit exactly on that endpoint. The chosen convention must therefore be stated and tested.

## Prepare a finite periodic complex field

Represent the input as

$$
E_{\rm in}(x)=e^{i\phi_g}\sum_{n}c_n e^{i2\pi n x/L},
\qquad \nu_n=\frac nL.
$$

The period is $L$, the integer orders are specified explicitly and each complex coefficient includes an amplitude and a relative phase. The preparation is an infinite periodic pattern described with finitely many modes. It has a well-defined mean intensity and integrated intensity per period, rather than finite total energy over the entire real line.

A finite series is useful because every passed or rejected component can be inspected individually. It is not an approximation to an arbitrary image unless an approximation rule and a convergence study are supplied. In particular, a few sidebands used to illustrate a weak phase object are not the exact Fourier expansion of an exponential phase grating.

## Combine equal frequencies before assigning energy

Two preparation entries at the same integer order refer to the same coherent frequency. They must be summed as complex amplitudes before computing an intensity or a Parseval mean. If $c_{1a}=1$ and $c_{1b}=-1$, the combined coefficient is zero, not two units of incoherent power.

For distinct combined orders, orthogonality over one period gives

$$
\langle I_{\rm in}\rangle=\frac1L\int_{-L/2}^{L/2}|E_{\rm in}|^2\,dx
=\sum_n|c_n|^2.
$$

A common phase changes the complex field relative to a reference but leaves this mean and every intensity unchanged. Relative phases can change the spatial intensity distribution. Treating the sum of entry powers as the answer would erase exactly the coherent interference we are trying to teach.

## Filter the field before squaring it

The relay acts on each complex coefficient:

$$
d_n=H_c(\nu_n)c_n,\qquad
E_{\rm red}(x)=e^{i\phi_g}\sum_n d_n e^{i2\pi\nu_n x}.
$$

Only after this sum do we form $I_{\rm red}=|E_{\rm red}|^2$. Filtering the input intensity with the amplitude mask is a different operation and generally produces the wrong result. Detector intensity contains cross terms between retained coherent modes.

The mean output is $\sum_n|d_n|^2$. A passive pupil gives mean transmission

$$
T_{\rm coh}=\frac{\sum_n|H_c(\nu_n)c_n|^2}{\sum_n|c_n|^2}
$$

when the denominator is positive. This transmission depends on the actual prepared spectrum. It is not generally equal to the open fraction of the pupil or the reference throughput of an incoherent point-source experiment.

## Retain parity and normalization in actual image coordinates

With a one-dimensional flux-preserving relay convention,

$$
E_{\rm actual}(x')=\frac1{\sqrt{|m|}}E_{\rm red}(x'/m),
\qquad I_{\rm actual}(x')=\frac1{|m|}I_{\rm red}(x'/m).
$$

The actual image period is $|m|L$. Integrating over it gives the same transmitted period integral as integrating the reduced intensity over $L$. Magnification redistributes a fixed transmitted quantity across physical image length; it does not create or destroy it.

This normalization concerns an ideal one-dimensional relay with the longitudinal carrier and an overall relay phase removed. A two-dimensional flux-preserving field uses a different amplitude scaling. Actual Fresnel constants, source calibration and detector units must be restored before interpreting a numerical intensity as irradiance.

## Worked low-pass example: a DC component and two sidebands

Take $L=100$ µm and

$$
E_{\rm in}(x)=1+\frac12\cos(2\pi x/L).
$$

The combined coefficients are $c_0=1$ and $c_{\pm1}=1/4$. The mean input intensity is $1+2(1/4)^2=1.125$. At the crest, intensity is $1.5^2=2.25$; half a period away it is $0.5^2=0.25$.

With the clear cutoff $20{,}000$ cycles/m, both $\pm10{,}000$ sidebands pass. Reduce the aperture fraction to $0.4$, making the cutoff $8{,}000$ cycles/m. Only DC remains: the output field is $1$, the intensity is constant $1$, and coherent mean transmission is $1/1.125=8/9$.

The aperture fraction is $0.4$, while this preparation transmits about $0.8889$ of its mean intensity. There is no contradiction: the coherent input is concentrated at three frequencies, and most of its power was already in the retained DC component. Do not infer a universal power rule from aperture width alone.

## A central stop reveals why intensity can contain higher frequencies

Keep the clear outer cutoff and obstruct the central fraction $\epsilon=0.2$. DC is removed while the two sidebands remain. The output is

$$
E_{\rm red}(x)=\frac12\cos(2\pi x/L),\qquad
I_{\rm red}(x)=\frac14\cos^2(2\pi x/L).
$$

Mean intensity is $0.125$ and transmission is $1/9$. The intensity contains a harmonic at $2/L=20{,}000$ cycles/m, even though the complex field contains only the two $\pm10{,}000$ components. That harmonic comes from their coherent beat.

It would be wrong to claim that a new complex field component passed through the excluded outer boundary. A statement about the bandwidth of the field and a statement about its squared observation refer to different quantities. This is also why comparisons with an incoherent intensity cutoff require careful preparation and observable definitions.

## A linear pupil phase shifts rather than attenuates

Within the transmitted band, multiplying a spectrum by $\exp(i2\pi\nu d)$ shifts the reconstructed field to $E(x+d)$. Its crest moves to reduced coordinate $x=-d$. For the linear phase above,

$$
d=\frac{b}{2\pi f_c}.
$$

With $b=\pi/2$ and $f_c=20{,}000$ cycles/m, the crest moves to $-12.5$ µm in reduced coordinates. If $f_2=200$ mm and $f_1=100$ mm, then $m=-2$ and the actual image crest is at $+25$ µm. Dropping inversion would predict the wrong direction.

A phase-only mask leaves the magnitudes of the retained coefficients unchanged and hence preserves their mean output power. It can still change the spatial observation substantially. The phase is relative to other components or a reference; it is not a second physical transverse displacement.

## Convert a weak complex preparation into visible intensity contrast

Consider the finite three-mode preparation

$$
E_{\rm in}(x)=1+i\beta\cos(2\pi x/L),\qquad\beta=0.2.
$$

Its intensity is $1+\beta^2\cos^2(2\pi x/L)$: variation begins at second order in $\beta$. This is a useful weak-phase-like preparation, but it is not the exactly unit-modulus object $\exp[i\beta\cos(2\pi x/L)]$.

Choose quadratic pupil phase strength $q=2\pi$. At the two sidebands, $\nu/f_c=\pm1/2$, so the added phase is $\pi/2$; DC receives zero phase. The sidebands turn from imaginary to negative real, giving $E_{\rm red}=1-\beta\cos(2\pi x/L)$. Intensities at the two opposite extrema become $0.64$ and $1.44$.

The mean is still $1+\beta^2/2=1.02$. The mask has converted relative phase into a first-order intensity signal without increasing the mean. A learned image-to-image system that ignores illumination coherence could confuse this deterministic phase effect with a change in object absorption.

## Zero fields, hard boundaries and model validity

When combined input coefficients are all zero, transmission relative to input is undefined. A zero complex sample has no observed phase. A blocked output can likewise have zero intensity while the nonzero input still defines a zero transmission ratio. These cases should not all be encoded as the same normalized state.

Exact pupil edges and nearby frequencies are different inputs. The implementation tests the declared strict edge rather than snapping a neighbourhood to the boundary. Common-phase rotations are evaluated separately from mode interference, and exact quadrantal phases are represented without deliberately changing neighbouring phase values.

The experiment assumes ideal scalar paraxial lenses, one refractive index, coherence and shift invariance. Pupil slope $a/f_1$ and field-of-view slope are separate diagnostics. Finite source support, vector focusing, lens aberrations beyond the chosen mask, detector pixels and noise need additional models. A small quadrature refinement difference or passing finite reference set is not a universal approximation-error bound.

The hard pupil also limits differentiability. A discrete prepared mode crossing a strict cutoff changes its transmission discontinuously, even though coefficient and phase variations within a fixed passband can be smooth. Gradient-based inverse design must address that idealization explicitly, for example with a physically justified smooth pupil or finite object spectrum. Automatic differentiation alone does not make a hard edge differentiable.

## Explained exercises and an AI research path

### 1. Locate a frequency physically

For $\lambda=500$ nm and $f_1=100$ mm, substitute $\nu=10{,}000$ cycles/m into $\xi=\lambda f_1\nu$ to obtain $0.5$ mm. Replacing cycles/m by rad/m without moving the $2\pi$ changes the mask position. Check this conversion before training a model on pupil coordinates.

### 2. Compare power with aperture fraction

For the three-mode amplitude example, compute the combined coefficients and the mean $1.125$. The narrow pupil retains DC power $1$ and discards two powers of $1/16$, giving $T=8/9$. Its width fraction $0.4$ does not determine this spectrum-weighted ratio. Explain why another coherent object with only rejected sidebands would instead have zero transmission.

### 3. Detect a mistaken intensity filter

With DC blocked, calculate $|\tfrac12\cos(2\pi x/L)|^2$ at $x=0$ and $x=L/4$. The answers are $0.25$ and $0$. A program that applies the amplitude mask directly to the input intensity is not implementing the same physical operation. Identify which coherent cross terms it loses or misassigns.

### 4. Track the sign of a shift through the relay

The linear-phase example moves the reduced crest to $-12.5$ µm. Applying $m=-2$ yields $+25$ µm at the physical output. The peak intensity density is halved by the chosen one-dimensional normalization, while the period length doubles. Their integrated transmitted quantity stays fixed.

### 5. Separate zero input from zero output

Prepare two equal entries at the same order with phases $0$ and $\pi$. Their combined field is exactly zero, so input-normalized transmission is unavailable. In a different preparation, place one nonzero mode outside the pupil: input power is positive, output is zero and transmission is zero. A single zero-intensity label cannot distinguish these experimental causes.

### 6. Design a phase-aware inverse problem

Use known complex preparations and several masks to predict complex output samples or calibrated intensity. Validate field calculations against independent continuous-pupil or period integrals and test actual coordinate inversion. Intensity-only observations can leave phase and object/pupil ambiguities; do not claim unique recovery merely from a small reconstruction loss. Add calibration, multiple illuminations or explicit identifiability assumptions before interpreting fitted phase as a physical discovery.

## Sources and further scope

[MIT 2.71 lecture 22](https://ocw.mit.edu/courses/2-71-optics-spring-2009/bebe78c6a2ae2603a0d1a3ea85e0b4ca_MIT2_71S09_lec22.pdf) discusses coherent 4F imaging and the distinction between actual and reduced coordinates. [TU Delft, Scalar Diffraction Optics](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html) connects lens Fourier planes with spatial filtering. The examples here are derived for the stated one-dimensional preparation and normalization; their formulas and numbers should not be transferred unchanged to a two-dimensional camera.

The next lesson derives positive incoherent point-source imaging, the PSF and pupil autocorrelation OTF. Partially coherent fields require their mutual coherence rather than a switch between these two extremes. Sampling and sensors are a subsequent topic, not an unverified feature hidden inside the continuous model.
