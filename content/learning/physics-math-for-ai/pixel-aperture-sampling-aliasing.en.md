## A camera pixel measures a region, not a mathematical point

Imagine photographing a striped wall. Two cameras with the same pixel pitch can produce different stripe contrast because their sensitive regions and exposure times differ. A denser numerical grid cannot restore light that the physical measurement already averaged away.

The previous optics lessons described how an optical system creates an intensity pattern. This lesson adds a declared measurement operator: integration over a rectangular pixel and over an exposure interval, followed by a regular spatial sampling grid. We will distinguish the continuous scene, the filtered scene, the discrete measurements and a finite reconstruction. This is a one-dimensional stripe pattern on a two-dimensional sensor, not a full camera model.

## Declare units and a nonnegative preparation

Let $x$ be metres along the stripes' varying direction. The irradiance on the sensor surface is

$$
E(x,t)=E_0[1+\eta\cos(2\pi\nu(x-vt)+\phi)].
$$

Here $E_0$ has units W/m², $0\leq\eta\leq 1$, spatial frequency $\nu$ has units cycles/m, velocity $v$ has units m/s and phase $\phi$ is in radians. The model is uniform in the transverse direction. Nonnegative irradiance is essential: an intensity measurement cannot collect negative energy. A negative modulation coefficient after filtering will mean contrast inversion, not negative total irradiance.

Calibrated irradiance is a new physical input. Relative squared-field values in the preceding optical model do not become W/m² simply by changing an axis label. Connecting those models requires a separately stated calibration.

## Specify the pixel lattice and active rectangle

Pixel centres are $x_n=x_0+np$, where $p$ is the pitch and $x_0$ is the origin. An active rectangle has width $a\leq p$ and height $b$. Its area is $A=ab$. The integration region extends from $x_n-a/2$ to $x_n+a/2$; the illumination is uniform over height $b$.

Pitch controls where measurements are taken. Aperture width controls what each measurement averages. They are separate parameters. A narrow active region can retain high-frequency modulation that later aliases on a coarse lattice. It also collects less energy under the same irradiance and exposure.

## Integrate over space and exposure

For exposure duration $T$ centred at $t_c$, the collected energy is

$$
U_n=b\int_{t_c-T/2}^{t_c+T/2}\int_{x_n-a/2}^{x_n+a/2} E(x,t)\,dx\,dt.
$$

The units are $(\mathrm{W/m^2})\mathrm{m^2}\mathrm{s}=\mathrm J$. For positive area and duration, the measured average irradiance is $\overline E_n=U_n/(abT)$. It is an average, not an extra amount of energy.

At zero area or zero duration, collected energy is zero and division by $abT$ is undefined. We may calculate a point or instantaneous limiting expression for comparison, but the model labels the physical average as undefined. A limit is not a working detector with finite throughput.

## Derive the two sinc factors

Use the normalized convention $\operatorname{sinc}(q)=\sin(\pi q)/(\pi q)$ with value one at zero. Integrating the sinusoid over symmetric space and time intervals gives

$$
\overline E_n=E_0[1+\eta H_x H_t\cos(2\pi\nu(x_n-vt_c)+\phi)],
$$

$$
H_x=\operatorname{sinc}(\nu a),\qquad H_t=\operatorname{sinc}(\nu vT).
$$

The spatial aperture smooths across stripes; temporal exposure smooths across motion. The two factors arise from different integrals. Static illumination has $H_t=1$ even for a long exposure. A moving pattern can lose contrast without any change in spatial pitch.

## Keep the sign of the transfer

For $\nu a=1.5$, $H_x\approx-0.2122$. Bright input crests become troughs relative to the mean. Taking an absolute value would preserve the modulation magnitude but erase a half-cycle phase reversal.

This is the same reason the previous lesson kept complex OTF information alongside MTF. A contrast magnitude is useful, but it does not specify the entire forward measurement operator. The total irradiance remains nonnegative because the modulation magnitude does not exceed the original bound.

## Work through an averaging example

Take $p=a=10\,\mu\mathrm m$ and $\nu=30{,}000$ cycles/m. Then $\nu a=0.3$ and

$$
H_x=\frac{\sin(0.3\pi)}{0.3\pi}\approx 0.8584.
$$

With input modulation $\eta=0.8$, static output modulation is approximately $0.6867$. The mean irradiance remains $E_0$; collected energy still scales with area and exposure. Reducing aperture width increases the retained modulation but decreases collected photons. That tradeoff becomes important when the next lesson includes shot noise.

## Sampling creates equivalence classes of frequencies

The sampling frequency is $\nu_s=1/p$. On a grid with origin zero,

$$
e^{i 2\pi(\nu+k\nu_s)np}=e^{i 2\pi\nu np}
$$

for every integer $k$. The discrete samples cannot distinguish these frequencies without an additional prior restriction. Aliasing is therefore an ambiguity in the inverse problem, not a failure of plotting software.

The pixel aperture acts before sampling. Two original frequencies can share a point-sampled pattern while having different box-averaged amplitudes. To construct an exact ambiguity for the whole measurement operator, their original amplitudes must account for those different transfers. An arbitrary pair with equal original amplitudes need not produce identical integrated measurements.

## Track alias phase when the grid origin moves

Choose a signed representative $\nu_a=\nu-k/p$ in the half-open interval $[-\nu_s/2,\nu_s/2)$. The phase of its equivalent sampled sinusoid is

$$
\phi_a=\phi-2\pi\nu vt_c+2\pi kx_0/p.
$$

The last term matters when the grid origin is nonzero. Moving the sensor lattice can change which stripe phase it samples even though the pitch is unchanged. The model preserves the signed representative instead of silently reflecting it to a positive frequency and losing phase information.

## Nyquist requires assumptions and a boundary discussion

For an arbitrary unknown real sinusoid, a safe unambiguous band lies strictly below $\nu_s/2$. At the boundary, a sine component with the wrong phase can disappear at every grid point. For example, $\cos(\pi n+\pi/2)=0$ although the continuous modulation between points is not zero.

With $p=10\,\mu\mathrm m$, the Nyquist frequency is $50{,}000$ cycles/m. A pattern at $80{,}000$ cycles/m has signed alias $-20{,}000$ cycles/m. The finite box aperture does not generally eliminate it: its sinc is nonzero there. An aperture is not an ideal brick-wall anti-alias filter.

## Exposure can erase a moving pattern

Let $\nu=25{,}000$ cycles/m, $v=0.001$ m/s and $T=0.04$ s. Then $\nu vT=1$, so $H_t=0$. During the exposure, a pixel sees exactly one temporal cycle. Its integrated value contains the mean but no modulation.

A shorter exposure can preserve motion contrast while reducing collected energy. Increasing digital gain later cannot undo temporal averaging. Nor can it recreate the missing pattern phase from an observation that contains only its mean.

## Convert collected energy to a mean photon count

For monochromatic light of wavelength $\lambda$, each photon has energy $hc/\lambda$. Thus

$$
\mu_{p,n}=\frac{U_n\lambda}{hc}.
$$

Using $A=25\,\mu\mathrm m^2$, $T=10$ ms, $E=0.001$ W/m² and $\lambda=550$ nm gives approximately $692.19$ incident photons. This is an expected count, not a promise that every exposure contains exactly that many photons.

For broadband light, divide each spectral energy contribution by its own photon energy before integrating. A single arbitrary wavelength does not describe an arbitrary spectrum. Sensor quantum efficiency belongs to the next conversion, from incident photons to collected electrons.

## Reconstruct a finite window honestly

For retained samples with indices $-N$ through $N$, one can draw

$$
\widetilde E_N(x)=\sum_{n=-N}^{N}\overline E_n\operatorname{sinc}\!\left(\frac{x-x_0}{p}-n\right).
$$

This finite sum interpolates retained nodes. It is not the infinite-sample sampling theorem. Missing samples create edge effects, and a reconstruction can overshoot or become negative between nodes even when the measured intensities are nonnegative. Such an interpolation artefact is not negative collected light.

Display sampling is a third grid: it approximates curves on a screen. The implementation resolves the original highest continuous frequency rather than drawing an already aliased display curve and calling that a physical result.

## Use the measurement operator in an AI inverse problem

A useful forward chain is scene parameters, optical intensity, aperture/exposure integration and discrete samples. If a learned reconstruction is trained with point samples while the camera collects box integrals, the training operator is misspecified. Apparent texture recovery may then reflect a prior rather than information uniquely present in the observation.

A good research experiment changes pitch, aperture and exposure separately, records physical units and asks which distinct scenes remain observationally equivalent. A held-out phase or grid origin is a stronger check than reusing one convenient crest-aligned pattern.

## Explained exercises

### 1. Compare pitch with aperture

A sensor has pitch 10 µm and width 5 µm. Does its Nyquist frequency double? No. It remains 50,000 cycles/m because the lattice spacing is unchanged. The box transfer changes from sinc(ν·10 µm) to sinc(ν·5 µm); collected energy is halved if height and exposure remain fixed.

### 2. Follow a high-frequency alias

For pitch 10 µm, what does 130,000 cycles/m alias to? Subtract one sampling frequency 100,000 cycles/m to obtain 30,000 cycles/m. If the origin is 3 µm, include an additional phase 0.6π. Ignoring that phase makes the predicted samples disagree.

### 3. Explain the sign

For νa=1.5, why can the response be negative? The aperture averages regions whose modulation contributions cancel with a reversed net sign. The DC level is still positive. The sign encodes a phase reversal of the modulation, not negative energy.

### 4. Test an erased boundary component

At ν=νs/2 and phaseπ/2, all ideal grid-centre modulation values vanish. Moving the origin changes this result. Therefore a recovery claim based only on one boundary-phase preparation is not valid for all unknown phases.

### 5. Separate a limit from a measurement

At T=0, the instantaneous limiting irradiance can be finite. The collected energy and photon mean are nevertheless zero, and the physical exposure average U/(AT) is undefined. Reporting the finite limit as actual detected photons confuses density with an integral.

### 6. Assess a reconstruction claim

A finite sinc sum passes through every measured pixel. Does that prove it recovered the scene? No. Finite-window truncation, aperture zeros and frequency aliases remain. Compare independent continuous preparations and held-out measurements, and state the prior that selects among ambiguous solutions.

## Sources and scope

[MIT Signals and Systems, Lecture 16](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/8708ec068ebdea2c4ee2f38fad39fb83_MITRES_6_007S11_lec16.pdf) supports the band-limited sampling and aliasing discussion. The rectangular moving-stripe preparation, integration formulas, finite-window implementation and examples above are this lesson's explicit model. The calibrated photon conversion connects to the separate sensor lesson; this lesson does not simulate a full color camera, demosaicing, rolling shutter or arbitrary spatial spectra.
