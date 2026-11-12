## Replace a zero-width focus with a finite field

A geometrical ray family can cross at one point. A real beam has a transverse field distribution and a finite wavelength. Its propagation must account for interference among contributions from that distribution. A thin lens changes phase and can concentrate the beam, but it does not generally produce a zero-width spot or place the smallest spot exactly at the ray focal plane.

We will propagate an axisymmetric Gaussian field through free space, an ideal thin lens and another free interval. The Gaussian can be handled analytically, checked by direct Fresnel integration and compared with the same ray matrix used in the previous lesson. This gives a useful bridge from geometric modelling to differentiable wave simulation for AI.

The experiment uses an ideal coherent scalar beam in one homogeneous medium. A Gaussian extends to infinity and has finite integrated squared field. A physical finite pupil truncates it and generally changes the answer. This lesson does not silently add an aperture, aberration, vector polarization correction or calibrated detector.

## Declare the field, the carrier and the dimensionality

Use propagation convention $\exp[i(kz-\omega t)]$, with $k=2\pi/\lambda$ and wavelength $\lambda$ in the background medium. The complex envelope at its input waist is

$$
U_0(x,y)=A_0\exp\left[-\frac{x^2+y^2}{w_0^2}\right],
\qquad r^2=x^2+y^2.
$$

Here $A_0\geq0$ is a relative field amplitude. The radius $w_0$ is the amplitude $1/e$ radius and the intensity $1/e^2$ radius. It is not an intensity half-maximum radius, diameter or one-dimensional standard deviation. Those alternatives differ by fixed factors that matter in experimental comparisons.

There are two transverse dimensions. We often draw a cut through the axis, but its line integral is not the total two-dimensional beam integral. A one-dimensional Gaussian propagation factor and a two-dimensional Gaussian propagation factor also have different powers and Gouy phases. Declaring the dimensionality prevents an otherwise plausible factor-of-two error.

## Define the actual integrated quantity

The relative intensity is $|U|^2$. At the input,

$$
\mathcal N=\int_{\mathbb R^2}|U_0|^2\,dx\,dy
=2\pi A_0^2\int_0^\infty r e^{-2r^2/w_0^2}\,dr
=\frac{\pi A_0^2 w_0^2}{2}.
$$

If relative field is dimensionless and coordinates are in metres, this quantity has units m². It is not automatically optical power in watts. Converting an actual electric phasor to irradiance requires a medium impedance, a field calibration and the real-field cycle-average factor. We preserve the integrated squared field without claiming that missing calibration.

The normalized fraction inside radius $R$ is $1-\exp(-2R^2/w^2)$ for a nonzero Gaussian. At $R=w$ it is approximately $0.864665$. For zero amplitude, both the contained and total integrals are zero; their conditional ratio is undefined. The reference mode width can still be calculated, but it is not evidence of a nonzero beam.

## Fresnel propagation supplies an independent physical check

With the carrier removed, forward paraxial propagation over $z>0$ is

$$
U_z(x,y)=\frac1{i\lambda z}\iint U_0(\xi,\eta)
\exp\left[\frac{ik}{2z}\big((x-\xi)^2+(y-\eta)^2\big)\right]d\xi\,d\eta.
$$

The positive quadratic phase follows from our declared carrier convention. At $z=0$, use the identity map rather than inserting zero into this integral kernel. The full monochromatic field would also carry $\exp(ikz)$; the simulator keeps that longitudinal carrier separate from the envelope.

For a separable Gaussian, two one-dimensional Fresnel integrals multiply to produce this two-dimensional field. That provides a numerical check that does not import the product's analytic propagator. Direct quadrature must still control oscillatory error and tails. The reference calculation integrates in dimensionless lens-width coordinates and records its finite truncation rather than describing a numerical integral as exact.

## Derive the complex Gaussian coefficient

Represent the radial envelope as $U=p\exp(-a r^2)$ with $\operatorname{Re}a>0$. Initially $a_0=1/w_0^2$ and $p_0=A_0$. Completing the square in the Fresnel integral gives

$$
a' = \frac{a}{1+2iza/k},\qquad
p' = \frac{p}{1+2iza/k}.
$$

The prefactor is for two transverse dimensions. In a single transverse dimension it would involve a square root. Using the single-axis factor while integrating the field over an axisymmetric area would fail conservation of $\mathcal N$.

A thin positive lens multiplies the envelope by $\exp[-ik r^2/(2f)]$. Consequently

$$
a^+=a^-+\frac{ik}{2f},\qquad p^+=p^-.
$$

The intensity and width are continuous across the ideal phase-only lens. The wavefront curvature changes there. An on-axis envelope phase is also continuous because the quadratic lens phase is zero at $r=0$.

## Recover free-space width, curvature and Gouy phase

Define the Rayleigh length $z_R=\pi w_0^2/\lambda=k w_0^2/2$. In free space after the input waist,

$$
U_z(r)=\frac{A_0}{1+iz/z_R}
\exp\left[-\frac{r^2}{w_0^2(1+iz/z_R)}\right].
$$

It follows that

$$
w(z)=w_0\sqrt{1+(z/z_R)^2},\qquad
\frac1{R(z)}=\frac{z}{z^2+z_R^2}.
$$

At the waist, inverse curvature is zero. Curvature radius itself has no finite value there; report it as unavailable as a finite radius instead of replacing it by zero. A zero curvature radius would mean a completely different wavefront.

The on-axis magnitude is $A_0 w_0/w(z)$ and its phase is $-\arctan(z/z_R)$ after removing the carrier. This is the full axisymmetric Gouy phase for the chosen convention. The transverse phase is $+kr^2/(2R)$, consistent with expanding phase fronts after a waist. Comparing sources requires translating their time, propagation and complex-parameter signs together.

## Define q with the correct half-plane

Write the radial exponential as $\exp[ik r^2/(2q)]$. In our convention,

$$
q_0=-iz_R,\qquad
\frac1q=\frac1R+i\frac{2}{kw^2},\qquad
 a=-\frac{ik}{2q}.
$$

Thus a physical finite-width Gaussian has $\operatorname{Im}q<0$, while the imaginary part of its inverse is positive. Choosing the opposite sign without also changing the field convention would turn a decaying Gaussian into an exponentially increasing transverse field.

Free propagation gives $q'=q+z$. A lens gives $1/q^+=1/q^- -1/f$. The beam parameter after a ray matrix $M$ is

$$
q'=\frac{Aq+B}{Cq+D}.
$$

This formula uses the same position-then-angle ordering as the previous lesson. It is a transformation of the Gaussian shape, not a substitute for its complex amplitude prefactor. Keeping only q would lose amplitude normalization and the phase needed to compare coherent fields.

## Why the finite-width map does not become singular at the ray focus

For a real matrix with determinant one,

$$
\operatorname{Im}q' =
\frac{\operatorname{Im}q}{|Cq+D|^2}.
$$

A negative imaginary part is preserved. The denominator cannot vanish for a finite physical q: if $C\neq0$, its imaginary part contains $C\operatorname{Im}q\neq0$; if $C=0$, determinant one requires $D\neq0$. This explains why a finite Gaussian stays in the physical half-plane even when geometrical rays cross.

The same identity is useful numerically. Directly subtracting two almost equal terms in the imaginary part of a complex quotient can lose precision near strong focusing. Calculate the transformed imaginary part with the determinant identity while retaining the original input. Do not project an invalid q into a preferred half-plane.

This argument applies to the stated finite Gaussian and ideal real paraxial matrix. It does not show that every point-source or infinite-plane-wave idealization has a finite field at every caustic. It also does not prove that the paraxial approximation remains accurate at large divergence.

## Focus a waist placed at a positive lens

Place the original waist at the lens and take $f>0$. Immediately after the lens,

$$
q_L=\frac{-iz_R}{1+iz_R/f}.
$$

Writing subsequent q as $z-z_* - i z_{R*}$ identifies the new waist:

$$
z_* = \frac{f z_R^2}{z_R^2+f^2},\qquad
z_{R*}=\frac{z_R f^2}{z_R^2+f^2},\qquad
w_*^2=\frac{w_0^2 f^2}{z_R^2+f^2}.
$$

The smallest beam lies before the geometrical focal plane unless a limiting approximation makes the difference negligible. The geometric result $z_*\approx f$ and the usual narrow-focus estimate $w_*\approx\lambda f/(\pi w_0)$ require $z_R\gg f$.

If $z_R\ll f$, the initial beam already spreads strongly on the lens scale; the weak lens shifts a waist only a short distance downstream and changes its radius modestly. Treating every positive lens as if it received a perfectly collimated wide beam would miss this regime.

## Worked comparison at a common wavelength and focal length

Take $\lambda=500$ nm, $f=20$ mm, $A_0=1$ and put the input waist at the lens. The following two preparations differ only in input width.

| Input radius | Rayleigh length | New waist coordinate | Minimum radius | Radius at $z=f$ |
| --- | --- | --- | --- | --- |
| 40 µm | 10.0531 mm | 4.03400 mm | 35.7391 µm | 79.5775 µm |
| 120 µm | 90.4779 mm | 19.0683 mm | 25.9006 µm | 26.5258 µm |

The narrow input is not well approximated as a long collimated beam: its smallest radius occurs far before 20 mm. The wider input approaches the geometric focal coordinate and forms a smaller minimum spot, even though its starting radius was larger.

For the 40 µm preparation, $\mathcal N=2.513274\times10^{-9}$ m² in the relative-field normalization. At its new waist the on-axis relative intensity is approximately $1.252662$, while the integrated quantity is unchanged. The larger peak results from redistribution over transverse area, not creation of total power.

## Move the input waist before the lens or use a negative lens

Let the initial waist lie a forward distance $u\geq0$ before the lens. Then $q_-=u-iz_R$ and $q_L=q_-/(1-q_-/f)$. For any subsequent forward distance $z$, $q=q_L+z$. The local waist coordinate relative to the lens is $z_*=-\operatorname{Re}q_L$ and its Rayleigh length is $-\operatorname{Im}q_L$.

The new waist can lie downstream, at the lens or at a negative extrapolated coordinate. A negative coordinate characterizes the Gaussian that the output field would trace backward in free space. It is not a negative actual propagation distance inserted into this experiment. Our output plane remains $z\geq0$.

For the 40 µm waist placed directly on a lens with $f=-20$ mm, the extrapolated waist is at $-4.03400$ mm and has radius 35.7391 µm. The actual output beam grows as it travels forward. Calling the negative coordinate a focus that the forward beam will later reach would reverse the physical meaning.

## Separate geometric and diffraction contributions without changing the preparation

For an input waist and a real determinant-one matrix, the output radius obeys

$$
w^2=(A w_0)^2+
\left(B\frac{\lambda}{\pi w_0}\right)^2.
$$

The first term is the squared radius of a parallel geometric family prepared at the input plane. The second retains the finite angular spread associated with the coherent Gaussian. It is a mathematical decomposition of the same beam, not a subtraction of two unrelated experiments or a pupil filter.

At a waist-on-lens geometrical focal plane, $A=0$ and $B=f$. The ray contribution vanishes but $w(f)=\lambda|f|/(\pi w_0)>0$. For the 120 µm preparation this is 26.5258 µm. The actual minimum lies slightly before that plane, as the table shows.

The signed geometric radius $A w_0$ can cross zero and reverse orientation, whereas a physical width is nonnegative. A plot should keep the sign of the geometric ray coordinate and draw the positive and negative Gaussian envelope boundaries explicitly. It must not depict a negative beam width after focus.

## Second moments, conservation and the short-wavelength limit

A transverse slice of the normalized two-dimensional intensity has position variance $w_0^2/4$. The Gaussian angular spectrum has angle variance $[\lambda/(\pi w_0)]^2/4$ at the waist. Form

$$
\Sigma_0=\operatorname{diag}\left(\frac{w_0^2}{4},
\frac{\lambda^2}{4\pi^2w_0^2}\right),\qquad
\Sigma=M\Sigma_0M^T.
$$

Then $\Sigma_{xx}=w^2/4$ and $\det\Sigma=[\lambda/(4\pi)]^2$. Narrowing position is accompanied by changes in angular variance and correlation. This second-moment identity belongs to the ideal coherent Gaussian in the declared paraxial coordinates; it is not a certification of every experimental beam or an invocation of a quantum measurement model.

For a strongly correlated covariance, subtracting $\Sigma_{x\theta}^2$ from $\Sigma_{xx}\Sigma_{\theta\theta}$ can lose the tiny determinant in floating-point arithmetic. The model retains the factor $M\operatorname{diag}(w_0/2,\lambda/(2\pi w_0))$, reports the raw determinant and its residual against the theoretical invariant, and exposes a cancellation ratio. It does not project the raw moments into a repaired matrix or present the expected invariant as a directly measured numerical determinant. The most extreme allowed preparations can also be outside the paraxial physical regime; both limitations must be kept visible.

At fixed input width and a fixed matrix, reducing wavelength reduces the diffraction contribution. Away from $A=0$, the width approaches $|A|w_0$. At the geometrical focus, the residual spot shrinks with wavelength but remains positive for each finite wavelength. Limit statements require keeping the preparation and the observation plane fixed.

## Read phase and validity diagnostics honestly

The phase of a nonzero complex sample is wrapped modulo $2\pi$. Its jumps at the branch cut are not physical discontinuities in the field. An unwrapped accumulated Gouy advance can be computed from the propagation denominators; the on-axis envelope carries its negative. With zero amplitude, an observed field phase is unavailable even though a formal reference mode has geometric parameters.

The far-field Gaussian divergence scale is $\lambda/(\pi w_*)$. The lens quadratic-phase slope at the radius incident on it is $w_L/|f|$. Both should be small for a comfortable paraxial interpretation. They are diagnostics, not a universal binary error bound: wavelength, aperture, longitudinal components and required precision also matter. A code path should report an extreme combination instead of silently clamping it to a different beam.

For a diagram, label all physical axes and any longitudinal/transverse stretching. Compare peak intensity together with the integrated quantity. Retain one preparation when moving the observation plane. Direct Fresnel references, q transport, the width decomposition and conservation each test a different aspect; agreement between two copies of the same formula is weaker evidence.

## Research extension: sharp edges change which beam moments exist

The Gaussian above has finite second moments. Finite integrated intensity alone does not guarantee that property after a hard aperture. We construct a separate one-dimensional Fresnel example, then regularize its edge to show that field-norm convergence does not control a whole-line RMS width. This does not add an aperture to the native two-dimensional Gaussian calculator.

### Distinguish norm from derivative regularity


$$
\widehat\psi(p)=\frac1{\sqrt{2\pi}}\int\psi(x)e^{-ipx}dx,\quad N=\int|\psi|^2dx:\qquad \langle p^2\rangle=\frac{\int p^2|\widehat\psi(p)|^2dp}{N}=\frac{\int|\psi\prime(x)|^2dx}{N}\quad(\psi\in H^1).
$$

Here p is transverse wave number in m⁻¹, not mechanical momentum, and N has relative-field-squared metre units. The Sobolev condition H1 means that the field and its weak derivative are square integrable. Plancherel and the Fourier derivative identity give the displayed equality. For a nonzero even real field, the mean transverse wave number is zero, so the second moment is its variance. Merely belonging to L2 controls norm, not this weighted spectral integral.

### A hard box has finite norm but infinite spectral variance


$$
\psi_0(x)=\mathbf1_{\{|x|<a\}},\quad N_0=2a,\quad\widehat\psi_0(p)=\frac{2\sin(ap)}{p\sqrt{2\pi}}:\qquad\frac1{N_0}\int_{-P}^{P}p^2|\widehat\psi_0(p)|^2dp=\frac1{\pi a}\left[P-\frac{\sin(2aP)}{2a}\right]\longrightarrow\infty.
$$

The limiting value of the transform at p=0 is 2a/sqrt(2pi). Away from zero its amplitude has a 1/p envelope. Multiplication by p² removes the decay from squared amplitude, leaving a nonnegative oscillation with positive mean. Equivalently, the two jumps give delta distributions in the weak derivative, not an L2 derivative. A finite numerical spectral cutoff always returns a finite value, but that is a different, cutoff-dependent statistic.

### A finite pointwise field need not have a finite RMS


$$
k=2\pi/\lambda,\quad\widehat{U_z\psi}(p)=e^{-izp^2/(2k)}\widehat\psi(p),\qquad xU_z\psi=U_z\left(x\psi+\frac zk(-i\partial_x)\psi\right),\qquad
|U_z\psi_0(x)|\le\sqrt{\frac{k}{2\pi z}}\,2a<\infty\quad(z>0),\qquad \|xU_z\psi_0\|_2=\infty\quad(z\ne0).
$$

Differentiate the spectral propagation phase to obtain the position identity. Since x psi0 is L2 but its weak derivative is not, a finite x Uz psi0 would imply an L2 derivative after applying the inverse unitary map, a contradiction. The pointwise bound instead uses the Fresnel kernel of constant magnitude and the finite L1 norm 2a. Thus a bounded displayed field can coexist with a divergent whole-line second moment; no infinite peak is being predicted. The identity extends from smooth fields in the distributional sense for the box.

### Regularize the edge with an explicit finite-norm family


$$
\psi_\varepsilon(x)=\begin{cases}1,&|x|\le a,\\(a+\varepsilon-|x|)/\varepsilon,&a<|x|<a+\varepsilon,\\0,&|x|\ge a+\varepsilon,\end{cases}\quad\varepsilon>0,\qquad N_\varepsilon=2a+\frac{2\varepsilon}{3}.
$$

These are different preparations with the same plateau amplitude, not independently peak-normalized propagated images. Their integrated norms differ by the small added ramps. The field is continuous and piecewise linear; its weak derivative has finite steps rather than delta contributions.


$$
\int|\psi_\varepsilon\prime|^2dx=\frac2\varepsilon,\qquad \sigma_p^2=\frac1{\varepsilon(a+\varepsilon/3)}.
$$

Each ramp has length epsilon and derivative magnitude 1/epsilon, giving the factor two. Symmetry makes both position and transverse-wave-number means zero. A real input also has zero symmetrized position-wave-number covariance: the real part of the cross derivative integral vanishes. Therefore


$$
\sigma_x^2(0)=\frac{2a^3/3+2a^2\varepsilon/3+a\varepsilon^2/3+\varepsilon^3/15}{2a+2\varepsilon/3},\qquad\sigma_x^2(z)=\sigma_x^2(0)+\left(\frac zk\right)^2\sigma_p^2.
$$

The numerator comes from integrating x² times the squared plateau and ramps. The propagation formula follows from the position identity and the zero covariance. Every epsilon>0 gives finite moments, but at fixed nonzero z and k the second term diverges as 1/epsilon when the edge becomes sharp. A separable two-dimensional field psiEpsilon(x) g(y) with finite nonzero y norm has the same normalized x moments; this connects the example to a transverse slice without replacing the native axisymmetric Gaussian preparation.

### Prove norm convergence without inferring moment convergence


$$
\frac{\|\psi_\varepsilon-\psi_0\|_2^2}{\|\psi_0\|_2^2}=\frac{\varepsilon}{3a}\longrightarrow0,\qquad\|U_z\psi_\varepsilon-U_z\psi_0\|_2=\|\psi_\varepsilon-\psi_0\|_2.
$$

Only the two added ramps contribute to this norm difference. Unitary propagation preserves it exactly. Nevertheless the squared-position observable is unbounded and its moments are not continuous under L2 convergence alone. This is why a small integrated field loss does not certify a fitted RMS beam width. The table scales lengths by a and fixes z/k=a². The ramp integrals are recomputed in Python; columns are dimensionless and rounded to nine significant digits.

| epsilon / a | relative L2 field distance | a² sigma_p² | propagated RMS / a |
|---|---|---|---|
| 0.5 | 0.40824829 | 1.71428571 | 1.47720779 |
| 0.2 | 0.25819889 | 4.6875 | 2.25144398 |
| 0.1 | 0.182574186 | 9.67741935 | 3.1676337 |
| 0.05 | 0.129099445 | 19.6721311 | 4.47401468 |

```python
from math import sqrt

def integrate(f, lo, hi, n=2000):
    step = (hi-lo)/n
    return step/3*(f(lo)+f(hi)+sum(
        (4 if j % 2 else 2)*f(lo+j*step) for j in range(1, n)))

# Scale lengths by a; z/k=a^2. All four columns are dimensionless.
a = 1
for ratio in [.5, .2, .1, .05]:
    epsilon = a*ratio
    ramp = lambda x: (a+epsilon-x)/epsilon
    norm = 2*a+2*integrate(lambda x: ramp(x)**2, a, a+epsilon)
    x2 = (2*a**3/3+2*integrate(
        lambda x: x*x*ramp(x)**2, a, a+epsilon))/norm
    p2 = (2/epsilon)/norm
    distance = sqrt((norm-2*a)/(2*a))
    propagated_rms = sqrt(x2+a**4*p2)/a
    print(ratio, *(format(value, '.9g') for value in
          [distance, a*a*p2, propagated_rms]))
```

### Keep finite detector windows and physical validity explicit


$$
\sigma_{x,R}^2(z)=\frac{\int_{-R}^{R}x^2|U_z\psi(x)|^2dx}{\int_{-R}^{R}|U_z\psi(x)|^2dx}\le R^2\quad\text{for the centered even profiles and a nonzero window signal}.
$$

This is a windowed RMS statistic of an even intensity profile, not the whole-line moment. Its denominator is the captured signal and must be reported; zero captured signal has no normalized width. A finite camera, spectral cutoff or other measurement model changes the observable. The formal sharp-edge limit at fixed wavelength is a statement about the mathematical paraxial model: extremely narrow edges introduce large transverse wave numbers and do not certify full Maxwell propagation. No infinite experimentally measured width or calibrated power is claimed.

The [MIT Fourier/Sobolev notes](https://math.mit.edu/~rbm/18-155-F17/L4.pdf) provide the regularity framework and [TU Delft scalar diffraction](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html) the propagation context. The box, ramp family and moment limits above are derived for the explicitly stated one-dimensional model.

![Continuous aperture ramps approach a hard box in relative L2 norm while their propagated whole-line RMS grows; the two diagnostics are explicitly distinguished.](/assets/learning/waves-ray-aperture-moments.en.svg)

### Research questions with explanations

**Does finite integrated intensity imply finite RMS width?** **Answer:** No. The hard box has finite norm, but its spectral derivative moment and any nonzero-distance Fresnel whole-line position moment diverge. Pointwise boundedness is a separate property.

**Can a small field-norm loss certify second moments?** **Answer:** Not by itself. The ramp family converges in L2 while its propagated second moment diverges. Additional weighted-norm or derivative control is needed.

**Will a finite detector report the mathematical infinite RMS?** **Answer:** No. It reports a captured, window-dependent statistic. State the window, signal fraction and physical propagation assumptions before comparing it with a Gaussian width or whole-line moment.

## Explained exercises and AI research connections

### 1. Identify the width convention

At $r=w_0$, the amplitude is $A_0/e$ and intensity is $A_0^2/e^2$. The intensity half-maximum radius is $w_0\sqrt{\ln2/2}$, not $w_0$. A reported spot diameter must be converted before it is compared to this model. Otherwise a constant convention error can be mistaken for failed physics or a learned correction.

### 2. Check the waist phase and curvature

For an input waist with no lens and zero propagation, q is purely negative imaginary, inverse curvature is zero, the prefactor is $A_0$ and the envelope phase is zero if $A_0>0$. The finite curvature radius is unavailable. If $A_0=0$, the field phase is unavailable too; returning a numeric phase would describe a reference expression rather than a measured field.

### 3. Test when the geometric focal distance is adequate

Using the table, compare $z_*/f$ for the two widths: approximately 0.2017 and 0.9534. The condition is a large Rayleigh length relative to focal length, not merely a small wavelength written in nanometres. State what relative location error your application tolerates before replacing $z_*$ by $f$.

### 4. Predict the effect of wavelength at a fixed plane

At $z=f$ with waist on the lens, doubling wavelength while holding $w_0$ and $f$ fixed doubles the output radius. The minimum-waist coordinate also changes because $z_R$ changes. Do not use the radius scaling at one fixed plane as if it automatically compared two separately optimized waist planes.

### 5. Explain increased peak without increased integral

For the 40 µm preparation, compute the new peak using $A_0^2(w_0/w_*)^2\approx1.252662$. Multiplying it by $\pi w_*^2/2$ gives the original $\mathcal N$. This area integral is the correct conservation check. Comparing only two on-axis samples can falsely suggest gain or loss.

### 6. Design a meaningful differentiable-optics benchmark

A forward task can map wavelength, input width and lens position to sampled complex fields or calibrated intensity. Compare against direct integration at held-out physical parameters, not only analytic q values generated by the same implementation. Include zero field, negative lens, the exact ray focal plane and large-slope diagnostics. If intensity alone is observed, investigate parameter ambiguities before claiming that a network recovered a unique optical system. Finite pupils, noise and aberrations need explicit extensions and independent validation.

## Sources and further scope

[MIT 6.974 course notes](https://ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/bf3bdeb971819ddc9c68719d28bb2745_classnotes.pdf) discuss paraxial Gaussian beams and ABCD propagation. Their phasor/q convention must be translated consistently before comparison with the positive Fresnel phase used here. [TU Delft, Geometrical Optics](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap2_Geometrical_optics/GeometricalOptics_2022_01Clean.html) supplies the geometric counterpart and its assumptions. The displayed examples follow the stated convention and independent integrations rather than an undeclared mix of signs.

Fourier optics will develop spectral transfer, pupils and spatial-frequency filtering as their own next topic. Vector focusing, non-Gaussian propagation, measured beam quality, finite apertures and calibrated power remain additional research work. A verified ideal Gaussian case is a foundation for those extensions, not evidence that the full optics curriculum is complete.
