## Blurring a velocity is different from blurring its nonlinear transport

Imagine a camera that deliberately blurs a velocity pattern. The blurred velocity keeps the broad shape but reduces short oscillations. To predict its evolution, should we insert that blurred velocity into the original equation? For a linear diffusion term the answer can be yes under suitable conditions. For a nonlinear velocity product, an extra term remains.

This example makes that term measurable. We know a complete periodic velocity field, apply a specified Gaussian filter and compare the filtered nonlinear flux with the flux formed from filtered velocity. Their difference is a residual stress. We also compute its spatial derivative, its signed energy transfer and the work of the actual forcing.

The example is a manufactured scalar Burgers family. It is not a full three-dimensional large-eddy simulation, and it does not generate an inertial range. Its purpose is to make filtering and closure precise before using a large simulation or training an AI model.

## Define the domain and the actual filter

Use a periodic interval of length L, positive physical wavenumber k and a Gaussian standard deviation ell. The filter averages periodically continued values over a normalized Gaussian displacement. At zero width it is the identity operator.

$$
\bar q(x)=\int_{-\infty}^{\infty}\frac{e^{-z^2/2}}{\sqrt{2\pi}}
q(x+\ell z)\,dz,\qquad L=\frac{2\pi}{k},\quad \ell\ge0.
$$

A Fourier mode with integer index n is multiplied by g-n. Constant fields are preserved. Higher spatial frequencies are attenuated more strongly.

$$
g_n=\exp\left[-\frac{n^2(k\ell)^2}{2}\right],\qquad
\overline{\sin(nkx)}=g_n\sin(nkx).
$$

Filter width is a physical modeling scale here. It is not a count of pixels or the numerical mesh spacing. Translation invariance, constant width and periodic continuation let differentiation commute with filtering. Walls, truncated kernels and variable widths require additional treatment. A discretized convolution can also introduce quadrature or aliasing errors that this analytic filter does not contain.

## Manufacture a solution and declare its moving frame

Let nu be nonnegative molecular kinematic viscosity. Two velocity amplitudes decay with their respective diffusion rates. A constant speed C specifies a Galilean-transformed family. The coordinate shifts as well as the velocity: adding C to velocity alone would give a different time derivative.

$$
A=A_0e^{-\nu k^2t},\quad B=B_0e^{-4\nu k^2t},\quad
\theta=k(x-Ct)+\alpha,\qquad
u=C+v,\quad v=A\sin\theta+B\sin2\theta.
$$

The actual force is v times its spatial derivative. With this force, the chosen field satisfies the complete scalar equation, including the moving-coordinate time derivative.

$$
u_t+\partial_x(u^2/2)=\nu u_{xx}+f,\qquad f=v v_x,
\qquad u_t=\nu v_{xx}-Cv_x.
$$

Velocity has units of metres per second and f has units of acceleration. For dimensional whole-cell quadratic energies, we specify a positive reference density rho and cross-sectional area S. They are fixed weights for this scalar benchmark, not a claim that a varying one-dimensional velocity with constant density solves compressible mass conservation. Burgers transport and a complete compressible fluid model are different physical models.

Position and initial phase are entered as cycle fractions. Exact quarter and eighth cycles follow declared exact trigonometric symmetries. The moving phase is composed separately so a tiny time shift is not discarded when it is added to a large phase.

## Derive the residual stress from the filtered product

The filtered velocity contains g-one times A and g-two times B. Define tau as the difference between the filtered square and the square of filtered velocity. Since the Gaussian kernel is a probability density, tau is also a local variance of velocity values sampled by the filter.

$$
\tau=\overline{u^2}-\bar u^2\ge0,\qquad
\overline{\partial_x(u^2/2)}
=\partial_x(\bar u^2/2)+\frac12\tau_x.
$$

Thus the correct filtered equation contains the gradient of tau. A model that evolves filtered velocity while keeping the original nonlinear expression must represent that additional term or accept a closure defect.

$$
\bar u_t+\partial_x(\bar u^2/2)
=\nu\bar u_{xx}+\bar f-\frac12\tau_x.
$$

The factor one half comes from the Burgers flux u-squared divided by two. It should not be copied into a Navier–Stokes Reynolds-stress formula without checking that equation's convention. Tau has units of velocity squared; minus half its gradient has units of acceleration.

## Do not apply Reynolds projection rules to a Gaussian filter

A second Gaussian filtering does not generally leave the first result unchanged. It multiplies each mode by g-n a second time. The two applications are equivalent to a Gaussian whose standard deviation is square root of two times ell.

$$
\overline{\bar u}=C+g_1^2A\sin\theta+g_2^2B\sin2\theta,
\qquad \overline{u-\bar u}=\bar u-\overline{\bar u}.
$$

For positive finite width this filtered residual is generally nonzero. Reynolds ensemble averaging in the previous example has different projection properties. Calling both operations “averaging” does not make all of their algebraic rules interchangeable.

A uniform velocity addition does leave the variance unchanged at corresponding material positions. In our moving-frame comparison, stress and transfer agree after accounting for the coordinate shift. Local velocity, energy flux and fixed-position time derivatives need not agree numerically between frames.

## Compute the variance as four nonnegative terms

Directly subtracting two nearly equal squares can erase a small variance or even give a negative rounding artifact. For this two-mode family an exact finite factorization avoids that subtraction. Set z to exp of minus the square of k times ell and introduce the following nonnegative coefficients.

$$
z=e^{-(k\ell)^2},\quad
V_e=\frac{(1-z)^2}{2},\quad V_o=\frac{1-z^2}{2},\quad
D_e=\frac{(1-z)^4(1+z)^2(1+z+z^2)}{2},\quad
D_o=\frac{(1-z)^3(1+z)(1+z^2)(1+z+z^2)}{2}.
$$

Two correlation ratios complete the expression. They arise by separating the even and odd parts of Gaussian displacement and completing a square in each two-dimensional covariance matrix.

$$
r_e=\sqrt z(1+z)^2,\qquad r_o=\sqrt z(1+z^2),
$$
$$
\tau=V_e(A\sin\theta+r_eB\sin2\theta)^2+D_e(B\sin2\theta)^2
+V_o(A\cos\theta+r_oB\cos2\theta)^2+D_o(B\cos2\theta)^2.
$$

Each term is nonnegative, including at zero width. This is a closed identity for the declared two modes, not a truncated infinite expansion and not a numerical clipping of negative values. It is verified independently against actual Gaussian convolution and a separate Fourier-polynomial calculation.

## Small width can reveal more than one power law

At an ordinary location a smooth field's filter variance usually begins with ell-squared times the square of its local gradient. At a stationary point that leading coefficient vanishes. A single sine mode at its maximum instead has a fourth-order variance. Our two-mode combination can cancel additional derivatives.

$$
\tau\big|_{B=0,\,\theta=\pi/2}
=\frac{A^2}{2}\left(1-e^{-(k\ell)^2}\right)^2,
\qquad
\tau\big|_{A=1,\,B=-1/2,\,\theta=0,\,t=0}
\sim\frac{15}{4}(k\ell)^6.
$$

For the second case, expand sine theta minus half sine twice theta. The linear terms cancel and the local field begins with theta-cubed divided by two. A centred Gaussian has sixth moment fifteen, giving the displayed coefficient. This is a concrete reason to retain small residual terms after a leading cancellation.

Zero width has exactly zero variance. A tiny positive width can instead have a positive variance whose ordinary floating-point value is unavailable. The implementation retains its sign and natural logarithm. It also computes tiny amplitude changes separately from rounded endpoint amplitudes. A flat-looking curve on a large vertical scale does not prove a derivative or a tail is zero.

## Write the local energy budget before interpreting transfer

Let w denote filtered velocity. We define positive Pi as transfer from the resolved quadratic energy toward the residual contribution. It is not the same quantity as molecular loss.

$$
\Pi=-\frac12\tau w_x,\qquad
[\Pi]=\mathrm{m^2/s^3},\qquad D_{\mathrm{local}}=\nu w_x^2\ge0.
$$

Multiplying the filtered equation by w gives a local balance with a conservative energy flux. Both forcing work and the stress contribution to the flux are required.

$$
\partial_t(w^2/2)+\partial_x\left(\frac{w^3}{3}-\nu w w_x+\frac{w\tau}{2}\right)
=-\nu w_x^2+w\bar f-\Pi.
$$

Tau being nonnegative does not make Pi nonnegative: its sign also depends on the filtered gradient. Negative Pi is return toward resolved energy under this convention. It does not imply reversible overall dynamics, negative molecular viscosity or a violation of the full energy balance.

## Integrate the whole-cell energies and powers

Multiply the periodic spatial integrals by rho times S. Orthogonality gives total energy K, resolved energy K-r and a nonnegative residual K-s. Their units are joules under the declared reference weighting.

$$
K=\frac{\rho SL C^2}{2}+\frac{\rho SL}{4}(A^2+B^2),\quad
K_r=\frac{\rho SL C^2}{2}+\frac{\rho SL}{4}(g_1^2A^2+g_2^2B^2),\quad
K_s=\frac{\rho SL}{4}\left[A^2(1-g_1^2)+B^2(1-g_2^2)\right].
$$

The resolved molecular loss D-r and integrated transfer P have different signs and different amplitude dependence. The transfer changes direction when B changes sign, while molecular loss does not.

$$
D_r=\frac{\rho SL\nu k^2}{2}(g_1^2A^2+4g_2^2B^2),\qquad
P=\rho S\int_0^L\Pi\,dx=-\frac{\rho SLkA^2B}{4}(g_1^2-g_2^2).
$$

For this particular manufactured family, the resolved forcing work F-r equals P. The full forcing work is zero, while the residual forcing work is minus P. Consequently the actual resolved energy decay is minus D-r, not minus P. This cancellation is a feature of the chosen forcing, rather than a universal turbulence law.

$$
\dot K_r=F_r-D_r-P=-D_r,\qquad F_r=P,\qquad
\dot K_s=F_s-D_s+P=-D_s,\quad F_s=-P,\qquad
\dot K=-D.
$$

## Inspect a worked example

Take k = 1 per metre, ell = 0.5 m, nu = 0.1 square metres per second, A-zero = 1 m/s, B-zero = minus 0.5 m/s, C = 0 and time = 0.3 s. Set rho and S to one in their SI units. The period is two pi metres.

| Quantity | Approximate value |
| --- | --- |
| Current amplitudes A, B | 0.970446, −0.443460 m/s |
| Filter multipliers g-one, g-two | 0.882497, 0.606531 |
| Total, resolved, residual energies | 1.788228, 1.265737, 0.522492 J |
| Resolved molecular loss | 0.321332 W |
| Integrated resolved-to-residual transfer | 0.269572 W |
| Resolved forcing work | 0.269572 W |

Resolved energy decreases at approximately 0.321332 J/s. Using only transfer would predict the wrong decay. Reversing B changes the integrated transfer and forcing work signs while preserving the three energies and molecular losses. Local transfer can have either sign even when its integral is positive.

## Keep numerical conditioning separate from physical closure

The code uses different equivalent expressions where their numerical conditioning differs. Positive variance factors retain small-width limits. Stable amplitude combinations retain differential diffusion after the initial amplitudes cancel. For a broad filter, direct Fourier coefficients preserve small stress gradients and forcing modes that disappear when nearly equal terms are subtracted.

The small-width filtered force uses an exact moment identity. That identity makes its own equation residual an implementation check, not independent proof. Reference calculations instead multiply and filter Fourier polynomials, evaluate physical periodic quadrature and, for selected ordinary cases, integrate the actual infinite-line Gaussian kernel. The direct Fourier-force diagnostic can round differently from the stable force in a difficult limit; it is not an alternative physical model.

## Understand what an AI closure would need to learn

Within a noiseless continuous two-mode model at finite width, Gaussian multipliers are nonzero. The filter is mathematically invertible on those modes. We should not claim that two different exact fields must produce the identical complete filtered field in this restricted family. Inversion becomes poorly conditioned as multipliers shrink, and real data also introduce noise, finite resolution and unresolved spectral content.

An AI closure must therefore state its observation operator, available resolution and training distribution. Filter width, molecular viscosity, forcing, physical units and coordinate conventions belong in that specification. Test residual stress, its divergence and signed energy transfer separately: a good stress fit need not give an accurate gradient. Check different widths and frames without leaking the target modes into the inputs. A two-mode benchmark is a controlled unit of evidence, not proof of a universal closure.

## Six exercises with explained answers

1. **Why does a wider filter suppress the second mode faster?** Its wavenumber is twice as large, so the Gaussian exponent is four times as large. Equal initial amplitudes do not imply equal filtered amplitudes.
2. **Can a positive stress produce negative energy transfer?** Yes. Pi is minus half tau times the resolved gradient. A positive gradient gives negative Pi wherever tau is positive.
3. **Why is the filtered residual not generally zero?** Applying the filter twice gives g-n squared, whereas applying it once gives g-n. A finite-width Gaussian is not a projection onto an unchanged set of Fourier modes.
4. **Why does the worked resolved energy decay at D-r rather than P?** The actual forcing supplies F-r = P. Those two terms cancel in the whole-cell resolved budget, leaving minus D-r.
5. **What happens at zero width?** All multipliers are one; stress, residual energy and transfer vanish. Filtering returns the original field and equation, including the declared moving-frame derivative.
6. **Does exact agreement here validate an AI model on real turbulence?** No. This family has only two modes and prescribed forcing. Broader evidence must cover other spectra, boundaries, noise, resolutions, forcing and physical regimes, while retaining the same declared observables.

## Source and limitations

The [University of Washington ME544 notes on LES](https://courses.washington.edu/mengr544/handouts/hd-les.pdf), especially the filter discussion on pages 5–8, provide background on spatial filtering. Their width convention is converted explicitly in the authoring notes. The manufactured Burgers family, positive variance factorization and energy calculations here are our own specialization. Historical computing-capability statements in those notes are not used as current claims.
