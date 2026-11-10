## A lower concentration peak does not mean less material

Imagine releasing a smooth cloud of dissolved material into a very long tube. The cloud spreads and its peak becomes lower. A uniform background flow can also carry the cloud downstream. These are different changes: advection moves its mean, while diffusion increases its variance. Neither process destroys material in the model considered here. Looking only at the highest concentration can therefore produce a false conclusion about mass loss.

We construct a Gaussian concentration field with a finite, explicitly specified mass and physical tube area. This connects familiar Gaussian probability formulas to a conservation law without treating concentration as a probability or a cloud center as a particle trajectory. Every plotted density, flux, integral and moment needs its own definition and units.

## Prerequisites, assumptions and physical units

Review derivatives, Gaussian integrals, mean and variance, and finite-window conservation. The tube is an infinite straight line with constant area. Diffusivity is constant and nonnegative; the prescribed advection velocity is spatially uniform. There are no reactions, sources, walls, variable coefficients or feedback from concentration to flow. Initial standard deviation is strictly positive, so time zero is an ordinary smooth field rather than a singular point source.

| Quantity | Unit | Meaning |
| --- | --- | --- |
| M | kg | total material mass |
| c | kg/m³ | mass concentration |
| area | m² | physical tube cross-section |
| U | m/s | prescribed advecting velocity |
| D | m²/s | diffusivity |
| j | kg/(m² s) | signed total mass flux |

$$
M=A\int_{-\infty}^{\infty}c(x,t)\,dx,\qquad[c]=\mathrm{kg/m^3},\qquad[j]=\mathrm{kg/(m^2s)}.
$$

These variables describe a dilute scalar teaching model. They do not automatically supply the density, pressure or momentum equations of a mixture.

## Derive the equation from advection and Fick flux

Fick's constitutive law assigns diffusive flux opposite to the concentration gradient. Advection contributes velocity times concentration. Their signed sum enters continuity. Because U and D are constant, expanding the flux derivative gives the usual advection-diffusion equation. The sign convention matters: in our convention a positive U moves the mean toward increasing x.

$$
j_{\rm adv}=Uc,\qquad j_{\rm diff}=-Dc_x,\qquad c_t+j_x=0,\qquad c_t+Uc_x=Dc_{xx}.
$$

Flux is not a speed. Dividing it by concentration can produce a local ratio where concentration is positive, but that ratio should not silently be assigned as the deterministic trajectory of every molecule. Diffusion describes spreading, including microscopic fluctuations that this continuum field does not explicitly simulate.

## Construct the smooth Gaussian family

Let the initial cloud have total mass M, mean mu-zero and standard deviation sigma-zero. Its mean translates uniformly and its variance grows linearly. Use variance, rather than standard deviation, in the growth law. Keeping the normalization factor makes spreading lower the peak without changing the integral. The area converts a mass-per-length profile into a physical volume concentration.

$$
\mu(t)=\mu_0+Ut,\qquad v(t)=\sigma_0^2+2Dt,\qquad c(x,t)=\frac{M}{A\sqrt{2\pi v(t)}}\exp\!\left[-\frac{(x-\mu(t))^2}{2v(t)}\right].
$$

The Gaussian has no sharp finite edge. Its positive tails are part of the field, not a plotting defect to remove and renormalize. A limited display interval is an observation window, not the physical boundary of the infinite tube.

## Differentiate the actual field

Write displacement from the current mean as delta. Differentiating the exponential gives the spatial gradient. A second derivative includes both a positive displacement-squared term and a negative normalization-scale term. The time derivative must account for both mean motion and variance growth. Ignoring the changing normalization factor would break mass conservation even if the profile broadened visually.

$$
\delta=x-\mu,\quad c_x=-\frac{\delta}{v}c,\quad c_{xx}=\left(\frac{\delta^2}{v^2}-\frac1v\right)c,\quad c_t=\left[\frac{U\delta}{v}+D\left(\frac{\delta^2}{v^2}-\frac1v\right)\right]c.
$$

Substitution verifies the PDE on this family. A residual computed from these shared formulas is an analytic consistency diagnostic. Independent differences of actual concentrations and independent spatial quadrature provide stronger checks of an implementation's signs, units and normalization.

## Interpret conserved mass and normalized moments

Multiply concentration by area and integrate over the entire line to recover total mass. With positive M, divide the first and central second mass moments by M. The resulting mean and variance are exactly the declared Gaussian shape parameters. They are global observations, not a statement that all material is located at the mean.

$$
A\int c\,dx=M,\qquad\frac{A}{M}\int xc\,dx=\mu,\qquad\frac{A}{M}\int(x-\mu)^2c\,dx=v\quad(M>0).
$$

The mean can move while the shape spreads, and the mass can remain constant while any particular finite window gains or loses material. Keep these scopes separate before comparing a network prediction with a conservation target.

## A hand calculation with a finite mass

Choose two kilograms of material, unit area, initial standard deviation half a metre, initial mean zero, advection speed half a metre per second and diffusivity one eighth square metre per second. After one second, the variance has grown from one quarter to one half square metre, and the mean has moved to half a metre. Peak concentration becomes two divided by the square root of pi kilograms per cubic metre. Total mass remains two kilograms.

$$
M=2,\ A=1,\ \sigma_0=0.5,\ \mu_0=0,\ U=0.5,\ D=0.125,\ t=1:\quad\mu=0.5\mathrm{\ m},\quad v=0.5\mathrm{\ m^2},\quad c_{\max}=\frac2{\sqrt\pi}\mathrm{\ kg/m^3}.
$$

At the current mean, diffusive flux is zero by symmetry. Advective flux there is still positive because U and concentration are positive. Zero diffusive flux does not imply zero total flux.

## Compare height, width and direction of flux

The peak is inversely proportional to standard deviation. Doubling standard deviation halves peak height at fixed mass and area; it quadruples variance. On the left side of a symmetric cloud the diffusive flux points left, and on the right it points right. A sufficiently large positive advection can make the total flux positive even on the left side.

$$
c_{\max}(t)=\frac{M}{A\sqrt{2\pi v}},\qquad j_{\rm diff}=\frac{D(x-\mu)}v c,\qquad j=\left(U+\frac{D(x-\mu)}v\right)c.
$$

Do not use a single color or arrow to hide the decomposition. A net flux can become small through cancellation while each contribution remains nonzero. Its sign concerns transport across an oriented surface, not the sign of concentration, which stays nonnegative.

## Measure the mass in a finite physical window

For fixed left and right positions, integrate concentration across the actual interval and multiply by area. The Gaussian formula can be written with a standard-normal CDF difference. This is a mathematical identity, not a requirement to implement the difference directly. In a far tail both CDF values may round to one, making their floating-point difference zero even when the interval contains nonzero material.

$$
M_W=A\int_a^b c\,dx=M\left[\Phi\!\left(\frac{b-\mu}{\sqrt v}\right)-\Phi\!\left(\frac{a-\mu}{\sqrt v}\right)\right].
$$

We evaluate the physical-width integral directly with log-scaled quadrature. The physical endpoints and their difference are preserved, even when standardized endpoints round together. Comparing two quadrature orders is a diagnostic, not a rigorous error certificate.

## Verify the window budget independently

The outward normal is negative on the left and positive on the right. Multiply the signed flux difference by tube area to obtain kilograms per second. Integrating the actual time derivative of concentration gives the storage rate. This can be checked separately from the two face fluxes and from finite differences of stored mass.

$$
\frac{dM_W}{dt}+A[j(b,t)-j(a,t)]=0.
$$

If the cloud is spreading out of a window around its mean, mass in that window can fall while total mass remains conserved. If the cloud is being advected into the window, it can rise. Omitting the boundary flux would create a false conservation failure. Assuming that every finite window contains all of M would create a different error.

## Separate exact limiting controls

At zero D, variance remains fixed and the Gaussian translates without broadening. At zero U, the mean remains fixed while variance grows. At positive D and time zero, the initial field is unchanged but its time derivative need not vanish: evolution begins immediately. These controls probe different terms of the equation.

$$
D=0:\ c(x,t)=c(x-Ut,0),\qquad U=0:\ \mu=\mu_0,\qquad\frac{dv}{dt}=2D.
$$

Neither control introduces a wall or a reaction. The initial positive standard deviation is retained; replacing it with zero would create a singular initial condition outside this implementation's domain.

## Zero mass has no mass-normalized mean

If M is exactly zero, concentration and all physical fluxes are zero. There is no mass by which to normalize moments. The parameterized shape can still have a declared center and variance, but these are not the mean and variance of observed material. Reporting zero as the “measured mean” would manufacture an observation rather than mark its absence.

$$
M=0:\quad c=j=0,\qquad\frac{A\int xc\,dx}{M}\ \text{is undefined}.
$$

This distinction is useful for data pipelines too: an empty observation and a nonempty observation centered at zero are different states. Preserve the physical rest control and the unavailable normalized channel separately.

## Retain far tails and logarithmic observations

For positive M, evaluate concentration by combining normalization and exponential decay before converting back to an ordinary number. A bare Gaussian exponential can underflow although the scaled concentration is representable. Conversely concentration itself may be outside arithmetic range while its derivative, multiplied by a large spatial factor, remains representable. Losing one channel does not justify erasing all others.

$$
\ell=\log\!\left(\frac c{c_*}\right)=\log\!\left(\frac{M}{A\sqrt{2\pi v}\,c_*}\right)-\frac{(x-\mu)^2}{2v},\qquad c_*=1\mathrm{\ kg/m^3}.
$$

The reference c-star makes the logarithm dimensionless. A finite log concentration with an unavailable ordinary value means a nonzero positive tail, not a physical zero. Likewise a finite log interval fraction can remain useful when the fraction itself underflows. Keep log mass readings referenced to one kilogram if they are displayed. Zero mass has no finite logarithm and must be handled as a separate exact case.

## Preserve small spreading below rounded equality

For tiny positive D or time, adding the variance increment to the initial variance can round back to the initial number. The increment can still be representable. Avoid subtracting nearly equal square roots to obtain the standard-deviation change. Rational identities also preserve a peak change that a rounded peak ratio might erase.

$$
\Delta v=2Dt,\qquad\Delta\sigma=\frac{\Delta v}{\sqrt{v_0+\Delta v}+\sqrt{v_0}},\qquad v_0=\sigma_0^2.
$$

With eta equal to relative variance growth, the peak ratio follows from the same normalization factor. The stable expression below retains its negative change even when the rounded square root equals one.

$$
\eta=\frac{\Delta v}{v_0},\qquad\frac{c_{\max}(t)-c_{\max}(0)}{c_{\max}(0)}=-\frac{\eta}{\sqrt{1+\eta}\,[\sqrt{1+\eta}+1]}.
$$

If a nonzero increment cannot be represented even after stable scaling, mark that channel unavailable. Do not replace it with the exactly zero-diffusivity control. Rounded mean motion and a stable mean displacement also need distinct labels when a small translation is lost in addition.

## Experiment sequence and AI research use

Start with zero advection and inspect left/right diffusive fluxes. Increase time, compare peak concentration with variance, and verify total mass using a sufficiently broad physical integral or an independent full-domain calculation. Then activate advection and separate mean translation from spreading. Track a fixed window as the cloud enters and leaves it. Compare zero diffusivity, zero mass, far-tail windows and tiny positive variance changes.

For a learned transport model, report errors in concentration, flux, total mass, mean and variance separately. A peak-fitting loss can reward the wrong width; a finite display window can hide missing tail mass; and normalizing every predicted frame can conceal an incorrect PDE flux. Hold out advection speeds and diffusivities, preserve actual area and units, and use boundary budgets before interpreting local mass changes. This Gaussian field is a controlled benchmark for constant coefficients, not evidence of correct reactions, walls, mixture dynamics or individual molecular paths.

## Exercises with explanations

### Exercise 1: variance or standard deviation

For diffusivity one eighth square metre per second, what grows by one quarter square metre each second?

**Answer:** Variance, because its growth rate is twice diffusivity. Standard deviation is the square root of variance and generally does not grow linearly in time.

### Exercise 2: a lower peak

Standard deviation doubles at fixed mass and tube area. What happens to peak concentration and total mass?

**Answer:** Peak concentration halves while total mass stays fixed. The cloud is wider. A lower peak alone is not evidence of reaction or mass loss.

### Exercise 3: symmetry and total flux

At the current mean, diffusive flux is zero. Must total flux also be zero?

**Answer:** No. Total flux includes U times concentration. It is nonzero there when advection and concentration are nonzero, even though the diffusive contribution vanishes by symmetry.

### Exercise 4: the empty cloud

With zero total mass, may the declared Gaussian center be reported as a measured mass-normalized mean?

**Answer:** No. The physical field is zero and normalization would divide by zero. Keep a declared shape parameter separate from an unavailable mean of material.

### Exercise 5: subtracting tail CDF values

Two standard-normal CDF values both round to one. Does their zero difference prove that no material lies between the physical endpoints?

**Answer:** No. Evaluate the interval directly or retain a stable log integral. Keep the original physical width; do not erase the tail or renormalize the remaining visible profile.

### Exercise 6: local mass change

Mass in a fixed window decreases while a full-domain reference retains the same total mass. What must you check before declaring a conservation failure?

**Answer:** Check the oriented total boundary flux, including advection and diffusion and multiplying by actual area. Outflow can explain the storage change. Also distinguish quadrature error and unavailable tiny channels from physical absence.

## Sources and scope

[Gilbert Strang's MIT OCW heat and convection-diffusion notes](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf) discuss Gaussian spreading and the fundamental solution. [MIT's diffusion illustration](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec09/heat/vorticity.html) motivates varying a small diffusivity. Here mass normalization, physical tube area, flux signs and arithmetic controls are derived explicitly. The illustration's initial-width parameter uses a different convention from our standard deviation; its old Java applet is not used as verification.
