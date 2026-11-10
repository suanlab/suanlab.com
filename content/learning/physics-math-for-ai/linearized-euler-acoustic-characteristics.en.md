## A sound signal does not simply travel with the fluid

Imagine speaking while standing beside a gently moving air stream. The air transports material, but a pressure disturbance also propagates relative to the air. Consequently, a laboratory observer can see information moving in three different ways: with the stream, downstream relative to it, or upstream relative to it. This distinction is essential when a neural model predicts the next frame of a flow field. A sequence can look smooth while transporting pressure at the wrong speed.

This chapter constructs a smooth, small-perturbation example that you can differentiate by hand. It separates the exact solution of a linear system from the approximation made when that system represents a nonlinear fluid. Neither a low training loss nor an attractive animation removes that modeling distinction.

## Prerequisites and the physical quantities

Review partial derivatives, conservation of mass, traveling waves and matrix eigenvalues. Consider a one-dimensional inviscid ideal gas without body forces or heat transfer. Let the uniform background have density rho-zero, velocity U and pressure p-zero. The ratio of specific heats gamma closes the pressure equation. Sound speed is a positive background property, rather than the instantaneous material velocity. Primes below mean perturbations, not derivatives; write derivatives with explicit subscripts.

| Quantity | Unit | Interpretation |
| --- | --- | --- |
| density and density perturbation | kg/m³ | mass per physical volume |
| U, sound speed and velocity perturbation | m/s | distinct speeds |
| pressure and pressure perturbation | Pa | force per area |
| wave number | rad/m | spatial phase gradient |

$$
\rho=\rho_0+\rho',\quad v=U+u',\quad p=p_0+p',\qquad c^2=\frac{\gamma p_0}{\rho_0},\quad D_0=\partial_t+U\partial_x.
$$

## Derive the linear system by keeping first-order terms

Start with continuity, the primitive momentum equation and the ideal-gas pressure equation. Substitute the background plus perturbations. Uniform background derivatives vanish. For example, the continuity product generates both the background-density times velocity-gradient term and products of two perturbations. Retain the former and omit the latter. The approximation concerns amplitude, not permission to delete any inconvenient derivative. Short wavelengths can still create large gradients and require careful interpretation of dimensional residuals.

$$
\rho_t+v\rho_x+\rho v_x=0,\qquad v_t+v v_x+\frac{p_x}{\rho}=0,\qquad p_t+v p_x+\gamma p v_x=0.
$$

The resulting equations are coupled: a pressure gradient changes velocity, and velocity divergence changes density and pressure. Omitting either coupling destroys acoustic propagation.

$$
D_0\rho'+\rho_0u'_x=0,\qquad D_0u'+\frac{p'_x}{\rho_0}=0,\qquad D_0p'+\rho_0c^2u'_x=0.
$$

## Change coordinates from fields to characteristic information

Add the velocity equation to the pressure equation divided by acoustic impedance. Subtraction produces the other acoustic combination. Subtracting sound-speed-squared times continuity from the pressure equation produces a third variable. Each combination now obeys a scalar transport equation. This is the practical meaning of diagonalizing the flux matrix: combinations of fields propagate independently even though the original fields are coupled.

$$
w_+=u'+\frac{p'}{\rho_0c},\quad w_-=u'-\frac{p'}{\rho_0c},\quad s=p'-c^2\rho'.
$$

The variable s has pressure units. It is an entropy-like contact variable in this linearization, not thermodynamic entropy measured in joules per kilogram-kelvin.

$$
(\partial_t+(U+c)\partial_x)w_+=0,\quad(\partial_t+(U-c)\partial_x)w_-=0,\quad D_0s=0.
$$

## Build an explicit three-mode field

Choose dimensionless signed amplitudes for right-going sound, left-going sound and a contact perturbation. A common phase makes a convenient example; it does not constrain the general solution to have identical phases. The wave number must stay positive. Changing the mean velocity translates all three characteristic speeds, while changing sound speed changes their separation. A negative amplitude reverses a mode's sign without reversing its propagation direction.

$$
A=a_+\cos(k[x-(U+c)t]+\phi),\quad B=a_-\cos(k[x-(U-c)t]+\phi),\quad C=a_0\cos(k[x-Ut]+\phi).
$$

Reconstruction makes the contact contribution explicit. It changes density while leaving pressure and velocity unchanged. That property provides a useful negative control for an acoustic visualization.

$$
u'=c(A+B),\qquad p'=\rho_0c^2(A-B),\qquad\rho'=\rho_0(A-B+C),\qquad s=-\rho_0c^2C.
$$

## Read a space-time sketch correctly

Read time on the horizontal axis and position on the vertical axis, as in the interactive figure. A tagged crest follows one of three straight lines. Its slope is displacement per unit time: the laboratory characteristic speed. Keep this axis convention when comparing traces. A stationary density contact is possible when the background velocity is zero; pressure waves still move in both directions.

$$
x_+(t)=x_+(0)+(U+c)t,\quad x_-(t)=x_-(0)+(U-c)t,\quad x_0(t)=x_0(0)+Ut.
$$

Do not label these crest paths as particle trajectories. A material particle follows the total local velocity evaluated at its changing position. Even in small-amplitude acoustics, that is a different differential equation. Smooth curves alone do not identify which object is being tracked.

## A hand calculation with physical units

Take the background density as one kilogram per cubic metre, sound speed as two metres per second and mean velocity as half a metre per second. Activate only a right-going amplitude of one hundredth. At a crest, velocity perturbation is two hundredths of a metre per second and pressure perturbation is four hundredths of a pascal. The characteristic moves at two and a half metres per second in the laboratory. This deliberately scaled teaching gas is not a calibration for room-temperature air.

$$
\rho_0=1,\ c=2,\ U=0.5,\ a_+=0.01:\quad u'=0.02\mathrm{\ m/s},\quad p'=0.04\mathrm{\ Pa},\quad\rho'=0.01\mathrm{\ kg/m^3}.
$$

## Check an actual finite observation window

A local equation also predicts a budget for a fixed spatial interval. Integrate the time derivative over that interval and compare it with the oriented difference of flux at its two ends. An arbitrary interval need not contain an integer number of wavelengths; periodic cancellation must not be assumed. For density perturbation, both background advection and the perturbed velocity carry the linearized mass flux.

$$
\frac{d}{dt}\int_a^b\rho'\,dx+[U\rho'+\rho_0u']_a^b=0.
$$

The integral has mass-per-area units in this one-dimensional model. Multiply by an explicitly specified cross-sectional area to obtain mass. Reporting it directly as kilograms would silently introduce missing geometry.

## A quadratic acoustic energy and its flux

The linear equations conserve a positive acoustic energy for the velocity-pressure subsystem. Multiply the velocity equation by background density times velocity perturbation, and multiply the pressure equation by pressure perturbation divided by background density times sound-speed-squared. Add them and use the product rule. Background advection contributes a flux, while pressure work contributes another. The contact mode has no energy in this particular quadratic definition; that does not make its material density variation nonexistent.

$$
q=\frac{\rho_0u'^2}{2}+\frac{p'^2}{2\rho_0c^2},\qquad q_t+\partial_x(Uq+p'u')=0,\qquad[q]=\mathrm{J/m^3}.
$$

This quantity is distinct from the complete nonlinear gas energy, which includes internal energy and background motion. A plot must identify which definition it uses before comparing budgets.

## Insert the same field into nonlinear Euler

Evaluate the original nonlinear equations on the reconstructed total fields. The linear residual should be at rounding scale, but the nonlinear residual generally is not zero. In continuity the remaining terms are products of perturbations. In the pressure equation they are also quadratic. The primitive velocity residual contains division by total density, so its amplitude scaling approaches quadratic behavior rather than obeying an exact factor-four relation at every finite amplitude.

$$
R_\rho^{NL}=u'\rho'_x+\rho'u'_x,\qquad R_p^{NL}=u'p'_x+\gamma p'u'_x,\qquad R_v^{NL}=u'u'_x+\left(\frac1{\rho_0+\rho'}-\frac1{\rho_0}\right)p'_x.
$$

Halving all amplitudes quarters the first two residuals. Changing wavelength changes derivative scales as well. These are diagnostics of a chosen family, not rigorous global error bounds for arbitrary Euler solutions.

## Positivity, floating-point observations and limits

Small amplitudes preserve positive density and pressure. The teaching range limits the absolute value of each dimensionless amplitude to three hundredths; gamma stays between 1.1 and 1.67. Density and sound speed remain strictly positive. A bound on the sum of mode magnitudes gives a global positivity check without sampling every position. Positive totals alone do not establish that a perturbation is sufficiently small for every intended experiment.

$$
\frac{|\rho'|}{\rho_0}\le |a_+|+|a_-|+|a_0|,\qquad\frac{|p'|}{p_0}\le\gamma(|a_+|+|a_-|).
$$

A tiny contact variable can disappear when subtracting two rounded pressure-scale observations. Keep the stable expression from the contact mode alongside that subtraction. Likewise, a squared tiny amplitude can underflow even while the velocity perturbation remains representable. An unavailable quadratic observation must not be displayed as evidence of a physically zero disturbance. This smooth example provides no shock formation, entropy-production or wall boundary-layer calculation.

Availability belongs to each reported quantity. For example, multiplying a very small mode by background density can produce a density perturbation below the display range. Adding that exact contribution to the positive background can still give a representable total density. The same distinction applies to pressure and a nonzero mean velocity. We combine the declared mode contributions before rounding each total, derivative and quadratic energy separately. An unavailable perturbation therefore does not automatically make the total field unavailable. The finite-window quadrature is a separate approximation using sampled observations, and can retain its own arithmetic limitations.

## Experiment sequence and connection to AI research

First activate a single acoustic mode and compare its crest displacement with its characteristic speed. Then add the opposite acoustic mode to see interference without changing either speed. Activate only the contact mode and verify that density moves without an acoustic pressure signal. Change the background velocity through the point where the left-going laboratory speed becomes zero. Finally halve all amplitudes and compare linear and nonlinear residuals on the same coordinates and time.

For a physics-informed model, record units and normalization separately for the three equation residuals: adding their dimensional magnitudes is not an invariant score. Hold out a mean-flow or sound-speed setting and test propagation, not only pointwise interpolation. A network can exploit a restricted training phase or confuse pressure-free contact with missing data. Check characteristic motion, finite-window budgets and the negative control as separate observations. These tests probe this family; they do not certify a general compressible-flow surrogate.

## Exercises with explanations

### Exercise 1: a stationary laboratory signal

Let U equal the sound speed. Which characteristic has zero laboratory speed?

**Answer:** The minus acoustic characteristic has speed U minus c, which is zero. The plus characteristic still moves at twice the sound speed, and the contact variable moves at U. Stationary acoustic information does not imply stationary fluid.

### Exercise 2: a pressure-free disturbance

Set both acoustic amplitudes to zero and keep a nonzero contact amplitude. Describe all three perturbation fields.

**Answer:** Velocity and pressure perturbations vanish. Density perturbation equals background density times the contact mode and translates with U. Its acoustic energy q is zero by definition, although the density field remains nonuniform.

### Exercise 3: impedance and units

For a single right-going mode, divide pressure perturbation by velocity perturbation away from their common zeros.

**Answer:** The ratio is background density times sound speed, with units kg/(m² s), or Pa divided by m/s. It is acoustic impedance, not a dimensionless correlation coefficient. At a simultaneous zero, use the mode relation rather than a numerical zero-over-zero division.

### Exercise 4: amplitude refinement

Halve all three amplitudes at fixed background and wave number. What happens to the nonlinear mass and pressure residuals?

**Answer:** Each becomes one quarter of its previous value because every remaining term contains two perturbation factors. The linear residual remains at arithmetic noise. This result does not justify quartering a residual after changing wave number as well.

### Exercise 5: a finite interval

The integral of density perturbation is increasing in a fixed window. What sign must the right-minus-left linearized mass flux have?

**Answer:** It must be negative: more linearized mass flux enters than leaves. A nonzero change inside a finite window is consistent with conservation; setting the boundary contribution to zero without checking it would manufacture an apparent violation.

### Exercise 6: a misleading energy measurement

A nonzero velocity perturbation is representable, but its square underflows in the acoustic energy calculation. Can you conclude there is no acoustic disturbance?

**Answer:** No. Preserve the nonzero field observation and mark the energy channel unavailable when its nonzero product cannot be represented. Distinguish arithmetic range from physical absence, and keep the acoustic energy definition distinct from total nonlinear gas energy.

## Sources and scope

[NASA Glenn's Euler equations introduction](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/euler-equations-2/) explains the coupled fields and inviscid approximation. The characteristic reconstruction and diagnostics above are derived here for the stated uniform ideal-gas linearization. NASA's displayed steady two-dimensional equations are not being cited as this unsteady three-equation derivation. The companion Taylor–Green chapter treats a different, incompressible nonlinear family.
