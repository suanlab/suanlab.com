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

## Research extension: incoming boundaries and a norm that sees contact

A neural predictor may fit every interior equation while importing an unmeasured disturbance through an inappropriate boundary. Consider a **separate finite-interval problem** on 0 < x < L with the same uniform ideal-gas linearization. Here the background is strictly subsonic and rightward. Keep the acoustic characteristics from above, and rescale the contact variable to a velocity h. This rescaling lets all three squared amplitudes use one declared dimensional scale; it does not make the contact signal acoustic.

$$
w_+=u'+\frac{p'}{\rho_0c},\quad w_-=u'-\frac{p'}{\rho_0c},\quad h=\frac{s}{\rho_0c}=\frac{p'-c^2\rho'}{\rho_0c},\qquad 0<U<c.
$$

Each component is transported at its own constant speed. A plus characteristic reaches the interval through the left end; a minus characteristic reaches it through the right end. The contact signal also enters from the left. Initially compatible smooth data and boundary data are assumed. The original interactive experiment uses analytic waves on the line; it does not solve this new reflecting-boundary problem.

$$
(\partial_t+(U+c)\partial_x)w_+=0,\quad(\partial_t+(U-c)\partial_x)w_-=0,\quad(\partial_t+U\partial_x)h=0.
$$

The earlier acoustic density q sees velocity and pressure but misses a pure contact disturbance. Add a positive contact weight beta to obtain a norm of the **full linearized state**. This norm vanishes only when all three characteristic variables vanish. Beta is an arbitrary positive dimensionless comparison weight. Although the units match energy density, the new term is not a derived thermodynamic internal energy, and the whole expression is not the nonlinear gas energy.

$$
\mathcal E_\beta=\frac{\rho_0}{4}(w_+^2+w_-^2)+\frac{\beta\rho_0}{2}h^2,\quad\beta>0,\qquad[\mathcal E_\beta]=\mathrm{J/m^3}.
$$

Multiply each transport equation by its weighted characteristic amplitude and add. Constant coefficients turn every term into a derivative of a square. The acoustic flux has one positive and one negative speed contribution, whereas contact is advected with the background. This local identity is independent of a particular sinusoidal initial condition.

$$
\partial_t\mathcal E_\beta+\partial_x\mathcal F_\beta=0,\qquad\mathcal F_\beta=\frac{\rho_0}{4}\big[(U+c)w_+^2+(U-c)w_-^2\big]+\frac{\beta\rho_0U}{2}h^2.
$$

Integration gives the signed boundary budget. The integral has units J/m², and its time derivative has units W/m². A specified cross-sectional area would turn them into J and W. At the left end two independent incoming signals need rules; at the right end only one does. Prescribing density, velocity and pressure independently at both ends generally constrains outgoing information that is already determined by the interior.

$$
\frac{d}{dt}\int_0^L\mathcal E_\beta\,dx=\mathcal F_\beta(0,t)-\mathcal F_\beta(L,t),\qquad\text{incoming: }(w_+,h)|_{x=0},\ w_-|_{x=L}.
$$

Normalize acoustic amplitudes by their **norm flux**, rather than comparing raw velocities. Squared eta has units W/m². Equal eta magnitudes carry equal magnitudes of this acoustic norm flux even though the two laboratory propagation speeds differ. These are mathematical norm-flux variables, not an assumption about a physical sound-power meter.

$$
\eta_+=\sqrt{\frac{\rho_0(U+c)}4}\,w_+,\qquad\eta_-=\sqrt{\frac{\rho_0(c-U)}4}\,w_-,\qquad\mathcal F_\beta=\eta_+^2-\eta_-^2+\frac{\beta\rho_0U}{2}h^2.
$$

For a concrete homogeneous boundary contract, make each incoming acoustic eta a real scalar multiple of the outgoing eta at that end, with magnitude at most one. Set incoming contact to zero. An initial contact signal can leave at the right, but no contact signal is injected at the left. Negative reflection coefficients reverse sign and leave the squared budget unchanged.

$$
\eta_+(0,t)=r_L\eta_-(0,t),\quad\eta_-(L,t)=r_R\eta_+(L,t),\quad h(0,t)=0,\quad |r_L|,|r_R|\le1.
$$

Substitution makes every boundary contribution nonpositive. Thus the augmented integral cannot increase. Notice the reciprocal conversion factors for raw w amplitudes: the right-end incoming amplitude may exceed its outgoing counterpart while its incoming norm flux remains bounded. This calculation is a mathematical stability statement for the stated homogeneous contract. It does not identify a rigid-wall reflection law in a gas with nonzero mean throughflow, or certify thermodynamic passivity of a device.

$$
\begin{aligned}w_+(0,t)&=r_L\sqrt{\frac{c-U}{c+U}}\,w_-(0,t),\\w_-(L,t)&=r_R\sqrt{\frac{c+U}{c-U}}\,w_+(L,t),\\\frac{d}{dt}\int_0^L\mathcal E_\beta\,dx&=(r_L^2-1)\eta_-^2(0,t)+(r_R^2-1)\eta_+^2(L,t)-\frac{\beta\rho_0U}{2}h^2(L,t)\le0.\end{aligned}
$$

As a negative control, equate raw incoming and outgoing w at the left. The net acoustic flux is then positive: equal amplitudes have unequal transport speeds. If the right-end signal and contact signal vanish at that instant, the integrated norm grows. A pulse following a full acoustic round trip takes T; the normalized amplitude on its original branch is multiplied by rL times rR. This statement concerns noninteracting constant-speed characteristics and the declared contracts, not a simulation of nonlinear reflections.

$$
w_+(0,t)=w_-(0,t)=w\ \Longrightarrow\ \mathcal F_{\rm acoustic}(0,t)=\frac{\rho_0U}{2}w^2>0\quad(U>0,\ w\ne0),\qquad T=\frac{L}{U+c}+\frac{L}{c-U}.
$$

A second control isolates what the observation misses. Extend the initial contact profile by zero outside the interval and advect it at U with zero incoming contact. Both acoustic modes remain zero, so q is identically zero while density differs from the background. Since rho-prime = -rho-zero h/c in this control, choose |h-zero|/c small enough to retain positive density and the small-perturbation regime. The sin-to-the-fourth integral gives a strictly positive initial augmented norm. A pressure-only evaluation can therefore report perfect agreement while missing transported density information.

$$
w_+=w_-=0,\quad h(x,0)=h_0\sin^2(\pi x/L)\ (0\le x\le L):\qquad q=0,\quad\int_0^L\mathcal E_\beta(x,0)\,dx=\frac{3\beta\rho_0h_0^2L}{16}>0\quad(h_0\ne0).
$$

The table uses Mach number M = U/c. Columns are M, the left raw-amplitude conversion at rL = 1, the right conversion at rR = 1, round-trip time Tc/L, and the incoming/outgoing flux ratio of the incorrect equal-raw-amplitude rule at the left. All are dimensionless. A large right conversion near sonic flow does not contradict the norm contract; the outgoing and incoming speeds are different.

| M | left factor | right factor | Tc/L | raw flux ratio |
| --- | --- | --- | --- | --- |
| 0.10000000 | 0.90453403 | 1.10554160 | 2.02020202 | 1.22222222 |
| 0.30000000 | 0.73379939 | 1.36277029 | 2.19780220 | 1.85714286 |
| 0.60000000 | 0.50000000 | 2.00000000 | 3.12500000 | 4.00000000 |
| 0.90000000 | 0.22941573 | 4.35889894 | 10.52631579 | 19.00000000 |

```python
from math import sqrt
for mach in (0.1, 0.3, 0.6, 0.9):
    left = sqrt((1-mach)/(1+mach))
    right = 1/left
    trip = 2/(1-mach*mach)
    raw_flux_ratio = (1+mach)/(1-mach)
    print(" ".join(f"{x:.8f}" for x in (mach, left, right, trip, raw_flux_ratio)))
```

The [Guaily–Epstein paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4293881/) supplies incoming-characteristic boundary background. The augmented norm, speed-weighted contracts, contact control and table here are derived separately. They are not a new boundary control in the existing lab.

![Incoming and outgoing norm fluxes, speed conversions and the invisible acoustic contact direction](/assets/learning/continuum-subsonic-incoming-norm.en.svg)

### Research questions with explanations

**Can a zero acoustic loss certify zero density error?** **Answer:** No. In the contact-only control, q is zero but density is nonuniform. Measure a full-state quantity or report that the acoustic observation has a contact null direction.

**Does a raw reflection coefficient of one imply zero boundary injection?** **Answer:** No. At the left boundary with U positive it produces positive acoustic norm flux. Convert using the declared characteristic-speed weights before judging the contract.

**Can the same formula be used at a sonic point by clipping its denominator?** **Answer:** No. At U = c one characteristic has zero laboratory speed, and at U = 0 contact is stationary. Recount incoming and stationary information; do not invent a finite boundary conversion by clipping.

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
