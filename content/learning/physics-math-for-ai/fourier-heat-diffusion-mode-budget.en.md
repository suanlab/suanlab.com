## A rod can become more uniform without losing heat

Imagine an insulated circular rod with alternating warmer and cooler regions. Heat flows from each warm region toward its cooler neighbors. Eventually the temperature becomes almost uniform. Has heat disappeared? No: in a closed periodic model, the total sensible energy stays constant while temperature differences shrink. A falling curve labeled only “energy” can hide the distinction between actual thermal energy and a mathematical measure of nonuniformity.

This chapter follows a single spatial Fourier mode so every part of that distinction can be calculated. We first derive a flux and a local budget, then compare an arbitrary observation window with a complete period. The example also shows why a smooth animation or a small PDE residual is insufficient to identify which physical quantity a model conserves.

## Prerequisites, assumptions and physical units

Review gradients, integration, conservation budgets and exponential decay. Consider a one-dimensional material at rest with constant volumetric heat capacity C and constant nonnegative thermal conductivity kappa. There is no internal heat source, phase transition or radiation. Cross-sectional area is specified and constant. Temperature is absolute temperature in kelvins; a temperature difference of one kelvin has the same magnitude as one Celsius degree, but absolute temperature and a Celsius reading are different quantities.

| Quantity | Unit | Meaning |
| --- | --- | --- |
| T | K | absolute temperature |
| C | J/(m³ K) | energy required per volume per temperature change |
| kappa | W/(m K) | thermal conductivity |
| q | W/m² | signed heat flux through a unit area |
| area | m² | physical rod cross-section |

$$
C=\rho c_p,\qquad [C]=\mathrm{J/(m^3K)},\quad[\kappa]=\mathrm{W/(mK)},\quad[q]=\mathrm{W/m^2}.
$$

The material parameters used for hand calculations are teaching scales, not a calibration for a named metal. Constant heat capacity is a local constitutive approximation over the stated temperature range.

## Derive the heat equation from a signed flux

Fourier's constitutive law states that heat flows down the temperature gradient. Combine it with a local energy balance. Positive q means heat flowing toward increasing x; it does not mean heat entering every observation window. A fixed window has an outward normal of minus one on its left end and plus one on its right end. Tracking this orientation prevents a sign mistake when converting local flux into net power.

$$
q=-\kappa T_x,\qquad C T_t+q_x=0,\qquad T_t=\alpha T_{xx},\qquad\alpha=\frac{\kappa}{C}.
$$

Thermal diffusivity has units of square metres per second. It controls a time scale, whereas conductivity also controls the flux for a specified gradient. Two materials can share a diffusivity yet carry different heat powers because their heat capacities and conductivities differ by a common factor.

## Construct a positive-temperature Fourier mode

Place a signed cosine perturbation on a positive uniform mean. Its wave number is positive and its phase fixes the positions of warm and cool regions. A negative contrast exchanges those regions. Temperature stays positive because the magnitude of the relative contrast is bounded by one fifth. This is a periodic rod model, not a finite rod whose ends are held at prescribed temperatures.

$$
T(x,t)=\overline T+B(t)\cos(kx+\phi),\quad B(t)=\overline T a e^{-\alpha k^2t},\qquad |a|\le0.2.
$$

One period has length two pi divided by k. The mean is constant because the cosine integrates to zero over a full period. An arbitrary shorter interval generally has a nonzero perturbation integral and exchanges heat with its surroundings.

## Verify the decay instead of guessing it

Differentiate the spatial cosine twice. Both derivatives contribute the negative wave-number-squared factor. The same factor appears when the exponential amplitude is differentiated in time. Substitution therefore verifies the heat equation point by point. Increasing wave number makes finer spatial features decay faster, even when the initial temperature contrast is unchanged.

$$
T_t=-\alpha k^2B\cos(kx+\phi),\qquad T_{xx}=-k^2B\cos(kx+\phi),\qquad\tau=\frac1{\alpha k^2}.
$$

The decay time tau is finite only for positive diffusivity. At zero conductivity this family has no temporal decay, even when the initial temperature is nonuniform. That limiting control is different from setting the initial contrast to zero.

## Read flux and temperature as different observations

Flux depends on a gradient, not simply on the local temperature. A temperature maximum has zero gradient and therefore zero local flux, while a mean-temperature crossing can carry a large flux. The flux has a sine spatial pattern, shifted relative to the cosine temperature pattern. Do not expect their maxima to occur at the same coordinate.

$$
T_x=-kB\sin(kx+\phi),\qquad q=\kappa kB\sin(kx+\phi),\qquad qT_x=-\kappa T_x^2\le0.
$$

The last inequality checks the direction of conduction. It does not assert that every finite window must gain or lose heat with the same sign. The two boundary fluxes determine that window's net exchange.

## A worked calculation in watts

Choose mean temperature three hundred kelvins, contrast one tenth, heat capacity two joules per cubic metre-kelvin, conductivity one watt per metre-kelvin, wave number one radian per metre and area two square metres. At time zero and x equal to one half pi metres, the perturbation is zero but the gradient is minus thirty kelvins per metre. Consequently the positive-x heat flux is thirty watts per square metre. Multiplying by the actual area gives sixty watts through that cross-section.

$$
B(0)=30\mathrm{\ K},\quad\alpha=0.5\mathrm{\ m^2/s},\quad x=\pi/2:\quad T=300\mathrm{\ K},\quad T_x=-30\mathrm{\ K/m},\quad q=30\mathrm{\ W/m^2}.
$$

At the temperature maximum x equal to zero, the local flux is zero. Yet a symmetric window around that maximum loses heat through both of its boundaries. A single local flux measurement does not replace an integrated budget.

## Integrate a real finite window

Specify a center and a nonnegative half-width in physical metres. The sensible energy in the window is heat capacity times area times the integral of temperature relative to a stated reference. Energy rate balances the right-minus-left heat power. A positive outward power corresponds to decreasing energy in the window. This budget uses watts, not watts per square metre.

$$
E_W=CA\int_{x_c-h}^{x_c+h}(T-T_{\rm ref})\,dx,\qquad\frac{dE_W}{dt}+A(q_R-q_L)=0.
$$

The perturbation integral can be evaluated with a symmetric identity, avoiding subtraction of nearly equal sine values. The sinc function is defined as one at zero. Retaining the specified half-width also matters when floating-point endpoint coordinates round to the same number.

$$
E'_W=CA\,2hB\,\operatorname{sinc}(kh)\cos(kx_c+\phi),\qquad\operatorname{sinc}(z)=\frac{\sin z}{z},\quad\operatorname{sinc}(0)=1.
$$

Here the prime marks the perturbation contribution to stored energy, not a derivative. With the worked parameters and a window from minus one half pi to plus one half pi metres, this contribution is two hundred forty joules. Its initial rate is minus one hundred twenty watts, balanced by one hundred twenty watts flowing outward.

## A full period conserves sensible energy

The period cell has a physical volume: area times length. We choose a reference temperature of 273.15 kelvins and report sensible energy relative to that reference. Its zero is a convention. A negative reference-relative energy for a colder positive-temperature material would not imply a negative absolute temperature. No extrapolation of constant heat capacity to absolute zero is required.

$$
L=\frac{2\pi}{k},\quad V=AL,\quad E_{\rm cell}=CV(\overline T-T_{\rm ref}),\quad\frac{dE_{\rm cell}}{dt}=0,\quad T_{\rm ref}=273.15\mathrm{\ K}.
$$

For this exact family the perturbation contributes zero net heat over a period. Heat still moves locally. Conservation of total energy and the presence of nonzero internal fluxes are compatible.

## A quadratic norm decreases, with different units

Measure nonuniformity by one half of the volume integral of squared temperature deviation. Integrating the cosine squared gives its mean value of one half. Differentiating this norm or integrating the squared gradient leads to the same decay relation. This is a useful mathematical stability diagnostic, but its units are kelvin-squared times cubic metres, not joules.

$$
N=\frac12\int_V(T-\overline T)^2\,dV=\frac{VB^2}{4},\qquad\dot N=-\alpha\int_VT_x^2\,dV=-2\alpha k^2N.
$$

For the worked parameters the initial norm is nine hundred pi in these norm units. Its decrease expresses smoothing. Labeling that decrease as destroyed thermal energy would contradict the conserved sensible-energy budget. Entropy production is another distinct physical quantity and is not computed by silently renaming this norm.

## Negative controls and what a residual can establish

Turn conductivity off while retaining a nonuniform initial temperature: the selected constitutive law then produces zero heat flux and no evolution. Set contrast to zero instead: temperature is uniform, so flux and evolution vanish even with positive conductivity. Finally keep the actual nonuniform conducting field but omit its boundary power from a window budget. The remaining energy rate exposes the missing contribution; it is not evidence that the original field violates conservation.

$$
R_{\rm omit\ boundary}=\frac{dE_W}{dt},\qquad R_{\rm full}=\frac{dE_W}{dt}+A(q_R-q_L).
$$

Analytically matching two expressions derived from the same formula is a consistency diagnostic. Independent differentiation, spatial integration and finite differences of actual temperature provide additional evidence. None of them certifies a general heterogeneous heat solver or an arbitrary learned constitutive law.

## Small changes and arithmetic observations

For a very small positive decay exponent, the displayed exponential factor can round to one. Computing its change with expm1 retains information that subtracting one from the rounded factor loses. A stable relative change and a stable absolute temperature-amplitude change have different arithmetic ranges: multiplying a tiny relative change by a larger temperature amplitude can recover a representable absolute observation.

$$
z=\alpha k^2t,\qquad\Delta B=B(0)\operatorname{expm1}(-z),\qquad\Delta N=N(0)\operatorname{expm1}(-2z).
$$

At the opposite extreme, the bare exponential can underflow while its product with a larger amplitude is still representable. Combine scaling in log space. Preserve unavailable nonzero observations as unavailable, alongside a rounded total temperature that may equal the mean. A zero-width window has exactly zero stored perturbation and rate; a positive width whose endpoints round together is a different case and retains its specified physical width.

Availability belongs to each observation separately. Suppose the mean is 300 K and a positive temperature perturbation has decayed below the smallest representable number. The total temperature still rounds to 300 K; its value is available even though the perturbation is unavailable. This does not establish an exactly uniform field. Likewise, a finite window can retain a measurable reference-relative sensible energy while its separate mode contribution is below range. Combine the baseline and mode before projecting the total energy. At exactly the reference mean of 273.15 K the baseline vanishes, so a nonzero mode below range leaves the total window energy unavailable too. An exact zero-contrast or zero-width control is a different case.

## Experiment sequence and AI research use

Start with temperature and flux profiles at time zero. Identify a temperature maximum and a mean crossing, then predict where heat flows. Follow a finite window around the warm region and compare its energy rate with oriented boundary power. Expand the observation to a full period and contrast conserved sensible energy with decreasing quadratic norm. Double wave number at fixed diffusivity and predict four times the decay rate. Compare zero conductivity, uniform initial temperature, missing boundary power and tiny nonzero decay.

For a learned model, evaluate the physical flux separately from the temperature reconstruction. A model can predict a plausible smoothed profile while using the wrong conductivity, area or boundary orientation. Hold out diffusivities and wave numbers, and report conservation and nonuniformity as separate tests. The macroscopic Fourier model has instantaneous mathematical propagation; this does not make it a microscopic or relativistic law. Variable materials, walls, sources and coupled fluid transport require additional assumptions and later models.

## Exercises with explanations

### Exercise 1: equal diffusivity

Multiply both conductivity and volumetric heat capacity by two while keeping the initial temperature fixed. Which observations change?

**Answer:** Diffusivity and the temperature history stay the same. The heat flux, stored sensible energy and boundary power double. Matching a temperature movie alone cannot identify both material coefficients independently.

### Exercise 2: finer spatial structure

Double wave number at fixed diffusivity and contrast. How does the exponential decay rate change?

**Answer:** It becomes four times larger because the rate contains wave number squared. The physical period becomes half as long; keep that geometry change explicit when comparing cell energies.

### Exercise 3: local zero flux

The temperature gradient at the center of a warm window is zero. Must the window's energy rate also be zero?

**Answer:** No. Flux at the two boundaries determines the budget. In the worked symmetric window, heat leaves at both ends even though flux at its center is zero.

### Exercise 4: the falling norm

The quadratic temperature norm decreases while total sensible energy is constant. Is conservation violated?

**Answer:** No. The two quantities have different definitions and units. Heat redistributes and temperature becomes more uniform. A norm in K² m³ must not be reported as thermal energy in joules.

### Exercise 5: missing boundary power

A fixed window loses energy, but an evaluation omits boundary transport. What residual remains?

**Answer:** The nonzero energy rate remains. Restoring the outward boundary power balances it. The omission control changes the evaluation, not the actual temperature field.

### Exercise 6: an apparently unchanged factor

The exponential factor rounds to one at positive conductivity. May the experiment be replaced by the exactly zero-conductivity control?

**Answer:** No. Retain the stable expm1 change, the material parameters and any unavailable arithmetic channels. Rounded equality, a sub-range nonzero change and an exactly zero physical decay are distinct observations.

## Sources and scope

[Gilbert Strang's MIT OCW notes on the heat equation](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf) discuss Fourier-mode decay and the distinction between integral conservation and squared-norm decay. Here the dimensional constitutive model, reference-relative sensible energy and finite-window controls are derived explicitly. The companion Gaussian diffusion chapter applies a related mathematical equation to concentration with different physical units.
