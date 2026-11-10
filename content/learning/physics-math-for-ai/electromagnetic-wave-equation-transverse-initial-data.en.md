## Same initial electric field, different futures

If you know the electric-field shape perfectly at one instant, do you know its future? Generally you also need initial magnetic field or compatible time-change information. The same initial Gaussian E can propagate one way or split in two. We first study field compatibility and propagation; quantitative energy-flow analysis comes later.

The preceding sinusoidal plane wave was one exact solution. Smooth nonperiodic pulses now connect the wave equation, first-order Maxwell relations, and initial data. The goal is to avoid solving a scalar wave equation and attaching an unrelated magnetic field afterward.

## Constant coefficients and transverse assumptions

The positive coefficients ε and μ are constant, with c=1/√(εμ), in a vacuum-form SI teaching model. Default ε is derived from the rounded teaching μ and SI c; arbitrary coefficient choices do not constitute a complete real-material model.

E has only an x component, B only a y component, and spatial variation is only along z. These fields are transverse to the propagation axis. This does not assert that every electromagnetic field is transverse: near sources, in waveguides, and at material boundaries, other components and constraints matter.

$$
\mathbf E=(E_x(z,t),0,0),\qquad\mathbf B=(0,B_y(z,t),0),\qquad c=\frac{1}{\sqrt{\epsilon\mu}}.
$$

## Start from the two first-order evolution laws

With these components both divergences vanish structurally. Faraday and Ampere–Maxwell reduce to the following first-order equations. Each field restricts the spatial and temporal variation of the other. Check their signs directly from curl components.

The initial z derivative of B determines the initial time derivative of E, and the initial z derivative of E determines the initial time derivative of B. You cannot prescribe all four initial fields and rates independently.

$$
\partial_z E_x=-\partial_tB_y,\qquad-\partial_zB_y=\frac{1}{c^2}\partial_tE_x.
$$

## Derive the wave equation and state its limits

Differentiate the first equation in time and use the z derivative of the second to derive the E wave equation. B satisfies the same form. This proves that first-order Maxwell compatibility implies the second-order wave equations; it does not prove an unconditional converse.

Differentiation can lose the original first-order relation between the fields. Just as Gauss constraints and initial or boundary data need checking, small componentwise wave residuals cannot replace full Maxwell verification.

$$
\partial_t^2E_x=c^2\partial_z^2E_x,\qquad\partial_t^2B_y=c^2\partial_z^2B_y.
$$

## Two profiles moving in opposite directions

The profile f(z−ct) translates along +z and g(z+ct) along −z. Adding them produces E, while B requires a relative sign that depends on propagation direction. That sign makes both first-order equations hold.

With fixed field axes, a sign change in B reverses its component direction. Where the profiles overlap, the single-direction relation B=E/c cannot be imposed on the total sum. Apply it to each travelling component before superposing.

$$
E_x=f(z-ct)+g(z+ct),\qquad B_y=\frac{f(z-ct)-g(z+ct)}{c}.
$$

## Split one Gaussian initial profile

Amplitude A is in V/m, width σ in m, and β is dimensionless between zero and one. Define the travelling components to share one initial shape. β=1 gives right-only propagation, β=0 left-only, and β=1/2 symmetric splitting.

A Gaussian becomes very small far away but has no exactly zero mathematical tail. This plane field is also infinite in x and y. Longitudinal localization does not make it a finite three-dimensional antenna field or a sharp causal wavefront.

$$
G(\xi)=A\exp\left(-\frac{\xi^2}{2\sigma^2}\right),\qquad f=\beta G,\qquad g=(1-\beta)G.
$$

## Read compatible initial data

At t=0 all three cases have E=G(z), but their B and ∂tE differ as shown below. One electric-field snapshot cannot identify propagation direction.

| β | Initial E | Initial B | Initial ∂tE |
| --- | --- | --- | --- |
| 1 | G | G/c | −cG′ |
| 1/2 | G | 0 | 0 |
| 0 | G | −G/c | +cG′ |

Even for the symmetric split with initial B=0, initial ∂tB=−G′ is generally nonzero. Replacing zero initial B with zero B for all time changes the model.

$$
E_x(z,0)=G(z),\quad B_y(z,0)=\frac{(2\beta-1)G(z)}{c},\quad\partial_tE_x(z,0)=c(1-2\beta)G'(z).
$$

## Differentiate at one point

Use normalized c=1, A=σ=1 at z=1 and t=0. Then G=e⁻¹ᐟ²≈0.60653066 and G′=−G. For right propagation, E=B=G, ∂zE=∂zB=−G, and ∂tE=∂tB=+G. Check both first-order signs.

The symmetric split has the same E and ∂zE but B=∂zB=∂tE=0, while ∂tB=+G. Comparing initial field values alone misses this change. Do not confuse the normalized numerical B with B obtained using default SI coefficients.

## Values after the split develops

With normalized c=A=σ=1, β=1/2, t=2, and z=2, the right-moving component is at its center and the left component is four units away. Their tails still overlap, so E is not exactly one half. Retain the small contribution shown below.

For a sketch, comparing E and cB in the same units is convenient, while original B should be read separately. Overlapping pulses can have different E and cB magnitudes; a large E does not always imply B in a fixed single-direction ratio.

$$
E_x=\tfrac12(1+e^{-8})\approx0.50016773,\qquad cB_y=\tfrac12(1-e^{-8})\approx0.49983227.
$$

## Zero B at the center with nonzero curl

At z=0 in the symmetric split, B is zero at every time. This does not mean B vanishes throughout space. It has opposite signs on the two sides and can have a nonzero spatial slope at the center.

For normalized c=A=σ=1 at t=1, central E=e⁻¹ᐟ², B=0, ∂tE=−e⁻¹ᐟ², and ∂zB=+e⁻¹ᐟ². Thus the x component of curl B, −∂zB, matches changing E. Distinguishing a point value from surrounding variation was also essential in the earlier charge-density example.

## A wrong magnetic field that passes wave equations

Keep the right-moving E=G(z−ct) but replace B by zero everywhere. E still satisfies its scalar wave equation, and zero B satisfies its own wave equation. Both divergences vanish too. Nevertheless, generic nonzero ∂zE violates Faraday, and the required magnetic curl for ∂tE is missing.

This is an explicit negative control, not a complete source-free Maxwell solution. At a pulse center the derivatives can happen to vanish, giving zero sampled residual. Test multiple positions and times. The complete zero field A=0 remains a valid special case.

$$
B_y=0:\qquad r_F=\partial_zE_x,\qquad r_A=-\frac{\partial_tE_x}{c^2}.
$$

## Numerical cancellation between two pulses

For β=1/2 at a very small time, G(z−ct) and G(z+ct) can round to the same floating-point value. Direct subtraction may suggest B=0 although their mathematical difference is nonzero. Compute the exponent difference first and use expm1 to retain it.

The function expm1(x) evaluates exp(x)−1 accurately for small x. Avoid multiplying a huge sinh by a tiny exponential, which can overflow at an intermediate step. The expression below covers zct≥0; the opposite-sign branch uses the other Gaussian as reference. Exact symmetry zeros at z=0 or t=0 differ from numerical loss.

$$
d=\frac{2zct}{\sigma^2},\qquad G(z-ct)-G(z+ct)=G(z-ct)\big[-\operatorname{expm1}(-d)\big]\quad(d\ge0).
$$

## Separate zero amplitude, tails and supported range

If A=0 the fields are exactly zero. If a nonzero Gaussian tail rounds to zero outside arithmetic range, the model reports it unavailable rather than declaring a mathematical zero field. An exact symmetric B=0 can sometimes remain known even when the E tail is unavailable. Track availability separately for each quantity.

The pure model supports A∈[−4,4] V/m, σ∈[0.1,4] m, z∈[−4,4] m, t∈[−4,4] s, and |ct|≤8 m. With default SI c, seconds exceed the displacement budget, so nanosecond-scale times are needed. Inputs are not silently clamped or wrapped into another pulse. This is a declared computational domain, not a limitation of the physical law.

## Checks for an AI field model

Training only E values and wave residuals can miss an incompatible B. Check initial E and B, first-order evolution, divergence, and boundary data separately. Loss terms have different units, so document nondimensionalization and weighting. Distinguish a pointwise residual from a family-wide classification and confirm that negative controls actually fail.

Derivatives in the product model are diagnostics derived from the same analytic family. Independent references use separate Python five-point spatial and temporal differences, actual-field line integrals and flux rates, and160-digit mpmath calculations. Do not confuse differentiation error, cancellation, or tail representation limits with mathematical defects or physical zeros. Energy, Poynting flow, and material boundaries remain subsequent original topics.

## Exercises with explanations

Write initial values, time derivatives, and spatial derivatives separately.

### Exercise 1: Same initial E

If initial E=G and B=G/c, what is β in this family? **Answer:** β=1, giving right propagation and initial ∂tE=−cG′. The same E with B=0 corresponds to β=1/2.

### Exercise 2: Initially zero B

For β=1/2, does initial B=0 imply initial ∂tB=0? **Answer:** Generally no. The derivative is −G′ and can be nonzero away from z=0. Distinguish initial value from initial rate.

### Exercise 3: Passing both wave equations

Is a travelling E paired with B=0 sufficient if both wave residuals vanish? **Answer:** No. Check Faraday ∂zE=−∂tB and Ampere −∂zB=∂tE/c² separately.

### Exercise 4: Zero B at the center

In a symmetric split, if B at z=0 is always zero, is curl B there also zero? **Answer:** No. Opposite B values on the two sides can produce nonzero ∂zB, supporting the central ∂tE.

### Exercise 5: A small pulse difference

For β=1/2, c=σ=A=z=1 and t=10⁻¹⁸, is B exactly zero if direct subtraction gives zero? **Answer:** No. Stable evaluation and high precision retain cB≈e⁻¹ᐟ²×10⁻¹⁸. Distinguish this from exact symmetry at t=0.

### Exercise 6: Width and curvature

If A stays fixed and σ is halved, what happens to the initial central second spatial derivative of E? **Answer:** Ezz(0,0)=−A/σ², so its magnitude grows fourfold. In the compatible family Ett=c²Ezz changes with it.

## References and next connections

Compare §20–1, equations20.24–20.25 of [Feynman Lectures II Chapter 20](https://www.feynmanlectures.caltech.edu/II_20.html), with axes relabeled to z propagation, x electric field, and y magnetic field. The following energy and Poynting-flow topic will connect propagation to quantitative energy transport.
