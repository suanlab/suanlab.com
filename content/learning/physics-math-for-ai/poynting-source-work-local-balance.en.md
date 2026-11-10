## Follow the energy, not just the field arrows

A changing magnetic field can store more energy inside a region even when no charged particle crosses its boundary. What supplies that energy? To answer, draw a fixed region and count three things: the energy stored inside, energy crossing the boundary, and work done on matter inside. A water tank gives a useful bookkeeping analogy, but electromagnetic energy flux is a vector field, not a stream of material particles.

This lesson reuses the bounded prescribed induction field. Review dot and cross products, divergence, the divergence theorem, and time derivatives first. We will connect a local equation in W/m³ to an integrated equation in W. The key skill is reading the sign of every term before trying to make a residual vanish.

## State the assumptions before defining energy

Use constant positive scalar permittivity ε and permeability μ, with a medium at rest. In this teaching model the field energy density and Poynting vector take the following forms. The familiar vacuum formulas follow by using the vacuum coefficients.

$$
u_E=\frac{\epsilon|\mathbf E|^2}{2},\qquad u_B=\frac{|\mathbf B|^2}{2\mu},\qquad u=u_E+u_B,\qquad \mathbf S=\frac{\mathbf E\times\mathbf B}{\mu}.
$$

Here u has units J/m³ and S has units W/m². These formulas do not automatically describe energy stored in a dispersive, hysteretic, nonlinear, or moving material. Such systems need constitutive and mechanical terms. Calling every occurrence of J·E “heat” would also be too strong: it is field-to-matter power density, whose eventual destination depends on the matter model.

## Derive the local balance

Dot Ampere–Maxwell with E and Faraday with B/μ. The vector identity below converts their curl terms into a divergence. Constant coefficients let us identify εE·∂tE and B·∂tB/μ as time derivatives of the two energy densities.

$$
\nabla\cdot(\mathbf E\times\mathbf B)=\mathbf B\cdot(\nabla\times\mathbf E)-\mathbf E\cdot(\nabla\times\mathbf B),\qquad \partial_tu+\nabla\cdot\mathbf S+\mathbf J\cdot\mathbf E=0.
$$

If J·E is positive, the field transfers energy to matter. If it is negative, matter or an imposed source supplies field energy. Positive divergence means a net outward flow locally. Thus increasing field storage needs inward flow, negative field-to-matter work, or both. No single term must vanish by itself.

## Integrate over a fixed region

Let the normal n point outward. For a fixed volume, differentiating the volume integral introduces no moving-boundary transport term. The divergence theorem gives the integral statement.

$$
U=\int_Vu\,dV,\qquad P_{\rm out}=\oint_{\partial V}\mathbf S\cdot\mathbf n\,dA,\qquad W_{\rm matter}=\int_V\mathbf J\cdot\mathbf E\,dV,\qquad \frac{dU}{dt}+P_{\rm out}+W_{\rm matter}=0.
$$

U is in J; its derivative, Pout, and Wmatter are in W. Wmatter here names an instantaneous power, not an accumulated work in J. Negative Pout means energy enters. A moving control volume requires an additional transport calculation and is outside this fixed-cylinder example.

## Reuse the prescribed induction fields

Inside the supported bounded region, prescribe Bz=b(t) and a circulating E. The current required by Ampere–Maxwell opposes the displacement current. The magnetic field is spatially uniform, so its curl is zero.

$$
b=b_0+b_1t+\tfrac12b_2t^2,\quad \mathbf B=(0,0,b),\quad \mathbf E=(b'y/2,-b'x/2,0),\quad \mathbf J=(-\epsilon b_2y/2,\epsilon b_2x/2,0).
$$

These fields require prescribed sources and boundary data. They are not a solution for an isolated finite solenoid, a complete circuit, or the mechanical response of a conductor. Using them is useful because each energy term can be derived without hiding a numerical PDE solver.

## Work out storage and flow at a point

Write r²=x²+y². Squaring E gives b′²r²/4. Taking E×B gives an inward radial flux when bb′>0. Reversing the sign of b without changing b′ can reverse that flow.

$$
u_E=\frac{\epsilon b'^2r^2}{8},\quad u_B=\frac{b^2}{2\mu},\quad \mathbf S=-\frac{bb'}{2\mu}(x,y,0),\quad \mathbf J\cdot\mathbf E=-\frac{\epsilon b'b_2r^2}{4}.
$$

For b′>0 the electric arrows circulate clockwise, but the Poynting arrows point radially inward when b is also positive. The two arrow diagrams represent different quantities. A cross product changes both direction and units; rotating an E arrow visually does not supply a quantitative energy flux.

## Read the two cancellations separately

The magnetic storage rate cancels the divergence of S. The electric storage rate cancels source work. Keeping these pairings visible prevents a large term from hiding a small missing contribution.

$$
\partial_tu_B=\frac{bb'}{\mu},\quad \nabla\cdot\mathbf S=-\frac{bb'}{\mu},\qquad \partial_tu_E=\frac{\epsilon b'b_2r^2}{4},\quad \mathbf J\cdot\mathbf E=-\partial_tu_E.
$$

If b₂=0, electric storage is constant and J·E is zero. Changing magnetic storage can still be supplied through the boundary. Zero local source work does not imply zero electromagnetic energy transfer.

## Calculate the fixed cylinder

Choose radius R and height h with its axis along z. The flux through the top and bottom is zero because Sz=0. Integrating the radial energy density requires the volume element 2πrh dr. The side surface has area 2πRh and an outward radial normal.

$$
U_E=\frac{\epsilon\pi hR^4b'^2}{16},\quad U_B=\frac{\pi R^2hb^2}{2\mu},\quad P_{\rm out}=-\frac{\pi R^2hbb'}{\mu},\quad W_{\rm matter}=-\frac{\epsilon\pi hR^4b'b_2}{8}.
$$

Differentiate UE and UB and compare with the two powers. The R⁴ scaling of electric storage differs from the R² scaling of magnetic storage. Doubling R therefore does more than multiply every quantity by the same volume factor.

## A worked example with normalized coefficients

To see the arithmetic, choose ε=1 F/m, μ=1 H/m, b₀=1 T, b₁=0.5 T/s, b₂=0.2 T/s², and t=1 s. These are pedagogical coefficients, not the vacuum constants. Then b=1.6 T and b′=0.7 T/s. At r=1 m, uE=0.06125 J/m³, uB=1.28 J/m³, the electric storage rate is 0.035 W/m³, the magnetic rate is 1.12 W/m³, divergence is −1.12 W/m³, and J·E=−0.035 W/m³.

For R=h=1 m, UE=0.0962113 J and UB=4.021239 J. Their rates are 0.0549779 W and 3.518584 W. Pout=−3.518584 W and Wmatter=−0.0549779 W, so both pairs close. Distinguish the local point values from the integrated cylinder values.

## Remove the required current deliberately

Keep E and B but set conduction current to zero. Faraday still holds, while Ampere–Maxwell generally fails if b₂≠0. The energy residual becomes the electric storage rate. At r=0 it happens to vanish, so one successful probe cannot certify the entire field family.

$$
R_{\rm energy}^{J=0}=\partial_tu_E=\frac{\epsilon b'b_2r^2}{4}.
$$

With physical SI coefficients and microtesla-scale b, the magnetic rate can be about 7.125×10⁻⁷ W/m³ while the missing electric term is about 3.683×10⁻²⁶ W/m³. Ordinary floating-point addition can hide the latter and report a zero total residual. The implementation displays paired components and stable grouping alongside ordinary summation. A rearranged analytic identity still needs an independent reference; it is not a numerical proof by itself.

## Why an energy residual is insufficient

A static E=(0,x,0), B=J=0 has zero energy time derivative, zero Poynting vector, and zero source work. Its energy residual is zero, but curl E has z component 1, violating Faraday. Energy balance is a consequence of the complete compatible equations; it is not a replacement for every equation.

For an AI field predictor, examine vector Maxwell residuals, source consistency, boundary data, energy components, and independent integral observations. Normalizing only by a dominant magnetic term can erase an important electric defect from a training objective.

## Numerical meaning and independent checks

The Python reference generator calculates energy from vector dot products and S from an actual cross product. It compares five-point time and space differences, radial volume quadrature, and oriented side-surface quadrature over 81 selected conditions. A separate 180-digit calculation checks the small SI defect. These measured discrepancies are not rigorous error bounds.

The TypeScript family marks lost nonzero intermediates unavailable rather than replacing them with physical zero. Empty cylinders have exact zero integrals; a nonzero cylinder whose volume cannot be represented is a different case. The supported coordinates and coefficients are explicit bounds, not silently clamped inputs.

## Read a storage-and-flow diagram

Draw the cylinder as a box in an energy-accounting diagram. Attach boundary arrows to its surface and source-work arrows inside. Storage changes belong inside the box; Pout and Wmatter are signed outgoing channels. For the normalized example, the following powers make the diagram quantitative.

| Channel | Power in W | Interpretation |
| --- | ---: | --- |
| Magnetic storage change | +3.518584 | Magnetic energy increases |
| Outward boundary flow | −3.518584 | Energy enters through the side |
| Electric storage change | +0.0549779 | Electric energy increases |
| Field-to-matter work | −0.0549779 | The prescribed source supplies energy |

Start by removing b₂ and observe which pair vanishes. Then restore b₂ and remove only J: the source-work arrow disappears while electric storage still grows. This comparison connects the missing term to a physical accounting failure rather than merely a red residual label.

## Exercises with explanations

### Exercise 1: Interpret a sign

Pout=−3 W and Wmatter=1 W. What is dU/dt? **Answer:** 2 W. Three watts enter, one watt goes to matter, and two watts increase field storage.

### Exercise 2: Remove acceleration

Set b₂=0 while bb′>0. Must field energy be constant? **Answer:** No. Electric storage is constant, but magnetic storage grows through inward boundary flux.

### Exercise 3: Change the cylinder radius

Double R with all other quantities fixed. How do UE and UB change? **Answer:** UE multiplies by16 and UB by4 because the electric field grows with radius.

### Exercise 4: Omit current

At r=1 with normalized ε=1, b′=0.7 and b₂=0.2, what energy defect remains if J=0? **Answer:** 0.035 W/m³. Faraday alone does not supply the missing source term.

### Exercise 5: Certify a model from one residual

Does the static field E=(0,x,0), B=J=0 become a Maxwell solution because energy balance passes? **Answer:** No. Curl E is nonzero while ∂tB=0, so Faraday fails.

### Exercise 6: A rounded zero

Why might a total SI energy residual be zero for the omitted-current model? **Answer:** A small electric rate is lost when added to a much larger magnetic rate before cancellation. Inspect components and compare stable grouping with independent high-precision arithmetic.

## References and next connection

The balance and sign conventions follow [Feynman II Chapter27, §§27–2–27–3](https://www.feynmanlectures.caltech.edu/II_27.html) and [Haus–Melcher Chapter11, §§11.1–11.4](https://ocw.mit.edu/courses/res-6-001-electromagnetic-fields-and-energy-spring-2008/0f814aee1c72517df7d7e7e96b7365e1_11.pdf). Next, apply the same bookkeeping to transverse pulses and distinguish finite-region energy from energy per transverse area.
