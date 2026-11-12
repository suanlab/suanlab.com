## Does a fast reaction reach the inside of a material?

Imagine a porous catalyst exposed to a reactant. Molecules diffuse inward and are consumed along the way. Increasing reaction speed sounds like a straightforward way to use more of the catalyst. Yet a very fast reaction can consume most molecules near the surface, leaving the interior poorly supplied. A faster local reaction does not automatically produce a proportionally larger total reaction rate.

The relevant comparison is between the time diffusion needs to replenish the material and the time the reaction needs to consume it. We will derive that comparison, solve a finite slab exactly and check the result using actual mass storage and consumption. This provides a small, interpretable model for thinking about scientific AI inputs, parameter inference and transfer between geometrically similar experiments.

## State the geometry and physical assumptions

Our domain runs from an exposed surface at x equals zero to a sealed or symmetry plane at x equals L. For a symmetric slab exposed on both outer faces, L is the half-thickness. Area A is the cross-sectional area of this half-slab. Concentration at the surface is prescribed as c-zero, while the inner plane has zero normal diffusive flux.

Diffusivity D is positive and constant. The first-order volumetric consumption rate is k times concentration, with constant k greater than or equal to zero. There is no fluid advection, temperature change, external transfer resistance, nonlinear saturation or changing pore geometry. The concentration is a dilute nonnegative scalar, not a full multicomponent chemical description.

$$
Dc_{xx}-kc=0,\qquad
c(0)=c_0,\quad c_x(L)=0,\qquad
[D]=\mathrm{m^2/s},\quad[k]=\mathrm{s^{-1}}.
$$

Concentration has units kg/m³. The local consumption density has units kg/(m³ s), whereas its volume integral has units kg/s. A zero-flux boundary does not impose zero concentration. Keeping these distinctions explicit prevents common mistakes when inspecting a concentration animation.

## Nondimensionalize with a scale that survives zero reaction

Choose xi equals x/L and use the diffusion time L²/D. If c-zero is positive, divide concentration by that surface value. The stationary equation then contains a single reaction-to-diffusion group.

$$
\xi=\frac{x}{L},\qquad C=\frac{c}{c_0},\qquad
C_{\xi\xi}-\mathrm{Da}\,C=0,\qquad
\mathrm{Da}=\frac{kL^2}{D}=\phi^2.
$$

Here Da is a diffusion-based Damköhler number and phi is the slab Thiele modulus under our first-order, constant-property convention. Other problems define Damköhler numbers using an advection time or a different reaction law. Compare definitions before comparing numbers.

If c-zero is exactly zero, the physical solution is the zero field. The measured ratio c/c-zero is undefined. We may still calculate the unit-surface-concentration basis function as a declared mathematical experiment, but must not display it as a measured ratio of that zero field.

## Explain the clocks and penetration length

The reaction clock is the inverse of a positive first-order rate. Comparing it with the diffusion clock yields Da. A diffusion distance associated with one reaction time gives a penetration length.

$$
t_D=\frac{L^2}{D},\qquad t_R=\frac1k\quad(k>0),\qquad
\mathrm{Da}=\frac{t_D}{t_R},\qquad
\ell_p=\sqrt{\frac Dk},\qquad \phi=\frac L{\ell_p}.
$$

For small Da, diffusion replenishes the slab before much consumption occurs, so concentration is nearly uniform. For large Da, consumption outruns replenishment over the full thickness and the surface region supplies most of the reaction. Penetration length is a characteristic scale; concentration does not abruptly become zero at that depth.

At k equals zero there is no finite reaction clock or finite reaction penetration length. The physical solution and the diffusion clock remain well defined. Assigning an arbitrary large finite number would conceal the absence of the reaction mechanism.

## Derive the finite-slab solution

The characteristic equation has roots plus and minus the square root of k/D. Combining the two exponentials into a hyperbolic cosine centered at the sealed plane makes the derivative condition immediate. The surface concentration fixes the remaining coefficient.

$$
c(x)=c_0\frac{\cosh[\phi(1-\xi)]}{\cosh\phi},\qquad
c_x=-\frac{c_0\phi}{L}\frac{\sinh[\phi(1-\xi)]}{\cosh\phi}.
$$

At the surface, the cosine ratio is one. At the inner plane, the gradient is zero but concentration is c-zero divided by cosh phi, which is positive for positive c-zero and finite phi. The profile decreases smoothly inward and has positive curvature when reaction and concentration are positive. Positive curvature is required by D times curvature equals k times concentration; it is not evidence that the material produces reactant.

When k is exactly zero, solve the original equation with the same boundary conditions. It gives uniform concentration, zero gradient and zero consumption. The limiting hyperbolic formula agrees with this result.

## Recover the actual mass balance

Define diffusive flux in the positive x direction as minus D times the concentration gradient. Differentiating that flux gives minus the local consumption. Integrate over the physical half-slab and multiply by its actual cross-sectional area.

$$
j=-Dc_x,\qquad j_x=-kc,\qquad
A j(0)=A\int_0^L kc(x)\,dx.
$$

The sealed-end flux is zero, so all incoming material is consumed before it can leave through that end. This is a steady balance: inlet rate equals consumption rate, while the slab can retain a nonzero constant mass of reactant. Flux is not spatially constant here, unlike the nonreacting advection–diffusion example.

The stored mass and total consumption rate follow from an actual spatial integral. They have different dimensions even when their numerical values happen to agree.

$$
M=A\int_0^L c(x)\,dx
=Ac_0L\frac{\tanh\phi}{\phi},\qquad
R=kM=Ac_0\sqrt{Dk}\,\tanh\phi\quad(k>0).
$$

For a symmetric slab exposed identically on both faces, the whole slab has twice this stored mass and twice this consumption. Do not replace L by the full thickness inside phi and also double the integrals: that would count the geometry incorrectly.

## Distinguish effectiveness from penetration length

An ideal reference assumes that the entire half-slab remains at the surface concentration. Its consumption would be k c-zero A L. Dividing actual consumption by that ideal rate defines the effectiveness factor when the denominator is nonzero.

$$
\eta_{\rm eff}=\frac{R}{kc_0AL}
=\frac{\tanh\phi}{\phi},\qquad
\lim_{\phi\to0}\eta_{\rm eff}=1.
$$

Effectiveness is dimensionless and compares reaction rates; penetration length has dimensions of length. For large phi, effectiveness is approximately one divided by phi. Increasing k still increases total consumption in this model, but at large k the leading rate grows proportionally to the square root of k rather than k itself.

At k equals zero or c-zero equals zero, the actual measured rate ratio is 0/0. Its analytic basis function has a useful continuous extension equal to one at phi zero, but that extension must be labeled separately from an observed rate ratio. Area is assumed positive, so it does not introduce another zero denominator here.

## Work through a physical example

Take L equals 1 m, D equals 0.25 m²/s, k equals 1/s, c-zero equals 2 kg/m³ and A equals 1.5 m². Then diffusion time is 4 s, reaction time is 1 s, Da is 4 and phi is 2. The penetration length is 0.5 m.

At the sealed plane the concentration is approximately 0.53160446 kg/m³. Effectiveness is approximately 0.48201379. The surface flux is approximately 0.96402758 kg/(m² s), so total consumption is approximately 1.44604137 kg/s. Stored mass is approximately 1.44604137 kg. The last two numbers agree only because k was chosen as 1/s; they are not the same physical quantity.

The ideal uniformly supplied reaction rate is 3 kg/s. Its ratio with the actual rate gives the same effectiveness. Neither the nonzero inner concentration nor a constant stored mass violates the zero-flux boundary condition or the steady mass balance.

## Preserve depletion when the field rounds to uniform

For small Da, directly subtracting the nearly equal cosine ratio from one can erase a real depletion. A product identity provides a stable alternative.

$$
1-C=
\frac{2\sinh[\phi(1-\xi/2)]\sinh(\phi\xi/2)}{\cosh\phi}
=\frac{\mathrm{Da}}2\xi(2-\xi)+O(\mathrm{Da}^2).
$$

At the inner plane, leading depletion is Da/2. At the midpoint it is three Da divided by eight. At the surface it is exactly zero. A stable depletion calculation can retain a positive value even when the displayed concentration rounds to c-zero. Likewise, effectiveness can round to one while its positive deficit remains separately available; the leading deficit is Da/3.

Products such as kL²/D can underflow if calculated too early as ordinary numbers. Computing their logarithms preserves the nonzero group and allows phi to remain representable even when Da itself is not. Ordinary values, exact zeros and nonzero quantities available only through a logarithm must be distinct states.

## Preserve the sealed-end tail at large phi

Both hyperbolic cosines can overflow in the direct formula. For large phi, evaluate their ratio through logarithms. A useful rearrangement isolates the attenuation and keeps the reflected contribution from the sealed plane.

$$
\log C=-\phi\xi+
\log\!\left(1+e^{-2\phi(1-\xi)}\right)
-\log\!\left(1+e^{-2\phi}\right),\qquad
C(1)\sim2e^{-\phi}.
$$

The factor two at the sealed plane is important. A semi-infinite exponential has a different far-boundary condition and cannot supply the exact finite-slab endpoint value. An extremely small positive concentration can be unavailable as an ordinary number while its logarithm is finite. This does not change the exact zero gradient at the wall.

## Compare similar slabs and connect other dimensionless groups

For positive a and b, changing L to aL, D to bD and k to b divided by a-squared times k preserves Da, phi and the normalized profile at matching xi. The diffusion and reaction clocks both scale by a-squared divided by b. With geometrically similar area, stored mass scales by a-cubed and total consumption by ab when the surface concentration is unchanged.

Matching this diffusion-based Da is insufficient if another slab has appreciable external transfer resistance, temperature-dependent rates, a different geometry or a different reaction order. Those changes introduce other parameters and possibly other equations. In AI training, collapsing measurements onto Da can be useful within the declared family; it should not erase evidence for mechanisms outside it.

For comparison with earlier mechanics, a viscoelastic Weissenberg number can be defined as relaxation time times a characteristic deformation rate, such as U/L. A Deborah number compares relaxation time with a declared observation time T. If T is a period, omega times relaxation time differs from relaxation time divided by T by a factor of two pi. Mach and Froude numbers compare still other terms. Each ratio requires its own specified scales and assumptions; our two scalar examples do not simulate all these systems.

## Research extension: external supply and an additional Biot number

The previous slab imposes its surface concentration directly. Now add a **finite external supply resistance** while preserving the same internal first-order consumption and sealed inner plane. A bulk reservoir has concentration cb, but the actual surface value cs must be determined. The mass-transfer coefficient hm has units m/s; it is not the mesh spacing h or a heat-transfer coefficient. Take hm nonnegative, D, L and area positive, and k nonnegative.

$$
Dc_{xx}-kc=0,\quad-Dc_x(0)=h_m[c_b-c_s],\quad c_x(L)=0,\quad c_s=c(0),\qquad[h_m]=\mathrm{m/s}.
$$

The internal differential equation and sealed condition give the same profile shape multiplied by the unknown surface value. Its surface flux is proportional to cs with an internal admittance a(k), also in m/s. At zero reaction this admittance is exactly zero. The concentration is still smooth through the slab; a finite penetration scale is not an abrupt reaction boundary.

$$
\phi=L\sqrt{k/D},\quad c(x)=c_s\frac{\cosh[\phi(1-x/L)]}{\cosh\phi},\quad a(k)=\sqrt{Dk}\tanh\phi,\quad a(0)=0,\qquad j(0)=a(k)c_s.
$$

Equate internal uptake a times cs with external supply hm times cb minus cs. For positive bulk concentration and nonzero total admittance, solving this scalar balance gives the surface fraction. A second dimensionless group appears: the mass-transfer Biot number. Two slabs with the same Da and geometry can now have different surface concentrations and different normalized profiles because their Biot numbers differ.

$$
\mathrm{Bi}_m=\frac{h_mL}{D},\qquad\frac{c_s}{c_b}=\frac{h_m}{h_m+a}=\frac{\mathrm{Bi}_m}{\mathrm{Bi}_m+\phi\tanh\phi}\quad(h_m+a>0).
$$

Integrate the actual consumption to obtain the total rate in kg/s. For two positive admittances the overall coefficient has additive reciprocal resistances. This is a consequence of the linear boundary law and first-order slab model, not a universal rule for arbitrary nonlinear kinetics. Physical area multiplies the flux only once; do not confuse the coefficient in m/s with the rate in kg/s.

$$
R=A\int_0^Lkc\,dx=Ac_b\frac{h_ma}{h_m+a}=Ac_bK_{\rm overall},\qquad\frac1{K_{\rm overall}}=\frac1{h_m}+\frac1a\quad(h_m,a>0).
$$

Define total effectiveness relative to a hypothetical slab everywhere at the bulk concentration, rather than everywhere at the actual surface concentration. It is the old internal effectiveness multiplied by the surface fraction. Both reference choices are legitimate when labeled, and generally give different numbers. This measured rate ratio requires positive k and cb. A zero-rate limit is a basis-function limit, not a measured nonzero denominator.

$$
\eta_{\rm total}=\frac{R}{kc_bAL}=\frac{\tanh\phi}{\phi}\frac{\mathrm{Bi}_m}{\mathrm{Bi}_m+\phi\tanh\phi}\quad(k,c_b>0).
$$

At fixed finite supply coefficient, arbitrarily rapid internal reaction cannot exceed the incoming supply ceiling. The surface concentration tends to zero as the internal admittance grows. In contrast, taking hm to infinity first recovers the earlier prescribed-surface problem. The faster-reaction behavior therefore depends on which boundary mechanism is held fixed; matching the original Da alone omits that information.

$$
\lim_{k\to\infty}R=Ah_mc_b\quad(0<h_m<\infty),\qquad\lim_{h_m\to\infty}R=Ac_b\sqrt{Dk}\tanh\phi.
$$

Differentiate the rate logarithm to quantify sensitivity. For positive k and hm, the internal chemical factor lies between one half and one. External resistance multiplies it by the surface fraction. When supply is limiting, a small surface fraction makes rate measurements weakly sensitive to the internal chemical constant, while sensitivity to hm approaches one. This is a practical inference issue even if the forward rate is computed accurately.

$$
\frac{\partial\log R}{\partial\log k}=\frac{h_m}{h_m+a}\left(\frac12+\frac\phi{\sinh(2\phi)}\right),\qquad\frac{\partial\log R}{\partial\log h_m}=\frac{a}{h_m+a}\quad(k,h_m,c_b>0).
$$

A single positive total rate also has an exact nonuniqueness. Let v be that rate divided by area and bulk concentration. Choose any chemical constant whose internal admittance exceeds v, then use the displayed positive hm. Infinitely many parameter pairs produce the same measured rate but different surface concentrations. The internal admittance increases continuously from zero to infinity. An additional surface measurement can separate supply and internal uptake under these declared assumptions; noise still affects the inversion.

$$
v=\frac{R}{Ac_b}>0,\quad a(k)>v:\qquad h_m=\frac{va(k)}{a(k)-v}>0,\quad\frac{c_s}{c_b}=\frac{v}{a(k)}.
$$

Keep the exact limiting controls distinct. No reaction with positive supply gives a uniform reservoir concentration. Reaction without supply gives the unique zero steady field. With neither supply nor reaction, both ends are sealed and every nonnegative constant field is stationary: the boundary reservoir is disconnected and initial inventory determines the eventual mean. The 0/0 surface formula must not select zero in this nonunique case. If cb is zero and total admittance is positive, the physical field is zero but its cb-normalized observation is undefined.

$$
\begin{cases}h_m>0,\ k=0:&c(x)=c_b,\ R=0,\\h_m=0,\ k>0:&c(x)=0,\ R=0,\\h_m=0,\ k=0:&c(x)=C\ge0\text{ arbitrary},\ R=0.\end{cases}
$$

For the uniquely determined steady states, similarity between the expanded experiments must preserve both Da and Biot number. The external transfer coefficient has to scale as diffusivity divided by length. With bulk concentration fixed and geometrically similar area, stored inventory and reaction rate scale differently as before. Holding hm fixed while applying only the old Da-preserving transformation generally destroys the normalized-profile collapse. The existing lab remains the prescribed-surface family; this Robin boundary is a supplemental model.

$$
L'=\alpha L,\quad D'=\beta D,\quad k'=(\beta/\alpha^2)k,\quad h_m'=(\beta/\alpha)h_m:\qquad(\mathrm{Da}',\mathrm{Bi}_m')=(\mathrm{Da},\mathrm{Bi}_m),\quad A'=\alpha^2A\Rightarrow R'=\alpha\beta R,\ M'=\alpha^3M.
$$

For the table use D = 1 m²/s, L = 1 m and k = 4/s, so phi = 2 and Da = 4. Columns are Biot number, surface fraction, rate divided by A cb D/L, total effectiveness and logarithmic k sensitivity. All columns are dimensionless. These are teaching scales, not fitted catalyst parameters. The figure uses the same family, with cb as its positive concentration scale.

| Bi | surface | rate | effectiveness | k sensitivity |
| --- | --- | --- | --- | --- |
| 0.10000000 | 0.04930832 | 0.09506917 | 0.02376729 | 0.02826783 |
| 1.00000000 | 0.34152362 | 0.65847638 | 0.16461909 | 0.19579110 |
| 10.00000000 | 0.83835964 | 1.61640363 | 0.40410091 | 0.48062080 |
| 100.00000000 | 0.98108416 | 1.89158437 | 0.47289609 | 0.56244293 |

```python
from math import tanh, sinh
# phi=2: D=1 m^2/s, L=1 m, k=4/s; all outputs dimensionless.
phi = 2.0
admittance = phi*tanh(phi)  # a*L/D
for bi in (0.1, 1.0, 10.0, 100.0):
    surface = bi/(bi+admittance)
    rate = bi*admittance/(bi+admittance)  # R/(A*cb*D/L)
    effectiveness = rate/(phi*phi)
    sensitivity = surface*(0.5+phi/sinh(2*phi))
    print(" ".join(f"{v:.8f}" for v in (bi, surface, rate, effectiveness, sensitivity)))
```

The [Finlayson nondimensionalization text](https://faculty.washington.edu/finlayso/ebook/reaction/rxndiff/nondim.htm) explains Thiele and mass-transfer Biot time scales. The planar Robin balance, inverse family and sensitivity above are derived separately. They do not include reaction heat, variable properties or nonlinear multicomponent kinetics.

![Same Damkohler number with different supply, reaction rates and chemical sensitivities](/assets/learning/continuum-damkohler-robin-supply-identification.en.svg)

### Research questions with explanations

**Does the same Damkohler number imply the same supplied slab?** **Answer:** No. Finite supply introduces a mass-transfer Biot number and changes the surface value and total rate.

**Does a limiting rate identify the internal reaction constant precisely?** **Answer:** Not when external supply dominates. Chemical sensitivity can be very small, and one steady rate admits infinitely many positive supply/reaction pairs.

**When reaction and supply both vanish, must the concentration be zero?** **Answer:** No. Any nonnegative constant is a steady state. Initial inventory or an additional constraint is needed; 0/0 is not a selected zero concentration.

## Six exercises with answers

### Exercise 1 — Whole thickness

A symmetric slab is 2 m thick and exposed identically on both outer faces. What length belongs in the present Da, and how is the whole consumption obtained?

**Answer:** Use L equals 1 m, the distance from one face to the symmetry plane. Compute one half-slab and double its consumption. Using 2 m inside Da would make that number four times too large for this convention.

### Exercise 2 — Sealed boundary

Does zero flux at the inner plane imply that its concentration is zero?

**Answer:** No. It imposes zero gradient for positive D. The concentration is c-zero divided by cosh phi and is positive for finite phi and positive surface concentration.

### Exercise 3 — Remove reaction

Set k to zero while keeping a positive surface concentration. What happens to storage, consumption and effectiveness?

**Answer:** Concentration is uniform, stored mass is A c-zero L and consumption is exactly zero. The measured effectiveness is undefined because both actual and ideal rates vanish; the analytic extension equals one. There is no finite reaction clock.

### Exercise 4 — Small depletion

At Da equals 10 to the power minus 20, estimate midpoint depletion and effectiveness deficit.

**Answer:** The leading normalized depletion is 3.75 times 10 to the power minus 21 and the effectiveness deficit is about 3.3333 times 10 to the power minus 21. Rounded values of one do not make these corrections exactly zero.

### Exercise 5 — Faster reaction

For a very large phi, what happens approximately when k is multiplied by four with other physical inputs fixed?

**Answer:** Phi doubles, penetration length halves, effectiveness approximately halves and total consumption approximately doubles. A fourfold local rate increase does not yield fourfold overall consumption in the diffusion-limited regime.

### Exercise 6 — Similarity and mass

Use a equals 2 and b equals 1, with geometrically similar area. What changes while Da stays fixed?

**Answer:** L doubles, k becomes one quarter and area grows by four. Both clocks grow by four, stored mass grows by eight and consumption grows by two. Shape agreement does not imply equality of actual inventory or reaction rate.

## Sources and limitations

[MIT 3.185, Transport Phenomena, Chapter 2](https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/a556573bb7fc3f3dd159f62cbd040359_chap2.pdf) derives steady first-order reaction–diffusion profiles in a symmetric slab. Its full-thickness length convention must be converted to our half-thickness. [Finlayson's author-provided reaction–diffusion text](https://faculty.washington.edu/finlayso/ebook/reaction/rxndiff/nondim.htm) explains the Thiele time-scale comparison and the role of geometry and boundary transfer. We derive the stated half-slab balance and similarity formulas directly. External transport resistance, heat release and nonlinear multicomponent kinetics remain outside this model.
