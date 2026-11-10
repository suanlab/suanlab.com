## Watch a pulse cross a measuring window

Imagine two light pulses moving in opposite directions through a fixed window. An electric-field plot can show a large peak where they overlap. Does that peak mean energy has been created? Does zero magnetic field at one instant mean there is no electromagnetic energy? Answer these questions by plotting electric storage, magnetic storage, and signed energy flux separately.

This lesson continues the smooth transverse Gaussian pulse. Review initial E/B compatibility, squared norms, Gaussian integrals, and the fixed-volume Poynting balance. The wave has infinite transverse extent: every integral along z is an energy per transverse area, not the finite total energy of a three-dimensional antenna.

## Fix the medium and the initial data

Use constant positive ε and μ, with c=1/√(εμ). Let G be a Gaussian with signed electric amplitude A and width σ. Split its initial electric profile into right- and left-moving components using β∈[0,1].

$$
G(\xi)=A\exp\!\left(-\frac{\xi^2}{2\sigma^2}\right),\quad f=\beta G(z-ct),\quad g=(1-\beta)G(z+ct),\quad E_x=f+g,\quad cB_y=f-g.
$$

At t=0 all choices of β have the same electric profile G. Their magnetic profiles differ. β=1 gives a right-moving pulse, β=0 a left-moving pulse, and β=1/2 an initially zero magnetic field with symmetric later splitting. These are different initial electromagnetic states.

## Separate electric and magnetic storage

Substitute the fields into the constant-coefficient energy formula. Since 1/(μc²)=ε, the magnetic term can be expressed using f−g. Both terms are nonnegative even when the signed field amplitude is negative.

$$
u_E=\tfrac12\epsilon(f+g)^2,\qquad u_B=\tfrac12\epsilon(f-g)^2,\qquad u=\epsilon(f^2+g^2).
$$

The +2fg term in electric storage and −2fg term in magnetic storage cancel in total energy. They do not disappear from the individual electric and magnetic plots. For an equal split at t=0, B=0 and all energy is electric at that instant. This is not an absence of field energy.

## The flux is a signed difference

The cross product points along z. Its sign tells us which direction dominates. There is no mixed term in the total flux either.

$$
S_z=\frac{E_xB_y}{\mu}=\epsilon c(f^2-g^2),\qquad |S_z|\le cu.
$$

If f²=g², local net flux is zero even though both traveling components can carry energy in opposite directions. This is analogous to equal opposing traffic counts producing zero net traffic; it does not mean every vehicle is stationary. The analogy describes accounting, not a particle interpretation of the field.

## A conditional energy-flow ratio

Where u>0, the signed ratio S/u has velocity units and lies between −c and c. A single right-moving component gives +c; a single left-moving component gives −c.

$$
v_{\rm energy}=\frac{S_z}{u}=c\frac{f^2-g^2}{f^2+g^2},\qquad u>0.
$$

This local energy-flow ratio is not automatically a particle velocity, a group velocity in a general medium, or an information velocity. At exact zero amplitude u=0, so the ratio is undefined. If absolute energy underflows numerically, distinguish a model-derived analytic ratio from a quotient of available observed numbers.

## Check the local conservation law

The right-moving component satisfies ft=−cfz, and the left-moving component satisfies gt=+cgz. Differentiating the total energy and flux shows the cancellation directly. This source-free family has J=0.

$$
\partial_tu=2\epsilon c(-ff_z+gg_z),\qquad \partial_zS_z=2\epsilon c(ff_z-gg_z),\qquad \partial_tu+\partial_zS_z=0.
$$

This calculation is an analytic consistency check using the same prescribed family. Independent finite differences and actual-field quadrature provide additional evidence. A shared derivative identity alone does not validate every implementation detail or every Maxwell equation.

## Draw a fixed slab and its outward normals

Choose a≤z≤b. The outward normal is −z at the left face and +z at the right face. Let UA be stored energy per transverse area. The two face fluxes therefore enter with different signs.

$$
U_A(t)=\int_a^b u(z,t)\,dz,\qquad P_{{\rm out},A}=S_z(b,t)-S_z(a,t),\qquad \frac{dU_A}{dt}+P_{{\rm out},A}=0.
$$

UA is in J/m² and the boundary power per area is in W/m². Multiplying both by a fixed transverse area gives J and W. We keep a and b fixed while the pulse travels. Moving the window along with the pulse would change the transport calculation.

## Calculate a finite window

With normalized ε=μ=1, A=1, σ=1, β=1, c=1, and t=0.3, choose a=−1 and b=1. Integrating the actual field energy gives UA≈1.42842866 J/m². The boundary values give Pout,A≈0.42810687 W/m² and the storage rate is about−0.42810687 W/m².

The energy in the window is decreasing because more leaves on the right than enters on the left. The entire pulse has not lost energy: part of it is outside the window. Always state whether “energy decreases” refers to a fixed region or the whole modeled axis.

## Integrate the entire axis

Integrating a squared Gaussian gives σ√π. Translation does not change this integral. The whole-axis energy per transverse area is time independent for the compatible family.

$$
U_{A,\mathbb R}=\epsilon A^2\sigma\sqrt\pi\,[\beta^2+(1-\beta)^2].
$$

For normalized unit parameters, β=1 gives √π≈1.77245385 J/m², while β=1/2 gives √π/2≈0.88622693 J/m². The same initial E does not imply equal initial energy because initial B changes. Comparing the two is not a redistribution experiment with the same energy budget.

## A local split-pulse example

For the equal split at z=1, t=1 with normalized unit parameters, f=0.5 and g≈0.06766764. Thus Ex≈0.56766764 V/m and cBy≈0.43233236 V/m. The total density is about0.25457891 J/m³ and flux is about0.24542109 W/m². Their ratio is about0.96402758 m/s, below c=1 m/s.

At t=0 and the same z, E≈0.60653066 V/m, B=0, u≈0.18393972 J/m³, and S=0. The later local increase does not violate conservation: energy has entered that neighborhood. A point value is not the energy of a material object that must remain attached to the same position.

## Distinguish actual and reference magnetic fields

For a negative control, keep a traveling E but force actual B=0 and J=0. The actual magnetic energy and Poynting flux are zero; the compatible B reference must not be substituted into those actual quantities. The scalar wave equation for E still holds, but first-order Maxwell relations and generally energy balance fail.

The actual zero-B model also has a different whole-axis energy. For the two Gaussian components its electric-only integral is the following expression; it need not be constant while the components overlap.

$$
U^{B=0}_{A,\mathbb R}=\frac{\epsilon A^2\sigma\sqrt\pi}{2}\left[\beta^2+(1-\beta)^2+2\beta(1-\beta)e^{-(ct/\sigma)^2}\right].
$$

For β=1 this integral happens to be constant, yet the local energy residual is generally nonzero because the profile travels with S=0. Whole-axis conservation alone is weaker than local conservation. At A=0 the nominal negative control reduces to the valid zero field; at selected peak points a sampled residual can also vanish without validating a nonzero family.

## Retain tiny fluxes without inventing tails

For β=1/2, computing f²−g² by subtracting rounded squares can erase a small nonzero flux. Use the stable Gaussian difference already derived in the preceding pulse lesson and multiply it by the Gaussian sum.

$$
S_z=\frac{\epsilon c}{4}(G_+-G_-)(G_++G_-),\qquad G_\pm=G(z\mp ct).
$$

At z=1, t=10⁻¹⁸ with normalized unit parameters, the flux is about3.67879441×10⁻¹⁹ W/m², while direct square subtraction may give0. Use expm1 for small exponent differences. A Gaussian tail below the representable range is unavailable, not a compactly supported physical zero. A=10⁻²⁰⁰ can leave representable fields while their squared energies are unavailable.

## Integrate thin windows carefully

Subtracting two nearly equal Gaussian antiderivatives can lose a thin slab's energy. Integrate in a unit coordinate while retaining the real physical width outside the integral. The implementation uses composite eight-node Gauss quadrature on subintervals no wider than σ/2. Stable differences of squared Gaussian endpoints provide a separate boundary formula.

$$
\int_a^b u(z)\,dz=(b-a)\int_0^1u\bigl(a+(b-a)s\bigr)\,ds.
$$

Independent 180-digit references check nine thin slabs and a tiny balanced flux. Actual-field SciPy quadrature and five-point derivatives check54 selected pulse conditions. These are observed checks, not a certified quadrature bound. A zero-width interval has exact zero integral; a positive width with lost arithmetic is a different state.

## Use the lesson to evaluate learned fields

A field-generating AI model may match an electric snapshot while predicting the wrong magnetic state and energy budget. Compare initial E and B, electric and magnetic storage, signed boundary flow, finite-region storage change, and first-order Maxwell relations. Do not reward a zero total-energy loss if the model merely hides local defects or confuses the compatible reference with its actual output.

## Exercises with explanations

### Exercise 1: Same E, different energy

Compare β=1 and β=1/2 at t=0 with the same A and σ. **Answer:** Initial E is identical, but the one-way state's whole-axis energy per area is twice as large because initial B is nonzero.

### Exercise 2: Equal opposing components

If f=g≠0, find u and S. **Answer:** u=2εf²>0 and S=0. Zero net flux does not imply zero stored energy.

### Exercise 3: Read the left boundary

If S(a)=2 W/m² and S(b)=5 W/m², find the storage rate. **Answer:** Pout,A=3 W/m² and dUA/dt=−3 W/m². The left face has outward normal−z.

### Exercise 4: Zero amplitude

What is S/u when A=0? **Answer:** Undefined, since both numerator and denominator are0. Assigning velocity0 would hide the missing normalization.

### Exercise 5: Traveling E with B=0

Does time-independent whole-axis electric energy certify this field? **Answer:** No. For a one-way E profile the local storage changes, S=0, and Maxwell compatibility generally fails.

### Exercise 6: Tiny energy

Can an analytic energy-flow ratio be meaningful when absolute energies underflow? **Answer:** Yes, as a separately labelled property of the compatible family. It is not a quotient of available double-precision energy measurements.

## References and next connections

Use [Feynman II Chapter27](https://www.feynmanlectures.caltech.edu/II_27.html) for local energy bookkeeping and [Haus–Melcher Chapter11](https://ocw.mit.edu/courses/res-6-001-electromagnetic-fields-and-energy-spring-2008/0f814aee1c72517df7d7e7e96b7365e1_11.pdf) for field storage and power flow. The explicit pulse calculations above follow our stated Gaussian initial data. Later lessons add material interfaces, where constitutive assumptions and boundary conditions become central.
