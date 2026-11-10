## One force, two legitimate stress measures

Pull a soft specimen until its length doubles. Its cross-section can become smaller at the same time. Dividing the applied force by the original area and dividing it by the current area then give different stresses. Neither division is automatically wrong. The error is to mix one stress definition with the area or work increment belonging to another.

We construct a finite-deformation elastic example where geometry, energy, stress and force can all be checked explicitly. The material is an incompressible isotropic neo-Hookean solid under homogeneous uniaxial loading with free lateral surfaces. A rigid rotation changes how its components look in the laboratory without changing its material properties. This example connects constitutive modeling to automatic differentiation, coordinate transformations and learned energy functions.

## Declare the physical assumptions and units

The shear modulus mu is positive and constant. Deformation is elastic, isothermal and spatially uniform. There is no viscosity, plastic flow, damage, inertia or temperature evolution. Incompressibility means that material volume stays constant, not that every length or area stays constant. The lateral boundary traction is zero, which will determine a pressure-like constraint variable.

| Quantity | Unit | Meaning |
| --- | --- | --- |
| lambda | 1 | axial stretch, current length divided by reference length |
| F | 1 | deformation gradient |
| W | J/m³ | stored energy per reference volume |
| mu, p | Pa | shear modulus and incompressibility pressure |
| sigma | Pa | Cauchy stress, based on current geometry |
| P | Pa | first Piola stress, based on reference area |

Use the reference configuration to identify the same material points throughout the experiment. A finite plot or a rotated drawing does not change that configuration's physical meaning.

## Construct an exactly incompressible deformation

Let the axial stretch be positive. The two equal transverse stretches must make the product of all three stretches one. Apply a proper rigid rotation R after this stretch. We use a rotation in the first two coordinate directions for the interactive geometry, while retaining all three stretches.

$$
F=R\operatorname{diag}(\lambda,\lambda^{-1/2},\lambda^{-1/2}),
\qquad \det F=1,\qquad B=FF^T.
$$

This is not a two-dimensional area-preserving replacement for a three-dimensional incompressible specimen. Both transverse directions matter. A doubled axial length gives each transverse length a factor of one over the square root of two, and the axial cross-sectional area a factor of one half.

## Define energy before stress

For the selected neo-Hookean law, stored energy is a scalar function of the deformation invariant. It is measured per reference volume. The undeformed state has zero energy, and a superposed rigid rotation does not change the trace of B.

$$
W=\frac\mu2(I_1-3),\qquad I_1=\operatorname{tr}(B),\qquad
W(\lambda)=\frac\mu2\left(\lambda^2+\frac2\lambda-3\right).
$$

The shear modulus is a material parameter; stretch is a state variable. Choosing one positive modulus gives a useful teaching family, but it does not make the law a precise fit to every rubber. A learned scalar energy can similarly generate stresses through differentiation, provided its domain, symmetries and physical constraints are stated.

## Pressure comes from the constraint and boundary condition

Incompressibility introduces a pressure-like variable rather than a volumetric energy penalty. The general Cauchy expression contains an undetermined isotropic part. Use zero lateral traction to determine it in this particular loading problem.

$$
\sigma=\mu B-pI,\qquad
\sigma_{\perp}=\frac\mu\lambda-p=0
\quad\Longrightarrow\quad p=\frac\mu\lambda.
$$

Pressure here is not an extra material constant to fit independently at every stretch. A different boundary-value problem can determine a different pressure. Setting it to zero merely because the energy is known would create lateral stresses inconsistent with the declared free surfaces.

## Derive axial Cauchy and nominal stresses

Subtract the pressure from the axial component of mu B. Convert to the first Piola measure using the inverse transpose of F and the unit Jacobian. In principal coordinates only the axial stress survives the free-surface condition.

$$
\sigma_{ax}=\mu(\lambda^2-\lambda^{-1}),\qquad
P_{ax}=\mu(\lambda-\lambda^{-2}),\qquad
P=\sigma F^{-T}.
$$

The axial nominal stress is also the derivative of the reduced energy with respect to axial stretch. This equality supplies an independent check linking material response to work. It does not mean that Cauchy and nominal stresses should be numerically equal at a finite stretch.

## Check force with the correct area

Let A-zero and L-zero be reference cross-section and length. The current geometry follows from the deformation. Use the actual current area with Cauchy stress, and the reference area with nominal stress.

$$
A=\frac{A_0}{\lambda},\qquad L=\lambda L_0,\qquad
AL=A_0L_0,\qquad F_{\rm axial}=A\sigma_{ax}=A_0P_{ax}.
$$

Multiplying Cauchy stress by the original area is a useful deliberately wrong control. It overestimates the force by a factor of lambda in extension. The same mistake can appear in a machine-learning dataset if force measurements are converted to stress without retaining which area was used.

## Objectivity changes components, not the material

Rotate the stretched specimen and its corresponding material surfaces together. Energy stays unchanged, while laboratory stress components transform covariantly. For a chosen current unit normal n, traction is stress times n. Its normal and tangential components describe different aspects of the same force-per-area vector.

$$
W(RF)=W(F),\qquad \sigma_{\rm rotated}=R\sigma R^T,\qquad
t(n)=\sigma n,\qquad \sigma_n=n^T\sigma n.
$$

When comparing two rotated experiments, rotate the observation normal as well if it represents the same physical face. Holding the laboratory normal fixed instead changes the face being observed. An apparent change in its normal stress then need not signal a change in the constitutive law. Objectivity and material isotropy are related but distinct requirements; the selected law has both.

## Mechanical work uses a conjugate pair

For this reduced path, nominal stress is conjugate to stretch in reference-volume energy. Integrate it over actual stretch, or include the stretch Jacobian if integrating over log stretch. Multiplying by reference volume converts energy density to specimen energy.

$$
\frac{dW}{d\lambda}=P_{ax},\qquad
\int_{\lambda_a}^{\lambda_b}P_{ax}\,d\lambda=W(\lambda_b)-W(\lambda_a),\qquad
E_{\rm specimen}=A_0L_0W.
$$

Cauchy stress times a stretch increment is not the same work density. A valid Cauchy power formulation uses the corresponding current-volume rate-of-deformation measure. For a closed elastic deformation path the net mechanical work is zero; along one open segment the material can either store or return energy.

## A worked doubled-length example

Choose mu=2 Pa and stretch two. The values are deliberately simple, with no claim that they represent a particular commercial material. Take reference area 2 m² and reference length 3 m to make the area and volume conversions visible.

$$
\lambda=2,\quad \mu=2\ \mathrm{Pa},\quad
W=2\ \mathrm{J/m^3},\quad P_{ax}=3.5\ \mathrm{Pa},\quad
\sigma_{ax}=7\ \mathrm{Pa}.
$$

Current area is 1 m² and current length is 6 m. Both force calculations give 7 N. Stored specimen energy is 12 J because the reference volume is 6 m³. Using 7 Pa with the original 2 m² would incorrectly predict 14 N. Rotating the specimen changes the force vector's components, while the corresponding axial force magnitude stays 7 N.

## Small-strain tangent and the limits of a stability claim

Differentiate the nominal response once more. Its positive slope is a property of this uniaxial path. At the undeformed state it gives the small uniaxial modulus for an incompressible solid.

$$
\frac{dP_{ax}}{d\lambda}=\mu(1+2\lambda^{-3})>0,\qquad
P_{ax}\approx3\mu(\lambda-1)\quad(\lambda\approx1).
$$

Do not turn this one-dimensional result into a proof of every multiaxial stability condition, global uniqueness or resistance to specimen buckling. Material tangent properties, geometric instability and boundary constraints answer different questions. The linear approximation is useful near stretch one and becomes an approximation to inspect, rather than enforce, at large stretch.

## Stable energy near the undeformed state

The direct energy formula subtracts quantities near three. Its true positive value can be lost or even appear slightly negative through rounding. Factor it and use log stretch as an input so that a small strain change can remain available when exp(log stretch) rounds to one.

$$
\ell=\log\lambda,\qquad \delta\lambda=\operatorname{expm1}(\ell),\qquad
W=\frac{\mu(\lambda-1)^2(\lambda+2)}{2\lambda}.
$$

Keep the stable energy and direct rounded subtraction as separate observations. Likewise, directly forming mu B minus p I can leave a tiny arithmetic stress under a nominal pure rotation. The analytic family has exactly zero stress at zero stretch change. A diagnostic of rounded matrix arithmetic should not be interpreted as physical residual stress.

## Tiny paths, cancellation and research experiments

Signed logarithms retain positive energy or signed force beyond linear floating-point range. Multiplying by specimen volume can recover a representable total energy even when the energy density alone is unavailable. For path work, positive and negative contributions can nearly cancel. Pairing symmetric integrand contributions before summation preserves a small physical difference without defining work from the endpoint energy it is supposed to check.

The accompanying references differentiate energy, construct full matrices, evaluate actual face traction and integrate in stretch coordinates independently from the product's scalar formulas. Start by changing stretch and comparing force through both areas. Then rotate specimen and normal together. Finally inspect small log stretches and a closed return path. These experiments are useful checks for learned energies and automatic differentiation pipelines.

## Exercises with worked answers

### Exercise 1 — Same specimen force

At stretch two, which area belongs with Cauchy stress?

**Answer:** The current area, one half of the reference area. Cauchy stress is twice the axial nominal stress here, so both area-weighted expressions give the same force.

### Exercise 2 — Determine pressure

May pressure be set to zero because the energy function is already known?

**Answer:** No. Incompressibility leaves pressure to be determined by the boundary-value problem. The declared free lateral surfaces require p equal to mu divided by stretch.

### Exercise 3 — Rotate the observation

Must a fixed laboratory normal report unchanged normal stress after rotating the specimen?

**Answer:** No. It observes a different relative face. Rotate the normal together with the specimen when comparing the same physical face; the covariant stress transformation then preserves the corresponding observation.

### Exercise 4 — Work conjugacy

Does integrating axial Cauchy stress against d lambda give the reference-volume energy change?

**Answer:** Generally no. The conjugate quantity on this path is axial nominal stress. A Cauchy power expression needs its associated current-volume deformation-rate measure.

### Exercise 5 — A rounded zero energy

If direct subtraction gives zero near stretch one, should stable positive energy be discarded?

**Answer:** No. The direct expression may have lost its small quadratic value. Preserve the stable factorization, the rounded control and any logarithmic observation separately.

### Exercise 6 — Positive tangent

Does positive uniaxial tangent certify that any specimen and boundary condition are stable?

**Answer:** No. It describes this loading path. Multiaxial material conditions and geometric instabilities require their own analysis and cannot be certified by one positive slope.

## Source and scope

[A. F. Bower, Applied Mechanics of Solids §3.5](https://solidmechanics.org/Text/Chapter3_5/Chapter3_5.htm) explains energy-based hyperelastic laws, stress measures and the role of incompressibility constraints. The free-side uniaxial family is derived explicitly here. It is not a general finite-element solution, a claim of universal material calibration or a certificate of global stability.
