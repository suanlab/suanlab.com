## A material boundary changes the question

Place two dielectric materials on opposite sides of a flat interface. An electric-field arrow can change its normal length when it crosses that interface, even if there is no free charge on the surface. This is a physical response, not automatically an error in a simulation. The right question is which component of which field must be continuous, and which surface charge accounts for a jump.

We start with stationary, isotropic, electrically linear materials. Each half-space has a constant positive permittivity. Review normal and tangential components, polarization, Gauss law, potential gradients, and oriented line integrals. The flat, nondecaying fields need remote sources or prescribed boundaries; this is not an isolated finite device.

## Define the normal and the three fields

Medium1 occupies z<0 and medium2 occupies z>0. Fix n=+z, from medium1 to medium2. Use ε₀ as the vacuum reference and distinguish electric field E, polarization P, and displacement D.

$$
\mathbf D=\epsilon_0\mathbf E+\mathbf P=\epsilon\mathbf E,\qquad \mathbf P=(\epsilon-\epsilon_0)\mathbf E.
$$

E is measured in V/m; D and P are in C/m². D is a macroscopic variable that puts polarization into the constitutive description. It is not a second kind of electric force. Do not compare the numerical lengths of E and D arrows without accounting for their different units.

## Integrate a pillbox across the interface

Draw a thin box straddling z=0. The upper face has outward normal+z and the lower face has outward normal−z. The side contributions cancel for our uniform tangential fields. Gauss law for D counts free charge; Gauss law for ε₀E counts all charge.

$$
\mathbf n\cdot(\mathbf D_2-\mathbf D_1)=\sigma_{\rm free},\qquad \epsilon_0\mathbf n\cdot(\mathbf E_2-\mathbf E_1)=\sigma_{\rm free}+\sigma_{\rm bound}.
$$

Multiplying either equation by the face area gives charge in C. In a general nonuniform configuration the vanishing-height argument makes side flux negligible. Here each side's cancellation can be checked directly at positive height because fields are uniform within each material.

## Locate the polarization charge

Uniform P has no bulk divergence in either open half-space. At the interface its discontinuity produces a surface distribution. The outward normal of medium1 is+n, while that of medium2 is−n.

$$
\rho_{\rm bound}=-\nabla\cdot\mathbf P,\qquad \sigma_{\rm bound}=\mathbf n\cdot(\mathbf P_1-\mathbf P_2).
$$

The sign is easy to reverse accidentally. Think of summing the two material surfaces at the same interface. “No free charge” does not mean “no total surface charge.” Polarization can still produce the normal E jump.

## Match the tangential field

In electrostatics, integrate E around a narrow rectangle crossing the interface. Shrinking its height leaves the two tangential edges. Their circulation must vanish, so tangential E is continuous. Equivalently, a regular electrostatic potential can be chosen continuous across this ideal interface, without an imposed dipole sheet.

$$
\mathbf n\times(\mathbf E_2-\mathbf E_1)=0,\qquad V_2(x,y,0)=V_1(x,y,0).
$$

Normal E continuity is not the corresponding condition. Normal D is continuous only when free surface charge is zero. Tangential D can jump when the permittivity changes even though tangential E matches.

## Specify a solvable family

Let d be the normal D in medium1, σf the specified free surface charge, and Et a common tangential electric component. Prescribe the normal D in medium2 using the charge jump.

$$
\mathbf E_1=(E_t,0,d/\epsilon_1),\qquad \mathbf E_2=(E_t,0,(d+\sigma_f)/\epsilon_2),\qquad \mathbf n=(0,0,1).
$$

Both bulk free and bound charge densities vanish. Surface charge and remote boundary sources supply the required field configuration. We use positive coefficients as a mathematical teaching family, not a claim that arbitrary constants describe every passive material at every frequency.

## Recover the potential on both sides

For each region, integrating the constant field gives a linear potential. Use one common additive reference. A change of reference shifts both potentials together and leaves all fields and jumps unchanged.

$$
V_j(x,z)=-E_t x-E_{jz}z+V_{\rm ref}.
$$

At z=0 the normal terms vanish, so the two potentials agree. The slope in z changes when En changes. A continuous function can have a discontinuous derivative: potential continuity does not require equal normal E.

## Work an example with physical units

Choose ε₁=ε₀, ε₂=4ε₀, d=2 nC/m², σf=0, and Et=1 V/m. Then E1z≈225.882 V/m and E2z≈56.4705 V/m. Normal D remains2 nC/m². P1z=0 and P2z=1.5 nC/m², giving σbound=−1.5 nC/m².

For a face area of0.6 m², the upper and lower D fluxes are+1.2 nC and−1.2 nC. Their sum is zero free charge. The ε₀E flux is−0.9 nC, equal to bound charge in that patch. At x=0.3 m on the interface, both potentials are−0.3 V if Vref=0.

| Quantity | Medium1 | Medium2 |
| --- | ---: | ---: |
| Normal E, V/m | 225.882 | 56.4705 |
| Normal D, nC/m² | 2 | 2 |
| Normal P, nC/m² | 0 | 1.5 |

The smaller E in medium2 and the unchanged D are compatible descriptions of one interface.

## Add free surface charge

Keep the preceding parameters but set σf=1 nC/m². Now D2z=3 nC/m² and E2z≈84.7057 V/m. Polarization gives σbound=−2.25 nC/m², so total surface charge is−1.25 nC/m².

A positive free charge can coexist with negative total charge because the polarization contribution is larger and negative. Reporting only one of these charges without naming it can make a correct flux appear inconsistent.

## Test an incorrect continuity rule

Suppose a model forces equal normal E while keeping unequal ε. Its required free charge changes. In our family with ε₂=2ε₁ and d=2 nC/m², equal normal E requires σfree=2 nC/m². If the requested free charge was zero, the model has changed the source problem.

If that nonzero free charge is explicitly allowed, equal En can be valid. The implementation distinguishes the specified source from the source required by the predicted fields. A failed source request is not proof that no Maxwell configuration could support those fields.

## Deliberately break tangential matching

Add a tangential jump δ to E2x while keeping E1x fixed. For a rectangle traversed along+x below the interface and−x above it, the normal is−y. The vertical edge integrals cancel.

$$
\oint\mathbf E\cdot d\boldsymbol\ell=-\delta L,\qquad V_2(x,0)-V_1(x,0)=-\delta x.
$$

For δ=0.3 V/m and L=0.6 m, circulation is−0.18 V. The potential jump at x=0.3 m is−0.09 V. One additive gauge constant cannot remove an x-dependent mismatch. This is an electrostatic negative control. Its static energy balance with B=J=0 could still pass, as the preceding energy lesson warned.

## Read limits without inventing an interface average

At z=0, the model returns the two side limits and no single averaged E or D. That is intentional: the value on an ideal mathematical surface does not replace the boundary jump conditions. A graph should show the two traces separately.

For positive box height, face fluxes and loop edges have ordinary geometric meanings. At exactly zero height, coincident faces and an interface path no longer define the same measurement. The API reports the collapsed measurement unavailable while keeping its labelled two-sided limit. A tangential-jump counterexample can have nonzero limiting circulation even as height approaches zero.

## Preserve small jumps numerically

Subtracting rounded face values can lose a small free charge on a large normal background. Cancel the common background algebraically first.

$$
\Delta E_n=\frac{\sigma_f}{\epsilon_2}+\frac{d(\epsilon_1-\epsilon_2)}{\epsilon_1\epsilon_2}.
$$

With equal vacuum coefficients, d=10⁻⁹ C/m² and σf=10⁻³⁰ C/m², rounded D2−D1 can be zero while the prescribed analytic jump is nonzero. Label this stable parameter-derived quantity separately from rounded face subtraction. Nonzero products lost to arithmetic are unavailable, not physical zero. Exact zero area also makes area-normalized charge undefined.

## Independent checks and AI field models

Independent Python references evaluate vector surface and edge integrals over243 static conditions. Two180-digit references examine tiny source and material jumps. These checks supplement the analytic formulas; shared jump identities are not independent certificates or universal error bounds.

For a learned field, test bulk residuals, constitutive relations, the declared normal, free and bound charges, interface traces, and potential differences. Training on only bulk points can miss a surface defect of measure zero. Smoothing a legitimate interface jump into an average may solve a different material problem.

## Exercises with explanations

### Exercise 1: No free charge

If ε₂=4ε₁ and σfree=0, must E2n=E1n? **Answer:** No. D2n=D1n, so E2n=E1n/4 in this family.

### Exercise 2: Bound charge sign

For P1n=0 and P2n=1.5 nC/m² with n=+z, find σbound. **Answer:** −1.5 nC/m². The two materials contribute opposite outward normals.

### Exercise 3: Which flux counts free charge?

Should ε₀E flux equal only free charge? **Answer:** No. D flux counts free charge; ε₀E flux counts free plus bound charge.

### Exercise 4: A tangential mismatch

Find circulation for δ=0.2 V/m and L=0.5 m using the stated traversal. **Answer:** −0.1 V. The sign follows the lower+x and upper−x edges.

### Exercise 5: Equal normal fields

Does equal En across unequal ε necessarily violate Maxwell? **Answer:** No. It requires a particular free surface charge. Compare that source with the requested one.

### Exercise 6: A rounded charge jump

Why can rounded D2−D1 be zero for nonzero prescribed σfree? **Answer:** Adding a tiny charge to a larger background may round away the increment. Use a stable analytic jump and independent high-precision references, and state its provenance.

## References and next connection

Compare the macroscopic conventions with [Haus–Melcher §6.2](https://web.mit.edu/6.013_book/www/chapter6/6.2.html) and [§6.5](https://web.mit.edu/6.013_book/www/chapter6/6.5.html). Next, match traveling E and H at the same interface and connect reflection amplitudes to energy flux.
