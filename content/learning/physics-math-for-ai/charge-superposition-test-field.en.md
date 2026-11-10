## A field is a question asked at every location

Imagine two charged objects held still. Before placing a third object, we can ask: what force would a small positive charge feel here? Move the observation point and repeat. The collection of these answers is an electric field. It is a vector-valued function of position, rather than one arrow attached to the entire apparatus.

Three roles must stay separate: **source charges** prepare the environment, an **observation point** specifies where we evaluate it, and a **test charge** turns the field into a force. Changing the test charge changes the force, while the field of the prescribed sources remains the same. A negative test charge reverses the force direction.

## Declare the model before drawing arrows

We use static point charges in vacuum and evaluate their electrostatic field. The sources are prescribed and held fixed. We do not simulate the force required to hold them, their mutual motion, radiation, or the disturbance caused by inserting a real measuring device.

| Quantity | Symbol | SI unit |
| --- | --- | --- |
| Source charge | $q_i$ | C |
| Source and observation positions | $\mathbf r_i,\mathbf r$ | m |
| Electric field | $\mathbf E$ | N/C, equivalently V/m |
| Test charge and force | $q_t,\mathbf F$ | C, N |
| Electric potential | $V$ | V |

The teaching calculation uses the deliberately rounded vacuum value $k=9\times10^9\,\mathrm{N\, m^2/C^2}$. This is an approximation, not an exact SI constant. Independent reference data factor out $k$, so the geometry and sign checks do not depend on this rounded number.

The diagram places sources and observations in a plane, but uses the **three-dimensional Coulomb law**. It is not the logarithmic potential of a genuinely two-dimensional electrostatic theory.

## Build one source contribution from a displacement

The displacement points from a source to the observation:

$$
\mathbf d_i=\mathbf r-\mathbf r_i,\qquad
R_i=\|\mathbf d_i\|,\qquad
\widehat{\mathbf d}_i=\frac{\mathbf d_i}{R_i}.
$$

For a nonzero source and $R_i>0$,

$$
\mathbf E_i(\mathbf r)=k\frac{q_i}{R_i^2}\widehat{\mathbf d}_i
=kq_i\frac{\mathbf d_i}{R_i^3}.
$$

The inverse cube in the last expression does not mean an inverse-cube magnitude: the numerator has length $R_i$. The magnitude is $k|q_i|/R_i^2$. A positive source points outward; a negative source reverses the same geometric vector. There is no separate rule to memorize for the horizontal and vertical signs.

For example, a positive source at the origin and observation $(1,1)$ have displacement $(1,1)$, distance $\sqrt2$, and direction $(1,1)/\sqrt2$. Add components, not just positive magnitudes.

## Superposition is a vector sum

With the source configuration fixed,

$$
\mathbf E(\mathbf r)=\sum_i\mathbf E_i(\mathbf r),\qquad
\mathbf F(\mathbf r)=q_t\mathbf E(\mathbf r).
$$

The horizontal components may cancel while the vertical ones reinforce. Summing the lengths of the arrows would erase that information. The norm of a sum is generally not the sum of the norms.

An arrow plot may compress lengths to keep a source-adjacent vector from covering the page. Such compression is a display operation. The original components, magnitude, units, and test force must remain in the numerical table. A normalized arrow alone cannot tell us whether a field doubled.

## Worked example: the same midpoint can have two very different meanings

Place sources at $(-1,0)$ and $(1,0)$ metres. First give each source charge $+1$ nC. At the origin, each field has magnitude $9$ N/C, but the directions oppose:

$$
\mathbf E(0,0)=(9,0)+(-9,0)=(0,0)\ \mathrm{N/C}.
$$

Now change the right source to $-1$ nC. Its contribution points toward that source, so both contributions point right:

$$
\mathbf E(0,0)=(9,0)+(9,0)=(18,0)\ \mathrm{N/C}.
$$

The second arrangement has zero **net source charge**, but a nonzero local field. Net charge alone does not specify a spatial field. For a test charge $q_t=-2$ nC, the second arrangement gives $F_x=-36$ nN. The test charge moves the force direction; it does not change the prepared field.

## Worked example: an off-axis vector needs both components

Return to the two positive sources and observe $(0,1)$ m. Both distances are $\sqrt2$ m. The horizontal components cancel and the vertical components add:

$$
\mathbf E(0,1)=\frac{kq}{2\sqrt2}(1,1)
+\frac{kq}{2\sqrt2}(-1,1)
=\left(0,\frac{kq}{\sqrt2}\right).
$$

For $q=1$ nC, $E_y\approx6.363961$ N/C. A $-2$ nC test charge experiences a downward force of approximately $12.727922$ nN. This is a useful counterexample to treating the nearest source direction as the direction of the total field.

## Potential is a scalar, not a field magnitude

With potential zero at spatial infinity,

$$
V(\mathbf r)=k\sum_i\frac{q_i}{R_i},\qquad
\mathbf E=-\nabla V.
$$

Potential has a signed scalar sum. For the two positive charges, the midpoint field is zero but $V=18$ V. For the opposite-charge pair, the midpoint potential is zero but $E_x=18$ N/C. Neither “zero potential” nor “zero net charge” implies “zero field.”

The gradient relation connects the two descriptions. The field points toward decreasing potential for a positive test charge. Potential energy is $q_tV$; a negative test charge changes the sign of that energy and of the force. Do not replace $V$ with $\|\mathbf E\|$: their units and distance dependence differ.

## A planar picture does not permit a planar Gauss-law shortcut

For one three-dimensional point source at the origin, the field away from it has zero three-dimensional divergence. On the plane $z=0$, however, differentiating only the two plotted components gives

$$
\partial_xE_x+\partial_yE_y=-\frac{kq}{(x^2+y^2)^{3/2}}\ne0.
$$

The missing derivative is $\partial_zE_z=kq/(x^2+y^2)^{3/2}$ at the plane. Although $E_z$ itself is zero on this plane, its normal derivative need not vanish. The three terms cancel away from the source.

This matters for a physics-informed learning loss: enforcing a two-dimensional source-free divergence constraint on a two-component slice of a three-dimensional Coulomb field imposes the wrong equation. Declare what is a true reduced-dimensional model and what is only a visualization slice.

## Separate cancellation, a source singularity, and numeric range

At an exact nonzero point source, this model has no finite field or potential. Replacing $R$ with $\max(R,\epsilon)$ silently changes the interaction law. A nearby nonzero distance remains a different observation: at $R=10^{-13}$ m, an enormous finite model value must not be mislabeled as an exact collision.

A declared source with $q_i=0$ contributes zero, including at its listed position. For raw coincident nonzero sources, this evaluator keeps the individual source singularity at the shared location, even when their signed charges sum to zero. Combining or replacing such sources is a separate modeling decision.

Some nonzero distances or charges exceed floating-point range. The evaluator reports unavailable quantities with a numerical-range state, rather than infinity or a fabricated zero field. Potential may remain representable after field magnitude overflows, so its availability is tracked separately. A test force is unavailable when its required field is unavailable, including a zero test-charge request at an undefined point. It is also unavailable when multiplying a finite nonzero field by a nonzero test charge loses the result below floating-point range. For example, a 9×10⁻²¹ N/C field and a 10⁻³¹⁰ C test charge require a nonzero force of 9×10⁻³³¹ N, which a double cannot represent. The field and potential can remain available while the force is unavailable. Exact zero test charge at a finite field still gives an exact zero force.

## Scaling and symmetry are useful negative controls

If all source charges change sign, both field and potential change sign. If only the test charge changes sign, only the force changes sign. Translating sources and observations together changes neither field components nor potential. Rotating the entire configuration rotates the field and leaves the scalar potential unchanged.

Under a common positive spatial dilation $\mathbf r\mapsto a\mathbf r$ with charges unchanged,

$$
\mathbf E\mapsto a^{-2}\mathbf E,\qquad V\mapsto a^{-1}V.
$$

These checks catch swapped displacements, lost charge signs, unit mistakes, and a wrong power of distance. A small residual on one grid is weaker evidence than several independent symmetry and scaling checks.

## Research and AI connections

A learned electrostatic surrogate should specify its geometry, source representation, boundary conditions, units, and treatment of singularities. Train/test splits should include changed source signs, observation distance scales, and rigid rotations. A model can interpolate familiar arrow patterns while violating all three.

Compare field components, force for several test charges, and potential-gradient consistency. Do not validate a vector predictor only by its magnitude. A normalized-direction score also misses amplitude errors. This introductory model supplies static sources; extending it to moving charges requires additional electromagnetic physics and is not justified by animating the observation point.

## Explained exercises

### Exercise 1 — Charge signs before arithmetic

For a negative source at the origin and observation $(1,0)$, which way does the field point? What about the force on a negative test charge?

**Answer.** The field points left, toward the source. Multiplication by the negative test charge makes its force point right. Source sign and test-charge sign enter different stages.

### Exercise 2 — A zero that keeps another observable

Give one configuration with zero midpoint field and nonzero potential, and another with the reverse behavior.

**Answer.** Equal positive sources at symmetric positions cancel the midpoint field but add their positive potentials. Equal opposite sources cancel the midpoint potential but reinforce the field from the positive source toward the negative source.

### Exercise 3 — Test charge zero

At a nonsingular point with $E_x=18$ N/C, what changes when $q_t$ becomes zero?

**Answer.** The source field remains $18$ N/C and the test force becomes zero. At a source singularity, the required field is outside this model's finite domain; multiplying a missing field by zero does not repair the observation.

### Exercise 4 — Double every length

Double all source positions and the observation position relative to the same origin. Keep the charges fixed.

**Answer.** Every source-observation distance doubles. Each field contribution falls by four, potential falls by two, and force on an unchanged test charge falls by four. Doubling only the observation coordinate does not generally implement this common dilation.

### Exercise 5 — Why a field slice can look divergent

Why can the displayed two-component divergence be nonzero away from a three-dimensional source?

**Answer.** The unplotted component can have a nonzero derivative normal to the plane, even when that component vanishes on the plane. Three-dimensional divergence uses all three derivatives; a planar visualization is not automatically a two-dimensional field theory.

### Exercise 6 — A misleading validation plot

A predictor gets every arrow direction right but doubles all field magnitudes. It is evaluated using unit arrows only. Has it passed a force prediction test?

**Answer.** No. Its force magnitudes are also doubled for every fixed nonzero test charge. Retain original component and magnitude comparisons, units, and independent potential-gradient or scaling checks.

## Sources and reproducible checks

[MIT 8.02T, Coulomb's Law, §§2.1–2.4](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/c31cd03c1b74a450ec131921f5625f0a_chap2coulomb_law.pdf) supports the charge, force, superposition and field definitions. The slice-divergence calculation and numerical examples above are derived explicitly for this lesson.

The repository's `learning-charge-current-fields.ts` evaluates the declared model. The independent Python generator uses direct NumPy sums and five-point derivatives of potential. Tests also cover exact cancellations, common translations and rotations, distance scaling, singular sources, zero sources and separate numerical-range states. These checks establish the supported introductory calculation, not a complete electrodynamic solver.
