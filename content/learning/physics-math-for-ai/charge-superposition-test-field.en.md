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

## Research extension: a finite probe can change the field it measures

### Declare the mechanical preparation

The fixed-source calculation above answers a well-defined question. A real probe can ask a different one by moving the source. Consider one positive point charge Q attached to a mechanical harmonic trap with stiffness kappa>0 and constrained to the x axis. Its unperturbed position is x=0. Hold a negative probe charge −q, q>0, fixed at R>0. Work in vacuum with k=9×10⁹ N m²/C², a deliberately rounded teaching constant. The source can move toward the probe but cannot pass it. Gravity, radiation, finite charge size, thermal escape and changes to the trapping apparatus are excluded.

The trap is mechanical; we are not assuming that a free charge is stably confined by an electrostatic potential in three dimensions. R and x have units m, kappa has units N/m, and U has units J. This is a supplementary quasistatic model. It adds no source-motion control to the native fixed-source experiment.


$$
U(x)=\frac{\kappa x^2}{2}-\frac{kQq}{R-x},\quad s=\frac{x}{R},\quad \eta=\frac{kQq}{\kappa R^3},\quad W(s)=\frac{U}{\kappa R^2}=\frac{s^2}{2}-\frac{\eta}{1-s},\qquad 0\le s<1.
$$

### Find both equilibria before selecting one

Differentiate the potential rather than equating unsigned force magnitudes without a direction. The attractive force on the source is positive x and the trap force is −kappa x. Their balance gives the following cubic.


$$
W^{\prime}(s)=s-\frac{\eta}{(1-s)^2}=0\quad\Longleftrightarrow\quad g(s)=s(1-s)^2=\eta,\qquad g^{\prime}(s)=(1-s)(1-3s).
$$

### Use curvature to identify the locally stable branch

The derivative of g is positive on 0<s<1/3 and negative on 1/3<s<1. Since g(0)=g(1)=0 and its maximum is 4/27, each 0<eta<4/27 has two interior equilibria. The lower-s equilibrium has positive potential curvature; the upper one has negative curvature. For a positive source mass, the former is locally stable under conservative dynamics and the latter is unstable. A small positive damping can relax a preparation toward the lower branch; no relaxation rate is computed here.


$$
\frac{U^{\prime\prime}(x)}{\kappa}=1-\frac{2\eta}{(1-s)^3}=\frac{1-3s}{1-s}\quad\text{at equilibrium}.
$$



$$
\eta_{\mathrm c}=\frac4{27},\quad s_{\mathrm c}=\frac13,\quad W^{\prime\prime}(s_{\mathrm c})=0,\quad W^{\prime\prime\prime}(s_{\mathrm c})=-\frac92,\qquad W(s)\longrightarrow-\infty\quad(s\to1^-).
$$

### Read the force on the actual probe

At the fold the curvature vanishes but the third derivative does not, so it is not a local minimum. Above the fold there is no interior equilibrium in this domain. Even below it, the lower branch is only a local minimum: the point-charge potential is unbounded toward collision. Local stability is not a global energy minimum or a promise of safety against large disturbances.

The force on the negative probe is negative x. Divide that signed force by its signed charge, not by its absolute value. The result is the source field at the probe in the disturbed configuration. E0 is the field of the unperturbed source at x=0.


$$
E_0=\frac{kQ}{R^2},\quad F_{\mathrm p}=-\frac{kQq}{(R-x)^2},\quad q_{\mathrm t}=-q,\quad \frac{F_{\mathrm p}/q_{\mathrm t}}{E_0}=\frac1{(1-s)^2}=:f(\eta)\quad(q>0).
$$



$$
\frac{ds}{d\eta}=\frac1{(1-s)(1-3s)},\qquad \frac{df}{d\eta}=\frac2{(1-s)^4(1-3s)}\qquad(0<s<1/3).
$$

### Turn small into a stated tolerance

On the lower branch both displacement and measured field increase with eta. The field ratio tends to 9/4 at the fold while its sensitivity to eta diverges. A finite field does not imply a well-conditioned preparation. For a requested relative field error epsilon, invert the field ratio and then use the increasing part of g to obtain an exact criterion within this model.


$$
0<\varepsilon<\frac54,\quad s_\varepsilon=1-\frac1{\sqrt{1+\varepsilon}}:\qquad f(\eta)-1\le\varepsilon\quad\Longleftrightarrow\quad\eta\le s_\varepsilon(1-s_\varepsilon)^2\quad\text{on the stable branch}.
$$



$$
\kappa,R,Q\text{ fixed},\quad q\to0^+:\quad\frac{s}{\eta}\to1,\quad\frac{f(\eta)-1}{\eta}\to2;\qquad \kappa=\frac{kQq}{\eta_*R^3},\quad 0<\eta_*<4/27:\quad f(\eta)=f(\eta_*)>1.
$$

### Calculate the stable branch in its explicit domain

The first limits hold at fixed trap stiffness, distance and source charge. They follow from eta=s(1−s)² and the derivative of (1−s)⁻² at zero. In the second preparation, the trap softens in proportion to q: eta remains eta*, so an arbitrarily small absolute probe charge retains a finite disturbance. State which quantities stay fixed when using a small-probe argument.

For Q=1 nC, R=0.1 m and kappa=10⁻⁴ N/m, E0=900 N/C and q=eta×(10⁻⁷/9) C. The table reports the lower equilibrium, field ratio and local curvature ratio. Bisection below brackets only the increasing branch, for 0<eta<4/27; it is not a generic cubic solver.

For a 5% field-error tolerance, s_epsilon≈0.0240999271 and eta≤0.0229523115. With the same physical parameters, this means q≤2.55025683×10⁻¹⁰ C. The eta=0.025 row already exceeds that tolerance. These thresholds apply to the locally stable branch, not the upper equilibrium.

| eta | stable s | field / E0 | curvature / kappa |
| --- | --- | --- | --- |
| 0.005 | 0.00505089421 | 1.01017884 | 0.989846929 |
| 0.025 | 0.0263726954 | 1.05490782 | 0.945825892 |
| 0.075 | 0.0907103305 | 1.20947107 | 0.800480895 |
| 0.14 | 0.246737798 | 1.76241284 | 0.344882041 |

```python
def stable(eta):
    lo, hi = 0.0, 1.0/3
    for _ in range(100):
        s = (lo + hi)/2
        if s*(1-s)**2 < eta:
            lo = s
        else:
            hi = s
    return (lo + hi)/2

for eta in [.005, .025, .075, .14]:
    s = stable(eta)
    print(*(format(x, '.9g') for x in
            [eta, s, 1/(1-s)**2, (1-3*s)/(1-s)]))
```

The field ratio at exactly q=0 is a limiting definition; the measured quotient 0/0 is undefined. This local point-charge model does not simulate collapse dynamics, contact or thermal switching. To connect with the existing evaluator, prescribe the source at x=R s and compare its force and field with the source prescribed at x=0. Such two fixed-geometry evaluations verify the Coulomb observable, not the existence or stability of a mechanically moving equilibrium.

[MIT’s electric-field chapter](https://ocw.mit.edu/courses/res-6-002-electromagnetic-field-theory-a-problem-solving-approach-spring-2008/7ee5ddbd5e0abae753e4212a9754ee23_MITRES_6_002S08_chp02_text.pdf) explains why a test charge should not rearrange source charges. The spring potential, fold, tolerance and coupled limits here are derived for this explicitly constrained supplement.

![Probe equilibrium and relative field disturbance](/assets/learning/electromagnetism-probe-backreaction.en.svg)

### Research questions with explanations

**Does dividing by the probe charge remove its disturbance?** **Answer:** No. Division removes the direct force factor, but not the source displacement caused by the probe.

**Is every equilibrium below the fold stable?** **Answer:** No. The upper branch is a local maximum; the lower minimum is only local and does not remove the collision singularity.

**Does a small absolute q always recover E0?** **Answer:** Only under stated fixed parameters. A trap softening with q can keep eta and the relative disturbance unchanged.

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
