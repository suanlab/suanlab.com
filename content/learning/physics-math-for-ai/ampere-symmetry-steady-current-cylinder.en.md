## The question: how can zero field coexist with nonzero current density?

Imagine a very long cylindrical conductor carrying current along its axis. The magnetic field circles the axis. At the exact axis, every transverse direction is equivalent: there is no preferred direction for a nonzero magnetic vector. The field there is zero, although current density need not be zero.

This is not a contradiction. Current density determines the **spatial curl** of the magnetic field in this static model. Curl describes neighboring values, not the value at one point. This distinction matters when an AI surrogate learns a field from sparse measurements: fitting the value at the center does not establish the correct local differential law.

We will derive the field, check its units and boundary behavior, and construct a background-field experiment that separates Ampere's law from the symmetry shortcut used to solve it.

## Specify the source before solving

The source is an infinitely long cylinder of radius $a>0$, centered on the $z$ axis. Its prescribed total current $I$ is uniformly distributed over its cross section. Positive current flows in the positive $z$ direction. Inside the cylinder,

$$
J_z=\frac{I}{\pi a^2},\qquad J_x=J_y=0.
$$

Outside, the prescribed current density is zero. Current density has units A/m²; total current has units A. We use a constant permeability $\mu$ with units T·m/A. The teaching default is $4\pi\times10^{-7}$ T·m/A, a rounded approximation, not an exact constant in modern SI. Changing $\mu$ in this model changes a linear coefficient; it does not introduce a nonlinear magnetic material model.

This is an idealized source calculation. We do not solve conductivity, carrier collisions, heating, magnetic material response or mechanical equilibrium. Nor does this infinitely long source describe the ends and return path of a finite circuit.

## Why steady current needs a continuity assumption

Charge conservation requires

$$
\frac{\partial\rho}{\partial t}+\nabla\cdot\mathbf J=0.
$$

For a steady source, the first term vanishes. Our current has only a $z$ component and is independent of $z$, so its divergence is zero. The radial jump at the cylinder surface does not create a radial current: the normal component through the lateral surface remains zero on both sides.

A finite wire with current entering but nowhere leaving would accumulate charge at an end. Calling that entire configuration steady would require an additional source, sink or return circuit. Also, zero net charge density does not imply zero current; oppositely charged carriers can have different velocities. These are source assumptions, not consequences of a plot looking stationary.

## A closed line integral is not a closed surface flux

Under the stated magnetostatic assumptions, Ampere's law is

$$
\oint_C\mathbf B\cdot d\boldsymbol\ell=\mu I_{\mathrm{enc}}.
$$

The left side is circulation around an oriented **closed line**, with units T·m. The enclosed current crosses a spanning surface whose boundary is that line. Its sign follows the right-hand relationship between the loop orientation and the surface normal.

Gauss's electric law instead integrates normal electric field over a **closed surface**. A circular loop is not a closed Gaussian surface, and circumference is not surface area. Confusing these geometries produces incorrect radial powers even when the algebra appears consistent.

Ampere's integral law does not generally determine the field at one point. We next use rotational and translational symmetry to make that inference for this particular isolated source.

## Symmetry turns circulation into a scalar equation

Choose a coaxial circle of radius $r>0$, oriented counterclockwise as viewed from positive $z$. The source field is tangent to the circle and has the same signed azimuthal component everywhere on it. Thus,

$$
2\pi r B_\phi(r)=\mu I_{\mathrm{enc}}(r).
$$

For an interior loop, only the fraction of the cross-sectional area inside that loop contributes:

$$
I_{\mathrm{enc}}(r)=
\begin{cases}
I(r/a)^2,&0\le r<a,\\
I,&r\ge a.
\end{cases}
$$

The resulting signed field is

$$
B_\phi(r)=
\begin{cases}
\dfrac{\mu I r}{2\pi a^2},&0\le r<a,\\
\dfrac{\mu I}{2\pi r},&r\ge a.
\end{cases}
$$

Inside, enclosed current grows as area, while loop length grows linearly, leaving a field proportional to $r$. Outside, enclosed current stops growing, so the field decreases as $1/r$. A uniform charged sphere also has a linear interior field, but its exterior electric field decreases as $1/r^2$: enclosed charge grows as volume and Gaussian surface area grows quadratically. Similar interior plots do not make the two laws interchangeable.

## Direction, sign and a regular Cartesian field

For $r=\sqrt{x^2+y^2}>0$, the azimuthal unit vector is

$$
\hat{\boldsymbol\phi}=(-y/r,x/r,0).
$$

Positive current gives a positive signed $B_\phi$. At positive $x$ the source field points along positive $y$; at positive $y$ it points along negative $x$. Reversing current reverses the vector, while its nonnegative magnitude is unchanged.

Inside the source, define $c=\mu I/(2\pi a^2)$. The Cartesian expression is especially useful:

$$
\mathbf B_{\mathrm{src}}=c(-y,x,0).
$$

At the axis this vector is exactly zero. The azimuthal basis is undefined there, but the physical vector is regular. No artificial small-radius cutoff is needed. At fixed $x,y$, changing $z$ leaves this infinite-cylinder solution unchanged.

## Zero at the axis does not mean zero curl

Differentiate the Cartesian field:

$$
\nabla\cdot\mathbf B_{\mathrm{src}}=0,\qquad
(\nabla\times\mathbf B_{\mathrm{src}})_z=2c=\frac{\mu I}{\pi a^2}=\mu J_z.
$$

This includes the axis. In cylindrical coordinates the same result reads

$$
(\nabla\times\mathbf B)_z=\frac{1}{r}\frac{d(rB_\phi)}{dr}
=\frac{dB_\phi}{dr}+\frac{B_\phi}{r}.
$$

The radial derivative alone is not the curl. Inside, both terms tend to $c$, giving $2c$. The apparent coordinate division by zero at the axis disappears in the regular Cartesian calculation. For a learned vector field, distinguish coordinate singularities from physical singularities before changing the loss or excluding data.

## A worked example with units

Take $a=1$ m, $I=1$ A and the teaching permeability $\mu=4\pi\times10^{-7}$ T·m/A.

| Distance from axis | Enclosed current | Source field magnitude | Circulation |
| --- | --- | --- | --- |
| 0 m | 0 A | 0 T | 0 T·m |
| 0.5 m | 0.25 A | 0.1 μT | $\pi\times10^{-7}$ T·m |
| 1 m | 1 A | 0.2 μT | $4\pi\times10^{-7}$ T·m |
| 2 m | 1 A | 0.1 μT | $4\pi\times10^{-7}$ T·m |

The same field magnitude at 0.5 m and 2 m comes from different enclosed currents and loop lengths. Field value alone does not identify whether the observation is inside or outside. Here $J_z=1/\pi$ A/m² and the interior curl is $4\times10^{-7}$ T/m. Check that multiplying permeability by current density indeed gives T/m.

At zero radius the loop has collapsed. Circulation is zero and physical source field is zero, but dividing circulation by loop length produces $0/0$. The model reports this normalized inference as unavailable, rather than pretending it measured a point field from a degenerate loop.

## The surface: continuous field, different one-sided derivatives

For nonzero current, the interior and exterior field values agree at $r=a$. Their radial derivatives do not:

$$
B_\phi'(a^-)=\frac{\mu I}{2\pi a^2},\qquad
B_\phi'(a^+)=-\frac{\mu I}{2\pi a^2}.
$$

The interior curl is $\mu I/(\pi a^2)$; the exterior curl is zero. There is no single classical derivative or classical curl at the interface. Assigning an average of the two derivatives would invent a pointwise quantity the field does not possess.

The current-density step may be assigned a value at the single boundary without changing the integrals; this lesson leaves that representative unassigned. The continuous field still satisfies the integral law and the differential law in the appropriate weak sense. When $I=0$, the jump disappears: the source field and its derivatives are zero everywhere, including the former interface.

In the worked example, the one-sided radial derivatives are $+2\times10^{-7}$ and $-2\times10^{-7}$ T/m. A centered stencil crossing the interface cannot be used as if the function were smooth there. Test one-sided limits and field continuity separately.

## A background experiment: the law survives, the shortcut fails

Add a uniform field $\mathbf B_{\mathrm{bg}}$. Its divergence and curl vanish, and its closed-loop circulation is zero:

$$
\mathbf B_{\mathrm{tot}}=\mathbf B_{\mathrm{src}}+\mathbf B_{\mathrm{bg}},\qquad
\oint_C\mathbf B_{\mathrm{bg}}\cdot d\boldsymbol\ell=0.
$$

At $(0.5,0,0)$ m, the example source field is $(0,0.1,0)$ μT. Set the background to $(0,0.3,0)$ μT. The total point field is now $(0,0.4,0)$ μT, while circulation divided by circumference remains 0.1 μT. The latter is the mean tangential component around the loop, not the new point value.

The total field no longer has the original rotational symmetry. Ampere's law remains valid; extracting a constant total tangential field from the integral is the invalid step. At the axis the source vector remains zero, but total field can be nonzero because of the background. Keep source, background and total vectors distinct in calculations and displays.

Adding this background changes a physical field. It is not a gauge transformation of a vector potential. Vector potentials and their gauge freedom require a separate lesson.

## How can exterior curl vanish but circulation remain nonzero?

For $r>a$, the source field has zero local curl. Nevertheless, a loop surrounding the cylinder has circulation $\mu I$. Stokes's theorem is consistent: the ordinary spanning disk crosses the current-carrying interior, where curl is nonzero.

If the domain is restricted to the exterior of the cylinder, that disk is not contained in the domain. A local zero-curl statement on the exterior does not imply zero circulation around its hole. This is a useful warning for AI reconstruction: local differential residuals can miss global circulation constraints and domain topology. Include oriented integral measurements or suitable boundary information when the problem requires them.

## What the infinite-source idealization cannot establish

The $1/r$ exterior profile applies to the prescribed infinite source. A finite circuit has ends, geometry and return current that can change the distant field. This model does not predict those corrections.

It also does not establish finite total magnetic energy for an isolated infinite current. Even energy per unit length outside the cylinder accumulates logarithmically as the outer integration radius grows. A finite apparatus or a return path must be specified before interpreting a total-energy calculation. Later energy and boundary-value lessons will supply the additional machinery.

## Numerical availability and independent verification

The implementation evaluates the regular Cartesian interior field directly. At an extremely small nonzero radius, squared-radius enclosed current may underflow to zero in floating-point arithmetic even when the linear field is representable. It reports the lost enclosed-current value as a numerical-range limitation, not a physical zero. Circulation, vector components, normalized inference and derivatives have separate availability states.

These extreme inputs test arithmetic contracts; they are not claims about a continuum cylinder below physically meaningful length scales. Exact zero current, a collapsed loop, an undefined coordinate direction and an unrepresentable nonzero number are distinct cases.

The analytic Ampere residual is useful for internal consistency, but is not independent verification when both sides share formulas. The reference generator instead integrates Biot–Savart contributions from infinite filaments over the cross section, using observer-centered ray/disk geometry. Separate tests numerically integrate the actual total vector field around a loop, including backgrounds. Compare these independent checks alongside symmetry, sign, units and interface behavior.

## Exercises with explanations

### Exercise 1 — Half the radius

At $r=a/2$, what fractions of the total current and boundary source field occur?

**Answer.** The enclosed area fraction is one quarter, so enclosed current is $I/4$. Dividing by a loop length half as large gives a field half the boundary value. Do not replace an area fraction with a radius fraction.

### Exercise 2 — Reverse the current

At positive $x$, where does the source field point when $I<0$?

**Answer.** Along negative $y$. Signed azimuthal field reverses, while magnitude remains nonnegative. A plot of magnitudes alone hides this change.

### Exercise 3 — A zero-field observation

Does a zero source field on the axis imply zero current density?

**Answer.** No. The vector value is zero by symmetry, but neighboring values have curl $\mu I/(\pi a^2)$. A single point value cannot determine a spatial derivative.

### Exercise 4 — An interface average

May the two radial derivatives at the surface be averaged to define the classical derivative?

**Answer.** No, unless the jump has disappeared at zero current. For nonzero current, preserve both one-sided values, mark the classical derivative unavailable and check the continuous field and integral law separately.

### Exercise 5 — Add a uniform field

In the worked example at positive $x=0.5$ m, add 0.3 μT along positive $y$. What changes?

**Answer.** The total point field becomes 0.4 μT along positive $y$. Circulation and its normalized mean remain unchanged; their source contribution is 0.1 μT after division by circumference. Total-field rotational symmetry has been broken.

### Exercise 6 — Design a research check

A predictor has almost zero exterior curl but incorrect circulation around the wire. Is this necessarily inconsistent with its local test?

**Answer.** No. The exterior has a hole around the source, and a local curl test does not enforce the global circulation. Check oriented loop integrals, source/interface data and domain assumptions. Before applying the result to a finite wire, specify ends and return current as well.

## Sources and reproducible checks

[MIT 8.02T, Sources of Magnetic Fields, §9.1 and §9.3, Example 9.3](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/4c84733e3977ba42a930a4c0414f70d0_ch9sourc_b_field.pdf) supplies Biot–Savart, Ampere's law and the uniform-current wire example. [NIST 2022 CODATA constants](https://physics.nist.gov/cuu/pdf/all.pdf) lists magnetic permeability with uncertainty; the historical teaching value is not asserted to be exact modern SI.

The product model is `learning-electro-magnetostatics.ts`; `generate-electro-magnetostatics-references.py` generates independent numerical references. In the interactive experiment, compare the axis, interface, exterior and symmetry-breaking background examples. Use the separate vector, circulation and availability tables to explain why a zero field value, zero curl and zero circulation are different claims.
