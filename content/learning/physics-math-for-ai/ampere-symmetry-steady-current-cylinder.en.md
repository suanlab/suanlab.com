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

## Research extension: skin effect separates uniform DC current from an Ohmic response

### Change from a prescribed DC source to a conducting response

The native cylinder prescribes a uniform steady current. It does not derive that current from an electric field. For this supplement take a long straight solid cylinder of radius R=a, with constant real permeability mu>0 and scalar conductivity sigma>0. Observe a cross section far from the ends, with a remote return path; neglect axial variation, proximity effects and end corrections. Prescribe a nonzero complex total-current amplitude calligraphic I, in A. Material parameters are ideal constants, not a fit to a specified metal over every frequency. Mu has units H/m, sigma has units S/m, and epsilon>0 is the permittivity in F/m.

Use peak phasors with exp(+i omega t), omega>0. Neglect displacement current relative to Ohmic current, chi=omega epsilon/sigma much less than one. This magnetoquasistatic (MQS) assumption is distinct from a uniform-current assumption: the magnetic diffusion time tau_d has units s, and Omega may be large even when chi is small. No frequency control is added to the native magnetostatic calculator.

$$
J_z(r,t)=\operatorname{Re}\{\widehat J_z(r)e^{i\omega t}\},\quad\widehat J_z=\sigma\widehat E_z,\quad\chi=\frac{\omega\varepsilon}{\sigma}\ll1,\qquad\tau_d=\mu\sigma R^2,\quad\Omega=\omega\tau_d.
$$

$$
\frac1r\frac{d(r\widehat B_\phi)}{dr}=\mu\sigma\widehat E_z,\qquad\frac{d\widehat E_z}{dr}=i\omega\widehat B_\phi\quad\Longrightarrow\quad\frac1r\frac d{dr}\left(r\frac{d\widehat E_z}{dr}\right)=i\omega\mu\sigma\widehat E_z.
$$

### Solve the regular radial diffusion equation

The sign follows from Faraday: the azimuthal curl component is −dE_z/dr=−i omega B_phi. Together with Ampere without displacement current it yields the radial diffusion equation. Its regular axis solution is a modified Bessel function I0; the second independent solution is singular at the axis and is excluded. Delta has units m and kappa has units 1/m. The functions I_n below are Bessel functions; they are not the prescribed total-current amplitude.

$$
\delta=\sqrt{\frac2{\omega\mu\sigma}},\quad\kappa=\frac{1+i}{\delta},\quad z=\kappa R,\quad z^2=i\Omega:\qquad\widehat E_z(r)=C I_0(\kappa r)\quad\text{regular at }r=0.
$$

$$
\mathcal I=2\pi\int_0^R\widehat J_z(r)r\,dr=\frac{2\pi\sigma C R}{\kappa}I_1(z),\qquad \mathcal I\ne0.
$$

The defining series below makes these functions concrete. Substituting it into the radial equation verifies the regular solution; termwise differentiation also gives the identity used for current normalization.

$$
I_n(w)=\sum_{m=0}^{\infty}\frac{(w/2)^{2m+n}}{m!(m+n)!}\quad(n=0,1),\qquad I_0(0)=1,\quad I_1(w)=\frac w2+O(w^3).
$$

### Check that current normalization has no spurious pole

Integrate d[r I1(kappa r)]/dr=kappa r I0(kappa r) to normalize the solution. It remains important to justify division by I1(z). Suppose a nonzero regular solution had zero total current. Ampere then gives B_phi(R)=0 and Faraday gives E_z'(R)=0. Multiply the radial equation by the conjugate of E_z and integrate by parts.

$$
\left[r\widehat E_z^{\prime}\overline{\widehat E_z}\right]_0^R-\int_0^Rr|\widehat E_z^{\prime}|^2\,dr=i\omega\mu\sigma\int_0^Rr|\widehat E_z|^2\,dr:\quad\mathcal I=0\Rightarrow\widehat E_z^{\prime}(R)=0\Rightarrow\widehat E_z\equiv0.
$$

### Read both amplitude and phase across the section

Under that homogeneous boundary condition the left side is real and nonpositive, while the right side is purely imaginary with a positive coefficient. Both integrals must vanish, so E_z is identically zero. Consequently a nonzero regular I0 solution cannot have I1(z)=0 on z²=i Omega with Omega>0. This is an energy argument, not an inference from failing to find a zero numerically.

Define dimensionless current and magnetic profiles. Their boundary/axis limits are b(1)=1 and b(0)=0. Complex values encode phase as well as amplitude, so a plot of absolute current alone does not completely specify the response.

$$
\xi=\frac rR,\quad j(\xi)=\frac{\widehat J_z(r)}{\mathcal I/(\pi R^2)}=\frac{z I_0(z\xi)}{2I_1(z)},\qquad b(\xi)=\frac{\widehat B_\phi(r)}{\mu\mathcal I/(2\pi R)}=\frac{I_1(z\xi)}{I_1(z)}.
$$

$$
Z_{\mathrm{int}}=\frac{\widehat E_z(R)}{\mathcal I}=R_{\mathrm{dc}}\frac{z I_0(z)}{2I_1(z)},\qquad R_{\mathrm{dc}}=\frac1{\pi\sigma R^2}\quad[\mathrm{\Omega/m}].
$$

### Recover the static model as a stated limit

The internal impedance per unit length has units ohm/m. It uses the surface axial electric field divided by total current. It excludes the external inductance, return path, terminal contacts and end fields. It is not the total impedance of an isolated infinite wire.

The convergent defining series for I0 and I1 gives the following small-Omega limit at fixed radius and material coefficients. Uniform current and the original linear interior B profile require Omega much less than one, not merely small chi. The low-frequency internal inductance per length is mu/(8 pi).

$$
\Omega\to0^+:\quad j(\xi)\to1,\quad b(\xi)\to\xi,\quad\frac{Z_{\mathrm{int}}}{R_{\mathrm{dc}}}=1+\frac{i\Omega}{8}+\frac{\Omega^2}{192}+O(\Omega^3),\qquad\frac{\operatorname{Im}Z_{\mathrm{int}}}{\omega}\to\frac\mu{8\pi}.
$$

### Verify passivity with the field energy balance

Differentiate r E_z conjugate(B_phi)/mu, substitute the two radial field equations and integrate over the cross section. The boundary term is E_z(R) conjugate(calligraphic I). This gives the complex power identity below. The factor 1/2 appears only when converting peak phasors to average real power. Magnetic storage accounts for the positive imaginary term.

$$
Z_{\mathrm{int}}|\mathcal I|^2=2\pi\int_0^R\left(\frac{|\widehat J_z|^2}{\sigma}+\frac{i\omega}{\mu}|\widehat B_\phi|^2\right)r\,dr,\qquad P_{\mathrm{in,avg}}=\frac12\operatorname{Re}Z_{\mathrm{int}}|\mathcal I|^2.
$$

$$
|\mathcal I|^2=\left|\int_A\widehat J_z\,dA\right|^2\le\pi R^2\int_A|\widehat J_z|^2\,dA\quad\Longrightarrow\quad\operatorname{Re}Z_{\mathrm{int}}\ge R_{\mathrm{dc}}.
$$

### Run a bounded numerical example

Cauchy–Schwarz applied to the cross-sectional current integral gives the last inequality: at the same total current, a nonuniform complex current cannot dissipate less than a uniform DC distribution with the same conductivity. This does not certify a real conductor with unknown losses or frequency-dependent material laws.

For orientation, ideal constants R=1 mm, mu=4 pi×10⁻⁷ H/m and sigma=10⁶ S/m give tau_d≈1.256637×10⁻⁶ s; Omega=1 corresponds to omega≈7.957747×10⁵ rad/s. This example is supplementary and is not a new native radius range. At Omega=32, R/delta=4, so magnetic diffusion strongly changes the current profile while the total current is held fixed.

The code uses 61 terms of each defining series only for n=0,1 and 0<Omega≤32. It is checked independently against angular-integral representations; it is not a general special-function library for arbitrarily large complex arguments. Table values are the center-current magnitude and real/imaginary internal-impedance ratios. The imaginary column is positive for the stated time convention.

| Omega | abs center j | Re Z / Rdc | Im Z / Rdc |
| --- | --- | --- | --- |
| 0.1 | 0.999973959 | 1.00005208 | 0.0124996745 |
| 1 | 0.997404616 | 1.00518673 | 0.124675942 |
| 10 | 0.803660705 | 1.3720872 | 1.02495076 |
| 32 | 0.323501888 | 2.27380107 | 1.97099225 |

```python
import cmath

def I(n, z):
    assert n in (0, 1)
    term = (z/2)**n
    total = term
    for m in range(1, 61):
        term *= z*z/(4*m*(m+n))
        total += term
    return total

for omega_scaled in [.1, 1, 10, 32]:
    z = cmath.sqrt(1j*omega_scaled)
    center = z/(2*I(1, z))
    impedance = z*I(0, z)/(2*I(1, z))
    print(*(format(t, '.9g') for t in
            [omega_scaled, abs(center), impedance.real, impedance.imag]))
```

For a learned frequency-response surrogate, matching total current and the boundary magnetic field does not validate the interior phase or dissipation. The boundary value b(1)=1 is identical across these different profiles. Compare complex radial values, integrated current, the resistance lower bound and power balance together; one small residual is not a general accuracy certificate.

The native cylinder checks only the zero-frequency limiting field and current density. It does not compute these complex profiles. The regular radial solution is exact within the declared magnetoquasistatic model, not an exact full-Maxwell description of all conductor geometries. A small series or quadrature error does not certify neglected displacement current, material laws or end effects.

[MIT’s skin-effect chapter](https://web.mit.edu/6.013_book/www/chapter10/10.7.html) supplies magnetic-diffusion context. [NIST DLMF’s definition](https://dlmf.nist.gov/10.25#E2) and [angular integral](https://dlmf.nist.gov/10.32#E3) supply the two independent representations used here. Current normalization, nonzero-denominator argument, low-frequency expansion and internal power balance are derived for the specified cylindrical family.

![Complex current profiles and internal impedance from magnetic diffusion](/assets/learning/electromagnetism-skin-diffusion.en.svg)

### Research questions with explanations

**Does negligible displacement current guarantee uniform current?** **Answer:** No. Chi controls that approximation; Omega controls radial magnetic diffusion. They are different dimensionless ratios.

**Why is a positive imaginary impedance not a sign error?** **Answer:** The convention is exp(+i omega t). Internal magnetic storage gives the positive imaginary term and the low-frequency inductive limit.

**Does this internal impedance determine a complete finite device?** **Answer:** No. Return paths, external inductance, contacts, ends and actual frequency-dependent materials require additional information and modeling.

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
