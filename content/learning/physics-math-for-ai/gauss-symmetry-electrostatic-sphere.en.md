## Why a larger Gaussian surface can see a stronger field

Begin with a ball whose electric charge is frozen uniformly throughout its volume. Imagine expanding an invisible spherical measuring surface from its center. Inside the ball, a larger measuring surface encloses more charge. Its area grows as the square of its radius, but enclosed charge grows as the cube. The resulting electric field therefore grows linearly with radius.

Outside the charged ball, the measuring surface encloses no additional charge. Its area keeps growing, so the field falls as inverse radius squared. These are two regions of one continuous field, not two unrelated laws joined by a numerical cutoff.

## Declare the source and the observation geometry

The source is a **prescribed nonconducting solid sphere** of radius $a>0$ and total signed charge $Q$. The source density is uniform inside and zero outside. We hold that distribution fixed rather than solve the mechanics needed to maintain it.

This is different from a conducting sphere in electrostatic equilibrium, where excess charge resides on the surface. It is also different from a mathematical point charge. Giving a source a physical volume changes the model; it is not merely a way to hide a point-source singularity.

| Quantity | Meaning | SI unit |
| --- | --- | --- |
| $a$ | Source radius | m |
| $r=\|\mathbf x\|$ | Observation distance from the center | m |
| $\rho$ | Volume charge density | C/m³ |
| $\mathbf E$ | Electric field | N/C |
| $\Phi_E$ | Electric flux through a closed surface | N m²/C |
| $V$ | Electric potential | V |

The calculation uses the previously declared rounded teaching value $k=9\times10^9\,\mathrm{N\,m^2/C^2}$. Write $4\pi k=1/\varepsilon_0$ consistently within this teaching model. Normalized independent references factor out $k$ rather than treating the rounded value as an exact physical constant.

## Gauss's law is an integral statement before it is a field shortcut

For an outward-oriented closed surface $S$ enclosing a volume,

$$
\Phi_E=\iint_S\mathbf E\cdot\mathbf n\,dA
=4\pi kQ_{\mathrm{enc}}.
$$

The surface is an imaginary measurement geometry. It does not push charge aside or impose a new physical boundary. The normal points outward, and signed flux is negative for a net inward field.

Gauss's law does not require spherical symmetry. Symmetry is what lets us turn this one scalar integral constraint into a pointwise field formula. Knowing the total flux alone does not normally reveal how the field is distributed over the surface.

## Derive the interior and exterior source field

Uniform volume density is

$$
\rho_0=\frac{3Q}{4\pi a^3},\qquad
Q_{\mathrm{enc}}(r)=
\begin{cases}
Q(r/a)^3,&0\le r<a,\\
Q,&r\ge a.
\end{cases}
$$

For the isolated spherical source, rotational symmetry makes the field radial and its signed radial component constant over a concentric measuring sphere. For $r>0$,

$$
4\pi r^2E_r(r)=4\pi kQ_{\mathrm{enc}}(r),\qquad
E_r(r)=
\begin{cases}
kQr/a^3,&0<r<a,\\
kQ/a^2,&r=a,\\
kQ/r^2,&r>a.
\end{cases}
$$

Positive $Q$ gives outward field; negative $Q$ reverses it. $E_r$ is a signed component, while field magnitude is $|E_r|$. Outside a truly spherical source, the point-charge expression is exact throughout the exterior, not just a far-field approximation.

## A zero vector does not imply zero divergence

At the origin, $\widehat{\mathbf r}$ has no unique direction. Writing a field as $E_r\widehat{\mathbf r}$ there would multiply zero by an undefined coordinate basis. Use the regular Cartesian interior expression instead:

$$
\mathbf E_{\mathrm{source}}(\mathbf x)=\frac{kQ}{a^3}\mathbf x
\quad (r<a),\qquad
\mathbf E_{\mathrm{source}}(\mathbf 0)=\mathbf 0.
$$

The field vector is well defined even though its direction at zero magnitude is not. Its divergence uses neighboring values:

$$
\nabla\cdot\mathbf E_{\mathrm{source}}
=3\frac{kQ}{a^3}=4\pi k\rho_0
\quad (r<a).
$$

The divergence is nonzero at the center when $Q\ne0$. It is not the field magnitude and is not just $dE_r/dr$. Away from the origin, spherical coordinates give $\nabla\cdot\mathbf E=E_r'+2E_r/r$. The two geometric terms contribute as well.

## Read the different radial profiles together

For $Q>0$, define $E_a=kQ/a^2$. The source-only profiles satisfy:

| Observation radius | $Q_{\mathrm{enc}}/Q$ | $E_r/E_a$ | $\Phi_E/(4\pi kQ)$ | $V/(kQ/a)$ |
| --- | --- | --- | --- | --- |
| $0$ | $0$ | $0$ | $0$ as a collapsed-surface limit | $1.5$ |
| $a/2$ | $1/8$ | $1/2$ | $1/8$ | $1.375$ |
| $a$ | $1$ | $1$ | $1$ | $1$ |
| $2a$ | $1$ | $1/4$ | $1$ | $1/2$ |

Field, enclosed charge, flux and potential are different observables with different units. These ratios divide by nonzero source quantities; for $Q=0$, use the actual zero source values instead of forming $0/0$. A planar drawing of the source or measuring sphere still represents a three-dimensional volume and surface.

## Worked example with physical units

Let $a=1$ m and $Q=1$ nC. At $r=0.5$ m, the enclosed charge is $0.125$ nC, the source field is $4.5$ N/C, and the measuring surface area is $\pi$ m². Thus

$$
\Phi_E=4.5\pi\approx14.137167\ \mathrm{N\,m^2/C}.
$$

At the source boundary the field is $9$ N/C. At $r=2$ m it is $2.25$ N/C. Both of these measuring surfaces enclose the same complete charge and therefore have flux $36\pi\,\mathrm{N\,m^2/C}$.

At the center the field is zero, yet the source density is approximately $2.387324\times10^{-10}$ C/m³ and divergence is $27\,\mathrm{N/(C\,m)}$. Reading only the central arrow would miss the nonzero source density encoded in spatial variation.

## Potential supplies a separate consistency check

For the isolated source, set potential zero at infinity. Integrating the source field gives

$$
V_{\mathrm{source}}(r)=
\begin{cases}
\dfrac{kQ}{2a}\left(3-\dfrac{r^2}{a^2}\right),&0\le r\le a,\\
\dfrac{kQ}{r},&r>a.
\end{cases}
$$

The worked example gives $V(0)=13.5$ V, $V(0.5)=12.375$ V, $V(1)=9$ V and $V(2)=4.5$ V. The potential is finite at the center and its gradient vanishes there. A large potential is not automatically a large field.

The relationship $\mathbf E=-\nabla V$ provides a useful independent numerical route. The reference generator integrates the Coulomb potential of spherical shells, then differentiates that potential. It does not simply recalculate the product field with the same branch formula.

## The interface has a continuous field but no common field slope

For nonzero $Q$, the radial field limits agree at $r=a$, but its one-sided derivatives are

$$
E_r'(a^-)=\frac{kQ}{a^3},\qquad
E_r'(a^+)=-\frac{2kQ}{a^3}.
$$

There is no classical radial field derivative at the interface. The one-sided divergences are $3kQ/a^3$ and zero. The piecewise volume-density model leaves the exact interface density representative unspecified; assigning a value on a zero-volume set would not alter the enclosed-charge integral.

The field is continuous, so this volume source does not introduce the field jump of a charged surface sheet. The integral law remains meaningful across the interface. Potential and its first derivative are continuous, while its second derivative lacks one common value there. If $Q=0$, the source vanishes and the supposed derivative jump also vanishes.

## Why a high-order finite difference can lose its expected order

A five-point potential-derivative stencil has high order only with sufficient smoothness around the observation. A stencil crossing the interface samples different second derivatives. It should not be judged by the same convergence expectation as a stencil entirely inside a smooth region.

For normalized $k=Q=a=1$, an independent stencil at $r=1$ with step $h=0.01$ gives approximately $0.995000647$ for $-V'$, rather than the exact field $1$. Halving the step gives approximately $0.997500082$. The error roughly halves rather than shrinking by a factor of sixteen.

This is evidence about the chosen numerical derivative near a nonsmooth interface, not evidence that the continuous field has an arbitrary average slope. Validate field continuity, one-sided derivatives, and integral conservation separately. Do not hide a difficult interface by changing its source model without saying so.

## Add a uniform background to test what flux cannot determine

Let a constant external electric field be $\mathbf E_0$. The total field is

$$
\mathbf E_{\mathrm{total}}=\mathbf E_{\mathrm{source}}+\mathbf E_0,\qquad
\iint_S\mathbf E_0\cdot\mathbf n\,dA=0.
$$

The same enclosed charge and closed flux now coexist with a different local field. In the worked example, add $\mathbf E_0=(3,0,0)$ N/C. At $(0.5,0,0)$ m the total field is $7.5$ N/C along $+x$, but flux divided by measuring-sphere area still gives $4.5$ N/C. That quotient is a surface average of the normal component, not this point's total field.

The source contribution stays spherically symmetric; the total field no longer does. Gauss's law remains valid, while the symmetry-based pointwise shortcut no longer describes the total field. Source data and an integral constraint do not replace boundary conditions.

## A changed background is not a potential gauge shift

The model uses $V_{\mathrm{background}}=-\mathbf E_0\cdot\mathbf x$, zero at the origin. Adding it changes the field. Adding a constant to potential would leave the field unchanged; that is a different operation.

The isolated-source contribution uses its zero-at-infinity reference. A nonzero uniform background does not have a total potential tending to zero in every direction at infinity. The model reports the sum with these declared reference choices and does not claim an impossible total boundary condition.

An ideal uniform field is a local/static comparison model, not a finite-energy device built from specified nearby electrodes. Its role here is to provide a concrete homogeneous-field ambiguity when boundary information is absent.

## Numeric range is different from physical zero

Enclosed charge scales as $r^3$ inside, whereas field scales as $r$. At extremely small formal radii, enclosed charge can underflow floating-point representation even when field remains representable. Computing the field by first rounding enclosed charge to zero and dividing by area would manufacture a wrong zero.

The implementation evaluates the regular Cartesian source field directly and tracks availability of density, enclosed charge, flux and normalized observations separately. These extreme parameters are numerical stress tests of a classical continuum formula, not claims about physically resolving arbitrarily small lengths.

At $r=0$, the measuring surface is collapsed rather than a regular Gaussian surface. Its area and limiting flux are zero, but their ratio is unavailable. The actual source field is still known from its regular Cartesian expression. Neither a coordinate singularity nor an unavailable normalization should be mislabeled as a physical point-charge singularity.

## Research and AI experiments

For a learned static field, combine independent potential gradients, closed-flux integrals, local derivatives away from the interface, and explicit interface traces. Separate source-only and background contributions. A Gauss-residual-only loss cannot distinguish all fields with the same sources and closed flux.

Keep the source distribution and observation geometry fixed when comparing methods. A conducting shell, a uniform volume sphere and a softened point charge can all look smooth in a plot, yet encode different interior physics. Near an interface, report the stencil, support region and convergence behavior rather than treating a small scalar residual as a universal accuracy guarantee.

## Research extension: exterior observations can hide an interior density

### Change the density while preserving its physical preparation

A fixed total charge need not specify the charge density inside a volume. To make that failure explicit, replace the uniform nonconducting sphere by a one-parameter radial family. Write its known radius as R, the same radius called a above. Keep its center, positive total charge Q, rounded Coulomb coefficient k and zero external background fixed. The source is prescribed rather than obtained from conductor equilibrium or mechanical relaxation. Alpha is dimensionless; rho0 has units C/m³.

For −1≤alpha≤1 the density remains strictly positive inside: the factor r²/R²−3/5 lies between −3/5 and 2/5. Positive alpha moves charge toward the outside; negative alpha concentrates it toward the center. No surface-charge sheet is added. The value assigned to density at the single interface does not affect a volume integral. This is a supplemental family, not a new density-profile control in the uniform-sphere lab.

$$
R>0,\quad Q>0,\quad \rho_0=\frac{3Q}{4\pi R^3},\quad -1\le\alpha\le1:\qquad\rho_\alpha(r)=\rho_0\left[1+\alpha\left(\frac{r^2}{R^2}-\frac35\right)\right]\quad(0\le r<R),\qquad\rho_\alpha=0\quad(r>R).
$$

$$
4\pi\int_0^R\rho_\alpha(r)r^2\,dr=Q\left[1+3\alpha\left(\frac15-\frac35\frac13\right)\right]=Q.
$$

### Integrate enclosed charge before dividing by area

The alpha term has zero total radial integral. Integrating it only as far as r, however, usually gives a nonzero contribution. This distinction is the source of the inverse ambiguity.

$$
\xi=\frac rR:\qquad\frac{Q_{\mathrm{enc},\alpha}(r)}Q=\begin{cases}\xi^3\left[1+\frac{3\alpha}{5}(\xi^2-1)\right],&0\le\xi\le1,\\1,&\xi\ge1.\end{cases}
$$

$$
E_R=\frac{kQ}{R^2}:\qquad\frac{E_{r,\alpha}}{E_R}=\begin{cases}\xi\left[1+\frac{3\alpha}{5}(\xi^2-1)\right],&0\le\xi\le1,\\\xi^{-2},&\xi\ge1.\end{cases}
$$

### Include potential as an independent observable

Gauss symmetry determines the signed radial field after the source family is declared. At the center use the regular vector expression E=(E_R/R)[1+(3 alpha/5)(r²/R²−1)] times the position vector, so E(0)=0 without an undefined radial direction. Integrating the field from infinity fixes the potential reference.

$$
V(\infty)=0:\qquad\frac{V_\alpha(r)}{kQ/R}=\begin{cases}\dfrac{3-\xi^2}{2}-\dfrac{3\alpha}{20}(1-\xi^2)^2,&0\le\xi\le1,\\\xi^{-1},&\xi\ge1.\end{cases}
$$

$$
\alpha\ne\beta:\quad\rho_\alpha\ne\rho_\beta,\qquad(E_\alpha,V_\alpha)=(E_\beta,V_\beta)\quad(r\ge R),\qquad\partial_\alpha(E_\alpha,V_\alpha)=0\quad(r\ge R).
$$

### State what exterior data cannot identify

Both exterior field and exterior potential are independent of alpha at every radius, including the boundary. This is stronger than a single matching flux: any number of exact exterior observations, in any direction, cannot distinguish these static spherical sources. The densities remain different inside. A learning loss using only those observations is flat along this family; regularization supplies a preference, not missing information.

This construction proves nonuniqueness for the declared family. It is not a classification of arbitrary three-dimensional charge distributions. Radius, center, total charge and background are known here. Unknown nuisance parameters require a different identifiability analysis.

### Put a field probe where it is sensitive

For one radial field reading strictly inside, solve the linear dependence on alpha. The formula below gives both the estimate and its exact response to an additive signed reading error. The measured component and E_R use N/C. A noisy unconstrained estimate can leave [−1,1]; do not silently interpret an out-of-family estimate as a valid positive-density preparation.

$$
0<\xi<1:\quad\widehat\alpha=\frac{E_R\xi-E_{\mathrm{obs}}}{(3/5)E_R\xi(1-\xi^2)},\qquad E_{\mathrm{obs}}=E_{r,\alpha}+\delta E\quad\Longrightarrow\quad |\widehat\alpha-\alpha|=\frac{5|\delta E|}{3E_R\xi(1-\xi^2)}.
$$

$$
\max_{0\le\xi\le1}\xi(1-\xi^2)=\frac{2}{3\sqrt3}\quad\text{at }\xi=\frac1{\sqrt3}:\qquad\min_{0<\xi<1}\frac{5}{3E_R\xi(1-\xi^2)}=\frac{5\sqrt3}{2E_R}.
$$

### Work a normalized and a physical example

The sensitivity factor xi(1−xi²) is zero at the center and surface. Its derivative 1−3 xi² vanishes only at xi=1/sqrt(3) in the interior; the derivative changes from positive to negative there. This is the best location only for this known one-parameter family and the same absolute field-reading uncertainty at every candidate location. It is not a universal sensor placement result.

A center field reading is blind, but a referenced center potential is not: the formula for V(0) depends on alpha. Different observables have different nullspaces. With R=1 m and Q=1 nC, E_R=9 N/C and kQ/R=9 V. For alpha=1, the half-radius field is 2.475 N/C, center potential is 12.15 V, and the field at 2R is still 2.25 N/C. At the best field-probe radius, a 0.01 N/C error gives an alpha-error bound of about 0.00481125.

The table uses those nonzero normalization scales; its last column stays constant although the interior columns change.

| alpha | E(R/2) / E_R | V(0) / (kQ/R) | E(2R) / E_R |
| --- | --- | --- | --- |
| -1 | 0.725 | 1.65 | 0.25 |
| 0 | 0.5 | 1.5 | 0.25 |
| 0.5 | 0.3875 | 1.425 | 0.25 |
| 1 | 0.275 | 1.35 | 0.25 |

```python
for alpha in [-1, 0, .5, 1]:
    x = .5
    field_half = x*(1+3*alpha/5*(x*x-1))
    potential_center = 1.5-3*alpha/20
    field_outside = 1/2**2
    print(*(format(t, '.9g') for t in
            [alpha, field_half, potential_center, field_outside]))
```

The native uniform-sphere model checks the alpha=0 case and the common exterior field/potential. It cannot evaluate a nonuniform density internally. The independent supplement instead integrates shell potential and differentiates it. [MIT’s Gauss-law notes](https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide04.pdf) provide the symmetry framework; this density family, inverse ambiguity and probe-error calculation are derived above under their stated assumptions.

![Radial densities, common exterior field and interior probe sensitivity](/assets/learning/electromagnetism-radial-density-inverse.en.svg)

### Research questions with explanations

**Do infinitely many exact exterior readings recover alpha?** **Answer:** No. Every member has the same complete exterior field and referenced potential, not merely the same total flux.

**Is the central observation always useless?** **Answer:** No. The center field is blind to alpha, but a properly referenced center potential depends on it. State the observable.

**Is R/sqrt(3) the best place for every sensor?** **Answer:** No. It minimizes the stated scalar error gain for known Q, R, center and background, with the same absolute radial-field error. Different noise and nuisance parameters change the design.

## Explained exercises

### Exercise 1 — Half-radius observation

For positive $Q$, what fractions of total charge and surface field are observed at $r=a/2$?

**Answer.** Enclosed charge is $Q/8$, but field is $E_a/2$. The enclosed volume grows as $r^3$ and the measuring area as $r^2$; their quotient grows as $r$.

### Exercise 2 — A central zero with a nonzero source

Does $\mathbf E(0)=0$ imply $\rho(0)=0$?

**Answer.** No. The interior field is $c(x,y,z)$ with $c=kQ/a^3$. The vector is zero at the origin, but divergence is $3c$, encoding the nonzero volume density. Divergence is a spatial derivative, not a point magnitude.

### Exercise 3 — A boundary derivative

In the physical worked example, may the interface field slope be filled with the average of $9$ and $-18\,\mathrm{N/(C\,m)}$?

**Answer.** No. There is no common classical slope there. Preserve the two limits and use an appropriate weak/integral or one-sided comparison. The field itself remains the unambiguous continuous value $9$ N/C.

### Exercise 4 — Double the source size

Double $a$ and the observation coordinates while holding total $Q$ fixed. How do source field, potential, density and enclosed charge change?

**Answer.** Field falls by four, potential by two, density by eight, and enclosed charge remains the same at the same radius fraction. Closed flux is unchanged. Keeping observation position fixed would be a different comparison.

### Exercise 5 — Same flux, different field

Add the worked example's $3$ N/C uniform background. Why can the field at $(0.5,0,0)$ become $7.5$ N/C without changing closed flux?

**Answer.** The uniform contribution integrates to zero over the outward normals of a closed surface. It changes the local field and breaks total spherical symmetry. This is a physical background change, not a constant potential gauge shift.

### Exercise 6 — Design an interface test

A predictor matches the Gauss identity and a centered finite difference poorly at the interface. What should be checked next?

**Answer.** Check the independently calculated field and potential, continuity, one-sided derivatives, source/geometry assumptions, and stencil convergence. The integral identity can be satisfied by construction, and the centered stencil loses its smooth-region order at this interface. Neither observation alone establishes general prediction accuracy.

## Sources and reproducible checks

[MIT 8.02, Gauss's Law, §4.2 and Example 4.4](https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide04.pdf) provides the integral law and uniform nonconducting-sphere example. The Cartesian regularity, background counterexperiment and numerical-interface discussion are developed explicitly for this lesson.

The model is `learning-electro-magnetostatics.ts`. Independent references use SciPy shell-potential quadrature and five-point gradients, including three-dimensional observations and an interface convergence sequence. Additional tests integrate the actual total vector field over closed measuring spheres and distinguish exact zeros from numerical-range states. The rounded static coefficients do not establish precision for a later electromagnetic-wave calculation; consistent constants and model assumptions must be declared there.
