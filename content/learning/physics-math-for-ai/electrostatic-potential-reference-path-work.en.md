## A map of height and the work of walking

Imagine two maps of the same hill. One labels the valley as height zero; the other labels it as 100 m. The slopes and the height gained between two locations are unchanged. Electric potential has a similar reference freedom: adding a spatial constant changes its labels but not the electrostatic field.

The analogy has a limit. Electric potential is measured in volts, not meters, and a test charge can be negative. A negative charge has a force opposite to the electric field. It is safer to derive the signs than to rely on a picture of a positive object rolling downhill.

Our questions are: what can be inferred from a potential, how does a path integral recover a potential difference, and which quantities remain reliable when large reference labels hide a small difference?

## State the static model and its domain

On the bounded region with each Cartesian coordinate between −4 and 4 m, prescribe

$$
V(\mathbf x)=\frac12\mathbf x^T H\mathbf x+\mathbf b\cdot\mathbf x+C,
\qquad H=H^T.
$$

Here $H$ is a constant symmetric 3×3 matrix with units V/m², $\mathbf b$ has units V/m, and $C$ has units V. Symmetry is an explicit input condition: the implementation rejects a nonsymmetric matrix rather than silently changing it.

This is a local teaching field, not the full-space solution of an isolated finite charge distribution. Quadratic potentials generally grow at large distances. We therefore do not impose zero potential at infinity or claim that an unmodeled apparatus automatically supplies the required boundaries. The field is static; an induction field will need the additional time-dependent vector-potential term in a later lesson.

## Differentiate a scalar to obtain a vector

For symmetric $H$,

$$
\mathbf E=-\nabla V=-H\mathbf x-\mathbf b.
$$

The reference $C$ disappears because its spatial derivatives vanish. Off-diagonal entries couple coordinates: changing $y$ can change $E_x$ even with $x$ fixed. Each field component has units V/m, equivalent to N/C.

For smooth fields, mixed partial derivatives commute. Consequently this construction has zero curl. Its divergence is

$$
\nabla\times\mathbf E=\mathbf0,\qquad
\nabla\cdot\mathbf E=-\operatorname{tr}H=4\pi k\rho.
$$

The final equality uses vacuum electrostatics with $k=1/(4\pi\epsilon_0)$. Our rounded teaching value is $k=9\times10^9$ N·m²/C². Thus the prescribed Hessian also fixes the local charge density compatible with Poisson's equation. A convex potential bowl has negative charge density in this model; a drawing of a bowl does not by itself determine a stable trajectory for every charge sign.

## Potential is not test-charge energy

For a prescribed test charge $q$,

$$
\mathbf F=q\mathbf E,\qquad U=qV,\qquad \Delta U=q\Delta V.
$$

Changing $q$ changes force and energy, not the prepared field or potential. With $q=0$, force and energy change are zero even when the field is nonzero. Reversing $q$ reverses force and signed energy change.

An absolute $U$ depends on the chosen reference just as $V$ does. For motion between two locations at fixed charge, the energy difference is the useful reference-independent quantity. The model prescribes the source field and does not include the test charge's backreaction.

## The sign of a path integral

Let a path run from point A to point B. Electrostatics gives

$$
V_B-V_A=-\int_A^B\mathbf E\cdot d\boldsymbol\ell,
\qquad W_{\mathrm{field}}=q\int_A^B\mathbf E\cdot d\boldsymbol\ell=-\Delta U.
$$

The orientation matters. Reversing the path reverses the field integral and the potential difference. The dot product selects the field component along each path element; integrating magnitudes instead would discard both geometry and sign.

An external agent moving the charge quasistatically supplies $W_{\mathrm{ext}}=\Delta U$. More generally, when the only forces doing work are the electric field and that agent, $W_{\mathrm{ext}}=\Delta U+\Delta K$. Do not call the negative of field work the external work of an arbitrary accelerating motion.

## Why a curved route gives the same result

For a differentiable path $\mathbf r(t)$, the chain rule gives

$$
\frac{dV(\mathbf r(t))}{dt}=\nabla V\cdot\mathbf r'(t)
=-\mathbf E(\mathbf r(t))\cdot\mathbf r'(t).
$$

Integration produces the endpoint formula without assuming a straight route. We compare a straight segment with a quadratic Bezier curve:

$$
\mathbf r(t)=(1-t)^2\mathbf x_A+2t(1-t)\mathbf c+t^2\mathbf x_B,
\qquad 0\le t\le1.
$$

The control point $\mathbf c$ bends the route while keeping its endpoints. All three points lie in the bounded cube, so the entire Bezier curve stays in their convex hull. Equal work does not mean equal distance, duration, speed or travel cost. Those require additional dynamics and constraints.

## A worked example with charge and units

Take $H=\operatorname{diag}(2,-1,0)$ V/m², $\mathbf b=(1,0,0)$ V/m and $C=5$ V. Then

$$
V=x^2-\frac12y^2+x+5,\qquad \mathbf E=(-2x-1,y,0).
$$

Move from A=(0,0,0) m to B=(1,1,0) m. The endpoint potentials are 5 and 6.5 V, so $\Delta V=1.5$ V. The field at B is (−3,1,0) N/C. For $q=2$ nC:

| Quantity | Value |
| --- | --- |
| Straight field integral | −1.5 V |
| Bezier field integral, control (0,2,1) m | −1.5 V |
| Test-charge energy change | +3 nJ |
| Work by the electric field | −3 nJ |
| External work for zero kinetic-energy change | +3 nJ |

Changing $C$ to 100 V raises both endpoint potentials by 95 V. All differences and work values above stay fixed. Changing charge to −2 nC reverses the energy and work signs while leaving both field integrals unchanged. Here divergence is −1 V/m² and compatible density is approximately −8.84×10⁻¹² C/m³ with the teaching $k$.

## Remove the reference before subtracting

Floating-point numbers have finite relative precision. Consider $V=x+100$ V along the x axis, moving from 0 to $10^{-16}$ m with the coefficient 1 V/m. Both separately evaluated endpoint potentials round to 100 V in ordinary double precision. Their direct difference is zero, although the reference-independent difference is $10^{-16}$ V.

For the quadratic field we evaluate the difference using

$$
\Delta V=\frac12(\mathbf x_B-\mathbf x_A)^TH(\mathbf x_B+\mathbf x_A)
+\mathbf b\cdot(\mathbf x_B-\mathbf x_A).
$$

The constant cancels symbolically. The implementation returns both this difference and the naive subtraction so their discrepancy can be observed. The improved expression does not create arbitrary precision: nearly canceling nonconstant contributions, rounded coordinates and underflow can still limit it. Gauge/reference invariance is also a useful numerical design principle.

## Local curl and global path independence

Our explicitly defined, single-valued smooth $V$ on the cube establishes global path independence there. In another domain, finding zero local curl is not always sufficient to construct a single-valued global potential. A puncture or excluded source can allow loops that cannot shrink within the domain.

Likewise, a time-varying magnetic flux can produce an electric field with nonzero circulation. Such a field is outside the static scalar-only model. A small local residual on a finite training set does not prove path independence for every loop or establish that the training domain has the required topology.

## Numerical availability and the meaning of a residual

The field is linear in position while the potential contains quadratic terms. At a sufficiently tiny nonzero coordinate, a quadratic term can underflow even when the field remains representable. The implementation keeps field and potential availability separate. A lost nonzero intermediate is marked as an arithmetic-range limitation, not a physical zero or a source singularity. This conservative arithmetic contract may decline an expression even if another evaluation order could recover a result.

For these affine fields, the integrand along a quadratic Bezier path is at most cubic. Four-point and two-point Gaussian quadrature are mathematically sufficient, but their floating-point difference is a diagnostic, not a certified error bound. Agreement of endpoint and integral expressions is stronger when the integral is actually computed; independent NumPy/SciPy references still check the implementation by another route.

## Connections to AI reconstruction and research

A potential parameterization enforces a conservative field structurally. It can reduce impossible predictions, but it also excludes induction fields and may misrepresent topology or source boundaries. A model that fits potential values can still have poor gradients, especially when reference offsets dominate its loss.

Compare field errors, oriented path integrals, charge-density compatibility and reference-shift experiments separately. Training on energy labels additionally requires the charge and energy reference to be defined. Report assumptions and independent numerical evidence rather than treating a zero analytic residual as a general physical validation.

## Exercises with explanations

### Exercise 1 — Shift the map

Add 40 V to every potential in the worked example. What changes?

**Answer.** Absolute endpoint potentials increase by 40 V and absolute test-charge energies by $40q$. Field, potential difference and work remain unchanged.

### Exercise 2 — Reverse the traveler

Use −2 nC instead of +2 nC on the same oriented path.

**Answer.** The field integral stays −1.5 V. Energy change is −3 nJ and field work is +3 nJ. The source field has not reversed.

### Exercise 3 — Take a detour

Does moving the Bezier control point necessarily change electric work?

**Answer.** No, with endpoints and this static scalar potential fixed. The route length and a realistic mechanical cost can change; those are different observables.

### Exercise 4 — Move fast

Can external work always be identified with $\Delta U$?

**Answer.** Only with the stated zero kinetic-energy change and work assumptions. Otherwise include $\Delta K$ and any other forces doing work.

### Exercise 5 — A missing difference

Two endpoint potentials round to 100 V. Is their physical difference necessarily zero?

**Answer.** No. Cancel the reference symbolically and inspect field integrals and coordinate precision. The small example above retains $10^{-16}$ V even when naive subtraction gives zero.

### Exercise 6 — Choose a validation test

A learned field has tiny sampled curl. What remains to be checked?

**Answer.** Domain topology, smoothness, boundaries, closed-loop integrals and whether the physical problem is static. For a potential model, independently check gradients and source compatibility as well.

## Sources and reproducibility

[MIT 8.02X, potential and test-charge energy](https://web.mit.edu/course/8/8.02x/www/faq.html) motivates the distinction between potential and energy. [MIT §29.5 Potentials](https://www.ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter29/section05.html) introduces scalar and vector representations; global statements here explicitly retain domain assumptions.

`learning-potentials.ts` implements the bounded static model. Independent references use NumPy matrix operations and SciPy integrals along straight and curved paths, with fixed random seed. The worked signs, small reference-independent difference, zero charge, reversed routes and arithmetic states have separate checks.
