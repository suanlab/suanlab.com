## Count what stays inside, not just what crosses one boundary

Imagine a cloud of charged carriers passing through a fixed observation interval. Watching the right boundary tells us what crosses that face. It does not tell us how much charge accumulates inside: charge may also enter through the left face. Conservation connects both faces to the amount stored between them.

This lesson builds that connection from a moving density. The charge is signed. A cloud of negative carriers can move right while its conventional current points left. When negative charge leaves a region, the stored charge increases toward zero rather than becoming more negative.

## Distinguish charge, density, current density and current

| Quantity | Meaning | SI unit |
| --- | --- | --- |
| $Q$ | Signed charge | C |
| $\rho$ | Charge per volume | C/m³ |
| $\lambda=A\rho$ | Charge per length of the stream | C/m |
| $\mathbf J$ | Signed charge flux per area | A/m² |
| $I$ | Charge flow through a directed surface | A = C/s |

A density value at one point is not the total charge of a region. Volume current density J is a vector, while the current I through a directed surface is a signed scalar. A sheet current density K is a different tangential vector, with units A/m, rather than the surface integral I in amperes. Neither current nor charge density must be positive.

## Declare one transported preparation

We prescribe a Gaussian cloud translating along $x$ with constant velocity $v$, constant width $\sigma>0$, and constant cross-section $A>0$. There is no lateral flow. The total signed charge is $Q_0$ and its initial center is $x_0$:

$$
x_c(t)=x_0+vt,\qquad
\rho(x, t)=\frac{Q_0}{A\sigma\sqrt{2\pi}}
\exp\!\left[-\frac{(x-x_c(t))^2}{2\sigma^2}\right].
$$

Integrating $A\rho$ over the entire real line gives $Q_0$. The normalized Gaussian shape is nonnegative, but multiplying by a negative $Q_0$ gives negative charge density. Charge density itself is not a probability density.

The visible spatial window is a view, not a physical boundary that truncates or renormalizes this cloud. Changing $\sigma$ selects another preparation;within each preparation it stays constant in time. This is not the spreading Gaussian wave packet from the optics lessons.

We prescribe the motion rather than derive it from electric and magnetic forces. No constitutive law, source of driving force, or self-consistent electromagnetic field is solved here. Satisfying charge conservation is necessary for a physical electromagnetic model, but is not sufficient to provide one.

## From carrier motion to a current-density vector

For the single common transport velocity in this preparation,

$$
\mathbf J(x, t)=\rho(x, t)v\,\widehat{\mathbf x}.
$$

Positive charge moving right gives positive $J_x$. Negative charge moving right gives negative $J_x$. Reversing either the charge sign or velocity reverses current;reversing both keeps current unchanged.

For several carrier species, the more general expression is

$$
\rho=\sum_s\rho_s,\qquad
\mathbf J=\sum_s\rho_s\mathbf v_s.
$$

There need not be one common velocity for which $\mathbf J=\rho\mathbf v$. For example,$\rho_+=\rho_0$ moving at $+v$ and $\rho_-=-\rho_0$ moving at $-v$ give net density zero but $J_x=2\rho_0v$. Do not infer “no current” from local charge neutrality. The interactive preparation in this lesson is the single-velocity case.

## A surface normal specifies the reported sign

For a flat patch over which current density is assumed uniform, with area $A_s$ and chosen unit normal $\mathbf n$,

$$
I=\int_S\mathbf J\cdot\mathbf n\, dA
=A_s\mathbf J\cdot\mathbf n.
$$

If $\mathbf J=J_x\widehat{\mathbf x}$ and the normal makes angle $\theta$ to $+x$, then $I=A_sJ_x\cos\theta$. A normal reversal changes the reported current sign without changing carrier motion. A tangent current contributes no flux through that patch.

The patch area $A_s$ and the stream cross-section $A$ have different roles. For the full end faces of our control interval,$A_s=A$ and their outward normals are fixed as $-\widehat{\mathbf x}$ on the left and $+\widehat{\mathbf x}$ on the right. An independently rotated measurement patch does not redefine those end-face normals. The rotated-patch calculator applies the query current density uniformly over that patch;it is not an exact integral of the spatially varying Gaussian across an arbitrarily tilted finite patch.

## Derive a fixed control-volume balance

Let $a<b$ be fixed boundaries. The stored charge and outward currents are

$$
Q_{[a, b]}(t)=A\int_a^b\rho(x, t)\, dx,\qquad
I_{a,\mathrm{out}}=-AJ_x(a, t),\qquad
I_{b,\mathrm{out}}=AJ_x(b, t).
$$

Positive outward flow removes positive charge, so

$$
\frac{dQ_{[a, b]}}{dt}
=-\left(I_{a,\mathrm{out}}+I_{b,\mathrm{out}}\right)
=A\left[J_x(a, t)-J_x(b, t)\right].
$$

The left contribution enters with the opposite sign from the right. Subtracting absolute current magnitudes would fail for negative charge and reversed flow. The expression uses signed charge throughout.

## Obtain the local continuity equation

Substitute the charge integral into the balance and use the fundamental theorem of calculus:

$$
A\int_a^b\left(\partial_t\rho+\partial_xJ_x\right)dx=0.
$$

For arbitrary fixed intervals and sufficiently smooth fields, this yields

$$
\partial_t\rho+\partial_xJ_x=0.
$$

Our Gaussian translates rigidly, so $\partial_t\rho=-v\partial_x\rho$ and $\partial_xJ_x=v\partial_x\rho$. They cancel. In three dimensions the second term becomes $\nabla\cdot\mathbf J$. If only one nonconserved carrier species is modeled, an appropriate species source term may be required;that is a different balance from total electric-charge conservation.

## Compute the stored Gaussian charge

Let $\Phi$ be the standard normal CDF and $\phi$ its density. Define $z_a=(a-x_c)/\sigma$ and $z_b=(b-x_c)/\sigma$. Then

$$
Q_{[a, b]}=Q_0\left[\Phi(z_b)-\Phi(z_a)\right],\qquad
\frac{dQ_{[a, b]}}{dt}=\frac{Q_0v}{\sigma}\left[\phi(z_a)-\phi(z_b)\right].
$$

The cross-section cancels from total stored charge and full-face current. It does not cancel from $\rho$ or $J_x$:at fixed total charge, a smaller stream area means a higher magnitude of density. This distinction catches an easy units mistake in reduced one-dimensional models.

## Worked example: an instantaneous increase can precede a net decrease

Use $Q_0=1$ C,$\sigma=1$ m,$x_0=0$,$v=1$ m/s,$A=2$ m² and fixed interval $[-0.5,0.8]$ m. At $t=0$,

$$
Q_{[a, b]}(0)=\Phi(0.8)-\Phi(-0.5)\approx0.479607063\ \mathrm C.
$$

The positive-$x$ boundary currents are $AJ_x(a,0)\approx0.352065327$ A and $AJ_x(b,0)\approx0.289691553$ A. Thus the initial net inflow is approximately $+0.062373774$ C/s. The charge is initially increasing.

At $t=1$ s, the stored charge is approximately $0.353933089$ C, so its change from $t=0$ is $-0.125673973$ C. There is no contradiction:the instantaneous inflow changes during the interval. The center reaches the interval midpoint at $t=0.15$ s, where the two boundary currents are equal and stored charge reaches its maximum.

Multiplying the initial inflow by one second gives the wrong answer. The correct finite-time relationship is

$$
Q_{[a, b]}(t_1)-Q_{[a, b]}(t_0)
=\int_{t_0}^{t_1}A\left[J_x(a, t)-J_x(b, t)\right]dt.
$$

The numerical values are a declared transport example, not a claim that this one-coulomb cloud is an unforced realizable material system.

## Repeat the example with negative charge

Change only $Q_0$ to $-1$ C. Density, current density, stored charge and every signed current reverse. The initial stored charge is about $-0.479607063$ C and its initial derivative is $-0.062373774$ C/s. More negative charge enters than leaves at that instant.

Over the complete second, the charge changes by $+0.125673973$ C and becomes less negative. Saying that charge “increases” refers to the signed number, not automatically to the number or absolute charge of carriers inside. Always state which quantity is being compared.

## Moving boundaries require relative flux

The implemented interval has fixed boundaries. If its endpoints instead move at speeds $u_a, u_b$, the changing integration limits add terms:

$$
\frac{dQ}{dt}
=A\left[(J_x-\rho u_a)\big|_a-(J_x-\rho u_b)\big|_b\right].
$$

For a window translating with this rigid cloud,$u_a=u_b=v$, both relative fluxes vanish and the stored charge stays constant. Laboratory current $J_x$ can still be nonzero. Following the cloud with the camera is not the same measurement as watching a fixed laboratory interval.

## Numerics: preserve a small interval rather than subtract two ones

The CDF formula is a useful analytic reference, but direct subtraction can lose a narrow positive tail. For a unit-width Gaussian, the fraction between $8$ and $8+10^{-10}$ standard deviations is about $5.0522715\times10^{-25}$, while two rounded CDF values close to one can subtract to zero.

The implementation integrates the normalized Gaussian density over the original interval using positive Gauss-Legendre weights and panels no wider than one standard deviation. It retains the original physical interval width before standardization. No plotted window is normalized to unit charge and no tail is silently clipped.

An order-16/order-8 quadrature comparison is a diagnostic, not a certified error bound. Very distant positive Gaussian mass can still underflow double precision; this returns an unavailable value, labelled numerical underflow, rather than an exact physical absence. The zero-total-charge preparation is a separate physical zero.

Point density, current, interval charge and conservation residuals have separate availability. A representable density can produce an unrepresentable nonzero current when velocity is extremely small. Conversely, exactly zero velocity gives zero current even if the distant density is unavailable. Normalization and the Gaussian exponential are combined in log space so that a representable density is not lost merely because the bare exponential underflows.

Unavailable values appear as gaps or dashes; an unknown current direction is not drawn as a zero-current arrow. If interval charges or sampled flow terms are unavailable, their balance is not certified by inserting zeros. This conservative calculation may decline an expression that another evaluation order could recover. Exactly zero duration or a stationary cloud still has zero charge change and integrated net inflow. Two unavailable quadrature values do not establish a zero order difference.

A nonzero projected current or nonzero current through a tiny patch can also fall below numerical range. The patch calculator preserves unavailable values instead of labeling that product an exact zero;an exactly zero patch area still has zero through-current. This is separate from the spatial and temporal integration diagnostics.

The displayed local residual also needs interpretation. Substituting two algebraically opposite analytic derivatives gives zero by construction. Independent finite differences and separately integrated boundary flows provide additional evidence. A zero from the same formula on both sides is not independent verification.

## Research and AI connections

A current predictor should be checked jointly with density evolution and control-volume charge. Evaluate both local residuals and integrated charge changes on intervals not used in training. Flip carrier signs, reverse velocity, change cross-section and move the observation window as separate experiments.

Keep interval geometry, physical density units and derivative conventions fixed when comparing models. A density renormalized separately inside every view can destroy the conservation problem. A continuity-preserving prediction can still violate a force law, constitutive law, Maxwell equation or experimental observation. Report which equations are actually evaluated.

## Explained exercises

### Exercise 1 — Negative carriers moving right

Take $\rho=-2$ C/m³,$v=0.3$ m/s, and a directed patch of $0.5$ m² with normal $+x$. Find $J_x$ and $I$.

**Answer.** $J_x=-0.6$ A/m² and $I=-0.3$ A. The current direction opposes the carriers' velocity. Reversing the patch normal reports $+0.3$ A without changing the transport.

### Exercise 2 — One boundary is not enough

A positive-charge stream has $AJ_x(a)=0.4$ A and $AJ_x(b)=0.1$ A. What is the stored-charge derivative?

**Answer.** It is $0.4-0.1=0.3$ C/s. The outward currents are $-0.4$ A on the left and $+0.1$ A on the right;negative their sum gives the same result.

### Exercise 3 — Change the cross-section

Halve $A$ while keeping $Q_0,\sigma, x_0, v$ and the interval fixed. Which observations change?

**Answer.** $\rho$ and $J_x$ double, while $A\rho$, full-face currents and interval charge remain unchanged. This comparison holds fixed the total cloud charge, not a density supplied by an external reservoir.

### Exercise 4 — Zero density with nonzero current

Can two carrier species have zero net charge density but nonzero conventional current?

**Answer.** Yes. Equal positive and negative densities moving oppositely have cancelling charge densities and reinforcing currents. Sum species currents before inferring current from net density;the single-velocity preparation does not cover this case.

### Exercise 5 — The sign of the finite-time change

In the worked example, the initial inflow is positive. Must the charge change over the next second be positive?

**Answer.** No. The inflow later reverses. Its time integral is approximately $-0.125673973$ C even though its initial value is positive. An initial derivative is not an average over a finite interval.

### Exercise 6 — A perfect residual that proves too little

A model sets $\partial_t\rho=-v\partial_x\rho$ and $\partial_xJ_x=v\partial_x\rho$ from the same expression. Is a zero sum evidence for a complete electromagnetic solution?

**Answer.** It confirms an imposed analytic identity. Check independent numerical derivatives, stored-charge integrals and boundary flows. Even those checks establish this transport balance only;they do not supply forces, constitutive behavior or electromagnetic fields.

## Sources and reproducible checks

[Haus and Melcher §1.4](https://web.mit.edu/6.013_book/www/chapter1/1.4.html) defines sheet current density K as a tangential vector in A/m,distinct from current I through a surface.

[MIT 8.02T, Current and Resistance, §§6.1–6.1.1](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/6b2cf7ec2d597d7459930a7e6f700bea_chapter6current.pdf) supplies the directed-current and carrier-current definitions. The fixed-interval continuity derivation, Gaussian preparation, moving-window comparison and numerical examples are developed explicitly above.

`learning-charge-current-fields.ts` evaluates the prescribed model. Independent SciPy CDFs and space/time quadratures cover signed charge, flow reversal, stationary clouds and area changes;five-point finite differences check local derivatives. Narrow-tail references test relative accuracy rather than allowing a small nonzero result to pass as zero under a large absolute tolerance.
