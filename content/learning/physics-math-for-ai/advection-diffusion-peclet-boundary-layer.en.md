## Why changing units does not change the transport regime

Imagine dye moving through a straight channel. Flow carries dye along the channel, while diffusion smooths concentration differences. A velocity written as 50 centimeters per second sounds numerically larger than 0.5 meters per second, but these describe the same motion. A useful comparison must survive a consistent change of units. The Peclet number compares the two transport mechanisms rather than the sizes of numbers in a chosen unit system.

There is a second, different operation: building a physically larger experiment while changing velocity and diffusivity to preserve the same dimensionless equation. This can preserve the shape of a concentration profile without preserving its travel time or actual mass transport. We will derive both statements from a boundary-value problem that can be solved exactly.

## Declare the family before interpreting its plots

We consider a dilute passive scalar with steady one-dimensional transport through a uniform interval. Velocity U and diffusivity D are constant; D and length L are positive. Concentrations at both ends are independently imposed and nonnegative. Cross-sectional area A is positive. There is no reaction, internal source, temperature coupling or feedback from concentration to velocity. The one-dimensional equation assumes that transverse structure has already been excluded or appropriately averaged.

$$
Uc_x=Dc_{xx},\qquad 0\le x\le L,\qquad
c(0)=c_L,\quad c(L)=c_R.
$$

Concentration has units kg/m³, U has units m/s and D has units m²/s. Flux density has units kg/(m² s); multiplying it by A gives a mass rate in kg/s. The word steady means that storage no longer changes in time. It does not mean that transport stops or that the interval contains no material.

## Derive the dimensionless equation

If the boundary contrast is nonzero, use interval length as the spatial scale and the actual concentration contrast as the concentration scale. Applying the chain rule introduces one factor of 1/L for each spatial derivative.

$$
\xi=\frac{x}{L},\qquad
\theta=\frac{c-c_L}{c_R-c_L},\qquad
\mathrm{Pe}\,\theta_\xi=\theta_{\xi\xi},\qquad
\mathrm{Pe}=\frac{UL}{D}.
$$

The endpoint conditions become theta-zero equals zero and theta-one equals one. Pe is signed here because U is signed. Its magnitude compares mechanisms; its sign records flow direction. Some texts define a positive Peclet number using a speed and specify direction elsewhere. These conventions are compatible once declared.

If the two boundary concentrations agree, the physical solution remains well defined: concentration is constant and total flux is U times that concentration. The measured normalization above is then 0/0. We can still display a declared unit-contrast basis function, but cannot label it as the normalized observation of the constant field.

## Recover the clocks instead of memorizing the ratio

Diffusion crosses a distance L on a characteristic scale proportional to L²/D. Advection crosses it on L divided by speed. These are characteristic comparison scales, not exact arrival times of every molecule.

$$
t_D=\frac{L^2}{D},\qquad
t_A=\frac{L}{|U|}\quad(U\ne0),\qquad
|\mathrm{Pe}|=\frac{t_D}{t_A}.
$$

For large magnitude Pe, advection crosses the interval faster than diffusion smooths it over the full interval. Near a boundary, however, the relevant diffusion distance can be much smaller than L. Diffusion can therefore remain essential in a thin layer even when it is weak on the global scale.

At U equals zero, there is no finite advection crossing time. The diffusion clock remains valid. If we restore the transient equation and choose tau equals Dt/L², its dimensionless form is c-tau plus Pe times c-xi equals c-xixi. Choosing an advection clock would instead become singular at zero velocity. This is a failure of that scale choice, not a failure of diffusion physics.

## Solve the boundary-value problem

First solve the first-order equation for the concentration gradient, then integrate once more and impose both endpoint concentrations. For nonzero Pe the resulting profile is exponential. At exactly zero velocity the original equation has a linear solution.

$$
\theta(\xi)=\frac{e^{\mathrm{Pe}\xi}-1}{e^{\mathrm{Pe}}-1},\qquad
c=c_L+(c_R-c_L)\theta,\qquad
\mathrm{Pe}=0:\ \theta=\xi.
$$

For positive Pe, most of the interval resembles the left boundary concentration and the right condition is accommodated near the downstream end. Reversing flow and exchanging the two boundary values reflects the profile about the midpoint. The corresponding thin-layer scale is D divided by speed. It is a useful asymptotic scale when that length is much smaller than L, not a separate sharp interface.

Setting D exactly to zero changes the order of the equation. A first-order transport equation generally cannot satisfy two arbitrary endpoint concentrations. A visually flat outer solution therefore cannot replace the finite-D solution while retaining both original boundary conditions.

## Evaluate large Pe without destroying its small tail

The direct exponential formula can overflow before a perfectly valid ratio is formed. For positive P equal to Pe, compute two positive boundary weights instead.

$$
w_R=e^{-P(1-\xi)}\frac{1-e^{-P\xi}}{1-e^{-P}},\qquad
w_L=\frac{1-e^{-P(1-\xi)}}{1-e^{-P}},\qquad
c=c_Lw_L+c_Rw_R.
$$

Their exact sum is one. We do not force separately rounded weights to sum to one by changing the field. For negative Pe, use Q equals minus Pe: the right weight is the unattenuated ratio with xi, and the left weight includes exp of minus Q times xi. Stable expm1 evaluations retain differences between nearby exponentials.

An exponentially small positive weight can lie below the smallest representable positive floating-point number. In that case its ordinary value is unavailable, while its positive sign and finite logarithm remain meaningful. It must not be replaced with an exact zero boundary or an invented diffusion-free region.

## Check actual flux, including cancellation

The steady equation implies a spatially constant total flux. Its advective and diffusive parts can vary strongly, and either part alone generally fails the conservation test.

$$
j=Uc-Dc_x,\qquad j_x=0,\qquad
j=U\frac{c_L-c_Re^{-\mathrm{Pe}}}{1-e^{-\mathrm{Pe}}}\quad(\mathrm{Pe}>0),\qquad
U=0:\ j=-\frac{D(c_R-c_L)}L.
$$

If cL is zero, cR is positive and Pe is large positive, the total flux is a tiny negative number. At some interior points, large terms in Uc minus Dcx can cancel when rounded. The stable boundary expression retains the actual counter-flow flux. The interface should show this result separately from a direct rounded subtraction. A zero diagnostic residual obtained from the same rounded terms would not independently prove accuracy.

## Work through a dimensional example

Choose U equals 0.5 m/s, L equals 2 m and D equals 0.1 m²/s. Then Pe equals 10, diffusion time is 40 s and advection time is 4 s. With cL equals 1 and cR equals 0.2 kg/m³, the exact total flux is approximately 0.50001816 kg/(m² s). An area of 1.5 m² gives approximately 0.75002724 kg/s.

At the right endpoint, advection contributes only 0.1 kg/(m² s). The negative concentration gradient supplies a positive diffusive contribution of approximately 0.40001816. The sum agrees with the left-end flux. Checking only Uc would incorrectly suggest that material disappears inside the channel.

## Preserve small corrections near Pe zero

The profile approaches a line continuously. Its first correction is nevertheless nonzero whenever velocity and the interior coordinate are nonzero.

$$
\theta=\xi+\frac{\mathrm{Pe}}2\xi(\xi-1)+O(\mathrm{Pe}^2).
$$

At the midpoint, the leading correction is minus Pe divided by eight. It can survive in a stable correction calculation even when adding it to 0.5 rounds back to 0.5. A rounded plot that looks straight does not establish that velocity is exactly zero. Conversely, at an endpoint the correction is exactly zero because the prescribed concentration is enforced there.

## Compare similar experiments and numerical grids

Let a and b be positive scale factors. Changing physical parameters as follows preserves Pe and the unit-contrast profile at matching xi.

$$
L'=aL,\qquad D'=bD,\qquad U'=\frac baU,\qquad
t_D'=\frac{a^2}b t_D,\qquad j'=\frac ba j.
$$

If geometry is also scaled with A-prime equals a-squared A, the total mass rate scales by ab. This is physical similarity between different experiments. Expressing the original experiment in centimeters instead of meters changes numerical parameter values consistently but changes none of its physical observables.

For a numerical mesh with spacing h, a common grid Peclet convention is absolute U times h divided by 2D. It compares transport over one cell, not over the entire domain. A large physical Pe can coexist with a small grid Pe on a sufficiently fine mesh. Resolving a boundary layer and changing the physical transport regime are different actions; the present exact model is not itself a discretization error estimate.

## Locate this number among other AI-relevant scales

In a constant-density Newtonian momentum balance, Reynolds number compares inertia and viscosity. With kinematic viscosity nu, mass diffusivity D and thermal diffusivity alpha, the identities are Re equals speed times L divided by nu, Sc equals nu/D and Pr equals nu/alpha. Using the same speed and length gives mass Pe equals Re Sc and heat Pe equals Re Pr. Thermal and mass diffusivities cannot be interchanged merely because both have units m²/s.

The following map uses a positive reference speed V, density rho, dynamic viscosity mu, gravity g and sound speed cs. The reference speed is a chosen scale, not necessarily the local instantaneous velocity.

| Group | Definition | Comparison and limitation |
| --- | --- | --- |
| Reynolds | Re = rho V L / mu | Inertia versus Newtonian viscosity; a different rheology needs its own balance. |
| Mach | Ma = V / cs | Speed versus acoustic speed; small Ma alone does not prove incompressibility under every forcing. |
| Froude | Fr = V / sqrt(g L) | Inertial versus gravitational scales; the coefficient of gravity under inertial scaling is 1/Fr². |
| Prandtl | Pr = nu / alpha | Momentum versus heat diffusivities, with specified material properties. |
| Schmidt | Sc = nu / D | Momentum versus mass diffusivities for the selected species. |
| Peclet | Pe magnitude = V L / D | Global advection versus scalar diffusion; direction is specified separately here. |
| Damköhler | Da = k L² / D | Diffusion time versus first-order reaction time in the next slab family. |

An inertial pressure scale rho V² and an advection clock L/V require a positive chosen V. Substituting an observed zero velocity into these scales can create a singular normalization without making the physical equations invalid. Matching Pe alone does not match compressibility, reaction, geometry or all boundary conditions. For AI surrogates, this distinction helps decide which input combinations can support transfer between experiments and which omitted mechanisms may invalidate it.

## Six exercises with answers

### Exercise 1 — Change units

Convert the worked example to centimeters and seconds. Does Pe change?

**Answer:** U becomes 50 cm/s, L becomes 200 cm and D becomes 1000 cm²/s. Their ratio remains 10. Flux numbers must also be converted with their mass and area units; the physical transport stays the same.

### Exercise 2 — Reverse the experiment

What happens if velocity is reversed and the two endpoint concentrations are exchanged?

**Answer:** The profile is reflected: the new concentration at xi is the old one at one minus xi. Its gradient changes sign, and total flux changes sign. The absolute transport time scales are unchanged.

### Exercise 3 — Constant concentration

Do equal nonzero endpoint concentrations imply zero flux?

**Answer:** No. Concentration is constant, diffusive flux is zero and total flux is Uc. The contrast-normalized measured profile is undefined, although a declared basis function is available.

### Exercise 4 — Similarity and actual mass rate

Use a equals 2 and b equals 1 with geometrically similar area. How do velocity, diffusion time, flux density and total mass rate change?

**Answer:** Velocity halves, diffusion time grows by four, flux density halves and total mass rate doubles because area grows by four. The normalized profile remains the same if endpoint data correspond.

### Exercise 5 — A straight-looking curve

At Pe equals 10 to the power minus 20, may a midpoint that rounds to 0.5 be reported as proof of zero velocity?

**Answer:** No. The leading midpoint correction is minus 1.25 times 10 to the power minus 21. Preserve this separately from the rounded field and retain the nonzero input velocity.

### Exercise 6 — A larger mesh

Does increasing grid spacing increase the physical Peclet number?

**Answer:** Physical Pe stays fixed when U, L and D stay fixed. Grid Pe increases, and the mesh may fail to resolve the downstream layer. A numerical change is not a physical parameter change.

## Source and scope

[Langtangen and Linge, Finite Difference Computing with PDEs](https://hplgit.github.io/fdm-book/doc/pub/book/html/._fdm-book013.html) presents a stationary one-dimensional convection–diffusion example and its exponential solution. We restore dimensional coefficients, signed velocity, actual boundary concentrations and area, then derive the flux and similarity relations here. This scalar family does not solve multidimensional turbulence, concentration-dependent transport or a general fluid momentum system.
