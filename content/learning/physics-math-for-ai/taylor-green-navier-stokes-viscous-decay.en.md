## A vortex can keep its shape while losing energy

Imagine a grid of smoothly rotating fluid cells. Over time their motion weakens, but the pattern stays recognizable. This is a useful setting for asking a precise question: which part of the governing equation changes the speed, and which part bends the motion? The answer is not simply that viscosity balances everything. In this example pressure balances nonlinear convection, while viscosity balances the time decay of velocity.

The two-dimensional Taylor–Green family gives an exact smooth solution of a specified incompressible equation. It lets us compare velocity, pressure, derivatives, physical budgets and particle paths without confusing a numerical solver with the field being studied. It is a demanding consistency check for a learning model, although it is not a representation of general three-dimensional turbulence.

## Prerequisites, assumptions and units

Review velocity gradients, divergence, material acceleration, stress and kinetic energy. Density is constant and positive. Kinematic viscosity is constant and nonnegative. There are no walls or body forces in our periodic square. A prescribed depth turns a two-dimensional cell into a physical volume. The pressure offset is arbitrary within this incompressible model; pressure gradients, rather than its constant reference, enter momentum.

| Symbol | Unit | Meaning |
| --- | --- | --- |
| rho | kg/m³ | constant mass density |
| U | m/s | initial velocity amplitude |
| k | rad/m | positive spatial wave number |
| nu | m²/s | kinematic viscosity |
| depth | m | out-of-plane physical thickness |

$$
\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u=-\frac{\nabla p}{\rho}+\nu\Delta\mathbf u,\qquad\nabla\cdot\mathbf u=0,\qquad\mu=\rho\nu.
$$

## Construct the actual velocity field

Let the amplitude decay exponentially and give the velocity components a quarter-turn spatial symmetry. The signs matter: with the convention below the vorticity near the origin is positive. The scale U is not a uniform background translation. The velocity has zero cell mean, and individual particles move through a spatially varying field. Reading U as a mean flow would give the wrong interpretation of both transport and energy.

$$
A(t)=Ue^{-2\nu k^2t},\qquad u=-A(t)\cos(kx)\sin(ky),\qquad v=A(t)\sin(kx)\cos(ky).
$$

The physical square has side two pi divided by wave number. The field repeats across its boundaries. This is a periodic identification, not a no-slip wall condition. A drawing with a square frame must not imply that fluid is confined by a solid wall at every edge.

$$
L=\frac{2\pi}{k},\qquad V=L^2d,\qquad M=\rho V.
$$

## Verify incompressibility and read the local rotation

Differentiate the two components with respect to their own coordinates. Their divergence contributions cancel at every point, not just after averaging over the square. Incompressibility constrains volume change; it does not mean all components of the velocity gradient vanish. The antisymmetric gradient part produces vorticity, and the symmetric part describes deformation rate.

$$
u_x=Ak\sin(kx)\sin(ky),\quad v_y=-Ak\sin(kx)\sin(ky),\quad\omega_z=v_x-u_y=2Ak\cos(kx)\cos(ky).
$$

A positive vorticity at the origin is consistent with counterclockwise local rotation under the conventional x-right, y-up axes. Reflecting a display's vertical axis without adjusting the interpretation can visually reverse that sense.

## Compute nonlinear convection before canceling it

Evaluate the directional derivative of each velocity component along the local velocity. The result is generally nonzero even though the flow pattern only decays. A particle accelerates while turning through the pattern. This is the reason that a time-independent Eulerian field can still have nonzero material acceleration. Keep the two acceleration terms separate in a visualization.

$$
(\mathbf u\cdot\nabla)\mathbf u=-\frac{A^2k}{2}\begin{pmatrix}\sin(2kx)\\\sin(2ky)\end{pmatrix}.
$$

The factor of two inside each sine comes from products of velocity and its gradient. A model using only a single-frequency pressure ansatz would miss the pressure response even if its velocity reconstruction looked convincing.

## Pressure supplies the turning acceleration

Choose a pressure change with twice the spatial frequency and twice the exponential decay rate of velocity. Differentiate it before comparing signs. The pressure acceleration is minus the gradient divided by density; comparing convection to the positive pressure gradient would reverse the force. Both coordinates contribute, so a one-dimensional pressure profile is insufficient for the full field.

$$
p-p_0=-\frac{\rho A^2}{4}\big[\cos(2kx)+\cos(2ky)\big],\qquad-\frac{\nabla p}{\rho}=-\frac{A^2k}{2}\begin{pmatrix}\sin(2kx)\\\sin(2ky)\end{pmatrix}.
$$

Changing p-zero adds a constant without changing motion. A negative pressure change relative to that reference is not a claim of negative thermodynamic absolute pressure. The incompressible model does not provide a complete equation of state or a cavitation model.

## Viscosity supplies the temporal decay

Each velocity component contains one sine or cosine factor in each coordinate. Its Laplacian therefore supplies minus two wave-number-squared times that component. Differentiating the exponential amplitude in time supplies the same factor multiplied by viscosity. These are equal vector fields. Together with the previous pressure-convection balance, they establish the full momentum equation point by point.

$$
\partial_t\mathbf u=-2\nu k^2\mathbf u,\qquad\Delta\mathbf u=-2k^2\mathbf u,\qquad\partial_t\mathbf u=\nu\Delta\mathbf u.
$$

At zero viscosity the field becomes a steady nonlinear Euler vortex. Pressure and convection remain nonzero in general. Turning off viscosity does not turn the fluid into stationary particles or eliminate pressure forces.

## Hand calculation at a chosen point

Set density and wave number to one, initial amplitude to two and viscosity to one tenth. At time zero and position one quarter pi in each coordinate, velocity is minus one and plus one metres per second. Convection points equally toward negative x and negative y. Temporal acceleration is opposite to the velocity and has a smaller magnitude. These numbers let you check arrows and labels without relying on a rendered animation.

$$
\rho=k=1,\ U=2,\ \nu=0.1,\ t=0,\ x=y=\pi/4:\quad\mathbf u=(-1,1),\quad(\mathbf u\cdot\nabla)\mathbf u=(-2,-2),\quad\mathbf u_t=(0.2,-0.2).
$$

Velocity has units m/s, whereas both acceleration vectors have units m/s². Equal arrow lengths across these quantities would need a stated scaling; a single unexplained vector scale cannot make them physically comparable.

## Integrate energy over the physical periodic cell

Kinetic energy density is one half density times speed squared. Integrate both trigonometric contributions over an entire period cell, including depth. The squared cosine and sine factors each average to one half. This gives a quarter, rather than a half, times density times amplitude squared for the mean energy density. Pressure and convection redistribute energy locally; their net transport across a full periodic cell cancels.

$$
\overline e_K=\frac{\rho A^2}{4},\qquad K=\frac{\rho U^2V}{4}e^{-4\nu k^2t},\qquad[K]=\mathrm J.
$$

An arbitrary subwindow does not have this cancellation. For that window, boundary transport must be included before interpreting an energy change as viscous dissipation.

## Dissipation comes from strain rate, not rotation alone

Form the symmetric velocity gradient D. In a Newtonian incompressible fluid the local conversion of kinetic energy into internal energy is twice dynamic viscosity times the squared tensor norm of D. Count both off-diagonal entries in that norm. In this special field their sum vanishes, while the two diagonal entries remain generally nonzero. Vorticity measures local rotation and cannot simply replace the local strain tensor in the dissipation formula.

$$
D=\frac{\nabla\mathbf u+(\nabla\mathbf u)^T}{2},\qquad\Phi=\int_V2\rho\nu(D:D)\,dV=4\nu k^2K,\qquad\frac{dK}{dt}=-\Phi.
$$

Here Phi is measured in watts. It removes kinetic energy; total energy is not destroyed. The teaching model does not evolve temperature, so it does not track the resulting internal-energy field explicitly.

## Distinguish streamlines, particle paths and numerical error

A streamline is tangent to the velocity field at a fixed time. A particle trajectory solves a time-dependent initial-value problem. Since this particular velocity family is a scalar time factor times a fixed spatial pattern, particles follow the same geometric streamline curves while slowing down. That coincidence of geometry is special: equal curves do not imply equal time parametrization, and it does not extend to general unsteady flows.

$$
\dot X=u(X,Y,t),\quad\dot Y=v(X,Y,t),\qquad\frac{d}{dt}\big[\cos(kX)\cos(kY)\big]=0.
$$

Our RK4 tracer approximates this equation using physical seconds and unwrapped positions. Halve its time step and compare the endpoint with an independently computed trajectory. Conservation of the geometric label is a useful additional check, but a tracer with a wrong clock can preserve that label too. Endpoint convergence and the time-dependent vector field are therefore separate requirements.

## Remove pressure as a negative control

Use the same actual velocity field and delete only the pressure acceleration when evaluating momentum. The remaining residual equals the convection term because the temporal and viscous terms already balance. It does not vanish merely because the velocity is divergence-free. This controlled change identifies a missing physical contribution without substituting another flow that might satisfy a different equation.

$$
R=\mathbf u_t+(\mathbf u\cdot\nabla)\mathbf u+\frac{\nabla p}{\rho}-\nu\Delta\mathbf u,\qquad R_{\mathrm{omit}\ p}=(\mathbf u\cdot\nabla)\mathbf u.
$$

If a diagnostic normalizes each vector independently, direction may remain readable while magnitude comparisons become invalid. Keep actual SI component values beside any direction-only schematic.

## Tiny changes and rounded equality

For tiny positive viscosity or time, the floating-point decay factor can round to one. Subtracting one from it then returns zero even though the mathematical change is nonzero. Evaluate the change with expm1, retain both observations, and label them separately. Energy has twice the velocity-decay exponent. A time-zero or viscosity-zero control has an exactly zero change, unlike a nonzero product too small to represent.

$$
z=2\nu k^2t,\qquad\frac{A(t)-U}{U}=\operatorname{expm1}(-z),\qquad\frac{K(t)-K(0)}{K(0)}=\operatorname{expm1}(-2z).
$$

These ratios assume nonzero U for their interpretation; the implementation also carries absolute changes, which remain zero for the rest state. If the positive exponent lies below the display range, the relative change can be unavailable while a weighted absolute change remains representable. Compute these channels separately. Squared amplitudes can also underflow while velocity remains representable. Small numerical residuals alone cannot prove accurate energy or motion at these scales.

Think of an energy density and the energy in a whole cell as two separate measurements. A tiny nonzero density can lie below the linear display range while multiplication by the actual cell volume yields a representable energy. The implementation combines that volume with the squared amplitude before rounding. It similarly combines the small decay with the initial amplitude or energy before reporting an absolute change. At the smallest binary64 scales, a second-order correction also determines which side of a rounding midpoint the exponential lies on; replacing every small exponential change by its linear term can select the wrong last bit. Coordinates and trigonometric factors retain their ordinary finite precision.

## Experiment sequence and AI research use

First inspect the spatial velocity, pressure and vorticity at time zero. Display convection and pressure acceleration separately, then time derivative and viscous acceleration. Set viscosity to zero and observe the steady Euler limit. Restore viscosity and compare velocity decay with the faster energy decay. Change the pressure offset and verify that forces and motion do not change. Remove the pressure term only in the residual control. Finally compare particle endpoints at multiple time steps and examine tiny-viscosity changes alongside their rounded counterparts.

For a learned PDE model, reserve unseen viscosities and wave numbers for evaluation. Report divergence, dimensional momentum residuals, pressure-gauge handling, integrated energy and actual tracer motion separately. A low residual can hide a low-amplitude collapse, and an attractive streamline plot does not verify particle timing. This exact family is a controlled benchmark, not evidence that a model handles walls, shocks, arbitrary initial data or three-dimensional turbulent cascades.

## Exercises with explanations

### Exercise 1: the physical cell

Take wave number two radians per metre and depth half a metre. What are the side length and volume of one full period cell?

**Answer:** The side is pi metres and the volume is one half pi-squared cubic metres. The depth must be included before multiplying by density to obtain kilograms or by energy density to obtain joules.

### Exercise 2: zero viscosity

Does setting viscosity to zero make every term in momentum zero?

**Answer:** No. Temporal and viscous terms vanish, while nonlinear convection and pressure acceleration generally remain and are equal. The Eulerian field is steady, but particles still turn and accelerate along its curves.

### Exercise 3: two decay rates

Velocity amplitude has fallen to one half its initial value. What fraction of the initial cell kinetic energy remains?

**Answer:** One quarter, because energy is quadratic in velocity amplitude. Density and cell geometry are fixed. The dissipative rate must be compared with energy in joules per second, not with velocity in metres per second.

### Exercise 4: a pressure gauge

Add three pascals everywhere without changing velocity. Which momentum term changes?

**Answer:** None. The pressure gradient is unchanged. The displayed pressure shifts, but pressure acceleration, trajectories, kinetic energy and dissipation are identical in this incompressible model.

### Exercise 5: streamline agreement

Two numerical tracers stay on the correct geometric curve, but reach different positions at the same time. Are both verified?

**Answer:** No. Preserving the geometric label only checks the curve. Compare the time-dependent ODE, endpoint reference and time-step refinement. A wrong clock can travel along the correct streamline with incorrect physical motion.

### Exercise 6: the rounded decay control

At positive viscosity the displayed decay factor equals one, while expm1 gives a negative change. Which observation should be removed?

**Answer:** Neither. They describe a rounded factor and a more stable change calculation. Preserve both with their definitions. If the nonzero change itself is outside arithmetic range, mark that channel unavailable; do not replace it with the exactly zero-viscosity interpretation.

## Sources and scope

The [NVIDIA Warp Taylor–Green example](https://github.com/NVIDIA/warp/blob/main/warp/examples/fem/example_taylor_green.py) supplies a primary implementation of the unit-square exact velocity and pressure used as a consistency reference. Here we derive dimensional scaling, physical budgets and controls separately. Our periodic teaching cell is an explicit modeling choice; it does not inherit every numerical boundary condition or solver choice from that example. No FEM solver code or solver results are imported as proof of this implementation.
