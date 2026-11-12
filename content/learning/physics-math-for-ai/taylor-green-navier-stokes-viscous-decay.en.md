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

## Research extension: base-flow decay and prediction-error growth

The original benchmark shows that a particular flow loses kinetic energy. Does a learned flow approaching that benchmark necessarily reduce its prediction error? Define another smooth, divergence-free solution v of the **same unforced equations**, with the same constant density, viscosity and periodic cell. It may have different initial data. The difference w is a velocity error, and its squared integral D is an error energy in J. This section derives a budget between solutions, not the residual of an arbitrary neural prediction.

$$
\mathbf w=\mathbf v-\mathbf u,\qquad\nabla\cdot\mathbf w=0,\qquad\mathcal D(t)=\frac\rho2\int_V|\mathbf w|^2\,dV,\qquad V=(2\pi/k)^2d.
$$

Subtract the two nonlinear momentum equations before taking a norm. Three transport terms appear: background transport of error, distortion of the background by error, and self-transport of error. The pressure difference is periodic. These assumptions allow integration by parts without wall flux or a pressure work contribution across an unmatched boundary.

$$
\partial_t\mathbf w+(\mathbf u\cdot\nabla)\mathbf w+(\mathbf w\cdot\nabla)\mathbf u+(\mathbf w\cdot\nabla)\mathbf w=-\frac{\nabla\pi}{\rho}+\nu\Delta\mathbf w,\qquad\pi=p_v-p_u.
$$

Dot the difference equation with rho times w and integrate. Background transport and self-transport vanish by periodicity and zero divergence; pressure does too. Viscosity gives a negative gradient integral. The remaining exchange samples the symmetric background strain S, since a vector contracted with an antisymmetric matrix gives zero. Its sign is not fixed. A decaying base flow can transfer energy into an error direction even though the kinetic energy of each full solution decreases.

$$
\dot{\mathcal D}=-\rho\int_V\mathbf w^TS(\mathbf u)\mathbf w\,dV-\rho\nu\int_V|\nabla\mathbf w|^2\,dV,\qquad S(\mathbf u)=Ak\begin{pmatrix}\sin(kx)\sin(ky)&0\\0&-\sin(kx)\sin(ky)\end{pmatrix}.
$$

Require **zero spatial mean error** for the next bound. The smallest nonzero Fourier wave number on this square is k, so Poincare supplies k². The pointwise largest absolute strain eigenvalue is at most |A|k. Combining these gives a scalar differential inequality. Both mean-zero status and the fundamental cell matter: a larger repeated domain permits longer waves and changes the Poincare constant.

$$
\int_V\mathbf w\,dV=0\ \Longrightarrow\ \int_V|\nabla\mathbf w|^2\,dV\ge k^2\int_V|\mathbf w|^2\,dV,\qquad\dot{\mathcal D}\le(2|A|k-2\nu k^2)\mathcal D.
$$

Integrate the inequality using the known exponential A. R is a Reynolds number for this amplitude and length 1/k; it is not U L/nu with L = 2pi/k. The dimensionless time z includes a factor two. For computation, evaluate 1-exp(-z) as -expm1(-z) when z is small. This is a conservative upper envelope, and its increase does not by itself prove growth of any actual error.

$$
\nu>0:\quad z=2\nu k^2t,\quad R=\frac{|U|}{\nu k},\qquad\mathcal D(t)\le\mathcal D(0)\exp\big[R(1-e^{-z})-z\big].
$$

For R at most one the envelope is nonincreasing. This is a sufficient condition from a norm estimate, not a universal physical transition Reynolds number. For R greater than one, differentiate the logarithm of the envelope: its derivative with respect to z is R exp(-z)-1. Its maximum is at log R. Treat zero viscosity separately rather than dividing by it. Even when the envelope eventually decreases, it need not be a sharp description of an individual solution.

$$
R>1:\quad z_{\rm peak}=\log R,\quad\log G_{\rm peak}=R-1-\log R;\qquad\nu=0:\quad\mathcal D(t)\le\mathcal D(0)e^{2|U|kt}.
$$

To demonstrate an actual positive **initial** budget, choose a two-mode stream function. The coefficients a and b have units m/s, and psi has units m²/s. Commuting derivatives makes this w divergence-free; both nonconstant modes have zero mean. It is a valid initial error, so u at time zero plus this w is admissible smooth initial velocity. Do not assume this two-mode form remains an exact solution at later times.

$$
\xi=kx,\quad\eta=ky,\quad\psi=\frac ak\cos\xi+\frac bk\cos(2\xi+\eta),\qquad\mathbf w=(\partial_y\psi,-\partial_x\psi)=\begin{pmatrix}-b\sin(2\xi+\eta)\\a\sin\xi+2b\sin(2\xi+\eta)\end{pmatrix}.
$$

Integrate the trigonometric products over one physical cell including depth. Orthogonality gives the first two integrals. Coupling to the background strain leaves the mixed a b term in the third, so changing the relative sign of the two modes changes production. Here U is the signed initial base amplitude, whereas R above uses its absolute value. The units are respectively m⁵/s², m³/s² and m⁵/s³ after the depth integral.

$$
\int_V|\mathbf w|^2\,dV=\frac V2(a^2+5b^2),\quad\int_V|\nabla\mathbf w|^2\,dV=\frac{k^2V}{2}(a^2+25b^2),\quad\int_V\mathbf w^TS(\mathbf u)\mathbf w\,dV=\frac{UkV}{2}ab\quad(t=0).
$$

Set b to minus one fifth of a, with a nonzero reference velocity. The strain contribution becomes positive when U is positive. At R = 20 with U positive the exact initial error budget is positive, although the base kinetic energy derivative remains negative. The value 10 here is specific to this chosen two-mode direction. It is not an instability threshold, an exponentially growing eigenmode, or a statement about its whole future trajectory. In contrast, two exact members of the Taylor–Green amplitude family have error energy decaying as exp(-4nu k²t).

$$
a=v_{\rm ref},\quad b=-v_{\rm ref}/5:\qquad\dot{\mathcal D}(0)=\rho Vv_{\rm ref}^2\big(Uk/10-\nu k^2\big)>0\quad\text{if }U>10\nu k,\ v_{\rm ref}\ne0.
$$

Finally remove the mean-zero assumption deliberately. A Galilean-shifted base flow plus a constant velocity b is another exact solution, with the pressure shifted by the same spatial argument. Its mean error is b at every time. Jensen or Cauchy–Schwarz gives a positive lower bound on D when b is nonzero. For a resting base, a constant velocity solution realizes the bound exactly. Simply adding b to the unshifted nonconstant base generally fails the equations. This control explains why a mean-free decay certificate cannot be reused for arbitrary mean error.

$$
\mathbf v(\mathbf x,t)=\mathbf u(\mathbf x-\mathbf b t,t)+\mathbf b,\quad p_v(\mathbf x,t)=p_u(\mathbf x-\mathbf b t,t),\qquad\overline{\mathbf w}=\mathbf b,\quad\mathcal D(t)\ge\frac{\rho V}{2}|\mathbf b|^2.
$$

The table takes U positive and nu positive. R is followed by the initial upper-bound rate divided by nu k², the exact two-mode initial budget divided by rho V v-ref² nu k², the envelope peak time z, and its peak gain. The gain is a bound on the squared-error integral, not on velocity amplitude. At R = 5 the bound can rise while this particular initial perturbation still decays; R = 20 gives positive initial production for it. No later trajectory of that perturbation is computed here.

| R | bound rate | initial budget | peak z | bound gain |
| --- | --- | --- | --- | --- |
| 0.50000000 | -1.00000000 | -0.95000000 | 0.00000000 | 1.00000000 |
| 1.00000000 | 0.00000000 | -0.90000000 | 0.00000000 | 1.00000000 |
| 5.00000000 | 8.00000000 | -0.50000000 | 1.60943791 | 10.91963001 |
| 20.00000000 | 38.00000000 | 1.00000000 | 2.99573227 | 8924115.04815937 |

```python
from math import exp, log
for reynolds in (0.5, 1.0, 5.0, 20.0):
    bound_rate = 2*(reynolds-1)
    initial_budget = reynolds/10-1  # U positive, a=v_ref, b=-v_ref/5
    peak_time = log(reynolds) if reynolds > 1 else 0.0
    gain = exp(reynolds-1-peak_time) if reynolds > 1 else 1.0
    print(" ".join(f"{x:.8f}" for x in (reynolds, bound_rate, initial_budget, peak_time, gain)))
```

The [Tanveer energy-method notes](https://people.math.osu.edu/tanveer.1/m865/lec5week.pdf) give integration-by-parts background. The finite-cell constants, Taylor–Green envelope and two-mode initial exchange are derived here. These estimates concern smooth solutions on their common existence interval and make no assertion about general three-dimensional regularity.

![Error upper envelopes, exact initial two-mode budgets and persistent mean error](/assets/learning/continuum-perturbation-energy-exchange.en.svg)

### Research questions with explanations

**Does an increasing error upper bound prove instability?** **Answer:** No. The bound discards directional information. The two-mode calculation separately exhibits a positive initial budget, and even that does not prove sustained growth or a spectral instability.

**Why cannot viscosity remove the mean error in this periodic problem?** **Answer:** The Laplacian of a spatial constant is zero, and the spatial mean momentum is conserved without forcing. A Galilean-shifted solution retains its nonzero mean difference.

**Is this budget valid for any predicted movie with small PDE residual?** **Answer:** Not as written. The derivation assumes two actual divergence-free solutions with compatible periodic pressure and the same parameters. A residual-bearing prediction adds a forcing term to the difference budget that must be estimated separately.

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
