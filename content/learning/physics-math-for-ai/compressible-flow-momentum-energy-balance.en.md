## Does following the same particles keep energy constant?

Consider an interval following the same gas particles. No particles cross its boundary, so its mass stays fixed. Must its total energy also stay fixed? Expanding gas can do work on its surroundings. Equating zero mass transfer with zero energy transfer would omit that pressure work. We select a smooth compressing or expanding flow in which each particle keeps a constant velocity. This makes it possible to track mass, momentum, internal energy and kinetic energy separately, and to identify exactly which energy channel a simplified flux leaves out.

## Prerequisites and model scope

Review moving-control-volume fluxes, Newton’s law, kinetic energy, pressure and work. Assume a one-dimensional ideal gas with constant cross-section A. Viscosity, heat conduction, body forces, reactions and shocks are excluded. Gamma is a constant heat-capacity ratio greater than one and supplies the relation between pressure and internal energy density. A later thermodynamics treatment develops its assumptions further. Here the aim is to understand three conservation budgets in a specified smooth family. This example is not a numerical solver for arbitrary Euler initial and boundary data.

$$
\mathcal I=\rho e=\frac{p}{\gamma-1},\qquad \mathcal K=\frac12\rho u^2,\qquad \mathcal E=\mathcal I+\mathcal K,\qquad \gamma>1.
$$

## Three stored quantities and their units

The conservation vector contains mass density, momentum density and total energy density. Momentum density carries the sign of velocity, whereas kinetic energy depends on velocity squared. Integrating each density over physical volume produces an amount stored in the selected interval. Pressure, energy per unit mass and energy per unit volume are related but different quantities. A learned model predicting one should not be checked against another without the required density or volume conversion. We retain separate units for the three budgets instead of combining their raw residuals.

$$
U=(\rho,\rho u,\mathcal E),\qquad [U]=(\mathrm{kg/m^3},\mathrm{kg/(m^2s)},\mathrm{J/m^3}).
$$

## Pressure contributes force and work

Mass flux is rho u. Momentum flux includes both the transported momentum rho u² and pressure p. Total-energy flux is not simply stored energy density multiplied by velocity: it also contains pressure-work rate p u. Energy storage is the sum of internal and kinetic energy, while the energy flux includes an enthalpy contribution. Confusing these definitions can produce a model that looks reasonable locally yet loses energy in a finite control-volume budget. Pressure work is mechanical transfer and should not be relabelled as conductive heat.

$$
F=(\rho u,\rho u^2+p,(\mathcal E+p)u),\qquad \partial_tU+\partial_xF=0.
$$

## Moving boundaries use F−wU

At a boundary moving with velocity w, relative flux is F−wU. The mass component becomes rho(u−w), momentum becomes rho u(u−w)+p, and energy becomes E(u−w)+p u. The final term must not be changed to p(u−w). Even at a material boundary with w=u, pressure work p u remains. Apply the left and right outward-normal signs after forming these relative fluxes. The difference of oriented boundary transport determines the rate of change inside the moving interval.

$$
F-wU=\bigl(\rho(u-w),\ \rho u(u-w)+p,\ \mathcal E(u-w)+pu\bigr),\qquad \frac{d}{dt}\int_{x_L}^{x_R}AU\,dx=A[(F-wU)_L-(F-wU)_R].
$$

## A smooth affine compression/expansion

Start with velocity linear in initial position. While the stretch s=1+at remains positive, density scales as 1/s and pressure as 1/s to the power gamma. Positive a produces expansion, negative a compression, and v0 adds a drift. The implementation restricts a and time so that s stays safely positive. It does not replace a collapsed s=0 configuration or vacuum with a small positive number. Such states require a different mathematical treatment. The selected pressure remains uniform in space but can change in time.

$$
s(t)=1+at>0,\qquad \rho=\frac{\rho_0}{s},\qquad u=\frac{v_0+ax}{s},\qquad p=p_0s^{-\gamma}.
$$

## Why is each particle velocity constant?

A particle initially at X follows x=sX+v0t. Differentiation gives v0+aX, independent of time. Substituting its current position into the velocity field gives the same result. Since pressure has no spatial gradient, material acceleration vanishes in this family. Velocity observed at a fixed spatial position can nevertheless change with time. This distinction between a fixed-position derivative and a derivative following a particle is central to continuum mechanics. Independently integrating particle trajectories provides a useful check on the proposed kinematics.

$$
x(X,t)=sX+v_0t,\qquad \frac{Dx}{Dt}=v_0+aX,\qquad \partial_tu+u\partial_xu=0,\qquad \partial_xp=0.
$$

## Check continuity and momentum

The velocity gradient H=a/s is spatially constant. Density changes at rate −H rho, balanced by mass-flux divergence. Momentum-density time change carries a factor −2H and cancels the spatial derivative of rho u²+p. Modifying density in time while keeping unrelated velocity and pressure fields fixed would define a different family. Check the derivatives of the actual fields together. The displayed analytic cancellations explain the construction; they are supplemented by independent finite differences rather than treated as independent numerical evidence.

$$
H=\frac as,\qquad \partial_t\rho=-H\rho,\qquad \partial_x(\rho u)=H\rho,\qquad \partial_t(\rho u)=-2H\rho u,\qquad \partial_x(\rho u^2+p)=2H\rho u.
$$

## Local changes of internal and kinetic energy

At a fixed position, internal energy density changes at −gamma H I and kinetic energy density at −3H K. Their sum is the total-energy time derivative. Spatial divergence of the energy flux compensates it. Constant velocity along each particle trajectory therefore does not imply constant kinetic energy density inside a fixed spatial volume. The mass and particles occupying that volume change. Always specify whether an energy statement concerns a density at a location, an amount inside a control volume, or a fixed material set.

$$
\partial_t\mathcal I=-\gamma H\mathcal I,\quad \partial_t\mathcal K=-3H\mathcal K,\quad \partial_t\mathcal E=-H(\gamma\mathcal I+3\mathcal K),\quad \partial_x[(\mathcal E+p)u]=H(\gamma\mathcal I+3\mathcal K).
$$

## Fixed, translating and material intervals

Choose an initial centre c and half-width h, then compare three observations. A fixed interval stays at the same coordinates, a translating interval moves at a selected constant observation speed, and a material interval follows the same particles. In affine flow the material interval changes width, so it cannot generally be replaced by a translating box. A control inactive in the selected mode does not change its physical boundary. Inspect the actual endpoint coordinates and both boundary velocities to identify which observation is being performed.

$$
x_{L,R}^{\rm fixed}=c\mp h,\qquad x_{L,R}^{\rm translating}=c+wt\mp h,\qquad x_{L,R}^{\rm material}=s(c\mp h)+v_0t.
$$

## Mass, momentum and energy of a material interval

Material-interval mass equals initial length times area times initial density and stays fixed. Equal pressures on the two sides give zero net momentum change. Each particle keeps its velocity, so kinetic energy of this material set is constant too. Internal energy nevertheless decreases during expansion. Forces can sum to zero while their powers do not, because the boundaries have different velocities. This example separates momentum balance from energy balance without requiring viscous dissipation, heat transfer or acceleration of individual particles.

$$
M_{\rm mat}=2Ah\rho_0,\quad \dot P_{\rm mat}=0,\quad \dot K_{\rm mat}=0,\quad I_{\rm mat}=\frac{2Ahp_0}{\gamma-1}s^{1-\gamma},\quad \dot E_{\rm mat}=-2Aahp.
$$

## Worked example: losing 2/3J without heat transfer

Choose A=1m², rho0=1kg/m³, p0=1Pa, a=0.5/s, v0=0 and gamma=2. The initial material interval is [−1,1]m. At t=0 its mass is2kg, internal energy2J and kinetic energy1/12J. At t=1s the stretch is1.5, boundaries are at±1.5m and pressure is4/9Pa. Internal energy has fallen to4/3J while kinetic energy is unchanged. Total energy therefore decreases by2/3J. The missing energy is pressure work delivered through the moving boundaries; it is not conductive heat or unexplained numerical loss.

$$
E(0)=2+\frac1{12}=\frac{25}{12}\ \mathrm J,\qquad E(1)=\frac43+\frac1{12}=\frac{17}{12}\ \mathrm J,\qquad \Delta E=-\frac23\ \mathrm J.
$$

## Use omitted pressure work as a negative control

Keep the actual fields fixed and remove only p u from the energy flux. At a material boundary its advective contribution also vanishes, so the incorrect model predicts no energy exchange. Actual storage still changes at −2Aahp. Do not modify the source or fields to conceal this discrepancy. A local test leaves residual −pH as well. Initially in the worked example, correct net energy inflow is−1W and the omitted-work prediction is0W. This controlled comparison isolates a missing term instead of comparing two unrelated physical systems.

$$
F_{E,\rm wrong}=\mathcal Eu,\qquad \partial_t\mathcal E+\partial_xF_{E,\rm wrong}=-pH,\qquad \dot E_{\rm mat}(0)=-1\ \mathrm W.
$$

## Exact zeros, tiny changes and validity limits

At a=0 the family becomes uniform drift or static pressure. At p0=0 it has positive density and zero pressure, which is not vacuum. A zero-width interval is distinguished from a regular face observation. For tiny a, endpoint energies can round to equality while a small internal-energy change remains. Log1p and expm1 retain that relative change when representable. Unavailable arithmetic quantities are distinguished from exact zeros. None of these residuals certifies arbitrary geometries, shocks or a general initial/boundary-value solution.

$$
\Delta I_{\rm mat}=I_{\rm mat}(t_0)\operatorname{expm1}\!\left((1-\gamma)\operatorname{log1p}\!\frac{a(t_1-t_0)}{1+at_0}\right).
$$

Using log1p and expm1 alone is insufficient if their dimensionless argument has already rounded to zero. The calculation factors out that argument, evaluates log1p(z)/z and expm1(z)/z with their continuous limits of one, and combines the remaining factors with the energy scale before projection. For example, with initial internal energy 2 J, gamma=1.5 and expansion rate equal to the smallest positive binary64 number per second, one second produces a negative energy change that still rounds to a nonzero value. The endpoints themselves both round to 2 J. This is an arithmetic distinction within the declared model, rather than a claim about experimental precision.

## How to test the three budgets of an AI model

First use material expansion to verify constant mass, momentum and kinetic energy alongside falling internal energy. Reverse to compression, then compare relative transport for fixed and translating intervals. Next inspect the omitted-pressure-work control, exact a=0 and a tiny nonzero a. A learned model needs a thermodynamic relation and initial, boundary and observation information in addition to conservation equations. Report units, scales and tolerances separately for each budget. Distinguish identities assembled from shared formulas from independent quadrature, finite-difference and particle-ODE observations.

## Exercises with answers

### Exercise 1

At a material boundary w=u, do all three relative fluxes vanish?

**Answer:** Mass flux vanishes, but momentum flux retains p and energy flux retains p u.

### Exercise 2

Find initial left/right outward momentum fluxes and net force in the worked example.

**Answer:** The outward values are−1N and+1N, giving zero net momentum change. Their powers need not cancel because boundary velocities differ.

### Exercise 3

Explain the total-energy change at t=1s and its cause.

**Answer:** Energy changes from25/12J to17/12J, a change of−2/3J. The cause is pressure work at expanding boundaries, not mass loss or conduction.

### Exercise 4

What happens to the sign of material internal-energy change when a becomes negative?

**Answer:** While s stays positive, compression increases internal energy. Particle velocities remain constant, so material kinetic energy does not change.

### Exercise 5

If a=0, v0=0 and p0>0, is all energy zero?

**Answer:** No. Static pressure and internal energy remain. Energy flux and time change vanish.

### Exercise 6

Do three zero conservation residuals uniquely identify a learned solution?

**Answer:** No. A thermodynamic relation and initial, boundary and observation data are needed. A check within a prescribed family does not certify a general solution.

## Sources and next connections

For general background on moving boundaries and material derivatives, consult the [MIT transport-theorem notes](https://ocw.mit.edu/courses/1-63-advanced-fluid-dynamics-of-the-environment-fall-2002/4bbeb3c0de9503422b6049d2f9760af9_13transthm.pdf). For stress, work and integral balances, consult the [Stanford conservation-equation notes](https://web.stanford.edu/~cantwell/AA200_Course_Material/AA200_Course_Notes/AA200_Ch_06_The_Conservation_equations_Cantwell.pdf). The selected solution families and worked observations are this textbook’s examples. Subsequent units extend to continuum assumptions, stress and strain, general Euler/Navier–Stokes equations, constitutive laws and conservative discretization.
