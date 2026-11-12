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

## Research extension: conservative jumps and entropy admissibility

### Declare a new model rather than inserting a shock into the smooth lab

The original affine experiment excludes shocks. To ask whether three conservation budgets are enough for research validation, introduce a separate planar ideal-gas discontinuity with positive density and pressure. The gas is calorically perfect: gamma and the positive specific heat cv are constant. The displayed dimensionless entropy uses the same positive pressure and density references on both sides. Its additive reference will cancel in an entropy difference. This is a jump model, not a resolved viscous shock profile or an additional native flow control.

$$
e=\frac{p}{(\gamma-1)\rho},\quad h=e+\frac p\rho,\quad \gamma>1,\quad c_v>0,\qquad \sigma=\frac{s_{\rm specific}-s_{\rm ref}}{c_v}=\log\frac p{p_{\rm ref}}-\gamma\log\frac\rho{\rho_{\rm ref}}.
$$

### Derive the jump condition from a shrinking interval

Let the discontinuity move at speed D, with constant state 1 on its left and state 2 on its right. Write square brackets for right minus left. Integrate each conservation equation over an interval around the interface and shrink its width. Boundary transport leaves the flux jump and motion of the stored jump contributes minus D times the state jump. Their cancellation is the Rankine–Hugoniot condition. It compares mass, momentum and energy separately, with their separate units; their raw residuals should not be added as one dimensional score.

$$
U=(\rho,\rho u,\mathcal E),\quad\mathcal E=\frac p{\gamma-1}+\frac12\rho u^2,\quad F=(\rho u,\rho u^2+p,(\mathcal E+p)u),\qquad [F]-D[U]=0.
$$

### Use the shock frame to expose three invariants

Set v equal to the laboratory velocity minus D. Assume flow enters from the left, so the common relative mass flux j is positive. The mass jump gives the first equality; the momentum jump then gives constant p+jv. Combine the energy jump with those two relations, or consistently transform energy into the shock frame, to obtain constant stagnation enthalpy h+v squared over two. A moving boundary removes advected storage, but pressure work remains. Replacing the laboratory energy flux by an unrelated shock-frame expression without transforming its state would mix frames.

$$
v_j=u_j-D,\quad j=\rho_1v_1=\rho_2v_2>0,\qquad p_1+jv_1=p_2+jv_2,\qquad h_1+\tfrac12v_1^2=h_2+\tfrac12v_2^2.
$$

### Select the compressive branch with supersonic upstream flow

Insert the ideal-gas relation into the three invariants. Besides the identical-state solution, eliminating downstream velocity and pressure yields the ratios below. Here upstream Mach number is measured relative to the shock. For M1 greater than one the density and pressure rise, velocity falls, and downstream relative Mach number is below one. At M1=1 this branch reaches the common-state limit, rather than a finite shock. Positive pressure and a nonzero sound-speed reference are required; vacuum and zero-temperature limits are outside this calculation.

$$
M_1=\frac{v_1}{\sqrt{\gamma p_1/\rho_1}}>1,\quad r=\frac{\rho_2}{\rho_1}=\frac{(\gamma+1)M_1^2}{(\gamma-1)M_1^2+2},\quad P=\frac{p_2}{p_1}=\frac{2\gamma M_1^2-(\gamma-1)}{\gamma+1},\quad v_2=v_1/r,\quad M_2^2=\frac{(\gamma-1)M_1^2+2}{2\gamma M_1^2-(\gamma-1)}<1.
$$

### Conservation does not include the entropy direction

For a smooth adiabatic ideal gas, specific entropy is transported with the flow. An admissible irreversible shock instead has nonnegative entropy production. The piecewise constant jump gives the distributional coefficient below for entropy divided by cv. The corresponding physical entropy production is cv times that expression. Because j is positive, the entropy must increase from left to right. This is an inequality with a direction, whereas the three conservative jump equations are equalities that by themselves do not select that direction.

$$
\Delta\sigma=\log P-\gamma\log r,\qquad \partial_t(\rho\sigma)+\partial_x(\rho u\sigma)=j\Delta\sigma\,\delta(x-Dt)\ge0.
$$

### Prove the sign instead of relying on a plotted example

Set z to upstream Mach number squared and substitute the pressure and density ratios into the entropy difference. Differentiation yields the positive expression below. All denominator factors are positive for gamma greater than one and z greater than one, and the difference starts from zero at z=1. Its integral is therefore strictly positive for every finite compressive shock in this family. Near Mach one the small increase can be numerically hard to resolve; a rounded zero is not a new exact isentropic shock. The symbolic sign proof and a floating-point evaluation play different roles.

$$
z=M_1^2:\quad \Delta\sigma(1)=0,\qquad \frac{d\Delta\sigma}{dz}=\frac{2\gamma(\gamma-1)(z-1)^2}{z[(\gamma-1)z+2][2\gamma z-(\gamma-1)]}>0\quad(z>1).
$$

### Swap the same states as a negative control

Keep D fixed and reverse which state lies on the left and right. Both flux differences and state differences change sign, so all three conservation jumps still vanish. The same positive relative mass flux now travels from the denser slower state to the lighter faster state, and the entropy production becomes negative. This expansion shock is excluded by the entropy condition. A physical smooth expansion fan is a different space-time solution; one must not relabel the reversed discontinuity as that fan merely because its endpoints look plausible.

$$
(U_L,U_R)=(U_2,U_1):\quad[F]-D[U]=0,\qquad j(\sigma_R-\sigma_L)=-j\Delta\sigma<0.
$$

### Check Galilean transformations using all stored quantities

A common coordinate boost adds b to the fluid and shock velocities. Pressure and density stay fixed, while momentum and total energy change as displayed. Transform the coordinate trajectory as well. Relative speeds, Mach numbers, ratios and entropy production remain the same. The laboratory jump conditions still hold because the transformed energy equation is a linear combination of the old mass, momentum and energy equations. Altering velocity while leaving momentum or energy unchanged is a different inconsistent request, not a symmetry check.

$$
u'=u+b,\quad D'=D+b,\quad \rho'=\rho,\quad p'=p,\quad (\rho u)'=\rho u+b\rho,\quad\mathcal E'=\mathcal E+b\rho u+\tfrac12b^2\rho,\qquad u'-D'=u-D.
$$

### Work one jump with exact rational arithmetic

Use density reference 1 kg/m³ and velocity reference 1 m/s, hence pressure reference 1 Pa and sound speed one in the upstream state. The numbers below are dimensionless relative to those declared scales; flux components use the corresponding mass, force-per-area and energy-per-area-time scales. With gamma=7/5 and M1=2, the three common shock-frame fluxes are 2, 33/7 and 9. Both sides have stagnation enthalpy 9/2 in the velocity-squared reference. The table also includes the identical-state Mach-one limit, plus three compressive examples. Python computes jump fluxes with exact fractions before evaluating logarithms for entropy.

$$
\gamma=\tfrac75,\quad M_1=2,\quad (\rho_1,v_1,p_1)=(1,2,\tfrac57),\quad(\rho_2,v_2,p_2)=(\tfrac83,\tfrac34,\tfrac{45}{14}),\quad D=0:\quad F_1=F_2=(2,\tfrac{33}7,9),\quad h_j+\tfrac12v_j^2=\tfrac92.
$$

### Keep energy conservation and irreversibility together

The slower downstream flow has higher specific enthalpy, while total transported enthalpy plus kinetic energy remains constant. An entropy increase does not mean total-energy flux disappeared. Conversely, using only advected energy E v drops pressure work and leaves the displayed nonzero jump in general. A real shock has dissipative microscopic structure; these inviscid jump states encode its net balances and admissibility without resolving that structure. This additional verification example retains the original affine lab, its no-shock assumptions and its independent pressure-work negative control.

$$
h_2-h_1=\tfrac12(v_1^2-v_2^2)>0,\qquad j(h_1+\tfrac12v_1^2)=j(h_2+\tfrac12v_2^2),\qquad (\mathcal E_2v_2-\mathcal E_1v_1)=-[pv]\quad(D=0).
$$

| Mach | density ratio | pressure ratio | entropy change / cv | energy flux jump |
| --- | --- | --- | --- | --- |
| 1.00000000 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| 1.25000000 | 1.42857143 | 1.65625000 | 0.00521109 | 0.00000000 |
| 2.00000000 | 2.66666667 | 4.50000000 | 0.13091644 | 0.00000000 |
| 3.00000000 | 3.85714286 | 10.33333333 | 0.44547751 | 0.00000000 |

```python
from fractions import Fraction as F
from math import log
# Dimensionless: rho_ref=1 kg/m^3, speed_ref=1 m/s, p_ref=1 Pa.
gamma = F(7,5)
def energy_flux(rho,u,p):
    energy = p/(gamma-1)+rho*u*u/2
    return (energy+p)*u
for mach in [F(1),F(5,4),F(2),F(3)]:
    density_ratio = (gamma+1)*mach**2/((gamma-1)*mach**2+2)
    pressure_ratio = (2*gamma*mach**2-(gamma-1))/(gamma+1)
    rho1,u1,p1 = F(1),mach,1/gamma
    rho2,u2,p2 = density_ratio,mach/density_ratio,p1*pressure_ratio
    jump = energy_flux(rho2,u2,p2)-energy_flux(rho1,u1,p1)
    assert jump==0
    entropy = log(float(pressure_ratio))-float(gamma)*log(float(density_ratio))
    values = [float(mach),float(density_ratio),float(pressure_ratio),entropy,float(jump)]
    print(" ".join(f"{x:.8f}" for x in values))
```

[NASA Glenn normal-shock relations](https://www.grc.nasa.gov/www/k-12/airplane/normal.html) give the ideal-gas ratio and irreversibility background. The [Clawpack Euler Riemann notes](https://www.clawpack.org/riemann_book/html/Euler.html) explain the conservation variables and wave families. The sign proof, reversed-state negative control and exact boosted jump checks here follow from the displayed assumptions; neither source certifies this supplementary implementation.

![Conservative shock states, reversed entropy sign and conserved energy transport](/assets/learning/continuum-shock-entropy-selection.en.svg)

### Research questions with explanations

**Do three exact jump conservation equations rule out an expansion shock?** **Answer:** No. Swapping the two states preserves those equations but reverses the entropy-production sign. An admissibility condition is additional information.

**Does positive entropy production imply loss of total-energy flux?** **Answer:** No. Stagnation enthalpy and total-energy transport remain conserved while kinetic and internal contributions change.

**Can a boost change velocity without changing the stored energy variable?** **Answer:** No. Total energy includes kinetic energy. Transform momentum, energy and shock trajectory together before checking invariance.

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
