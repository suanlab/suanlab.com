## Why does mass change in a moving photograph?

Imagine photographing a short stretch of a river. A stationary camera repeatedly sees different water enter and leave. A drone following the current can keep the same water in view. A change in the amount of water inside the photograph is therefore not automatically a source or sink. Before interpreting conservation, specify what moves with the observation. This unit uses a smooth density pattern and an independently moving observation interval to turn that distinction into a calculation. The pattern is prescribed so that we can inspect the measurement geometry without also solving a general flow problem.

## Prerequisites and physical units

Review derivatives, definite integrals, velocity, density, outward-normal signs and the continuity equation. Flow is along x and the cross-section A is constant. Time is in seconds; position and half-width are in metres. Density times volume is mass, whereas density times velocity times area is a mass rate. A graph of stored mass and a graph of boundary transport consequently have different units. Do not subtract their raw vertical coordinates. Later we compare a mass change with the time integral of a mass rate, which does have matching dimensions.

$$
[\rho]=\mathrm{kg/m^3},\quad [A]=\mathrm{m^2},\quad [M]=\mathrm{kg},\quad [A\rho u]=\mathrm{kg/s}.
$$

## Material sets and control volumes

A material set follows the particles selected initially. A control volume is a spatial window chosen by the observer and may move differently from the fluid. Write v for fluid velocity and w for window velocity. Particles cross the window at relative velocity v−w. Only the special translating window with w=v follows the same material set in this example. A moving display is not sufficient to establish that an interval is material. Identify the trajectories of its boundaries and compare them with the trajectories of the fluid.

$$
\dot x_{\mathrm{particle}}=v,\qquad \dot x_{\mathrm{window}}=w,\qquad u_{\mathrm{relative}}=v-w.
$$

## The transport theorem counts moving boundaries

Differentiating an integral over a moving region has two contributions: the field changes at positions already included, and the moving boundary includes or excludes additional positions. The transport theorem expresses that geometric relationship. Here G is an amount per unit volume and the boundary velocity is the vector w. The theorem itself is not a conservation law. A physical statement about transport and sources is also required. Keeping these roles separate prevents a moving-domain formula from being mistaken for evidence that the chosen field satisfies the relevant physics.

$$
\frac{d}{dt}\int_{V(t)}G\,dV=\int_{V(t)}\partial_tG\,dV+\int_{\partial V(t)}G\,\mathbf w\cdot\mathbf n\,dS.
$$

## Left and right signs

Let xL and xR denote the left and right boundaries. The left outward normal points toward −x and the right normal toward +x. Positive-x transport entering through the left face increases stored mass; transport leaving through the right decreases it. Subtract boundary motion before applying those signs. The same principle applies to independently moving boundaries, although both boundaries share one velocity w in this density experiment. An outward flow is a signed quantity, so a negative outward flow is a legitimate inflow rather than an invalid observation.

$$
\dot M=A[\rho(x_L,t)(v-w)-\rho(x_R,t)(v-w)],\quad I_L^{\rm out}=-A\rho_L(v-w),\quad I_R^{\rm out}=A\rho_R(v-w).
$$

## A traveling positive-density family

Add a cosine contrast alpha to a positive reference density rho0. The bound |alpha|≤0.8 keeps density positive everywhere. For positive k, rho0 is the spatial mean over a period. At k=0 the cosine equals one, so the actual uniform density is rho0(1+alpha). The reference parameter must not be confused with every observed average. The infinite line contains positive density throughout and therefore has no finite whole-line mass. Our observable is the mass inside a finite interval. No artificial renormalization is performed inside the displayed window.

$$
\rho(x,t)=\rho_0[1+\alpha\cos(k(x-vt))],\qquad \rho_0>0,\quad |\alpha|\le0.8.
$$

## Check local continuity directly

The spatial derivative of the cosine is a negative sine; differentiating its time argument contributes an additional factor −v. Density change and mass-flux divergence therefore cancel. Deriving this identity by hand explains why this entire prescribed family satisfies continuity, rather than just one selected point. A zero residual assembled from shared analytic expressions is not an independent check. We separately compare finite differences of actual field observations and integrals over intervals. Neither a single point nor one successful integral uniquely identifies a density field.

$$
\partial_x\rho=-\rho_0\alpha k\sin(k(x-vt)),\qquad \partial_t\rho=-v\partial_x\rho,\qquad \partial_t\rho+\partial_x(\rho v)=0.
$$

## Window mass and sinc

The window centre is c+wt and its half-width h is constant. Integrating the cosine separates the centre phase from the half-width. Using sinc avoids dividing a small difference of sines by k and gives a well-defined exact k=0 limit. A wide window can average away spatial contrast even while the local density varies strongly. This is a loss of information in the observation, not disappearance of matter. The analytic average can also be evaluated as a limiting quantity when h=0, but that limit is distinct from an observed mass divided by zero volume.

$$
M(t)=2Ah\rho_0[1+\alpha\cos\phi(t)\operatorname{sinc}(kh)],\qquad \phi(t)=k[c+(w-v)t],\qquad \operatorname{sinc}(z)=\begin{cases}\sin z/z&z\ne0,\\1&z=0.\end{cases}
$$

## Cancel the common background before subtracting flows

Both boundary densities contain a common background. Subtracting their rounded values can erase a small contrast. A cosine-difference identity cancels the common background before the net inflow is evaluated. Compare this family-based value with ordinary face subtraction, but do not claim that it reconstructs information lost by real sensors. These are different arithmetic evaluations of the same specified inputs. Their agreement is useful, while disagreement calls for examining representation and cancellation rather than silently replacing one physical experiment with another.

$$
\dot M=2A\rho_0\alpha(v-w)\sin\phi(t)\sin(kh).
$$

## Change the observation frame consistently

Adding the same boost b to fluid and window velocities leaves relative velocity unchanged. Position must be transformed consistently to describe the same observation. Stored mass and relative net inflow then agree, while the laboratory-frame mass flux changes. Altering only v and holding the window fixed instead defines a different relative motion. This provides a useful transformation test for a learned model: transform the fields, coordinates and observation geometry together before deciding whether its outputs should remain invariant.

$$
x'=x+bt,\qquad v'=v+b,\qquad w'=w+b,\qquad v'-w'=v-w.
$$

## Worked example: 2kg stored and 1kg/s entering

Choose A=1m², rho0=1kg/m³, alpha=1/2, k=pi/2rad/m and v=1m/s. Set c=1m, h=1m and w=0, then observe t=0. The interval is [0,2]m, with left density 1.5kg/m³ and right density 0.5kg/m³. The cosine contribution averages to zero over this interval, giving mass 2kg. Nevertheless the instantaneous net inflow is 1kg/s. Stored amount and rate of change answer different questions. If the window instead follows the fluid at w=1m/s, its initial mass is still 2kg but its relative net inflow is zero.

$$
\rho_L=1.5,\quad \rho_R=0.5\ \mathrm{kg/m^3},\qquad M(0)=2\ \mathrm{kg},\qquad \dot M(0)=1\ \mathrm{kg/s}.
$$

## Finite-time change versus instantaneous slope

Compare endpoint mass change with the time integral of net inflow. Multiplying the last instantaneous slope by elapsed time generally gives a different quantity. In the worked example, mass increases by approximately 0.450158kg over 0 to 0.5s. Multiplying the initial slope of 1kg/s by 0.5s gives 0.5kg, also a different answer. Read a point on a rate curve separately from the area under it. Reversing the integration limits reverses the signed change; an exactly zero duration gives an exactly zero change.

$$
\Delta M=\int_{t_0}^{t_1}\dot M\,dt,\qquad M(t)=2+\frac{2}{\pi}\sin\!\left(\frac{\pi t}{2}\right),\qquad M(0.5)-M(0)=\frac{\sqrt2}{\pi}\ \mathrm{kg}.
$$

## Tiny changes and exact zeros

Two endpoint masses can round to the same floating-point number while the prescribed family retains a small nonzero change. Expressing the endpoint cosine difference as a product of sines preserves such changes when representable. Exact zero duration, contrast, wavenumber or material relative velocity instead gives a structural zero. An unrepresentable nonzero quantity remains unavailable. Quadrature-order differences and balance residuals are numerical observations for selected calculations, not rigorous global error bounds. A stable family calculation also cannot recover distinctions absent from the recorded input parameters.

$$
\Delta M=-4Ah\rho_0\alpha\operatorname{sinc}(kh)\sin\!\left(\frac{\phi(t_0)+\phi(t_1)}2\right)\sin\!\left(\frac{k(w-v)(t_1-t_0)}2\right).
$$

For an extremely short interval, computing half the time first can round a nonzero midpoint to zero. The implementation retains the small factors in the sine identity until they can be combined with the mass scale. For example, a stationary window centred at zero has a nonzero second-order mass change over a positive tiny duration, even though the initial slope is zero. A symmetric interval from minus that duration to plus it instead has exactly zero change. An unavailable nonzero result must not be used as evidence of conservation or stationarity.

## Zero length and coordinate resolution

At h=0 both mass and volume vanish, so an observed mean cannot be defined by their ratio. A positive h smaller than the coordinate resolution can also make the two floating-point endpoints appear identical. Record that condition instead of widening the interval or replacing it with exact zero width. The half-width parameter and rounded endpoint coordinates can retain different information. The analytic limiting average may remain meaningful, but it must be labelled separately from an actual normalized measurement on a nondegenerate interval.

$$
\overline\rho_{\rm observed}=\frac{M}{2Ah}\quad(h>0),\qquad h=0:\quad M=0,\quad \overline\rho_{\rm observed}\text{ undefined}.
$$

## AI verification and experiment sequence

Begin with the worked example to establish units and signs, then compare fixed and material windows. Next move the window across a stationary density pattern to expose what an incorrect laboratory-frame flux misses. Finally compare small k, exact zero width and a positive width below coordinate resolution. A vanishing continuity residual does not identify the initial density or unobserved regions. Specify initial/boundary information, observations and tolerances alongside the equation. Report pointwise checks and interval balances separately when evaluating learned fields or conservation-constrained models.

## Research extension: moving-window balances across a discontinuity

### Replace the smooth pattern by a declared contact

A river photograph can contain a sharp boundary between two densities. Keep a constant physical transport velocity v and cross-section A, but replace the cosine by the positive step below. H is zero on negative arguments and one on positive arguments; its value at zero will not determine any volume integral. Both boundaries may move independently, provided the interval remains ordered throughout the observation. This is a supplementary mass-transport model. With a common constant pressure and velocity it can also be embedded as an ideal-gas contact discontinuity, rather than a compressive shock.

$$
\rho(x,t)=\rho_L+[\rho]H(x-vt),\quad [\rho]=\rho_R-\rho_L,\quad \rho_L,\rho_R>0,\quad A>0,\quad L(t)=L_0+w_Lt<R(t)=R_0+w_Rt.
$$

### Conservation need not mean a classical derivative everywhere

The density has no classical spatial derivative at its jump. In a distributional derivative, differentiating the moving step gives a delta term in time change and an equal opposite delta term in flux divergence. The cancellation holds across the jump as well as away from it. A delta records a finite integrated contribution concentrated on a moving interface; it is not a very large point sample to assign to one grid cell without a declared discretization.

$$
\partial_t\rho=-v[\rho]\delta(x-vt),\qquad\partial_x(v\rho)=v[\rho]\delta(x-vt),\qquad\partial_t\rho+\partial_x(v\rho)=0\quad\text{as distributions}.
$$

### Integrate the stored mass first

The primitive of a step is its positive part. Integrating the actual density over the moving interval therefore gives the exact expression below. The bracket measures how much of the interval lies to the right of the interface. This formula covers an interface outside, inside or exactly on a boundary without inventing a new density at that one point. Density times area times length has units kilograms. The total mass on the infinite line remains infinite, so it is the finite-window quantity that we compare.

$$
z_+=\max(z,0),\qquad M(t)=A\left\{\rho_L[R(t)-L(t)]+[\rho]\bigl[(R(t)-vt)_+-(L(t)-vt)_+\bigr]\right\}.
$$

### Locate crossings before integrating a rate

A boundary crosses the interface when its position equals vt. Each nonparallel boundary has the crossing time below. Keep only crossings inside the time interval, split there, and use a constant density trace on each open subinterval. If w equals v, a separated boundary never crosses; a coincident boundary stays on the interface. The latter has zero relative mass flux regardless of which trace is assigned at equality. We require positive window width throughout; coincident left and right boundaries are outside this finite-window experiment.

$$
t_L=-\frac{L_0}{w_L-v}\quad(w_L\ne v),\qquad t_R=-\frac{R_0}{w_R-v}\quad(w_R\ne v).
$$

### A corner in the mass curve is allowed

Differentiating positive parts away from crossings recovers the relative face-flux formula. At an isolated crossing the one-sided mass slopes can differ, so a single instantaneous classical slope may not exist. The mass itself stays continuous and piecewise linear. Altering a bounded density value at the isolated crossing time cannot change a time integral. Do not compare a chosen one-point face trace with a derivative that does not exist, or treat that mismatch as a mass source.

$$
\dot M=A\left[\rho(L(t),t)(v-w_L)-\rho(R(t),t)(v-w_R)\right]\quad\text{for almost every }t.
$$

### Use the finite-time balance as the observation

Integrate over each crossing-free subinterval and add the contributions. This proves the finite-time identity, including intervals whose endpoints are crossings. Reversing time limits reverses the signed change; zero duration gives zero change. When both boundaries move with v, stored mass is fixed even if the selected boundary lies on the contact. Uniform density is a separate limit: an independently expanding window can gain mass through its geometry even with no density contrast. The transport theorem and the physical continuity law still play different roles.

$$
M(t_1)-M(t_0)=A\int_{t_0}^{t_1}\left[\rho(L(t),t)(v-w_L)-\rho(R(t),t)(v-w_R)\right]dt.
$$

### A wrong jump speed can evade point samples

Now move only the step at another speed s while retaining the flux v rho. Off the moving jump, the density and flux are constant and every classical residual sample is zero. At the jump the distributional defect below remains when s differs from v and the density contrast is nonzero. A finite set of ordinary collocation points away from the interface can therefore report zero loss for a nonconservative field. This is a negative control with the same declared flux, not a new source that is silently permitted to make the request valid.

$$
\rho_s=\rho_L+[\rho]H(x-st):\qquad r_s=\partial_t\rho_s+\partial_x(v\rho_s)=(v-s)[\rho]\delta(x-st).
$$

### A weak measurement sees the missed source

Pair the defect with a smooth compactly supported test function in space and time. Its value samples the entire interface trajectory through the integral below. Choose a nonnegative test function that is positive on part of that trajectory to detect a nonzero constant-sign defect. A test supported away from the interface will still see zero, so one weak observation does not certify every possible defect. Multiple test scales and regions, initial data and boundary observations must match the research question. A grid approximation to this pairing needs its own quadrature and resolution check.

$$
\langle r_s,\varphi\rangle=(v-s)[\rho]\int\varphi(st,t)\,dt,\qquad \varphi\in C_c^\infty(\mathbb R^2).
$$

### Calculate a crossing without an arbitrary midpoint trace

For the stated SI example, a stationary metre-wide window initially contains half of each density and stores 2 kg. As the interface moves right, the dense portion shrinks. At 0.5 s the interface reaches the right boundary and the mass becomes 1 kg; it stays there afterwards. Before that crossing the slope is minus 2 kg/s, and afterwards it is zero. The table reports stored mass and cumulative relative inflow, avoiding an undefined instantaneous slope at the corner. The code integrates flux on open subintervals with exact fractions before printing.

$$
A=1\,\mathrm{m^2},\ \rho_L=1,\ \rho_R=3\,\mathrm{kg/m^3},\ v=1\,\mathrm{m/s},\ L=-\tfrac12\,\mathrm m,\ R=\tfrac12\,\mathrm m:\quad \frac{M(t)}{1\,\mathrm{kg}}=1+2\left[(\tfrac12-\theta)_+-(-\tfrac12-\theta)_+\right],\quad\theta=t/(1\,\mathrm s).
$$

### Transform the observation together with the field

A common Galilean coordinate boost transforms the interface, both window paths and physical velocity together. Relative velocities and the same physical stored mass remain unchanged. Moving only the interface instead is the wrong-speed negative control, not a change of observer. For AI verification, report finite-window storage, integrated relative inflow, jump geometry and the test-function support together. The existing cosine lab retains its smooth source family and controls; this section does not add a discontinuous solver or a new jump-speed input.

$$
x'=x+bt,\quad v'=v+b,\quad w_L'=w_L+b,\quad w_R'=w_R+b:\qquad v'-w_j'=v-w_j,\quad M'=M.
$$

| t (s) | M (kg) | mass change (kg) | integrated inflow (kg) | residual (kg) |
| --- | --- | --- | --- | --- |
| 0.00000000 | 2.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| 0.25000000 | 1.50000000 | -0.50000000 | -0.50000000 | 0.00000000 |
| 0.50000000 | 1.00000000 | -1.00000000 | -1.00000000 | 0.00000000 |
| 0.75000000 | 1.00000000 | -1.00000000 | -1.00000000 | 0.00000000 |

```python
from fractions import Fraction as F
# SI: A=1 m^2, rho_L=1, rho_R=3 kg/m^3, v=1 m/s.
left, right, v = F(-1,2), F(1,2), F(1)
def mass(t):
    return right-left+2*(max(right-v*t,F(0))-max(left-v*t,F(0)))
def integrated_inflow(end):
    cuts = sorted({F(0), end} | {t for t in [left/v,right/v] if 0<t<end})
    total = F(0)
    for a,b in zip(cuts,cuts[1:]):
        t = (a+b)/2
        rho_left = F(1) if left<v*t else F(3)
        rho_right = F(1) if right<v*t else F(3)
        total += v*(rho_left-rho_right)*(b-a)
    return total
for t in [F(0), F(1,4), F(1,2), F(3,4)]:
    change = mass(t)-mass(F(0))
    inflow = integrated_inflow(t)
    assert change==inflow
    values = [t,mass(t),change,inflow,change-inflow]
    print(" ".join(f"{float(x):.8f}" for x in values))
```

The [Clawpack Euler Riemann notes](https://www.clawpack.org/riemann_book/html/Euler.html) supply contact and weak-conservation context. The positive-part storage, crossing-aware integration and missed-source example are derived here for the declared one-dimensional family.

![Moving contact, continuous window mass and crossing-aware cumulative inflow](/assets/learning/continuum-contact-weak-window.en.svg)

### Research questions with explanations

**Can zero classical residual away from a jump prove mass conservation?** **Answer:** No. A wrong jump speed leaves a nonzero distributional source on the interface even though off-interface residuals vanish.

**Must the mass derivative exist at the exact crossing time?** **Answer:** No. Its one-sided slopes can differ. The continuous stored mass and finite-time integrated balance remain defined.

**Does uniform density make mass in every moving window constant?** **Answer:** No. A changing window width changes volume. Constant material-window mass requires boundaries that follow the transport velocity.

## Exercises with answers

### Exercise 1

For a material window with v=w=1m/s, what are the relative face flows and mass change?

**Answer:** Both relative face flows and the mass change are zero because relative velocity is zero. Laboratory flux rho v can remain nonzero.

### Exercise 2

With uniform density 1kg/m³, A=1m², v=1 and w=−0.3m/s, find both outward flows and net inflow.

**Answer:** Left outward flow is −1.3kg/s, right outward flow +1.3kg/s and net inflow zero. Material can cross both faces while storage stays constant.

### Exercise 3

With rho0=1 and alpha=1/2, is actual density one when k=0?

**Answer:** No. Cosine equals one, giving actual density 1.5kg/m³. Distinguish reference density from the actual mean in this limit.

### Exercise 4

At h=0, may zero mass divided by zero volume be reported as zero mean density?

**Answer:** No. The observed mean is undefined. Label an analytic positive-width limit separately.

### Exercise 5

Find the worked example’s mass change from0 to0.5s and compare it with1kg/s times0.5s.

**Answer:** The exact change is sqrt(2)/pi≈0.450158kg, while the constant-slope estimate is0.5kg. The actual slope varies over time.

### Exercise 6

Two different uniform densities both satisfy continuity. Can a zero residual distinguish them?

**Answer:** No. Initial density or an additional mass observation is needed. Conservation and identification are different questions.

## Sources and next connections

For general background on moving boundaries and material derivatives, consult the [MIT transport-theorem notes](https://ocw.mit.edu/courses/1-63-advanced-fluid-dynamics-of-the-environment-fall-2002/4bbeb3c0de9503422b6049d2f9760af9_13transthm.pdf). For stress, work and integral balances, consult the [Stanford conservation-equation notes](https://web.stanford.edu/~cantwell/AA200_Course_Material/AA200_Course_Notes/AA200_Ch_06_The_Conservation_equations_Cantwell.pdf). The selected solution families and worked observations are this textbook’s examples. Subsequent units extend to continuum assumptions, stress and strain, general Euler/Navier–Stokes equations, constitutive laws and conservative discretization.
