## An ellipse needs a clock

An orbit equation gives possible positions, not the time spent at each position. Moving a dot around an ellipse at uniform drawing angle usually produces the wrong velocity, angular momentum and force. This lesson adds a physical clock to the inverse-square ellipse by solving Kepler's equation.

Picture a satellite passing close to its source. It crosses the near side quickly and spends more time far away. Equal time intervals sweep equal areas about the force center, not equal distances along the ellipse and not equal angles about its geometric center. We will calculate position and velocity together so this statement can be checked rather than merely illustrated.

## Choose a collision-free bound family

We keep the fixed-source inertial model q̈=−μq/r³ with positive μ and mass m. Choose a positive semimajor axis a and eccentricity 0≤e≤0.95. The initial point is periapsis, the closest point to the focus. For e=0, every point is equally close; the initial point then means a chosen phase rather than a distinguished periapsis.

The allowed family has

$$
r_p=a(1-e)>0,\qquad r_a=a(1+e),\qquad b=a\sqrt{1-e^2}.
$$

Consequently these trajectories never reach the singular origin. The upper eccentricity limit keeps all control combinations strictly elliptic. It is a teaching range, not an approximation that sets all larger eccentricities equal to 0.95. This lesson does not solve parabolic flight, hyperbolic flight or radial collision time; those require other parametrizations and domain decisions.

A fixed orientation angle rotates the ellipse in the laboratory. A direction selector changes prograde to retrograde motion. Neither operation changes a, e, the period or the specific energy.

## Distinguish the focus from the ellipse center

With the periapsis direction along positive x, the eccentric anomaly E parametrizes the ellipse as

$$
x=a(\cos E-e),\qquad y=b\sin E.
$$

The force center is (0,0), one focus. The geometric ellipse center is (−ae,0), which generally is not a force center. E is an auxiliary parameter measured through a related circle; it is not the polar angle of q about the focus. Adding the offset −ae is essential. An otherwise attractive-looking centered ellipse would implement a different physical geometry.

A fixed rotation matrix transforms both q and v into the chosen laboratory orientation. Rotating position while leaving velocity unrotated would violate the initial tangency and alter the measured angular momentum.

## Derive the time equation from the area law

Define the positive mean motion n and period T by

$$
n=\sqrt{\frac\mu{a^3}},\qquad T=\frac{2\pi}{n}.
$$

Let s=+1 for prograde and −1 for retrograde motion. Starting at periapsis at t=0, the signed mean anomaly is M=snt. Kepler's equation is

$$
E-e\sin E=M.
$$

One way to understand the sine correction is to compute the focus-centered area swept as the auxiliary angle changes. Differentiating q with respect to E gives q×dq/dE=ab(1−e cos E). Integrating from E=0 yields A(E)=ab(E−e sin E)/2. Requiring constant areal velocity then gives the time relation above.

This also explains why uniform E is incorrect when e>0: the factor 1−e cos E is not constant. Mean anomaly grows uniformly in time, while eccentric anomaly generally does not.

## Solve a monotone implicit equation safely

The derivative of the left side is 1−e cos E≥1−e>0. Therefore the equation has exactly one root for every finite M in this elliptic range. Since |e sin E|≤e, a guaranteed mathematical bracket is [M−e,M+e].

The numerical solver starts within that bracket. It attempts a Newton step using derivative 1−e cos E. If the proposed step leaves the bracket, it bisects instead. Each evaluation updates the appropriate bracket endpoint. The output reports the residual E−e sin E−M and iteration count.

A residual is evidence of solving the implicit equation, not by itself proof of the entire physical implementation. The position formula, velocity derivative, force center and time scaling can still be wrong independently. We therefore check them using high-precision roots and a separate Cartesian differential-equation integration.

The circular case e=0 returns E=M directly. The zero-time root is also exact. The solver retains unwrapped anomalies instead of reducing them to a principal angle. This matters for repeated revolutions and accumulated area.

## Differentiate with respect to actual time

Implicit differentiation gives

$$
\dot E=\frac{s n}{1-e\cos E}.
$$

The compatible velocity is

$$
v_x=-a\sin E\,\dot E,\qquad
v_y=b\cos E\,\dot E.
$$

These are derivatives of the displayed position, not an independently chosen tangent arrow. At periapsis, the denominator is 1−e, so the motion is fast. At apoapsis it is 1+e, so the motion is slow. Retrograde motion reverses the signed evolution of E; its positive-time position is the reflection across the periapsis axis of the corresponding prograde point before applying the laboratory rotation.

The radius identity r=a(1−e cos E) stays strictly positive. Acceleration and force are evaluated about the focus using −μq/r³ and −mμq/r³.

## Check energy, angular momentum and speed

For every time in this family,

$$
\epsilon=-\frac\mu{2a},\qquad
h=s\sqrt{\mu a(1-e^2)},\qquad L=mh.
$$

The instantaneous kinetic and potential energies are K=m|v|²/2 and U=−mμ/r. Their sum must equal mε. The speed also satisfies the vis-viva relation

$$
|v|^2=\mu\left(\frac2r-\frac1a\right).
$$

K and U change along an eccentric orbit, even though total energy is constant. A panel that displays only total energy would conceal that exchange. Conversely a changing kinetic energy alone would not imply dissipation. There is no drag or heat term in this model.

## Area is signed and accumulated

The focus-centered areal velocity and swept area are

$$
\dot A=\frac{q\times v}{2}=\frac h2,\qquad A(t)=\frac{ht}{2}.
$$

The second evaluation A(E)=ab(E−e sin E)/2 provides an independent geometric consistency check once E has been solved. After a full prograde revolution the area increment is πab; a retrograde revolution contributes −πab. After many revolutions it continues accumulating. Reducing E modulo 2π before evaluating this formula would incorrectly erase those full-ellipse contributions.

Here area means the oriented integral of q×dq/2 along the traversed path. Repeated traversal can exceed the area of the geometric ellipse. It is not the area of the union of visited pixels on the screen.

## Read the geometry and time plots separately

The full closed ellipse is reference geometry over one revolution. The actual path stops at the selected query time; at t=0 it contains only the initial point. A query dot identifies the computed state. A full reference ellipse must not imply that every point has already been visited.

Time panels span 0 to 20 seconds. Radius, speed, energy and accumulated area have different physical units and need separate axes or panels. The query is explicitly included in the samples, as are periapsides and apoapsides within the interval. Sampling uniformly in eccentric anomaly and converting each sample to its actual Kepler time places more samples near the short, rapid periapsis passage. A coarse uniform-time grid can miss that feature at high eccentricity.

Sampling is a rendering decision; it does not replace the implicit time solution. The queried state is solved directly, so dragging the query does not merely select a nearby precomputed frame.

## Research extension: a finite periapsis passage can coexist with a diverging orbital period

Hold periapsis radius fixed while an elliptic orbit approaches the parabolic boundary. Its full period grows without limit, but the outward flight time to a fixed nearby radius remains finite. We derive that limit and construct a separate parabolic time solution, without changing the native elliptic control family.

Keep positive mu and rp, start at periapsis at t=0, and choose the outgoing radial branch. The same dimensionless delta gives:

$$
\delta=1-e,\quad r_p=a\delta,\qquad
\epsilon=-\frac{\mu\delta}{2r_p},\quad h^2=\mu r_p(2-\delta),\qquad
T=2\pi\sqrt{\frac{r_p^3}{\mu}}\,\delta^{-3/2}.
$$

The question now concerns the clock. A large semimajor axis and a long return from a distant apoapsis need not imply slow passage through a fixed neighbourhood of periapsis. The native a≤4 m,e≤0.95 range does not contain this entire fixed-rp limiting family.

### Remove the radial turning-point square root

Define u=sqrt(r/rp−1), with u≥0 on the outgoing branch. Substitute the periapsis energy and angular momentum into the radial energy equation:

$$
r=r_p(1+u^2),\qquad
\dot r^2=\frac\mu{r_p}\frac{u^2[2-\delta(2+u^2)]}{(1+u^2)^2},\qquad
\frac{dt}{du}=2\sqrt{\frac{r_p^3}{\mu}}\frac{1+u^2}{\sqrt{2-\delta(2+u^2)}}.
$$

Although radial speed is zero at the initial turning point, the factor u cancels against dr/du in the time integral. The transformed integrand is regular while its remaining denominator stays positive. For an ellipse, u is generally not tan(f/2); conflating those variables would give the wrong clock.

### Integrate the parabolic limit

At delta=0 the conic has e=1 and nonzero angular momentum. Then u becomes tan(f/2), and the time integral is polynomial:

$$
\delta=0:\quad u=\tan(f/2),\qquad
t_B(U)=\sqrt{\frac{2r_p^3}{\mu}}\left(U+\frac{U^3}{3}\right).
$$

This is Barker's time relation with the periapsis time chosen as zero. It is not the h=0 radial collision branch and does not pass through the singular force center.

### Prove a bound over the entire passage interval

Choose a fixed target radius R>rp and its corresponding U. Rewrite the elliptic integrand as the parabolic one times (1−delta(1+u²/2))^(-1/2). On 0≤u≤U this extra factor lies between one and (1−delta A)^(-1/2):

$$
U=\sqrt{R/r_p-1}>0,\quad A=1+U^2/2,\quad 0\le\delta A<1,\qquad
t_B(U)\le t_\delta(U)\le\frac{t_B(U)}{\sqrt{1-\delta A}}.
$$

The strict upper condition places the target before apoapsis. Squeezing gives t_delta→tB at fixed R, with a difference bounded by tB[(1−delta A)^(-1/2)−1]=O(delta). This is a continuous analytic bound, not a conclusion from a finite quadrature grid. Sending R toward the distant apoapsis at the same time is a different limit.

For rp=1 m,mu=2 m³/s²,R=2 m, U=1 and tB=4/3 s. At delta=1/2, a=2,e=1/2,n=1/2. Radius two is reached at eccentric anomaly E=pi/2, giving the independent time pi−1 s.

| delta | passage to 2 m (s, approximate) | full period (s, approximate) |
|---|---:|---:|
| 1/2 | 2.141592654 | 12.566371 |
| 1/10 | 1.421506138 | 140.496295 |
| 1/100 | 1.341407313 | 4442.882938 |
| 1/1000 | 1.334134066 | 140496.294621 |
| 1/1000000 | 1.333334133 | 4442882938.158366 |

![The full period diverges near the parabolic boundary, while time to the fixed two-metre radius stays between its finite Barker limit and an analytic upper bound.](/assets/learning/classical-kepler-passage-limit.en.svg)

The plotted delta range is 0.0001 to 0.5. The top panel is full period; the bottom is fixed-radius passage time. Both use seconds but have separate axes. Orange is the analytic upper bound and the dotted horizontal line is the parabolic limit. Delta zero is not plotted as a point on a logarithmic axis.

### Recover a parabolic position and velocity from time

Write k for the positive Barker time scale, which has units of seconds. The hyperbolic-sine triple-angle identity inverts t/k=u+u³/3:

$$
k=\sqrt{\frac{2r_p^3}{\mu}},\qquad
u(t)=2\sinh\!\left(\frac13\operatorname{asinh}\frac{3t}{2k}\right),\quad
q=r_p(1-u^2,2u),\quad
v=\frac{2r_p}{k(1+u^2)}(-u,1),\quad r=r_p(1+u^2).
$$

These formulas have radius rp(1+u²)>0 and correctly differentiated velocity, rather than an independently drawn tangent arrow. With rp=1 m,mu=2 m³/s², k=1 s. At t=4/3 s they give u=1,q=(0,2) m,v=(−1,1) m/s,epsilon=0 and h=2 m²/s.

In the native inverse-square diagnostic with codes (2,2,0,20,20,1), radius two admits radial velocities −1 and +1 m/s and tangential speed one. The time solution above selects the outgoing branch. It does not add parabolic propagation to the elliptic widget.

### Relate an elliptic root residual to position error

A finite passage limit does not eliminate numerical conditioning. For an elliptic root E_star and an approximation Ehat, hold e,a and M fixed and let rho be the exact mathematical Kepler residual. Monotonicity with derivative at least delta gives the first bound below. The ellipse parametrization has derivative norm at most a, giving the second:

$$
F_e(E)=E-e\sin E,\quad F_e'(E)\ge\delta=1-e>0,\quad
\rho=F_e(\widehat E)-M,\quad F_e(E_*)=M,\qquad
|\widehat E-E_*|\le\frac{|\rho|}{\delta},\quad
\lVert q(\widehat E)-q(E_*)\rVert\le\frac{a|\rho|}{\delta}
=\frac{r_p|\rho|}{\delta^2}.
$$

For the fixed-rp family a=rp/delta, so the conservative position factor grows like delta^(-2). A small residual therefore needs its scale and domain before it supports a position-accuracy claim. A floating-point residual report is not automatically a certified enclosure of the mathematical residual. The native independent position/velocity checks remain necessary; this bound does not assert a defect in its existing solver.

### Reproduce with independent quadrature and explicit time states

The code compares composite Simpson grids of 512 and 1024 intervals for five delta values, checks the analytic passage bracket and the special elliptic time pi−1, and evaluates five parabolic states. The quadrature comparison is a numerical consistency check, not a certified integration-error bound. The six shifted elliptic roots at the end illustrate the analytic residual bound in floating point; they do not turn their computed residuals into rigorous certificates.

```python
from math import sqrt, pi, sinh, asinh, hypot, sin, cos

def passage(delta, steps):
    if not (0 <= delta < 2/3) or steps <= 0 or steps % 2:
        raise ValueError("Use an even grid before the target apoapsis")
    def integrand(u):
        return sqrt(2)*(1+u*u)/sqrt(2-delta*(2+u*u))
    dx = 1/steps
    total = integrand(0)+integrand(1)
    total += sum((4 if j % 2 else 2)*integrand(j*dx) for j in range(1, steps))
    return total*dx/3

tB = 4/3
for delta in [.5, .1, .01, .001, .000001]:
    t1, t2 = passage(delta, 512), passage(delta, 1024)
    upper = tB/sqrt(1-1.5*delta)
    assert abs(t1-t2) < 5e-11
    assert tB-2e-11 <= t2 <= upper+2e-11
    period = 2*pi/sqrt(2)*delta**(-1.5)
    print("delta, passage time, analytic upper, period (s):", delta, t2, upper, period)
assert abs(passage(.5, 1024)-(pi-1)) < 5e-11

def parabolic(t):
    u = 2*sinh(asinh(1.5*t)/3)
    udot = 1/(1+u*u)
    q, v = (1-u*u, 2*u), (-2*u*udot, 2*udot)
    radius = 1+u*u
    assert abs(u+u**3/3-t) < 2e-12*max(1, abs(t))
    assert abs(hypot(*q)-radius) < 2e-12
    assert abs((v[0]**2+v[1]**2)/2-2/radius) < 2e-12
    assert abs(q[0]*v[1]-q[1]*v[0]-2) < 2e-12
    return q, v

for t in [0, .5, 4/3, 2, 10]:
    print("parabolic time, position, velocity:", t, *parabolic(t))
q, v = parabolic(4/3)
assert max(abs(q[0]), abs(q[1]-2), abs(v[0]+1), abs(v[1]-1)) < 2e-12

# Floating illustrations of the analytic residual-to-position bound.
e, a, exact_E = .5, 2, pi/2
M = (pi-1)/2
for shift in [-.1, -.01, -.001, .001, .01, .1]:
    Ehat = exact_E+shift
    rho = Ehat-e*sin(Ehat)-M
    qhat = (a*(cos(Ehat)-e), a*sqrt(1-e*e)*sin(Ehat))
    position_error = hypot(qhat[0]+1, qhat[1]-sqrt(3))
    analytic_bound = a*abs(rho)/(1-e)
    assert position_error <= analytic_bound+2e-14
print("six residual-to-position illustrations passed")
```

The native elliptic family can bracket the special passage with codes (2,4,10,0,0,21,1) and (2,4,10,0,0,22,1): at 2.1 s its radius is below two and at 2.2 s above two. Its period is 4pi s. An arbitrary near-parabolic fixed-rp family or the new parabolic formula is instead reproduced by the independent script.

### Worked questions for orbital time prediction

1. **Does a huge full period imply a huge passage time to two metres?** Not in this fixed-rp family. The distant part of the orbit drives the divergence, while the finite window has the explicit Barker limit.
2. **Why not insert delta=0 into the native ellipse parameters?** At fixed rp it would require unbounded a. A separate time parametrization handles the parabolic branch without pretending it is an allowed elliptic control value.
3. **What evidence should a time-dependent surrogate retain?** State its radial branch, radius window, parameter uncertainties, physical time scale and independent state references. Root, invariant and geometric checks support different claims.

For Barker and Kepler time laws see [MIT 16.346, Lecture 3](https://ocw.mit.edu/courses/16-346-astrodynamics-fall-2008/379eae1a78cf9ad58247058dffbae3ac_lec_03.pdf). The interval bound and worked research examples here are derived directly for the declared fixed-source model.

## Worked examples and exercises with solutions

1. **Circular limit.** Set μ=a=1, e=0, prograde direction and t=1. **Solution:** n=1, E=M=1, q=(cos 1,sin 1), speed=1, radius=1 and T=2π. Area is 1/2. The force center and ellipse center coincide only because e=0.

2. **Initial eccentric state.** Set μ=2, a=2, e=1/2, m=3 and t=0. **Solution:** rp=1, ra=3, b=√3 and the initial speed is √3. Initial acceleration is (−2,0), total energy is −3/2 and angular momentum is 3√3. The ellipse center is (−1,0), while the force acts toward (0,0).

3. **Period scaling.** Double a while holding μ and e fixed. **Solution:** T increases by 2 to the power 3/2. Quadrupling μ at fixed a instead halves T. The invariant relation is T²/a³=4π²/μ. Eccentricity changes the shape and timing within a revolution, but does not enter this period at fixed a and μ.

4. **Equal areas, unequal angles.** Compare a high-eccentricity orbit near periapsis and apoapsis over equal short time intervals. **Solution:** each interval contributes hΔt/2, but the near-side interval traverses a larger angular change and higher speed. Uniform polar-angle animation would violate the area law.

5. **Reverse direction and rotate.** Reverse the direction selector, then rotate the orientation by π/2. **Solution:** direction reversal negates h, L and accumulated area while preserving energy and radius at equal positive time. Rotation maps (x,y) to (−y,x) and transforms velocity in the same way. Neither operation changes the period.

6. **Multiple revolutions.** Use μ=6, a=1/2, e=0.95 and t=20. **Solution:** n=√48, so the orbit completes more than 22 revolutions. E must remain greater than 40π in prograde motion, and accumulated area exceeds many ellipse areas. A principal-angle root would display a plausible current position but the wrong total area and revolution history.

## Research use and verification limits

A learned dynamics model should be checked at the focus, at high eccentricity and across long phase accumulation, not only on a nearly circular picture. Coordinate accuracy, energy conservation, timing accuracy and area accumulation are separate requirements. Small invariant residuals can coexist with a phase error, so independent position and velocity references remain necessary.

The accompanying calculation uses 70-digit bisection roots and independent Cartesian inverse-square integrations as references. Floating-point residuals are reported numerically rather than labeled exact. The physical assumptions remain fixed-source Newtonian attraction and a collision-free elliptic family. For the underlying construction, see the [MIT OpenCourseWare Kepler problem notes](https://ocw.mit.edu/courses/8-282j-introduction-to-astronomy-spring-2006/93b6bc699c653da0efd67170a90c7c81_kepler_prob.pdf).
