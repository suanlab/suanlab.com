## No work does not mean no force

Imagine a force that always pushes sideways relative to the current velocity. It changes the direction of motion without changing speed. Kinetic energy is constant, yet the vector momentum rotates. This is a concrete reason not to treat energy, linear momentum and angular momentum conservation as interchangeable statements.

We select one positive constant point mass in an inertial plane. A prescribed external velocity-dependent force acts; the mechanism producing it is outside the selected subsystem. The model is deliberately limited to its stated motion law. Zero work does not make this particle an isolated system.

## Specify the quarter-turn operator and sign convention

Define J(vx,vy)=(−vy,vx), a counterclockwise ninety-degree rotation. It obeys J²=−I and v·Jv=0. The declared force is

$$
F=m\Omega Jv,\qquad \dot v=\Omega Jv,
$$

where Ω is a signed constant with units s⁻¹. Positive Ω rotates velocity counterclockwise and negative Ω rotates it clockwise. Its magnitude is an angular frequency, not cycles per second.

This is a velocity-dependent force. The same position with different velocities generally has different forces, so it is not a global position-only potential-gradient model. The interface does not introduce an arbitrary scalar potential to explain away the velocity dependence.

## Derive kinetic-energy conservation directly

Power is force dotted with actual velocity:

$$
\frac{dK}{dt}=F\cdot v=m\Omega(Jv)\cdot v=0,
\qquad K=\frac12m|v|^2.
$$

Thus speed and kinetic energy remain their initial values. The accumulated work is zero. This statement concerns scalar power. The momentum law remains dp/dt=F, so nonzero force changes vector momentum even while its magnitude stays fixed.

The numerical query power may contain floating-point roundoff from cancellation. Work is analytically zero by the perpendicular-force identity; it is not defined as the negative of an unexplained energy residual. Independent numerical integration checks the work rate and external impulse.

## Solve the rotating velocity

Let R(θ)=cosθ I+sinθ J. The initial-value solution is

$$
v(t)=R(\Omega t)v_0
=v_0\cos(\Omega t)+Jv_0\sin(\Omega t).
$$

Rotation preserves vector length. The two momentum-component curves are generally sinusoidal and neither needs to remain constant. The zero-velocity initial state stays stationary, even if Ω is nonzero, because its force remains zero.

The exact conservation classification uses the input values: vector momentum is conserved throughout the motion only in the Ω=0 branch or the stationary state. A momentum component can have zero derivative at an individual instant without satisfying that whole-motion condition.

## Integrate position and handle the free branch

For Ω≠0,

$$
q(t)=q_0+\frac{\sin(\Omega t)}{\Omega}v_0
+\frac{1-\cos(\Omega t)}{\Omega}Jv_0.
$$

At Ω=0 the model explicitly uses q=q0+v0t and v=v0. It does not divide by zero or invent a finite circle center. For numerical stability at short times, the nonzero branch evaluates sine ratios with a sinc function and the cosine difference using a half-angle identity. The exact limit is the same free motion.

Query time ranges over ten seconds. The spatial panel uses one full period when Ω is nonzero and a ten-second line when Ω is zero. Its horizon is explicitly distinguished from the time panels' fixed ten seconds.

## Find the guiding center without adding another body

For Ω≠0 define

$$
C=q+\frac{Jv}{\Omega}.
$$

Differentiating gives Cdot=v+J(ΩJv)/Ω=v−v=0. Therefore the trajectory is a circle about the fixed C for nonzero speed, with radius |v0|/|Ω| and period 2π/|Ω|. C is derived from the initial state; it is not a second physical particle or a new applied force.

The center is undefined in this formula at Ω=0. If v0=0 and Ω≠0, C=q0 and the spatial orbit degenerates to a point. A reference marker does not imply a nonzero orbit radius.

## Mechanical angular momentum generally changes

About the fixed laboratory origin, let L=(q×mv)z. Then

$$
\dot L=(q\times F)_z=m\Omega(q\cdot v).
$$

This torque is generally nonzero despite zero work. Work uses a dot product with velocity; torque uses a cross product with a reference-dependent position. There is no contradiction in having one vanish and the other remain nonzero.

A special orbit whose guiding center is the origin has q perpendicular to v, making mechanical L constant. The interface distinguishes that case exactly from a generic displaced circle. Energy conservation therefore permits several combinations of other conserved quantities rather than imposing one universal pattern.

## Verify a modified angular invariant

Since d|q|²/dt=2q·v,

$$
I=L-\frac{m\Omega}{2}|q|^2,
\qquad \dot I=0.
$$

This is a different quantity from mechanical L. It has the same units kg m²/s, but a different definition. We establish its conservation by differentiation rather than invoking an unstated general theorem.

The torque integral also satisfies ΔL=mΩΔ|q|²/2. The numerical angular-balance residual checks that identity independently of treating L itself as conserved. In the free branch the correction vanishes and I reduces to the ordinary conserved angular momentum.

## State what is external and what is not modeled

The selected particle receives an external force and generally an external torque. The apparatus or field producing that force is not included. Preserving the particle's kinetic energy does not establish momentum conservation for the particle alone.

A uniform perpendicular magnetic field can motivate this type of velocity rotation, with the appropriate charge/field sign absorbed into Ω. We do not model field dynamics, radiation, electric acceleration, spin or relativistic corrections. A full apparatus-plus-field conservation budget is a separate problem.

## Read the four diagrams

The spatial panel shows actual motion in meters, the fixed origin and the guiding center when defined. The momentum panel compares Px and Py in kg m/s. The angular panel compares mechanical L and modified I in kg m²/s. The energy panel shows K in joules.

Initial K,L,I and guiding-center coordinates are exact rational calculations from control codes. Trigonometric motion is evaluated numerically. Reverse Ω to change bending direction, move q0 to change the reference-dependent angular history, and use Ω=0 to inspect the free limit. These controls change specified parts of the physical model rather than decorative animation parameters.

## Research extension: exact invariants can hide a numerical phase error

A numerical trajectory can conserve more than kinetic energy and still reach the wrong position at a specified time. We construct an example preserving the guiding center and the modified angular invariant as well. It explains why checking the shape of an orbit is different from checking motion along that orbit.

The native lab evaluates the analytic flow above. The implicit midpoint method below is an independent comparison, not a replacement for that evaluator or a newly added step-size control. Keep the same constant positive mass, fixed lab axes and signed constant Omega. Write the quarter-turn operator as J, with J squared equal to minus the identity.

### Solve the midpoint equations explicitly

For time step h>0 define the midpoint velocity and update both variables consistently:

$$
\bar v_n=\frac{v_n+v_{n+1}}2,\qquad
v_{n+1}-v_n=h\Omega J\bar v_n,\qquad
q_{n+1}-q_n=h\bar v_n.
$$

Moving the new velocity terms to the left gives (I−aJ)v_next=(I+aJ)v with a=h Omega/2. Because (I−aJ)(I+aJ)=(1+a²)I, the linear equation is invertible for every real a. Its velocity map and signed principal rotation angle are:

$$
a=\frac{h\Omega}{2},\quad
R_h=\frac{(1-a^2)I+2aJ}{1+a^2},\quad
v_{n+1}=R_hv_n,\qquad
\theta_h=2\arctan(h\Omega/2).
$$

Here I inside the matrix is the identity operator. The angle lies between −pi and pi and has the sign of Omega. At Omega=0 the velocity map is the identity and position advances by h v: exactly the free branch, without defining a finite guiding center.

### Prove kinetic energy preservation before looking at samples

The difference of two squared norms factors through the midpoint velocity:

$$
\Delta K=m\bar v_n\cdot(v_{n+1}-v_n)
=mh\Omega\bar v_n\cdot J\bar v_n=0,\qquad
\left(\frac{1-a^2}{1+a^2}\right)^2+\left(\frac{2a}{1+a^2}\right)^2=1.
$$

The coefficient identity also proves that R_h is an exact rotation matrix in real arithmetic. Thus constant energy is an algebraic property of this discrete method for every step size. Floating-point implementations may still show roundoff; the fraction code below removes that ambiguity for its rational preparations.

### Check the guiding center and the other angular invariant

For nonzero Omega, use the same C as the continuous model. The consistent position update makes its difference vanish too. In this section L=m(q cross v) is the single particle's mechanical lab angular momentum. Denote the original modified invariant by I_ang so it is not confused with the matrix identity:

$$
C=q+\frac{Jv}{\Omega},\qquad
\Delta C=h\bar v_n+\frac{J(h\Omega J\bar v_n)}{\Omega}=0,\qquad
I_{\mathrm{ang}}=L-\frac{m\Omega}{2}\lVert q\rVert^2
=\frac{m}{2\Omega}\lVert v\rVert^2-\frac{m\Omega}{2}\lVert C\rVert^2
\quad(\Omega\ne0).
$$

The identity follows by expanding the squared norm of C and using q dot Jv=−(q cross v). It shows that constant speed and C also give constant I_ang. The numerical positions lie on the same circle as the exact trajectory. These three invariant checks therefore do not determine the correct time-dependent point on that circle. For Omega=0 use free motion directly, rather than dividing by Omega in this representation.

### Identify the phase that the method actually advances

The exact velocity turns by h Omega per step; the method turns by theta_h. After n steps, with physical time t=n h:

$$
n\theta_h-\Omega nh
=-\frac{t\Omega^3h^2}{12}+O\!\left(t|\Omega|^5h^4\right),\qquad
 t=nh,\quad |h\Omega|\ll1.
$$

This small-step expansion applies when the dimensionless magnitude of h Omega is small. It describes accumulated signed phase error, not an energy drift. At a fixed elapsed time it is second order in h. For a large step use the exact arctangent formula, rather than the truncated series. A stable rotation with unchanged radius can still have a substantially wrong frequency.

### Work a four-step counterexample

Choose m=1 kg,Omega=1 s^-1,q0=(0,0) m,v0=(1,0) m/s and h=2 s. Then theta_h=pi/2: four steps return both position and velocity to their initial values at t=8 s. The analytic trajectory instead gives:

$$
q_4^{\mathrm{mid}}=(0,0),\qquad
q^{\mathrm{exact}}(8)=(\sin8,1-\cos8)\ \mathrm{m},\qquad
\lVert q^{\mathrm{exact}}(8)-q_4^{\mathrm{mid}}\rVert
=2|\sin4|\ \mathrm{m}\approx1.513604991\ \mathrm{m}.
$$

| diagnostic | midpoint at 8 s | exact flow at 8 s |
|---|---|---|
| position (m) | (0,0) | (sin 8,1−cos 8) |
| kinetic energy (J) | 1/2 | 1/2 |
| guiding center (m) | (0,1) | (0,1) |
| modified angular invariant (kg m²/s) | 0 | 0 |

![The analytic orbit and four midpoint positions lie on the same circle, but the exact eight-second star differs from the numerical return to the starting point.](/assets/learning/classical-midpoint-gyro-phase.en.svg)

Both position axes use metres with equal unit lengths. Blue is the continuous analytic path; orange squares are the discrete midpoint nodes numbered zero through four. The orange dashed segments only connect their order and are not a physical interpolation. A star marks the analytic position at 8 s. The start and numerical fourth point coincide, while their distance from the analytic endpoint is about 1.514 m.

The original lab can evaluate the exact side of this comparison: use mass code 1,Omega code 10,zero position,velocity codes (10,0),time code 80. It does not compute the orange midpoint nodes. A learned model with the wrong Omega could likewise conserve energy; separate an incorrect force coefficient from a time-discretization error before diagnosing the source of a bad prediction.

### Reproduce with rational updates and explicit free motion

The code uses m=1 and rational initial states. It checks four signed/zero frequencies and four step sizes, each for 100 steps, with exact K,I_ang,C comparisons. For Omega=0 it verifies the free positions and unchanged velocities without a center formula. The final sine evaluation illustrates the analytic endpoint error in floating point; it is not used to certify the rational invariant identities. These independent preparations need not fit the native bounded input grid.

```python
from fractions import Fraction as F
from math import sin, hypot

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def J(v):
    return (-v[1], v[0])

def midpoint(q, v, omega, h):
    a = h*omega/2
    c, s = (1-a*a)/(1+a*a), 2*a/(1+a*a)
    vn = (c*v[0]-s*v[1], s*v[0]+c*v[1])
    return tuple(x+h*(u+w)/2 for x, u, w in zip(q, v, vn)), vn

def invariants(q, v, omega):
    K = dot(v, v)/2  # mass = 1
    I = q[0]*v[1]-q[1]*v[0]-omega*dot(q, q)/2
    C = None if omega == 0 else tuple(x+y/omega for x, y in zip(q, J(v)))
    return K, I, C

for omega in map(F, [0, 1, -1, "1/10"]):
    for h in map(F, ["1/10", "1/2", 2, 4]):
        q0, v0 = (F(1), F(2)), (F(3), F(-1))
        q, v = q0, v0
        expected = invariants(q, v, omega)
        for n in range(1, 101):
            q, v = midpoint(q, v, omega, h)
            assert invariants(q, v, omega) == expected
            if omega == 0:
                assert q == tuple(x+n*h*u for x, u in zip(q0, v0))
                assert v == v0
print("16 preparations x 100 exact-rational invariant checks passed")
q, v = (F(0), F(0)), (F(1), F(0))
for _ in range(4):
    q, v = midpoint(q, v, F(1), F(2))
assert q == (0, 0) and v == (1, 0)
assert invariants(q, v, F(1)) == (F(1, 2), F(0), (F(0), F(1)))
error = 2*abs(sin(4))
assert abs(error-hypot(sin(8), 2*sin(4)**2)) < 1e-14
print("four-step position error (m):", error)
```

### Worked questions for a learned simulator

1. **Which diagnostics pass in the four-step example?** K,C and I_ang agree exactly, and every node lies on the right circle. A time-aligned position or velocity comparison still fails.
2. **Does all-step energy preservation imply all-step accuracy?** No. It fixes a radius but not theta_h=h Omega. Reverse the sign of Omega and the signed phase error reverses; the missing timing information remains.
3. **What extra evidence should a research result report?** Include time-aligned trajectories, step refinement, the declared force law and independent reference states. Invariant residuals support their stated identities rather than identifying the complete dynamics.

For the midpoint definition and geometric-integration background see [Hairer, Lecture 2](https://www.unige.ch/~hairer/poly_geoint/week2.pdf). The distinction between a numerical flow and the original flow is developed in [Lecture 3](https://www.unige.ch/~hairer/poly_geoint/week3.pdf). The rotation formulas and worked counterexample here are derived directly for this declared force law; they do not assert that every nonlinear midpoint problem conserves its physical energy.

## Six exercises with worked solutions

1. **Exercise: work-free bending.** Set m=1,Ω=1,q0=0,v0=(1,0). **Solution:** v=(cos t,sin t),q=(sin t,1−cos t),K=1/2J. The external force is nonzero and rotates momentum.
2. **Exercise: angular quantities.** For exercise1, find L and I. **Solution:** L=1−cos t and |q|²=2(1−cos t). Hence I=0 although mechanical L varies. External torque is sin t.
3. **Exercise: origin-centered special case.** Set q0=(1,0),v0=(0,1),m=Ω=1. **Solution:** C=0,q=(cos t,sin t),L=1 and I=1/2. Linear momentum still changes direction, demonstrating a different conservation combination.
4. **Exercise: free motion.** Set m=2,Ω=0,q0=(0,1),v0=(1,0). **Solution:** q=(t,1),p=(2,0),K=1J,L=I=−2. Force and work are zero; guiding center is undefined rather than set to the origin.
5. **Exercise: reverse orientation.** Replace Ω by−1 in exercise1. **Solution:** x=sin t,y=−(1−cos t),L=−(1−cos t). K remains1/2J. The modified invariant remains zero with the reversed Ω sign.
6. **Exercise: stationary state.** Take any initial q0, set v0=0 and choose a nonzero Ω. **Solution:** q stays at q0,F=0,K=0 and L=0. The modified invariant is the constant−mΩ|q0|²/2. A nonzero force coefficient does not imply a nonzero force when velocity vanishes.

## Connection to learned dynamical systems

A learned simulator can keep speed constant while predicting the wrong rotation frequency or direction. Energy-only losses cannot detect that error. Check vector momentum, external impulse, angular balance, the modified invariant and actual positions together. Conservation diagnostics must match the selected system and the declared force dependence; they do not replace validation of the force law against observations.

## Background and scope

For a physical perpendicular-force example, see [MIT OCW, Introduction to Magnetic Fields](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/90510e513af27a2dde30b890a61bbcd1_ch8magneti_field.pdf). The signed Ω model, guiding center and modified invariant above are derived directly here. General symmetry theorems and complete electromagnetic conservation are developed separately.
