## Name the reference point before asking about rotation

A particle moving along a straight track can carry angular momentum about a point beside the track. The same particle can have zero angular momentum about a point on the track. Nothing about its physical motion changed. Angular momentum includes a lever arm measured from a chosen reference point, so the question is incomplete until that point is named.

We study one constant positive point mass in an inertial laboratory. Its initial position and velocity are given, and one prescribed constant planar force acts. Two reference points O and Q=O+d are fixed in the same laboratory. The controls move the reference markers between experiments; they do not make those points move during an experiment or apply an additional physical force.

## Define the signed planar cross product

For two planar vectors a and b, their three-dimensional cross product has the z component

$$
(a\times b)_z=a_xb_y-a_yb_x.
$$

We use positive z for counterclockwise orientation in the displayed coordinate plane. Parallel vectors have zero cross product even when both are nonzero. A negative value records the opposite orientation; it is not a negative magnitude.

Let rO=q−O and p=mv. Angular momentum and torque about O are

$$
L_O=(r_O\times p)_z,\qquad \tau_O=(r_O\times F)_z.
$$

Angular momentum has units kg m²/s. Torque has units N m. Although N m has the dimensions of joules, torque is a cross-product quantity and work is an integral of a dot product. Equal dimensions do not make them interchangeable physical quantities.

## Solve the physical initial-value problem first

The force is constant, so Newtonian motion gives

$$
q(t)=q_0+v_0t+\frac{F}{2m}t^2,\qquad
p(t)=mv_0+Ft.
$$

These laboratory vectors do not depend on O or Q. Only afterward do we subtract the reference position to form each lever arm. This order prevents a common conceptual error: changing the point about which angular momentum is reported does not itself change acceleration.

The spatial panel uses meters for both axes and equal scales. It shows the physical path plus radius vectors from the two reference points to the query position. Those arrows are spatial vectors, not force arrows plotted with an undeclared conversion from newtons to meters.

## Derive the torque–angular momentum relation

Differentiate about a fixed O:

$$
\frac{dL_O}{dt}=(v\times mv)_z+(r_O\times\dot p)_z
=(r_O\times F)_z=\tau_O.
$$

The first term vanishes because v is parallel to mv. The derivation used a fixed reference point, constant mass and momentum measured in the same inertial laboratory. It did not assume circular motion.

If the reference point moves while lab momentum is retained, differentiating r introduces an extra term involving the reference velocity. The present experiment excludes that case. A formula derived for fixed O should not be transferred unchanged to a moving camera or accelerating pivot.

## A constant force can have a changing torque

Substitute the position solution into the lever arm:

$$
\tau_O(t)=((q_0-O)\times F)_z+(v_0\times F)_z t.
$$

The acceleration displacement is parallel to F, so its cross product with F vanishes. Define τ0 as the initial torque and s=(v0×F)z. Then torque is τ0+st, even though force itself is constant. The changing lever arm supplies the time dependence.

Integrating gives the exact angular impulse balance

$$
L_O(t)-L_O(0)=\tau_0t+\frac{s}{2}t^2.
$$

Integrated torque has units N m s, equivalent to angular momentum. The experiment compares this integral with the cross product calculated directly from the query position and momentum.

## Separate instantaneous conditions from conservation

Angular momentum is constant throughout this family of motions precisely when τ0=0 and s=0. Zero query torque alone only says that the derivative of L vanishes at that query instant. It does not say that torque vanished earlier or will vanish later.

Zero torque also does not imply zero angular momentum. Force-free straight motion has zero torque about every fixed point, but it can carry a nonzero conserved L about a point off the track. Conversely, zero L at a single instant does not require zero velocity: when the particle passes through the reference point, its lever arm vanishes.

These examples distinguish a quantity's value, its instantaneous rate of change and its conservation over a time interval.

## Translate the angular reference point

With Q=O+d and the same p and F,

$$
L_Q=L_O-(d\times p)_z,\qquad
\tau_Q=\tau_O-(d\times F)_z.
$$

Different reference points may therefore report different values and different conservation conditions. The two identities remain compatible with differentiation because p changes according to the same force. The exact shift residuals in the readout check this consistency.

This is a change in the point used to define angular quantities. All positions displayed in the physical path panel remain laboratory coordinates. The experiment does not rotate axes or insert a non-inertial force.

## Read the three panels and exact table

The first panel shows a zero-to-ten-second laboratory path, the two fixed reference markers and the current radius vectors. Its orange query dot selects one physical event. The angular-momentum panel compares LO and LQ in kg m²/s against time. The torque panel compares their derivatives in N m.

The table retains exact rational values for bounded integer control codes. Divide every nonmass control by ten to obtain physical positions, velocities, forces, shifts and time. Curves sample the same exact solution for display; the polynomial identities and flags do not depend on judging whether two plotted lines look close.

Try changing only d. The laboratory path, momentum and force stay fixed while comparison angular quantities change. Then set query time to zero in the first guided example: torque is zero at that instant, but the conservation flag correctly remains false.

## Research extension: choose the momentum when the reference point moves

The fixed-point derivation above uses lab momentum and a reference that stays still. With a moving reference, two quantities that are often both called “angular momentum about the point” differ: one retains lab momentum, while the other uses velocity relative to the point. Stating both choices prevents a false conservation test.

This is a standalone planar extension. Keep a constant positive particle mass and an inertial laboratory with fixed, nonrotating axes. Let the reference A(t) be twice differentiable. The native controls still select two fixed points during each experiment; changing a marker between experiments does not implement the moving path derived here.

### Define the two angular quantities

Use the same lever arm in both definitions:

$$
\mathbf r=\mathbf q-\mathbf A(t),\quad \mathbf p=m\mathbf v,\qquad
L_{\mathrm{lab}}=(\mathbf r\times\mathbf p)_z,\quad
L_{\mathrm{rel}}=[\mathbf r\times m(\mathbf v-\dot{\mathbf A})]_z.
$$

Both quantities have units kg m²/s, so units alone cannot tell which one a dataset contains. The position of the reference and the velocity convention are both part of the label. In this section, the cross product always means its signed z component in the fixed lab axes.

### Retain lab momentum and differentiate the lever arm

The radius changes at v−A', not v. With p'=F and constant mass:

$$
\dot L_{\mathrm{lab}}
=[(\mathbf v-\dot{\mathbf A})\times m\mathbf v]_z
+(\mathbf r\times\mathbf F)_z
=\tau_A-(\dot{\mathbf A}\times\mathbf p)_z.
$$

The v cross mv term cancels; the reference-velocity term remains. The torque tau_A is still the physical lever-arm cross force. It can be zero even when L_lab changes. Integrating supplies the appropriate residual for this label:

$$
L_{\mathrm{lab}}(t)-L_{\mathrm{lab}}(0)
=\int_0^t\tau_A(s)\,ds
-\int_0^t[\dot{\mathbf A}(s)\times\mathbf p(s)]_z\,ds.
$$

A fixed reference makes the correction zero. A moving reference can also make it vanish if its velocity is parallel to momentum. The condition is a cross-product statement, not simply a test of whether the reference position changes.

### Use relative momentum and obtain the acceleration term

Now r'=v−A'. Differentiating r cross m r' cancels r' cross m r'. The remaining term gives:

$$
L_{\mathrm{rel}}=L_{\mathrm{lab}}-m(\mathbf r\times\dot{\mathbf A})_z,
\qquad
\dot L_{\mathrm{rel}}=[\mathbf r\times(\mathbf F-m\ddot{\mathbf A})]_z.
$$

The relative equation contains the translating-frame inertial term −mA''. When A moves at constant velocity, that term vanishes and the relative quantity obeys the physical torque balance. An accelerating reference generally requires its contribution. Rotating axes would require their own transformation as well; they are excluded from these formulas.

The reference-velocity correction in the lab-momentum definition and the reference-acceleration correction in the relative definition are different, consistent consequences of their definitions. Selecting a different definition is not evidence for a new interaction with the particle.

### Work an unforced particle example

Choose m=1 kg and a force-free particle with q0=(1,0) m,v0=(0,1) m/s. Let A move right at 1 m/s. Using numerical SI time:

$$
\mathbf q=(1,t),\quad \mathbf v=(0,1),\quad
\mathbf A=(t,0),\quad \mathbf r=(1-t,t),\qquad
L_{\mathrm{lab}}=1-t,\quad L_{\mathrm{rel}}=1,\quad\tau_A=0.
$$

| t (s) | L_lab (kg m²/s) | L_rel (kg m²/s) | physical torque (N m) |
|---|---|---|---|
| 0 | 1 | 1 | 0 |
| 1/2 | 1/2 | 1 | 0 |
| 1 | 0 | 1 | 0 |
| 2 | −1 | 1 | 0 |

The naive residual L_lab(t)−L_lab(0)−integral tau_A is −t, although the force law is exactly satisfied. The missing integral of A' cross p equals +t and restores zero residual. At t=1, lab momentum is parallel to the lever arm, giving zero L_lab; the particle has not stopped.

![A force-free particle moves vertically while the reference moves horizontally; lab-momentum angular momentum changes linearly, whereas relative-momentum angular momentum remains constant.](/assets/learning/classical-moving-reference.en.svg)

The upper panel uses lab positions in metres with equal unit lengths on both axes. Markers and the radius arrow select t=1.5 s; the arrow is a spatial lever arm, not a force or velocity placed on a distance axis. The lower panel uses angular momentum versus time. Both curves refer to the same lab particle and reference path.

### Accelerate the reference without adding a lab force

Keep the same force-free particle but choose the following origin path:

$$
\mathbf A=(t^2/2,0),\quad\dot{\mathbf A}=(t,0),\quad\ddot{\mathbf A}=(1,0),
\quad L_{\mathrm{lab}}=1-t^2/2,\quad L_{\mathrm{rel}}=1+t^2/2,
\quad\dot L_{\mathrm{rel}}=t.
$$

Here tau_A=0 remains true. The lab-momentum derivative is −t, from the reference-velocity correction. The relative derivative is +t, because r cross (−mA'') equals +t. At 1 s the two angular quantities are 1/2 and 3/2 kg m²/s. Their different derivatives satisfy their respective balance laws.

This also explains why changing a moving camera can change angular labels while the lab force and trajectory remain fixed. To compare learned predictions, first reconstruct which origin path and momentum convention generated the labels.

### Verify complete polynomial identities

The code represents a polynomial by its coefficients in increasing powers of time. It differentiates and multiplies those coefficients with exact fractions. For three choices of constant force and quadratic origin path, both balance laws are checked coefficient by coefficient, so the check covers every time in each declared polynomial example. The general smooth-path identities were derived above; the three examples are not a proof of every possible path.

```python
from fractions import Fraction as F

def trim(p):
    while len(p)>1 and p[-1]==0:
        p.pop()
    return p

def add(a,b):
    return trim([(a[i] if i<len(a) else F(0))
                 +(b[i] if i<len(b) else F(0))
                 for i in range(max(len(a),len(b)))])

def scale(a,c):
    return trim([c*x for x in a])

def mul(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            out[i+j]+=x*y
    return trim(out)

def deriv(a):
    return trim([i*a[i] for i in range(1,len(a))] or [F(0)])

def cross(a,b):
    return add(mul(a[0],b[1]),scale(mul(a[1],b[0]),F(-1)))

def value(a,t):
    return sum(x*t**i for i,x in enumerate(a))

# Ascending polynomial coefficients; all numerical quantities use SI units.
for mass,U,G in [(F(1),(1,0),(0,0)),
                 (F(1),(0,0),(1,0)),
                 (F(2),(1,1),(-1,1))]:
    force=(F(0),F(0)) if mass==1 else (F(-1),F(0))
    q=[[F(1),F(0),force[0]/(2*mass)],
       [F(0),F(1),force[1]/(2*mass)]]
    origin=[[F(0),F(U[i]),F(G[i])/2] for i in range(2)]
    velocity=[deriv(x) for x in q]
    origin_v=[deriv(x) for x in origin]
    radius=[add(q[i],scale(origin[i],F(-1))) for i in range(2)]
    momentum=[scale(x,mass) for x in velocity]
    relative_p=[add(momentum[i],scale(origin_v[i],-mass))
                for i in range(2)]
    physical=[[x] for x in force]
    torque=cross(radius,physical)
    L_lab=cross(radius,momentum)
    L_relative=cross(radius,relative_p)
    effective=[add(physical[i],scale(deriv(origin_v[i]),-mass))
               for i in range(2)]
    assert deriv(L_lab)==add(torque,scale(cross(origin_v,momentum),F(-1)))
    assert deriv(L_relative)==cross(radius,effective)
    if mass==1 and U==(1,0):
        assert L_lab==[F(1),F(-1)] and L_relative==[F(1)]
        for t in [F(0),F(1,2),F(1),F(2)]:
            print("t, L_lab, L_relative, torque:",t,
                  value(L_lab,t),value(L_relative,t),value(torque,t))
    if mass==1 and G==(1,0):
        assert L_lab==[F(1),F(0),F(-1,2)]
        assert L_relative==[F(1),F(0),F(1,2)]
        print("accelerating origin at 1 s:",value(L_lab,F(1)),
              value(L_relative,F(1)),value(deriv(L_relative),F(1)))
print("Both balance laws verified coefficient by coefficient for three cases.")
```

### Worked questions for angular labels

1. **Why can the first example have zero physical torque and changing L_lab?** Its reference velocity crossed with lab momentum is 1 N m, producing L_lab'=−1 N m. Torque alone is the fixed-point balance.
2. **Which quantity is conserved for a free particle viewed from a constant-velocity translating origin?** L_rel, because the origin acceleration is zero. L_lab about that moving origin need not be conserved.
3. **What metadata should accompany angular-momentum supervision?** Specify the origin path, axes, lab or relative momentum, and signed orientation. Compute the residual from the corresponding law before interpreting it as force-model error.

All identities follow directly from the product rule and Newton's law. They extend the fixed-point analysis while preserving the fixed-reference meaning of the existing interactive controls.

## Six exercises with worked solutions

1. **Exercise: changing lever arm.** Use m=1,q0=(1,0),v0=(0,1),F=(−1,0),O=0. Find the state at1s. **Solution:** q=(1/2,1),p=(−1,1),L=3/2 and τ=1. Initially L=1; the torque integral is1/2.
2. **Exercise: zero torque now.** In exercise1 query t=0. Is L conserved? **Solution:** τ0=0 but s=v0×F=1, so τ(t)=t and L(t)=1+t²/2. A zero initial derivative does not imply a constant function.
3. **Exercise: two conservation answers.** Set q0=(1,0),v0=(0,1),F=(0,−1),m=1 and compare O=0 with Q=(1,0). **Solution:** LO=1−t and τO=−1. The whole path lies on the vertical line through Q, so LQ=τQ=0 at every time.
4. **Exercise: force-free straight motion.** Take q0=(0,1),v0=(1,0),F=0,m=1. **Solution:** LO=−1 about the origin and remains constant. About Q=(0,1), LQ=0. The path is straight rather than circular.
5. **Exercise: crossing the reference point.** Start q0=O=0 with v0=(1,0). What is initial L? **Solution:** it is zero because r=0, although velocity is nonzero. An angular coordinate at that point is undefined; the cross-product definition still makes sense.
6. **Exercise: double mass.** Double only mass in exercise1. **Solution:** at1s q=(3/4,1),p=(−1,2),L=5/2. Initial L becomes2, while angular impulse remains1/2. In this constant-force family the additional acceleration displacement is parallel to F and adds no torque.

## Connection to learned mechanics

A model trained to preserve angular momentum about one point can fail when evaluated about another if it predicts inconsistent linear momentum or forces. Check the shift identities together with the integrated torque balance. A small residual alone does not establish that a learned force law matches observations: the prescribed force and initial data remain assumptions to be tested against measurements.

## Background and limits

For the fixed-point and moving-point distinctions, consult [MIT OCW, Angular Momentum and Torque](https://ocw.mit.edu/courses/2-003sc-engineering-dynamics-fall-2011/pages/angular-momentum-and-torque/). The constant-force polynomial example is derived above. Rigid-body spin, inertia tensors and moving reference points need additional models.
