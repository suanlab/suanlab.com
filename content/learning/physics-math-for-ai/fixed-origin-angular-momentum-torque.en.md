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
