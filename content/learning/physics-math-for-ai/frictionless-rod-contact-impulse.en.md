## An impact is decided at the contact point

A rigid object can rotate while its center of mass translates. The velocity of its lowest endpoint can therefore differ from the velocity of its center. A center moving upward does not guarantee that every point is leaving the floor. To decide whether an impulse is needed, compute the normal velocity of the actual contact point first.

This lesson studies a single instantaneous impact of a uniform thin rod against a fixed horizontal floor. It connects compatible rigid velocities, linear impulse, angular impulse and energy loss. It does not integrate a later flight or infer that a rod with zero endpoint normal speed remains attached to the ground.

## Fix a geometry with only one initial contact

Place the center of mass C at(0, h), the lower endpoint O at(a, 0), and the upper endpoint at(−a, 2h). Assume a≥0, h>0 and positive constant mass m. The rod length and center-of-mass inertia are

$$
\ell=2\sqrt{a^2+h^2}, \qquad I=\frac{m(a^2+h^2)}3.
$$

Only the lower endpoint initially touches the floor; the rest of the segment lies above it. Changing a or h changes the declared rod geometry and inertia. A horizontal rectangle with two corners touching would require a different contact model.

The impact map freezes position and orientation during the velocity jump. Incoming center velocity is(Vx, Vy). Angular velocity ω is positive counterclockwise. The floor normal is n=(0, 1), and the center-to-contact offset is r=(a, −h). There is no tangential impulse. Finite forces such as gravity do not contribute a finite impulse over this ideal zero-duration event.

## Compute the compatible contact velocity

Planar rigid kinematics gives

$$
v_O=V+\omega\, \hat z\times r
=(V_x+h\omega, \;V_y+a\omega).
$$

Write the incoming normal component as c⁻=Vy⁻+aω⁻. A closing contact has c⁻<0 and receives a nonnegative normal impulse. For c⁻≥0, this map applies no impulse: positive speed is separating, while zero is nonclosing.

The zero-speed case may still require a persistent-contact force in a later dynamics calculation. An inactive impact law does not by itself decide that force. Nor does it declare a contacting body to be in static equilibrium.

## Write both impulse balances

Let jn be the scalar upward normal impulse, with units N s. The center velocity and counterclockwise angular velocity jump according to

$$
V_x^+=V_x^-, \qquad V_y^+=V_y^-+\frac{j_n}{m},
$$

$$
\omega^+=\omega^-+\frac{a j_n}{I}.
$$

The lever arm a multiplies the impulse to give angular impulse. The floor is external to the rod, so the rod's vector linear momentum generally changes. Its spin angular momentum about C changes as well. These are physical responses to an external impulse, not violations of Newton's laws.

Frictionless means that center tangential momentum is unchanged. Contact tangential speed can still change, because rotation changes. This distinction is visible in the term hω in vOx.

## Apply restitution to the contact normal speed

For a closing contact, choose a coefficient e in[0, 1] and impose

$$
c^+=-e c^-.
$$

Substituting the impulse balances gives

$$
c^+=c^-+Wj_n, \qquad
W=\frac1m+\frac{a^2}{I}>0,
$$

$$
j_n=-\frac{(1+e)c^-}{W}.
$$

W is an inverse effective normal mass, measured in1/kg. Its reciprocal is the effective mass seen at this contact. A normal impulse can generate both translation and rotation, so that effective mass need not equal the total mass.

Restitution is not a rule that sets Vy⁺=−eVy⁻. That simpler relation holds here only when the normal impulse has no rotational coupling, such as a=0. For general a, it is the endpoint normal speed that the law constrains.

## Derive the energy change instead of assuming it

Total kinetic energy includes both translation and spin:

$$
K=\frac12m(V_x^2+V_y^2)+\frac12I\omega^2.
$$

Using the impulse balances,

$$
K^+-K^-=j_nc^-+\frac12Wj_n^2
=-\frac{(1-e^2)(c^-)^2}{2W}
$$

for an active impact. Thus e=1 conserves kinetic energy while redistributing it, and e<1 loses a nonnegative amount. The inactive branch leaves all velocities and energy unchanged.

The loss is an energy budget for the declared restitution map. It does not predict temperature, deformation history or how the loss is partitioned among internal mechanisms. Those questions require a richer constitutive model.

## Find the angular momentum that stays conserved

About the fixed impact point O, the center offset is C−O=(−a, h). The total angular momentum is

$$
L_O=I\omega+(C-O)\times mV
=I\omega-a mV_y-hmV_x.
$$

The normal impulse acts through O and has zero moment about O. Substitution gives LO⁺=LO⁻ exactly. Spin momentum about C can change while total angular momentum about O is conserved. The orbital term is essential; using only Iω would test the wrong quantity.

This is an instantaneous statement at the fixed impact location. A later moving contact point or an accelerated reference would need its own derivative and transport analysis.

## Read the elastic and plastic limits carefully

An elastic contact does not require the center to bounce upward. Rotation may open the endpoint contact even while Vy⁺ remains negative. What is guaranteed is the outgoing contact normal speed and the conserved total kinetic energy.

At e=0, c⁺=0. The rod can retain tangential center motion, angular velocity and positive kinetic energy. Zero normal speed is one scalar condition, not a declaration that every velocity component vanishes. Whether subsequent forces maintain, release or change the contact is outside this instantaneous map.

A vertical rod has a=0 but still has positive inertia mh²/3. The normal impulse passes through C, so spin remains unchanged and the ordinary particle-like vertical restitution relation is recovered.

## Relate the example to a contact Jacobian

Using generalized velocity ν=(Vx, Vy, ω), define J=(0, 1, a) and mass matrix M=diag(m, m, I). Then c=Jν and

$$
M(\nu^+-\nu^-)=J^\mathsf Tj_n, \qquad
W=JM^{-1}J^\mathsf T.
$$

This exposes the structure behind effective contact mass. The example has one scalar normal contact and a positive definite mass matrix, so the normal solve is unambiguous. Adding friction, several simultaneous contacts or a compliant surface changes the closure problem. A tangential law cannot simply be appended without checking consistency and energy behavior.

## Interpret the geometry and comparisons

The rod geometry is identical before and after the ideal impact; only velocities jump. Drawn before/after configurations must therefore not imply a computed later trajectory. Velocity arrows use m/s, impulse arrows use N s, and angular quantities require their own units.

The calculation reports exact rational balances for the discrete controls. The rod length uses a numerical square root, while inertia, impulse, velocities and energy are rational. Active restitution residuals are defined only on the closing branch; an inactive law is not reported as a failed restitution equation.

## Worked examples and exercises with solutions

1. **A rising center, closing endpoint.** Set m=a=h=1, V=(0, 1/2), ω=−1 and e=1/2. **Solution:** I=2/3, c⁻=−1/2 and jn=3/10. The outgoing values are Vy=4/5, ω=−11/20 and c⁺=1/4. The endpoint impacts despite the rising center. Energy loss is3/80.

2. **Elastic redistribution.** Keep m=a=h=1, use V=(1, −1), ω=0 and e=1. **Solution:** W=5/2 and jn=4/5. After impact Vy=−1/5 and ω=6/5, so c⁺=1. The center still descends, but the endpoint opens. K is1 before and after; spin and translation redistribute energy.

3. **Plastic normal stop.** Use the second example with e=0. **Solution:** jn=2/5, Vy⁺=−3/5 and ω⁺=3/5. Endpoint normal speed is zero, but K⁺=4/5 and Vx=1 remain. The loss is1/5. No permanent-contact trajectory has been solved.

4. **Exactly nonclosing contact.** Take a=1, Vy=−1 and ω=1. **Solution:** c⁻=0, so the impact map applies jn=0 for every allowed e. The center moves downward while the endpoint has zero normal speed. Later support forces require a continuous contact calculation.

5. **Vertical limit.** Set m=4, a=0, h=3, Vy=−2, ω=2 and e=1. **Solution:** I=12, W=1/4 and jn=16. Vy⁺=2 while ω⁺=2. There is no normal-impulse spin torque because the lever arm is zero.

6. **Tangential endpoint speed.** Return to the elastic second example. **Solution:** Vx remains1 because tangential impulse is zero. Endpoint tangential speed nevertheless changes from1 to11/5 through hω. Angular momentum about the floor contact remains0 because its spin and orbital parts compensate.

## Research connection and reading

For a learned simulator, classifying impact from center velocity alone can select the wrong branch. Conserving the wrong angular momentum, applying restitution to the wrong speed, or erasing motion at e=0 produces distinct failures. Test contact kinematics, impulse balance, energy and the appropriate reference momentum separately.

The [contact and impact section of MIT's Underactuated Robotics notes](https://underactuated.mit.edu/multibody.html) provides a general mass-matrix and contact-Jacobian framework. The single-rod geometry, frictionless law, exact controls and examples above define this lesson's narrower experiment. The next lesson studies continuous wheel rolling and sliding with an explicitly declared tangential friction law.
