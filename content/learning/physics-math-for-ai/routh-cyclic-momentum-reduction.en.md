## Distinguish a conserved momentum from a fixed coordinate

A cyclic coordinate can be removed from part of a calculation without being physically frozen. If an angle is cyclic, its canonical momentum is conserved. Its angular velocity can still change as the radius changes. The reduced equations must retain that effect.

This lesson derives a Routhian by a partial Legendre transformation. It compares the result with naive substitution into the Lagrangian and with a different experiment in which a motor holds angular velocity fixed. These operations answer different questions. Agreement at a single radius is not a test of their equivalence.

The interactive model evaluates exact rational states and radius profiles. It does not integrate a trajectory. A plotted radius is a configuration parameter in meters, not physical time.

## Specify a regular mechanical model

Use radial position $r>0$ in m, radial speed $v_r$ in m/s and an angular speed $\omega$ in s⁻¹. Consider

$$
L=\frac m2(v_r^2+r^2\omega^2)+\frac{\beta r^2\omega}{2}-U(r),
\qquad U(r)=\frac{k r^2}{2}+\frac{a r^4}{4}.
$$

Here $m>0$ is in kg, $k$ is in N/m, the quartic coefficient $a$ is in N/m³, and $\beta$ is in kg/s. Every term in L has units J. The velocity-linear term is gyroscopic; its contribution cancels out of the physical energy. No electromagnetic interpretation or charge convention is needed to use this model.

Mass, stiffness and quartic coefficient use their integer controls. Momentum and beta codes are divided by two, in Js and kg/s respectively. The radius code is divided by ten in m; radial-speed and imposed-angular-speed codes are divided by two in their respective SI units. Radius zero is included only to expose a singular boundary.

## Identify the canonical conserved quantity

The angle does not appear in L, so its canonical momentum is

$$
p_\theta=m r^2\omega+\frac{\beta r^2}{2}=\ell.
$$

The constant $\ell$ has units Js. The mechanical angular momentum is instead $m r^2\omega=\ell-\beta r^2/2$. With nonzero beta and changing radius, mechanical angular momentum need not be constant even though the angle is cyclic.

This is a useful refinement of the slogan that rotational symmetry conserves angular momentum. Specify which momentum the chosen Lagrangian generates. A coordinate-dependent velocity-linear term changes the relation between canonical and mechanical momentum. It does not make the Noether argument disappear.

## Invert only on the regular radial domain

The angular velocity Hessian is $\partial^2L/\partial\omega^2=m r^2$. It is positive for $r>0$, so the equation for momentum has the unique inverse

$$
\omega(r,\ell)=\frac{\ell}{m r^2}-\frac{\beta}{2m}.
$$

Along a fixed-momentum evolution, $\dot\omega=-2\ell v_r/(m r^3)$. The angle can therefore speed up or slow down while its canonical momentum remains fixed. Reconstructing the angle would require integrating this rate and specifying an initial angle; this lesson does not invent a reconstructed trajectory from a radius profile.

The radial velocity Hessian of the reduced problem is m. These two Hessians have different units, kg m² and kg. They concern different coordinates and should not be combined into one numerical magnitude or compared as coordinate-invariant spectra.

## Subtract momentum times angular velocity first

Define the reduced Lagrangian with our sign convention by

$$
\mathcal R(r,v_r;\ell)=\left[L-\ell\omega\right]_{\omega=\omega(r,\ell)}.
$$

The partial derivative at fixed ell is

$$
\partial_r\mathcal R=\partial_rL+
 (\partial_\omega L-\ell)\partial_r\omega=\partial_rL.
$$

The extra chain-rule term vanishes because the momentum equation has been imposed. In contrast, differentiating $L_{\mathrm{sub}}=L(r,v_r,\omega(r,\ell))$ leaves $\ell\partial_r\omega$. That term changes the radial force. The problem is a change in what is held fixed during variation, not an algebraic mistake in the momentum inversion.

Some texts reverse the sign of the object called a Routhian and then use Hamilton-like signs for the eliminated variables. This lesson uses $\mathcal R=L-\ell\omega$ as the Lagrangian of the remaining radial variable. Keep the convention attached to the equation of motion.

## Read the amended potential and correct force

The Routhian is

$$
\mathcal R=\frac m2v_r^2-U_{\mathrm{eff}}(r),\qquad
U_{\mathrm{eff}}=U+\frac{(\ell-\beta r^2/2)^2}{2mr^2}.
$$

Thus the correct radial equation is

$$
m\ddot r=\frac{\ell^2}{m r^3}-\frac{\beta^2r}{4m}-U'(r).
$$

Direct substitution into L would instead predict

$$
m\ddot r\big|_{\mathrm{naive}}=-\frac{\ell^2}{m r^3}-\frac{\beta^2r}{4m}-U'(r).
$$

The force gap is $2\ell^2/(mr^3)$ N. It reverses the centrifugal contribution whenever ell is nonzero. It does not depend on the sign of ell, although the reconstructed angular velocity generally does.

Expanding the amended potential gives $U+\ell^2/(2mr^2)-\beta\ell/(2m)+\beta^2r^2/(8m)$. The middle constant does not affect radial force at fixed ell, but it does affect the absolute energy represented by this chosen convention. Retain it when comparing reconstructed physical energy with the reduced energy.

## Check the complete reconstruction identities

At each regular query, the model checks the original radial Euler-Lagrange equation, reconstruction of $p_\theta=\ell$, and $d p_\theta/dt=0$. The reconstructed angular acceleration is part of this check; simply holding omega constant would not suffice.

The physical energy is

$$
E=\frac m2v_r^2+\frac m2r^2\omega^2+U
  =\frac m2v_r^2+U_{\mathrm{eff}}.
$$

Its instantaneous derivative vanishes when the reduced acceleration and reconstructed angular acceleration are inserted. The mechanical angular-momentum rate is $-\beta r v_r$, which can be nonzero. These simultaneous results are consistent: a velocity-linear gyroscopic coupling redistributes components of motion without contributing to the physical energy.

The identity $-\partial_\ell\mathcal R=\omega$ provides the reconstruction rule from the partial Legendre transformation. The comparisons use exact rational expressions rather than finite-difference slopes.

## Compare a motor-controlled angular speed

A motor can impose $\theta(t)=\Omega t+\theta_0$. This is an externally imposed time-dependent coordinate condition, not a conserved-momentum level. Substitution of the prescribed constant Omega into L is appropriate for this different radial problem. Its radial force is

$$
F_{\Omega}=(m\Omega^2+\beta\Omega)r-U'(r).
$$

The motor must supply torque and power

$$
\tau_{\mathrm{motor}}=(2m\Omega+\beta)r v_r,
\qquad P_{\mathrm{motor}}=\Omega\tau_{\mathrm{motor}}.
$$

The physical energy obeys $\dot E_\Omega=P_{\mathrm{motor}}$. An autonomous radial quantity $E_\Omega-\Omega p_\theta$ remains constant, but the physical energy generally changes. Calling that radial conserved quantity the physical energy would hide the motor's work.

The beta term is part of the torque balance. At Omega zero the holding torque can be nonzero while its power is zero. Zero power at one setting is not a general statement that imposing angular speed requires no external interaction.

## Work through a misleading point of agreement

Use $m=1$ kg, $k=1$ N/m, $a=0$, $\ell=1$ Js, beta zero, $r=1$ m, $v_r=1$ m/s and imposed Omega 1 s⁻¹.

The reconstructed angular speed equals the imposed speed at this radius. Both the correct fixed-momentum force and the motor-controlled force are zero. The naive substituted force is -2 N. The Routhian is -1/2 J and reconstructed physical energy is 3/2 J.

Nevertheless, the two correct experiments are not the same model. Their force slopes are -4 N/m for the fixed-momentum reduction and zero for the imposed-speed model. Moving radius slightly distinguishes them. The motor-controlled physical energy has rate 2 W and its torque is 2 N m, whereas the closed fixed-momentum energy rate is zero.

This example suggests a research check: compare a neighborhood, derivatives and conserved conditions, rather than validating a reduced model at one matched state.

## Investigate zero canonical momentum with nonzero rotation

Take unit mass, zero potential, $\ell=0$, $\beta=1$ kg/s, $r=1$ m and $v_r=1$ m/s. The reconstructed angular speed is -1/2 s⁻¹. Mechanical angular momentum is -1/2 Js even though canonical momentum is zero.

Both the Routh and naive substituted forces equal -1/4 N in this special momentum sector. Their agreement does not establish the naive procedure at other momentum values. Turning ell on reveals the missing centrifugal sign immediately.

The physical energy is 5/8 J, its rate is zero, and the mechanical angular-momentum rate is -1 N m. Canonical momentum remains fixed because the beta contribution changes by the opposite amount. Zero canonical momentum is not a statement that the angle is stationary.

There is also a genuine special agreement with the motor comparison: set its imposed speed to -1/2 s⁻¹ in this zero-momentum example. Then $(2m\Omega+\beta)=0$, so the required motor torque and power vanish and both radial forces agree at every regular radius. This particular coincidence does not justify interchanging fixed momentum and fixed speed in general.

## Keep radius zero explicitly undefined

At radius zero, the angular Hessian is zero. Polar coordinates no longer specify an angle, and the momentum equation cannot be uniquely inverted for omega. If ell is nonzero there is no finite angular-speed solution; if ell is zero the equation fails to select a unique angular speed.

Accordingly, the query state and query dimension counts are unavailable at radius zero. The model does not divide by a tiny epsilon or silently choose the first positive plotted radius as the query. Radius profiles are drawn only on their declared regular range, 0.1 to 4 m.

Even when an ell-zero expression has a finite algebraic limit, that limit is not proof that the angular Legendre map or polar chart has become regular. Distinguish continuity of one formula from invertibility of the full coordinate-momentum relation.

## Count the dimensions and recover the discarded information

On the regular domain, the original phase space has dimension four: $(r,\theta,p_r,p_\theta)$. Fixing $p_\theta=\ell$ gives a three-dimensional level set. Quotienting by the cyclic angle gives a two-dimensional reduced phase space, or one radial configuration degree of freedom.

The momentum level alone has odd dimension and is not the final reduced symplectic phase space. Nor did we constrain theta to a fixed number. Its initial phase and reconstructed angular evolution are information outside the radial quotient.

This differs from the next lesson's second-class circle constraints, where a position relation and its compatible momentum relation select a two-dimensional surface in a four-dimensional ambient phase space. Count conditions and symmetry orbits separately rather than subtracting one dimension for every conserved quantity without checking the reduction being performed.

## Use radius profiles without inventing a time evolution

The force, force-slope, momentum, energy and power curves compare admissible states at different radii with the other controls fixed. They are not a single solution moving through those radii at the displayed constant radial speed. The local acceleration is the equation-of-motion value at each state.

An energy profile can vary with radius in this experiment while the directional derivative of energy along the correctly reconstructed dynamics is zero. There is no contradiction: the plotted family does not follow that dynamics. The same distinction explains why a motor-power profile is not the time integral of motor work.

For learned reduced models, record the fixed momentum, regularity domain, reconstruction equations and energy convention. A small force gap at ell zero or at one radius can conceal a wrong reduction. A dimension count without the quotient or a finite singular limit can conceal a missing variable or an invalid chart.

## Exercises with explained solutions

1. **Centrifugal sign.** In the first example, calculate the correct and naive radial forces at one meter.

   **Solution:** The correct force is $1-1=0$ N. Naive substitution gives $-1-1=-2$ N. Their difference is $2\ell^2/(mr^3)=2$ N. The missing partial Legendre term accounts for the discrepancy.

2. **Neighborhood check.** Compute the two force slopes in that example.

   **Solution:** At fixed ell the slope is $-3\ell^2/(mr^4)-k=-4$ N/m. At fixed Omega it is $m\Omega^2-k=0$. Equal forces at one radius do not imply equal nearby responses.

3. **Canonical versus mechanical momentum.** With ell zero and beta one, find omega, mechanical angular momentum and its rate at the stated unit state.

   **Solution:** Omega reconstructed from momentum is -1/2 s⁻¹, mechanical angular momentum is -1/2 Js, and its rate is $-\beta r v_r=-1$ N m. The change of $\beta r^2/2$ cancels this rate in canonical momentum.

4. **External work.** Why can the motor-controlled energy increase at 2 W even when its radial acceleration is zero at the first query?

   **Solution:** Radius is increasing at 1 m/s while the motor holds Omega fixed. It supplies torque 2 N m and hence power 2 W. The growing rotational kinetic energy and potential account for that power. Zero instantaneous radial acceleration does not remove external work.

5. **Singular inversion.** At radius zero and ell zero, does assigning omega zero repair the reduction?

   **Solution:** No. The angular Hessian is zero and the momentum equation does not determine omega. Assigning one value arbitrarily does not make the inverse unique or restore a polar angle at the origin. The query remains undefined.

6. **Dimension counting.** After fixing a cyclic momentum in a four-dimensional regular phase space, is the resulting three-dimensional level already the radial reduced phase space?

   **Solution:** No. One must also quotient the angle symmetry orbit, leaving dimension two. Recovering a full motion still requires an initial angle and the reconstruction equation. Fixing momentum is not the same as fixing the angle or its speed.

## Reading and verification scope

The classical partial Legendre reduction is stated in B. Langerock, F. Cantrijn and J. Vankerschaver, [Routhian reduction for quasi-invariant Lagrangians, introduction and Theorem 1](https://arxiv.org/html/0912.0863v1). David Tong's [Lagrangian Formalism, sections 2.5.3–2.5.4](https://davidtong.org/teaching/classical-dynamics/dynhtml/S2) explains why substituting a conserved angular momentum directly into L can reverse a force sign. The gyroscopic model, motor comparison and rational examples above are independent constructions. General nonabelian and quasi-cyclic reduction require further hypotheses and are not implemented here.
