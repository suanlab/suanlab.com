## A conservation statement needs a system and an exchange term

Momentum is not conserved merely because two particles exert internal forces on one another. The chosen system may also receive an external impulse. Kinetic energy is another quantity and obeys a different balance. A useful physical model names the system, the exchanged quantities and the assumptions before using an invariant as a constraint.

This lesson will use two constant positive point masses in one dimension and a declared impulse update. It will then compare that update with complete momentum-and-kinetic-energy constraint sets. The picture is a velocity-plane constraint diagram, not a time trajectory or a simulation of when two bodies actually contact.

## Specify the impulse update and its units

Let J be an internal impulse that adds momentum to particle 1 and removes the same momentum from particle 2. Let I be an external impulse delivered to particle 1. Both have units kg*m/s. The update is

$$
v_1'=v_1+\frac{J+I}{m_1},\qquad
v_2'=v_2-\frac{J}{m_2}.
$$

Velocities are signed and measured in m/s. The prime labels the post-update state; it does not denote a time derivative in this lesson. No positions, contact geometry or force-duration history have been specified. Calling the update an impulse map does not establish that every query corresponds to an actual elastic collision.

## Derive the momentum balance

Initial and final momenta are P0=m1*v1+m2*v2 and P1=m1*v1'+m2*v2'. Substituting the update cancels J and gives

$$
P_1-P_0=I.
$$

The internal pair cancels because the chosen system includes both particles. Under this update momentum is conserved when the external impulse is zero; for nonzero I the correct statement is a balance with the external contribution. A nonzero momentum change is not an error when it equals that contribution.

Choosing only particle 1 as the system would make the interaction impulse an external term for that smaller system. This connects the calculation to the earlier lesson about accounting boundaries: an exchange term can become internal when the system is enlarged.

## Kinetic energy does not follow from momentum conservation

Define K=(m1*v1 squared+m2*v2 squared)/2. With I=0, direct expansion gives

$$
K_1-K_0=J(v_1-v_2)+\frac{J^2}{2}
\left(\frac{1}{m_1}+\frac{1}{m_2}\right).
$$

Equal and opposite impulses therefore do not automatically preserve kinetic energy. Depending on J, the kinetic energy can increase, decrease or remain unchanged while momentum is conserved. Kinetic energy is also not the whole thermodynamic energy of an actual apparatus. A change in K can reflect work or conversion involving degrees of freedom absent from this point-mass description.

For equal unit masses with initial velocities 1 and0, choosing J=-1 gives final velocities 0 and1 and preserves K. Choosing J=-0.5 gives both final velocities 0.5 and lowers K from0.5J to0.25J. Choosing J=1 gives velocities2 and-1 and raises K to2.5J. All three internal-only queries preserve momentum.

## Verify an algebraic work balance

For any impulse query, the kinetic change can be written

$$
\Delta K=\frac{v_1+v_1'}{2}(J+I)
+\frac{v_2+v_2'}{2}(-J).
$$

This follows from the difference of two squares for each constant mass. It is an algebraic midpoint-velocity identity for the declared update. It does not reconstruct an actual force history or establish how an actuator or an internal potential supplies that work.

The distinction is useful in learned models: a balance relation can be checked exactly without claiming that every physical mechanism behind the relation has been identified. State what was modeled and what remains an unresolved constitutive or contact law.

## Solve the complete invariant constraint set

Suppose a candidate final velocity pair is required to have target momentum Pstar and kinetic energy equal to the initial K0. Introduce total mass M=m1+m2, reduced mass mu=m1*m2/M, center velocity V=Pstar/M and relative velocity d=v1'-v2'. The velocities can be reconstructed as

$$
v_1'=V+\frac{m_2}{M}d,\qquad
v_2'=V-\frac{m_1}{M}d.
$$

Substituting into K separates center motion from relative motion:

$$
K_0=\frac{P_\star^2}{2M}+\frac{\mu d^2}{2}.
$$

Define gap=2K0-Pstar squared/M. If gap is negative there is no real pair satisfying both targets. If it is zero there is one equal-velocity pair. If it is positive there are exactly two pairs, obtained from d equal to plus or minus sqrt(gap/mu). These are the complete intersections of the momentum line and kinetic-energy ellipse, including the degenerate point case when K0=0.

Exact rational gap signs can classify existence before numerical square roots are displayed. A few sampled points on an ellipse would not establish completeness of this constraint set.

## Recover both no-external roots

When I=0, take Pstar=P0. The initial pair already satisfies the targets. The two possible internal impulses are

$$
J=0,\qquad J=-2\mu(v_1-v_2).
$$

The first leaves the velocities unchanged. The second reverses the relative velocity, giving the usual nontrivial elastic exchange candidate in one dimension. If the initial velocities agree, the two roots merge into one.

Neither invariants alone nor the existence of the second candidate decides whether particles have collided, which way a contact impulse can act, or what time evolution led to the candidate. Positions, nonpenetration and an appropriate interaction law are additional information. Keeping both algebraic roots avoids silently using contact assumptions that were never stated.

## Interpret an incompatible externally shifted target correctly

For nonzero external I the momentum target dictated by the update is Pstar=P0+I. Asking at the same time for unchanged kinetic energy is an additional hypothetical constraint; it is not automatically the physical energy balance.

For example, two unit masses initially moving at speed 3 have P0=6 and K0=9. An external impulse 2 makes the momentum target8. The minimum kinetic energy at that momentum is16, so no final velocity pair can retain K0=9. This does not mean the forced motion is impossible. The update with J=0 gives velocities 5 and3 and kinetic energy 17, satisfying the correct momentum balance while changing kinetic energy.

A model should distinguish an impossible combination of requested invariant targets from an impossible physical process. External work and internal energy conversion can change the appropriate energy target.

## Connect the constraint geometry to AI models

A learned transition model can satisfy a momentum penalty and still make a wrong kinetic-energy prediction. It can also satisfy both invariants while choosing a transition inconsistent with contact geometry or the desired dynamics. Necessary constraints restrict outputs without always selecting a unique physical evolution.

Projecting an output onto an invariant set can be useful, but its existence, multiplicity and choice rule matter. A set may be empty or contain more than one point, even in this simple two-velocity example. A benchmark should report balance residuals, feasibility and the actual dynamical assumptions rather than interpreting one small loss as complete physical correctness.

## Exercises with worked solutions

1. **Exercise:** What is the total momentum change under the declared map? **Solution:** Internal J cancels, leaving exactly I.
2. **Exercise:** For unit masses and initial velocities 1,0, what does J=-0.5,I=0 do? **Solution:** Both velocities become0.5; momentum stays1 while kinetic energy decreases from0.5J to0.25J.
3. **Exercise:** Why are there at most two final pairs for fixed momentum and kinetic energy? **Solution:** The center-relative decomposition leaves the equation mu*d squared=gap, giving zero, one or two real values.
4. **Exercise:** Which impulses preserve both invariants for I=0? **Solution:** J=0 and J=-2*mu*(v1-v2), with a merged root for equal initial velocities.
5. **Exercise:** Does an empty unchanged-K target after external impulse rule out forced motion? **Solution:** No. The physical kinetic energy can change through the external or internal work balance.
6. **Exercise:** Do both invariants determine an actual collision event? **Solution:** No. Contact positions, timing and interaction assumptions are absent from this constraint diagram.

## Sources and further study

[MIT's impulse and momentum material](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/c7462af7761b1bf1ea6441fc151da0cc_Vg8t8_IOHDg.pdf) gives background on system choice and external impulse. The impulse map, complete target-constraint set and worked algebra above are direct derivations. 
