## Watch the masses and the energy separately

Two weakly connected oscillators can appear to pass motion back and forth. A slow envelope grows on one displacement while shrinking on the other. This is a useful picture of beats, but an envelope is not an energy measurement. Velocity and energy stored in the connecting spring also matter.

This chapter uses an exactly solvable two-mass system to compare the physical trajectory, envelope, local energy and signed energy flow. It also tests the familiar weak-coupling approximation against the exact solution over a stated observation time.

## Fix the system and the initial experiment

Let the two masses be equal to $m>0$, with identical anchoring stiffness $k\ge0$ and coupling stiffness $\kappa\ge0$. Use SI units: kg, N/m, m, s and J. Initially displace only mass 1 by $A\ge0$ and release both masses from rest:

$$
q_1(0)=A,\qquad q_2(0)=0,\qquad v_1(0)=v_2(0)=0.
$$

The undamped motion obeys

$$
m\ddot q_1=-(k+\kappa)q_1+\kappa q_2,\qquad
m\ddot q_2=\kappa q_1-(k+\kappa)q_2.
$$

All comparisons below preserve these initial conditions. Changing the initial velocity or exciting a single normal mode produces a different experiment.

## Find the symmetric and antisymmetric motions

The same-direction pattern $(1,1)$ does not deform the coupling spring. Its angular frequency is $\omega_-=\sqrt{k/m}$. The opposite-direction pattern $(1,-1)$ changes the relative displacement twice as much and has $\omega_+=\sqrt{(k+2\kappa)/m}$.

Define the average frequency and splitting:

$$
\bar\omega=\frac{\omega_++\omega_-}{2},\qquad
\Delta\omega=\omega_+-\omega_-.
$$

For positive coupling, compute the difference without subtracting close square roots:

$$
\Delta\omega=\frac{2\kappa/m}{\omega_++\omega_-}.
$$

At zero coupling the splitting is exactly zero. If both stiffnesses also vanish, use the free-motion limit directly rather than evaluating a $0/0$ expression.

## Reconstruct two physical displacements

The initial state is an equal sum of the two displacement patterns. Thus

$$
q_1(t)=\frac A2[\cos(\omega_-t)+\cos(\omega_+t)],\qquad
q_2(t)=\frac A2[\cos(\omega_-t)-\cos(\omega_+t)].
$$

Adding and subtracting these equations recovers one normal coordinate per frequency. The factors $A/2$ are fixed by the initial displacement. They are not fitted separately at each time.

The second mass can move even though it started at zero displacement. A normal-mode decomposition describes contributions to both masses; it does not assign one mode to each mass.

## Expose the fast carrier and slow envelope

The sum-to-product identities give

$$
q_1=A\cos(\bar\omega t)\cos(\Delta\omega t/2),\qquad
q_2=A\sin(\bar\omega t)\sin(\Delta\omega t/2).
$$

The signed envelope factors are $A\cos(\Delta\omega t/2)$ and $A\sin(\Delta\omega t/2)$. Their magnitudes describe the visible amplitude bounds. The fast carrier determines the actual position inside those bounds.

For close frequencies, the carrier completes many oscillations while the envelope changes slowly. When the frequencies are not close, these are still exact identities, but interpreting one factor as a slowly varying amplitude is less useful.

## Differentiate the envelope as well as the carrier

Write $C=\cos(\bar\omega t)$, $S=\sin(\bar\omega t)$, $c=\cos(\Delta\omega t/2)$ and $s=\sin(\Delta\omega t/2)$. Then

$$
v_1=-A[\bar\omega Sc+(\Delta\omega/2)Cs],\qquad
v_2=A[\bar\omega Cs+(\Delta\omega/2)Sc].
$$

Dropping the terms from differentiating the envelope can make a displacement picture appear correct while giving the wrong kinetic energy. It is a particularly visible error when an envelope reaches zero: the displacement may vanish while its envelope is changing.

## Identify envelope times without inventing a common state period

For $\Delta\omega>0$ and $A>0$, the first exchange of envelope magnitudes occurs at $t_{\rm swap}=\pi/\Delta\omega$. The magnitudes repeat after $2\pi/\Delta\omega$, while the signed envelopes repeat after $4\pi/\Delta\omega$.

These times are not generally a period of the entire position–velocity state. A common state period requires a suitable rational relation between the two modal frequencies. The carrier phase at an envelope swap must still be evaluated.

At zero coupling there is no finite exchange period. At $A=0$ there is no observed exchange, even though the system still has its intrinsic spectrum. The calculation keeps spectral envelope times separate from an observed nonzero-amplitude exchange.

## Keep the connecting spring in the energy ledger

The total mechanical energy is

$$
E=E_1+E_2+E_c,\quad
E_j=\frac12mv_j^2+\frac12kq_j^2,\quad
E_c=\frac12\kappa(q_2-q_1)^2.
$$

Initially $E=(k+\kappa)A^2/2$. Even though only the first mass was displaced, some initial energy belongs to the connecting spring. Adding only the two bare energies $E_1,E_2$ can therefore produce an apparent failure of conservation.

The exact modal energies remain constant. The changing physical-mass energies and changing coupling energy are a different decomposition of that same conserved total.

## State a local energy convention and derive its flux

To obtain two local energies that sum to the full energy, assign half the coupling energy to each side:

$$
E_1^*=E_1+\frac12E_c,\qquad E_2^*=E_2+\frac12E_c.
$$

Differentiate using the actual forces. The resulting signed flux is

$$
J_{1\to2}=\frac\kappa2(q_1-q_2)(v_1+v_2),\qquad
\frac{dE_1^*}{dt}=-J_{1\to2},\quad
\frac{dE_2^*}{dt}=J_{1\to2}.
$$

Positive flux means that this energy partition gains energy on side 2. Instantaneous flux can be negative, including early in an experiment initially displaced on side 1. A local partition is a stated convention; the total physical energy does not depend on that convention.

## Integrate the flux independently

For these initial conditions, $q_1-q_2=A\cos(\omega_+t)$ and $v_1+v_2=-A\omega_-\sin(\omega_-t)$. Therefore

$$
J_{1\to2}=-\frac{\kappa A^2\omega_-}{2}\cos(\omega_+t)\sin(\omega_-t).
$$

Define an integral that remains regular at zero frequency:

$$
I(\nu,t)=\int_0^t\sin(\nu u)\,du
=\frac{\nu t^2}{2}\operatorname{sinc}^2(\nu t/2).
$$

The signed net transfer is

$$
W_{1\to2}(t)=\frac{\kappa A^2\omega_-}{4}
[I(\Delta\omega,t)-I(\omega_-+\omega_+,t)].
$$

Compute this expression independently of endpoint energies, then check $E_2^*(t)-E_2^*(0)-W_{1\to2}(t)$. Defining transfer as the energy difference would force that residual to zero without testing the flux calculation.

## Worked example: an envelope swap with residual energy

Set $m=k=A=1$ in SI units and $\kappa=1/16$ N/m. Then $\omega_-=1$, $\omega_+=\sqrt{1.125}\approx1.060660$ rad/s and the first envelope swap occurs at approximately 51.790039 s.

At that instant, the exact state is approximately

| Quantity | Value |
| --- | --- |
| $q_1$ | 0 m |
| $q_2$ | 0.046223 m |
| $v_1$ | 0.030298 m/s |
| $v_2$ | −1.029229 m/s |
| bare $E_1$ | 0.000458974 J |
| coupling $E_c$ | 0.000066769 J |
| total $E$ | 0.531250 J |

The second envelope has magnitude 1 m, but the second displacement is near a carrier zero. The first displacement is zero while its velocity is nonzero. Most energy has moved to side 2, yet exact complete transfer cannot be inferred from the envelope alone.

## Worked example: a special exact displacement exchange

Choose $m=k=A=1$, $\kappa=1.5$. Now $\omega_-=1$ and $\omega_+=2$ rad/s. At $t=\pi$ s, $q=(0,-1)$ m and both velocities are zero. The first bare energy is exactly zero and the second bare energy is 0.5 J.

The coupling spring still stores 0.75 J, so total energy is 1.25 J. With the half-coupling convention, local energies change from $(0.875,0.375)$ J initially to $(0.375,0.875)$ J. Signed net transfer is 0.5 J. This particular frequency relation permits an exact bare-energy exchange; it does not establish a universal rule for every envelope swap.

## Compare a weak-coupling approximation over time

For $k>0$ and small $\kappa/k$, expand

$$
\omega_+\approx\omega_0+\frac{\kappa}{m\omega_0},\qquad
\omega_-=\omega_0=\sqrt{k/m}.
$$

Use these approximate frequencies to reconstruct both positions and velocities with the same initial conditions. The neglected frequency correction accumulates phase error with time. A small coupling ratio alone does not guarantee accurate long-time positions.

Report both the ratio $\kappa/k$ and an error over the stated observation window. At $k=0$ this expansion is unavailable, while the exact two-coordinate solution remains well defined.

## Treat uncoupled, zero-mode and zero-amplitude cases explicitly

If $\kappa=0$, mass 2 remains at its initial equilibrium and mass 1 oscillates freely when $k>0$. Net transfer is zero and there is no finite exchange period. If $k=0$ but coupling is positive, the center coordinate is a zero mode and the relative coordinate oscillates.

In the latter case with these initial conditions, the two half-coupling local energies are each $\kappa A^2/4$ and the flux is identically zero. Kinetic energy can grow on both masses while coupling energy falls. A changing displacement envelope does not imply energy flow under every local partition.

At $A=0$ all physical energies vanish. Energy fractions are unavailable because their denominator is zero. Intrinsic frequency information remains meaningful; it does not create a measured nonzero motion.

## Six exercises with explained solutions

### Exercise 1: Recover the initial state

Evaluate the two cosine formulas at $t=0$ and differentiate them once.

**Solution:** Both cosines equal one, giving $(q_1,q_2)=(A,0)$. Both sine derivatives vanish, so both initial velocities are zero. Choosing the modal coefficients without checking these four conditions can define a different experiment.

### Exercise 2: Count the initial energy

With $m=k=A=1$ and $\kappa=1/16$, what are $E_1,E_2,E_c$ initially?

**Solution:** They are 0.5, 0 and 0.03125 J. Total energy is 0.53125 J. The second side initially receives 0.015625 J under the half-coupling convention even though its bare energy is zero.

### Exercise 3: Differentiate at an envelope zero

At $\Delta\omega t/2=\pi/2$, does $q_1=0$ imply $v_1=0$?

**Solution:** No. The carrier derivative term vanishes with $c$, but the envelope derivative gives $v_1=-A(\Delta\omega/2)\cos(\bar\omega t)$. It vanishes only for an additional condition on the carrier phase or amplitude.

### Exercise 4: Interpret the sign of transfer

For small positive $t$, positive $k,\kappa,A$ and these initial data, what is the leading sign of $J_{1\to2}$?

**Solution:** Expand the exact flux to obtain $J_{1\to2}\approx-\kappa A^2\omega_-^2t/2<0$. Early local flow is toward side 1 under the half-coupling convention. The initial location of a displacement does not fix the sign of every later energy current.

### Exercise 5: Remove the anchoring springs

Set $k=0$, $\kappa>0$. Why can both kinetic energies change while $J_{1\to2}=0$?

**Solution:** The velocity sum is zero because the center of mass stays fixed. Thus the flux formula vanishes. Both masses receive energy released by the connecting spring, with equal local energies under the symmetric partition.

### Exercise 6: Assess an approximation fairly

A weak-coupling approximation looks accurate for 2 s. Does that verify it for 100 s?

**Solution:** No. Frequency error accumulates into phase error. Compare exact and approximate positions and velocities at the same times, report the coupling ratio and horizon, and inspect a full-window error measure. The exact initial conditions should be held fixed.

## Read the plots and connect them to AI validation

The plots separate physical positions from signed envelopes, bare and coupling energies from assigned local energies, instantaneous flux from its integral, and exact trajectories from the weak-coupling approximation. Mark the same selected time in each view. An envelope landmark outside the observation window should stay labeled outside; it should not move to the last drawn point.

The educational calculation uses quarter-kg masses, 1/16 N/m stiffness steps and amplitudes from 0 to 4 m. Observation windows reach 128 s; individual actual-time queries support 512 s. Analytic samples resolve the fastest phase by at most 0.125 rad per drawing interval. An envelope period is not a numerical step count.

A learned simulator can have a small position loss while misrepresenting velocity, coupling energy or signed flow. Test unseen coupling strengths, longer observation windows and the zero-mode boundary. Report the physical partition used for local energy; agreement with one arbitrary latent energy split is insufficient to establish physical energy transport.

## Primary sources and limits of the model

Normal-mode superposition and initial-value reconstruction follow [MIT 8.03SC chapter 3, sections 3.3.3–3.4.3](https://www.ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/29afe7e96aadefce1bd80486771aeae9_MIT8_03SCF16_Text_Ch3.pdf). The force formulation is also illustrated in [MIT 8.03SC lecture 4, pp. 1–5](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/b4f512207b1e991ffdfa8df5f519b6eb_MIT8_03SCF16_Lec4.pdf). The explicit energy partition, flux integral and counterexamples above are derived for the stated two-mass initial experiment. Damping, nonlinear coupling and many-body transport require further analysis. The next original topic develops the wave equation.
