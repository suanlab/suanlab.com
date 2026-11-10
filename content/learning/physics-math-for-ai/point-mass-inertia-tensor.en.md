## Why a rotating object needs a tensor

Angular velocity describes how a rigid object is moving at an instant. It does not by itself determine angular momentum: the positions of the masses also matter. Two objects can have the same mass and angular velocity but different rotational energies. Even for one object, angular momentum can point in a different direction from angular velocity.

Imagine a collection of small masses connected by ideal massless rods. A mass close to the selected rotation axis moves slowly; one far from the axis moves quickly. The inertia tensor packages all those distance effects into a linear map. Our first experiment evaluates that map at an instant. It does not yet solve the subsequent free motion or calculate a mechanism that maintains the imposed rotation.

## Declare the six-mass geometry

Use three antipodal pairs with positive masses m₁,m₂,m₃ on each member of its pair. Their positions in a reference frame are

$$
(\pm a,0,0),\qquad (0,\pm b,0),\qquad (s,0,c),\;(-s,0,-c).
$$

The lengths a,b,c are positive; the shear offset s can have either sign. Equal masses within each pair put the center of mass C at the origin. Total mass is M=2(m₁+m₂+m₃). Positive a,b,c make this family genuinely three-dimensional. A line of point masses can have a zero moment about that line, but that degenerate geometry is outside these positive-length controls.

Select a pivot O=(d,0,0) and angular velocity ω. All values are in SI units: positions in m, masses in kg and angular velocity in rad/s. The orientation control applies one common fixed rotation to every position and vector, rather than changing the body's shape.

## Start with compatible point velocities

Let ρᵢ=rᵢ−O be position relative to the selected pivot. The experiment imposes the instantaneous rigid velocity field

$$
v_i=\omega\times\rho_i.
$$

Every point shares the same angular velocity, but its linear velocity depends on its location. This compatibility is essential: assigning arbitrary independent velocities to the masses would generally deform the configuration and invalidate a rigid-body tensor description.

When d changes in this experiment, the physical velocity field changes as well. This is a comparison between rotations about different selected pivots. It is not merely a change of the observation origin for one unchanged motion. A pivot force, torque or support law that would sustain a chosen motion over time is not inferred from this instantaneous construction.

## Derive angular momentum by summing the masses

The angular momentum about O is

$$
L_O=\sum_i\rho_i\times m_i(\omega\times\rho_i)
=I_O\omega,
$$

$$
I_O=\sum_i m_i\left[(\rho_i\cdot\rho_i)\operatorname{Id}
-\rho_i\rho_i^\mathsf T\right].
$$

Here Id is the 3×3 identity matrix. The diagonal entry Ixx is the sum of m(y²+z²), the mass-weighted squared distance from the x axis. An offdiagonal entry such as Ixz is −Σmxz. Its sign records correlated geometry, not a negative amount of rotational resistance.

The same sum about C gives IC. For the displayed family, IC has only one potentially nonzero offdiagonal pair, Ixz=Izx=−2m₃sc. Changing shear can therefore rotate the principal directions even though the matrix stays symmetric and positive definite.

## Derive the kinetic-energy quadratic form

Sum the point kinetic energies using the same velocities:

$$
K_O=\frac12\sum_i m_i|\omega\times\rho_i|^2
=\frac12\omega^\mathsf T I_O\omega.
$$

For any vector u,

$$
u^\mathsf T I_Ou=\sum_i m_i|u\times\rho_i|^2\ge0.
$$

This proves positive semidefiniteness without inspecting the signs of individual entries. Our noncollinear three-dimensional family gives strict positivity for nonzero u. The tensor has units kg m², angular momentum has kg m²/s, and kinetic energy has J. A vector plot cannot use one unlabeled scale to compare ω and L as though their units matched.

A prescribed angular velocity along a unit direction n has axis inertia In=nᵀIOn. Then K=In|ω|²/2. When ω=0, the selected direction ω/|ω| is undefined; the experiment reports no queried axis inertia instead of dividing by zero. The tensor itself remains perfectly well defined.

## Separate center-of-mass translation and rotation

Since C=0, averaging the point velocities gives

$$
V_C=-\omega\times O,\qquad P=MV_C.
$$

The relative velocities about C remain ω×rᵢ. Thus

$$
K_O=\frac12\omega^\mathsf T I_C\omega
+\frac12M|V_C|^2.
$$

This is an actual energy decomposition of the selected velocity field. Moving the pivot generally adds center-of-mass translation. If ω is parallel to the displacement O, the cross product vanishes and that translation term is zero; a nonzero displacement does not automatically add energy for every spin direction.

## Apply the parallel-axis theorem from the center of mass

The two geometric tensors satisfy

$$
I_O=I_C+M\left[|O|^2\operatorname{Id}-OO^\mathsf T\right].
$$

The center-of-mass condition removes the first-moment cross terms in the expansion. This is why one may not blindly apply the same simple formula between two arbitrary noncentral reference points.

Angular momentum also satisfies the reference identity LO=LC−O×P. Here LC is computed from the relative center-of-mass motion. With our centered antipodal geometry it also agrees with the absolute momentum sum about C, because the first mass moment vanishes. Substituting VC=−ω×O reproduces LO=IOω and provides a separate consistency check.

## Understand principal axes and repeated moments

A real symmetric tensor has an orthonormal eigenbasis. Along a principal direction, angular momentum is parallel to angular velocity. Between principal directions, unequal eigenvalues generally tilt L away from ω. That tilt is not evidence that the tensor calculation failed.

The experiment computes the eigenvalues of the y block and symmetric xz block. Eigenvalues are invariant under a common rotation of coordinates. The sign of an eigenvector is conventional. Repeated eigenvalues allow an entire eigenspace of valid principal directions; an isotropic object does not select one unique set of three axes. The display flags this ambiguity rather than treating a chosen numerical basis as additional physical information.

The directional curve sweeps n=(cosθ,0,sinθ) in the reference xz plane. It is a slice of the full directional dependence, not a search over every direction on the sphere. A principal direction along y might therefore supply a global minimum or maximum that this particular curve does not attain.

## Transform all quantities in the same frame

For a fixed orthogonal rotation R,

$$
I_{\rm lab}=RI_OR^\mathsf T,\qquad
\omega_{\rm lab}=R\omega,\qquad L_{\rm lab}=RL_O.
$$

Positions, pivot and velocities transform by the same R. The scalar energy is unchanged. Rotating the tensor while leaving ω expressed in the old frame would mix coordinates and usually give the wrong energy and momentum.

A projected spatial diagram should be read as a projection, not as an exact ruler for three-dimensional lengths or angles. Physical calculations use the original three-dimensional coordinates. Numerical eigenvectors and rotated geometry are also distinct from the exact rational tensor entries and point sums.

## Worked examples and exercises with solutions

1. **Isotropic example.** Put one unit mass at each of the six positions ±x,±y,±z, with O=0 and ω=(1,1,0). **Solution:** IC=IO=4Id, L=(4,4,0) and K=4. Every direction is principal; a displayed eigenbasis is a choice rather than a unique geometric feature.

2. **Nonparallel momentum.** Use equal unit masses with a=1,b=2,c=3,s=0,O=0 and ω=(1,1,0). **Solution:** I=diag(26,20,10), L=(26,20,0) and K=23. The different x and y weights make L nonparallel to ω even though the tensor is diagonal.

3. **Shift the actual pivot.** Keep the previous geometry and spin, but select O=(1,0,0). **Solution:** IO=diag(26,26,16), VC=(0,0,1), P=(0,0,6) and LO=(26,26,0). K increases to26: rotational energy about C is23 and translational energy is3. This comparison changed the velocity field.

4. **A negative matrix entry.** In the second example set s=1/2 and keep c=3. **Solution:** Ixz=−3. All energy quadratic forms remain nonnegative. To see the cross term in K, give both ωx and ωz nonzero values; the sign of one entry alone cannot decide the sign of energy.

5. **Zero spin and a shifted pivot.** Set all angular-velocity components to zero with any allowed geometry and pivot. **Solution:** all point velocities, P,L and K vanish. The tensor and its eigenvalues remain defined, while ω/|ω|, the queried spin-axis inertia and the angle between ω and L are undefined.

6. **A common quarter-turn.** Rotate the coordinate orientation by π/2 about z in the second example. **Solution:** ωlab=(−1,1,0) and Llab=(−20,26,0). Their angle, the principal moments and K=23 are preserved. Transforming only one of these vectors would not describe the same physical state.

## Research connection and reading

Rigid-body learning and robotics require a model that respects frame covariance, mass geometry and the distinction between translation and spin. A network that predicts a plausible scalar moment can still fail to map vectors correctly, confuse a pivot change with an origin change, or invent a unique direction in an isotropic eigenspace. Point-sum energy, reference identities and transformed-vector checks expose these separate errors.

The next lesson adds actual torque-free time evolution. For background on three-dimensional angular momentum and inertia, see [MIT OpenCourseWare's rigid-body dynamics lecture material](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/resources/mit16_07f09_lec26/). The six-mass family, selected-pivot comparison and exact calculations here are the explicitly declared experiment above.
