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

## Research extension: identify the tensor without claiming to identify the object

### What information survives a rotational measurement?

Suppose a robot or learned simulator estimates a rigid object's inertia from motion. Recovering every entry of the matrix can feel like recovering the object. The matrix actually records a small set of spatial moments. A useful research question is therefore: which different objects produce exactly the same observations?

Keep the center of mass at the origin and use positive point masses. Define the mass second-moment matrix S. Its relation to inertia is

$$
S=\sum_i m_i r_i r_i^{\mathsf T},\qquad
I=\operatorname{tr}(S)\operatorname{Id}-S,\qquad
S=\frac{\operatorname{tr}(I)}{2}\operatorname{Id}-I.
$$

Taking a trace gives tr(I)=2tr(S), which proves the inverse formula. Knowing I determines S, but the individual positions and masses have already been summed out. This is a many-to-one compression of the mass distribution.

Physical realizability imposes more than symmetry and positive definiteness. For every direction n,

$$
n^{\mathsf T}Sn=\sum_i m_i(n\cdot r_i)^2\ge0,\qquad
\frac{I_j+I_k-I_i}{2}\ge0\quad(\{i,j,k\}=\{1,2,3\}).
$$

The second inequality follows by diagonalizing S; its eigenvalues are (I_j+I_k−I_i)/2. For example diag(1,1,3) is positive definite, but its inferred S is diag(3/2,3/2,−1/2). A negative squared-position moment is impossible. A positive-definite matrix output layer alone therefore does not ensure a physically realizable inertia tensor.

There is also a constructive converse for ideal point masses. Let M>0 be any chosen total mass and let the u_j be orthonormal eigenvectors of a positive-semidefinite S:

$$
S=\sum_{j=1}^{3}s_j u_j u_j^{\mathsf T},\quad s_j\ge0,
\qquad m_{j,+}=m_{j,-}=\frac{M}{6},\qquad
r_{j,\pm}=\pm\sqrt{\frac{3s_j}{M}}\,u_j.
$$

Each antipodal pair contributes s_j u_j u_jᵀ, all six masses sum to M, and their center of mass is zero. A zero s_j places that pair at the origin. This proves realizability in the ideal point-mass model. It does not specify a material, enclosing shape or manufacturable connecting structure.

### Two shapes that no inertia experiment distinguishes

In the following table each mass is the mass of **one** member of a ± pair; distances are in metres and masses in kilograms. The connecting rods remain ideal and massless.

| Shape | x pair: mass, position | y pair: mass, position | z pair: mass, position |
|---|---|---|---|
| A | 4, ±1/2 | 1, ±1 | 1, ±1 |
| B | 1, ±1 | 1, ±1 | 4, ±1/2 |

Direct sums give

$$
M_A=M_B=12\ \mathrm{kg},\quad S_A=S_B=2\operatorname{Id}\ \mathrm{kg\,m^2},
\quad I_A=I_B=4\operatorname{Id}\ \mathrm{kg\,m^2},
\qquad\sum_i m_i x_i^4=\begin{cases}1/2&\text{A},\\2&\text{B}\end{cases}\ \mathrm{kg\,m^4}.
$$

Both objects give L=4ω and K=2‖ω‖² with the stated SI coefficients. They remain indistinguishable under any common orientation. Because their total masses also agree, their tensors about any common displaced pivot agree by the parallel-axis theorem. Yet their fourth x moments differ. A density image or another spatial observation can distinguish them.

The two distributions can be related by a spatial rotation. That does not restore identifiability in a fixed laboratory frame: their isotropic inertia has lost that orientation information as well. This example concerns recovering a distribution in declared coordinates; it does not assert the two shapes are inequivalent under every possible equivalence relation.

![Different point distributions with the same mass and inertia](/assets/learning/classical-rotation-shapes.en.svg)

The drawing is an xz projection. Colored marker area is proportional to each point's mass. The hollow marker identifies the two y-axis points that overlap only in this projection. It does not add a seventh mass.

Reproduce the comparison in the actual experiment using the declared control order `[m1,m2,m3,2a,2b,2c,2s,10ωx,10ωy,10ωz,10d,orientation]`:

```text
A: [4,1,1,1,2,2,0,10,10,0,0,0]
B: [1,1,4,2,2,1,0,10,10,0,0,0]
```

Both produce K=4 J and L=(4,4,0) kg m²/s. Change control `10d` to 10 in both: O=(1,0,0), I_O=diag(4,16,16) kg m², K=10 J and COM translation energy=6 J. The distributions are still indistinguishable. This pivot comparison changes the imposed velocity field; it is a new physical rotation comparison, rather than a reference change for an unchanged motion.

### Six energy probes recover a matrix, not a distribution

Let ω_*>0 be a known angular-speed scale. Measure energy at three axis spins ω_*e_i and three pair spins ω_*(e_i+e_j). Expanding the quadratic form gives

$$
I_{ii}=\frac{2K(\omega_*e_i)}{\omega_*^2},\qquad
I_{ij}=\frac{K(\omega_*(e_i+e_j))-K(\omega_*e_i)-K(\omega_*e_j)}{\omega_*^2}.
$$

These six linear measurements span the six-dimensional space of symmetric matrices. Arbitrary choices of six directions need not work: their outer products ωωᵀ must span that space. A known diagonal or xz structure can reduce the number of unknowns, but that is additional prior information.

Use the native geometry `[1,1,1,2,4,6,1,10,10,0,0,0]` and vary only the three spin controls. It has

$$
I=\begin{pmatrix}26&0&-3\\0&41/2&0\\-3&0&21/2\end{pmatrix}\ \mathrm{kg\,m^2}.
$$

At ω_*=1 rad/s, the exact energies are:

| Spin direction | Energy (J) |
|---|---:|
| e₁ | 13 |
| e₂ | 41/4 |
| e₃ | 21/4 |
| e₁+e₂ | 93/4 |
| e₁+e₃ | 61/4 |
| e₂+e₃ | 31/2 |

For example K(e₁+e₃)−K(e₁)−K(e₃)=−3 in the stated units, recovering the negative cross term. Axis-only measurements would have missed it. This complete tensor recovery still cannot distinguish shapes A and B. With noisy observations, subtracting energies propagates measurement error; uncertainty, probe design and physical constraints must also be analyzed.

### Execute the independent point sums

This standard-library Python cell constructs both distributions, sums their moments using exact fractions, recomputes the shifted-pivot tensor and reconstructs the sheared matrix from six energies. It also rejects the proposed diag(1,1,3) tensor through its negative second moment.

```python
from fractions import Fraction as Q


def geometry(masses, lengths):
    points = []
    for j, (mass, length) in enumerate(zip(masses, lengths)):
        for sign in [-1, 1]:
            position = [Q(0)] * 3
            position[j] = sign * Q(length)
            points.append((Q(mass), position))
    return points


def second_moment(points):
    return [[sum(m * r[i] * r[j] for m, r in points)
             for j in range(3)] for i in range(3)]


def inertia(points):
    s = second_moment(points)
    trace = sum(s[j][j] for j in range(3))
    return [[(trace if i == j else 0) - s[i][j]
             for j in range(3)] for i in range(3)]


def energy(tensor, velocity):
    return sum(velocity[i] * tensor[i][j] * velocity[j]
               for i in range(3) for j in range(3)) / 2


def shape_checks():
    a = geometry([4, 1, 1], [Q(1, 2), 1, 1])
    b = geometry([1, 1, 4], [1, 1, Q(1, 2)])
    expected = [[Q(4 if i == j else 0) for j in range(3)]
                for i in range(3)]
    assert inertia(a) == inertia(b) == expected
    assert sum(m for m, _ in a) == sum(m for m, _ in b) == 12
    assert energy(expected, [1, 1, 0]) == 4
    fourth = [sum(m * r[0] ** 4 for m, r in p) for p in [a, b]]
    assert fourth == [Q(1, 2), Q(2)]
    # Recompute point sums about O=(1,0,0), not just a tensor formula.
    shifted = [[(m, [r[0] - 1, r[1], r[2]]) for m, r in p]
               for p in [a, b]]
    ip = [[Q(4), Q(0), Q(0)], [Q(0), Q(16), Q(0)],
          [Q(0), Q(0), Q(16)]]
    assert inertia(shifted[0]) == inertia(shifted[1]) == ip
    assert energy(ip, [1, 1, 0]) == 10

    # Six energies reconstruct a general symmetric tensor, not its geometry.
    original = [[Q(26), Q(0), Q(-3)], [Q(0), Q(41, 2), Q(0)],
                [Q(-3), Q(0), Q(21, 2)]]
    axes = [[Q(int(i == j)) for i in range(3)] for j in range(3)]
    diagonal_energy = [energy(original, w) for w in axes]
    recovered = [[Q(0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        recovered[i][i] = 2 * diagonal_energy[i]
        for j in range(i):
            w = [axes[i][k] + axes[j][k] for k in range(3)]
            recovered[i][j] = recovered[j][i] = (
                energy(original, w) - diagonal_energy[i] - diagonal_energy[j])
    assert recovered == original
    impossible = [Q(1), Q(1), Q(3)]
    s_diag = [sum(impossible) / 2 - x for x in impossible]
    assert s_diag == [Q(3, 2), Q(3, 2), Q(-1, 2)]
    print('Same mass/tensor, different fourth moments:', *fourth)
    print('COM/pivot energies (J): 4, 10; six-probe reconstruction passed')
    return a, b


shape_checks()
```

### Research questions with worked answers

1. **Can repeated rotated measurements distinguish A and B?** No. Their common isotropic tensor and total mass make the stated rigid inertia responses identical for every common spin, orientation and pivot. Rotating the measurement alone does not create a missing fourth-moment observation.
2. **Does physical realizability identify a unique shape?** No. The antipodal construction provides one realization; the two explicit native shapes show nonuniqueness even when total mass is supplied. Additional shape or material priors can restrict the answer but must be stated.
3. **Why use pair spins?** A pure axis probes only one diagonal entry. Pair spins expose cross terms through the energy difference. Matrix identifiability and distribution identifiability are separate questions.

For the underlying inertia definitions, see [MIT16.07 Lecture26](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf). The distributions, probe calculations and realizability construction above state their own assumptions and can be checked directly.

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
