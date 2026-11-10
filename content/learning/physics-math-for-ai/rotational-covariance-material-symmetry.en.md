## Changing a description is different from rotating an experiment

Imagine a spring system that is softer along one material axis than another. We can describe the same system using a rotated pair of coordinate axes. We can also rotate a displaced particle while leaving the spring system fixed. The first operation changes the description; the second changes the physical state relative to its surroundings.

Both involve rotation matrices, but they ask different questions. Coordinate covariance requires transforming the spring coefficients along with the state. A rotational symmetry asks whether the fixed model gives correspondingly rotated predictions without changing those material coefficients.

This lesson uses a two-dimensional quadratic potential, a constant positive scalar mass and fixed positive stiffnesses. It examines a constitutive force law at a state, rather than presenting a scatter diagram as an integrated trajectory. We will also derive the instantaneous torque and state exactly when angular-momentum conservation follows for this mechanical model.

## Start with the material model

Let q=(x,y) be a displacement in meters. Define

$$
V(q)=\frac12q^T Kq,
\qquad K=\begin{pmatrix}k_x&0\\0&k_y\end{pmatrix},
\qquad F(q)=-Kq.
$$

The stiffness coefficients have units N/m and the potential has units joules. Positive kx and ky make constant positive-potential contours ellipses. When the two stiffnesses agree, those ellipses become circles.

The matrix describes the material axes in these coordinates. An anisotropic matrix is not merely an inconvenient notation: it expresses a physical difference between directions. A coordinate change must retain that difference in its transformed components.

## Fix a rotation convention

Use a proper planar rotation

$$
R=\begin{pmatrix}c&-s\\s&c\end{pmatrix},
\qquad c^2+s^2=1.
$$

For the coordinate-description comparison, define the new components by q'=Rq and v'=Rv. This equation fixes the convention. A geometrically described rotation of the coordinate basis itself would use the inverse orientation; stating the component relation avoids a hidden sign ambiguity.

The relation R transpose R=I preserves Euclidean lengths and angles. Consequently, a scalar-mass kinetic energy m v transpose v/2 is unchanged under this spatial rotation of components. This spatial statement differs from the velocity shift of a Galilean observer, which need not preserve kinetic energy.

## Transform the stiffness as well as the state

To represent the same potential in the primed components, substitute q=R transpose q'. The correct coefficient matrix is

$$
K'=RKR^T.
$$

Writing its components explicitly gives

$$
K'_{11}=k_xc^2+k_ys^2,\quad
K'_{22}=k_xs^2+k_yc^2,\quad
K'_{12}=K'_{21}=(k_x-k_y)cs.
$$

An originally diagonal stiffness generally develops off-diagonal components. Those terms are required to express the same material axes in the new coordinates; dropping them changes the model.

For example, take kx=1 N/m, ky=2 N/m and c=3/5,s=4/5. Then K'11=41/25 N/m, K'22=34/25 N/m and K'12=-12/25 N/m. These are exact rational components. The stiffness eigenvalues remain1 and2, so the directional difference has not disappeared.

## Check coordinate covariance directly

The correctly transformed force is

$$
F'(q')=-K'q'=-RKR^TRq=R[-Kq]=RF(q).
$$

The orthogonality identity cancels the neighboring R transpose R. The potential also agrees:

$$
\frac12q'^T K'q'=\frac12q^T Kq.
$$

Together with the rotated velocity, this preserves the numerical total energy in this spatial coordinate comparison. These equalities hold for anisotropic as well as isotropic K. They establish a consistent description of the same model, rather than proving that the fixed material model has every possible rotational symmetry.

A practical check should compare the transformed force with R times the original force. Comparing component arrays directly without the rotation generally asks the wrong equality question.

## Ask the separate active-symmetry question

Now rotate the particle's state while holding K fixed. The force evaluated at the rotated state is F(Rq)=-KRq. An equivariant force response under that rotation would require

$$
F(Rq)=RF(q)\quad\text{for every }q.
$$

This is equivalent to KR=RK. The queried-point defect is F(Rq)-RF(q)=(RK-KR)q, but global model symmetry requires the matrix commutator to vanish, not only its product with one selected q.

For a diagonal two-dimensional K, a rotation with nonzero s commutes with K precisely when kx=ky. The identity and the half-turn R=-I commute even with anisotropic K. Thus symmetry under one selected rotation can be weaker than symmetry under the whole continuous planar rotation group.

The interface should separately report the queried force coincidence, symmetry for the selected rotation at every state, and symmetry under every planar rotation. Combining those flags would hide the main conceptual distinction.

## Work a force counterexample

Take kx=1 N/m, ky=2 N/m and q=(1,1) m. The original force is(-1,-2) N. A quarter-turn gives Rq=(-1,1) m and rotates the original force to RF=(2,-1) N.

If the material matrix stays fixed, its force at Rq is(1,-2) N. This differs from RF by(-1,-1) N, so the fixed model lacks quarter-turn symmetry. If instead the coefficients are transformed to K'=diag(2,1), the transformed force is(2,-1) N and coordinate covariance holds exactly.

In this particular state, the original and actively rotated potential energies are both1.5 J. Their coincidence does not rescue the failed force symmetry. One scalar output can match accidentally while the constitutive vector law differs. It is useful to test quantities that distinguish the claimed property, rather than selecting a convenient matching readout.

## Connect torque to angular momentum under the stated dynamics

Suppose the actual dynamical model is m q double-dot=-Kq with constant scalar mass. Angular momentum about the fixed origin is L=m(x vy-y vx). Differentiating and using the equation of motion gives

$$
\dot L=xF_y-yF_x=(k_x-k_y)xy.
$$

When kx=ky, the force is central and this torque vanishes at every state, so L is constant along every solution of this model. This is a dynamical conclusion supported by a derivative identity and the full constitutive assumption.

When kx differs from ky, a state with x=0 or y=0 still has zero instantaneous torque. That does not establish conserved angular momentum for arbitrary states or an entire future motion. A special trajectory confined to a material axis can have special behavior, but its existence is not global continuous rotational symmetry.

## A discrete half-turn does not supply the continuous conclusion

An anisotropic quadratic potential is unchanged by q going to-q. Its force also satisfies F(-q)=-F(q), so the half-turn is a genuine symmetry of the fixed model.

Nevertheless, with kx=1,ky=2 and q=(1,1), the torque is-1 N*m. Angular momentum changes at that instant under the specified dynamics. Symmetry under a single half-turn therefore does not imply the angular-momentum result associated with continuous rotations.

Noether's theorem connects appropriate continuous symmetries of an action to conserved quantities under its equations of motion. This example is an accessible place to see why the word continuous matters. We can derive the torque result directly here without treating every equality or discrete transformation as a sufficient conservation theorem.

## Interpret the diagrams and exact checks

A position-plane diagram may show a constant-potential ellipse and the selected position. A separate force-plane diagram can compare F(Rq) with RF(q). Positions have units meters; force vectors have units newtons. They should not be placed on one unlabeled numerical axis as if they were the same kind of quantity.

A transformed ellipse represents the same material model in another component convention. An ellipse retained while only the state is rotated represents the active fixed-model comparison. Captions must state which operation each panel performs. None of these static diagrams by itself supplies a time trajectory.

The laboratory can use rational rotations such as quarter turns and the3/5,4/5 pair. For the supported rational inputs, transformed coefficients, force defects, energy identities and torque can then be checked exactly with fractions. Decimal coordinates of sampled ellipses are illustrative numerical values, separate from the exact constitutive flags.

## Implications for equivariant learning

A learned force law can be tested for equivariance only after specifying which contextual quantities transform. If the physical environment includes an anisotropic stiffness tensor, rotating both the state and tensor is a coordinate-covariance test. Rotating only the state asks whether that fixed environment itself has the corresponding symmetry.

Enforcing every rotation as a symmetry while withholding the material orientation can erase real directional behavior. Conversely, a model can correctly accept an anisotropic tensor and transform its outputs consistently without claiming isotropic physics.

A useful evaluation reports context transformations, matrix-level or domain-level claims, and queried-point errors separately. A zero error at q=0 is especially weak: both linear forces vanish there even for a model lacking the tested global symmetry.

## Exercises with worked solutions

1. **Exercise:** What coefficient transformation represents the same model when q'=Rq? **Solution:** K'=RKR transpose. The resulting off-diagonal entries generally matter.
2. **Exercise:** Does coordinate covariance require kx=ky? **Solution:** No. Transforming K consistently gives force and energy covariance for any positive stiffnesses in this model.
3. **Exercise:** For kx=1,ky=2 and q=(1,1), compare a quarter-turn's two force predictions. **Solution:** F(Rq)=(1,-2) N and RF(q)=(2,-1) N. Their difference is(-1,-1) N.
4. **Exercise:** Does equal potential energy at that original and rotated state prove force equivariance? **Solution:** No. Both potentials happen to be1.5 J, while the vector predictions differ.
5. **Exercise:** An anisotropic model is symmetric under a half-turn. Must angular momentum be conserved at every state? **Solution:** No. Its torque is(kx-ky)xy, nonzero at a generic state despite that discrete symmetry.
6. **Exercise:** At q=0 the queried force defect is zero. What has been established? **Solution:** Only a coincidence at that point. The matrix commutator can still be nonzero, ruling out global symmetry under the selected rotation.

## Sources and scope

[David Tong's Classical Dynamics, section2](https://davidtong.org/teaching/classical-dynamics/dynhtml/S2) discusses coordinate changes and continuous symmetries. The quadratic stiffness model, tensor components, force counterexamples and torque calculations here are direct derivations. The lesson does not claim that a single sampled equality establishes a general conservation law.
