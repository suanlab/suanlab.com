## A position constraint also restricts momentum

A particle constrained to a circle cannot carry an arbitrary normal momentum. Imposing the position equation alone does not produce a consistent phase-space state. To keep the position relation true during motion, momentum must also lie in the tangent direction.

This lesson derives the Dirac bracket for that pair of constraints. It compares the full canonical bracket with a rank-two constrained structure and separates three questions: Is the bracket regular here? Is the current state on the desired constraint surface? Does the selected Hamiltonian conserve an observable? These questions have different answers on deliberately invalid inputs.

The model computes exact instantaneous fields and structural identities. It does not integrate a numerical trajectory or claim that a reference-circle drawing is a simulated motion.

## State the phase space, units and constraints

Use canonical coordinates $z=(q_x,q_y,p_x,p_y)$, with q in m and p in kg m/s. The canonical matrix in this order is

$$
J=\begin{pmatrix}0&I\\-I&0\end{pmatrix},\qquad
\{f,g\}=\nabla f^T J\nabla g.
$$

For a target radius $R>0$, define

$$
\phi_1=\frac{q\cdot q-R^2}{2},\qquad \phi_2=q\cdot p.
$$

The first has units m² and the second Js. They are distinct conditions, not two measurements with one common numerical tolerance. The desired physical phase space is their simultaneous zero set. With the canonical free velocity $\dot q=p/m$, differentiating phi one gives $\dot\phi_1=\phi_2/m$, which motivates the momentum condition.

Mass uses an integer kg control. Radius, Cartesian position, momentum and external force codes use input/2 in m, kg m/s and N respectively. Target radius stays positive. Position zero remains selectable as an explicit singular state outside the physical circle.

## Construct and invert the constraint matrix

Calculate brackets before substituting the constraints. With $s=q\cdot q$,

$$
C_{ab}=\{\phi_a,\phi_b\}
 =\begin{pmatrix}0&s\\-s&0\end{pmatrix},\qquad
C^{-1}=\begin{pmatrix}0&-1/s\\1/s&0\end{pmatrix}.
$$

The inverse exists when q is nonzero. Its determinant is $s^2$ in m⁴. On the target circle, s equals $R^2$, so the constraints form a second-class pair. This invertible constraint-bracket matrix is different from the rank-two ambient Dirac tensor derived below.

Zero entries still carry the units dictated by their row and column quantities. For example C11 has units m² s/kg while C22 has units Js; the nonzero off-diagonal entries have units m². Inverse off-diagonal entries use 1/m². A single magnitude heat scale across these entries would mix unlike units.

## Define the corrected bracket from its inverse

The Dirac bracket is

$$
\{f,g\}_D=\{f,g\}-\{f,\phi_a\}(C^{-1})^{ab}\{\phi_b,g\}.
$$

Let $G$ be the two-by-four matrix of constraint gradients. The code forms the complete matrix directly as

$$
D=J-JG^T C^{-1}GJ.
$$

It does not guess individual component signs from a drawing. Multiplication gives $GD=0$, so both constraints have zero Dirac bracket with every smooth observable on the regular domain. They are Casimirs of this degenerate Poisson structure.

Antisymmetry alone is not sufficient for a Poisson bracket. The independent reference generator also differentiates the arbitrary-state matrix and verifies all four independent coordinate Jacobi triples before inserting any numerical state. No finite list of sample states is presented as a substitute for that symbolic check.

## Derive the blocks and check the momentum sign

The position-momentum block is the tangent projector

$$
P=I-\frac{qq^T}{s}.
$$

The complete elementary brackets are

$$
\{q_i,q_j\}_D=0,\qquad
\{q_i,p_j\}_D=P_{ij},\qquad
\{p_i,p_j\}_D=\frac{p_iq_j-q_ip_j}{s}.
$$

The q-q block uses s/kg, q-p and p-q blocks are dimensionless, and the p-p block uses kg/s. This is not a four-dimensional Euclidean metric. Rank is meaningful, but a norm which silently adds block magnitudes with different units is not a physical error measure.

Verify the sign at $q=(R,0)$ and $p=(0,P_t)$. The result is $\{p_x,p_y\}_D=-P_t/R$. For free motion this produces inward momentum rate $\dot p_x=-P_t^2/(mR)$. Reversing that bracket sign would push the particle outward and fail the differentiated constraints. The definition, coordinate ordering and this test fix the convention.

## Distinguish two different projectors

The two-dimensional P satisfies $P^2=P$, has rank one, and projects configuration vectors onto the tangent line. The phase-space projector is instead

$$
\Pi=I-JG^TC^{-1}G.
$$

It satisfies $\Pi^2=\Pi$, $G\Pi=0$ and $D=\Pi J\Pi^T$. Its rank is two. It is not generally an orthogonal projector under a Euclidean dot product of all four phase coordinates, whose position and momentum components have different units.

Likewise the Dirac tensor D has rank two as a four-by-four matrix on the regular ambient domain. After restriction to a regular two-dimensional constraint leaf, the induced structure is nondegenerate there. A singular ambient matrix is therefore compatible with a proper reduced symplectic phase space. Count the surface dimension rather than demanding a rank-four inverse for D.

## Derive the constrained Hamiltonian field

Take a constant Cartesian external force F and Hamiltonian

$$
H=\frac{p\cdot p}{2m}-F\cdot q.
$$

The corrected field is

$$
\dot q=\frac{Pp}{m},\qquad
\dot p=PF+\frac{p(q\cdot p)-q(p\cdot p)}{ms}.
$$

Both constraint rates vanish exactly. On the physical zero surface, $q\cdot p=0$, so velocity equals $p/m$ and

$$
\dot p=PF-\frac{p\cdot p}{mR^2}q.
$$

This includes tangent force and inward centripetal momentum rate. The ideal reaction is $\dot p-F$ and its power is zero because the physical velocity is tangent. The code exposes a physical reaction only when both constraints are satisfied. An off-surface field is mathematically defined but cannot be interpreted blindly as the intended particle's Newton equation with p equal to m times velocity.

## Compare the canonical free step at a valid state

Use $m=1$ kg, $R=1$ m, $q=(1,0)$ m, $p=(0,1)$ kg m/s and zero external force. This is an admissible phase-space state.

The canonical free field gives $\dot q=(0,1)$ m/s and $\dot p=(0,0)$ N. The first constraint has instantaneous rate zero, but the second has rate 1 J. The next change therefore leaves the circle's compatible momentum condition. Initial tangency alone is insufficient.

The Dirac field gives the same initial velocity and $\dot p=(-1,0)$ N. Both constraint rates vanish. The tensor has rank two, $\{q_x,p_x\}_D=0$, $\{q_y,p_y\}_D=1$ and $\{p_x,p_y\}_D=-1$ kg/s. The reaction is inward and has zero power.

These statements concern the exact continuous vector field. A generic finite Euler step can still leave the circle even when its input velocity is tangent. An integrator needs its own constraint and convergence analysis; the continuous bracket identity is not a numerical trajectory certificate.

## Do not confuse preservation with repairing initial data

Now keep q at $(1,0)$ but set $p=(1,1)$. Position is on the target circle, yet $\phi_2=1$ Js, so the phase-space state is invalid. The bracket remains regular and its constraint rates remain zero.

The Dirac velocity is $(0,1)$ m/s rather than $p/m=(1,1)$ m/s. The momentum rate is $(-1,1)$ N. The flow preserves the current nonzero value of phi two; it does not reset that value to zero. The code does not silently replace the input momentum by its tangent projection.

A position error is similar. At $q=(1/2,0)$ with target radius one, phi one is -3/8 m². The regular field preserves that wrong level. A Casimir identity describes motion within the present leaf. Correct initial data, a consistent initialization algorithm or a separately specified projection is still required to reach the desired zero leaf.

## Keep energy and angular momentum tests separate

Antisymmetry gives $\dot H=\nabla H^TD\nabla H=0$ on every regular leaf, including leaves that do not represent the intended physical circle. Energy conservation alone therefore does not certify admissible initial conditions.

For angular momentum $L=q_xp_y-q_yp_x$,

$$
\dot L=q_xF_y-q_yF_x.
$$

A zero external force gives global rotational conservation in this force family. A nonzero constant force generally breaks it, although the torque can vanish at the query point. With $q=(1,0)$, $p=(0,1)$ and $F=(0,1)$ N, angular-momentum rate is 1 N m while energy remains conserved. Constraint reaction still has zero power.

Thus correct constraints, a regular bracket, conservation of H and conservation of L are different checks. The force controls change the Hamiltonian, not the bracket construction. Conservation statements must name their Hamiltonian and their domain.

## Use a local angle without mistaking its branch cut for a singular bracket

A local angle differential has gradient $(-q_y/s,q_x/s)$ in configuration coordinates. Combining it with angular momentum gives the canonical local relation $\{\theta,L\}_D=1$.

This does not create one globally single-valued angle on the circle. The displayed principal atan2 angle has a cut on the negative x axis. At a nonzero point on that cut, C is still invertible and a different local angle chart still gives bracket one. The principal-chart derivative is unavailable there, so its bracket display is left undefined.

On the physical zero leaf, the local canonical pair has reduced Hamiltonian $H_{\mathrm{red}}=L^2/(2mR^2)-R(F_x\cos\theta+F_y\sin\theta)$. It gives $\dot\theta=L/(mR^2)$ and the same external torque as the ambient Dirac field. The code compares this reduced energy with the ambient H only for admissible states. A local canonical pair does not make the whole cylinder globally identical to a single coordinate plane.

At the origin, the issue is stronger: C itself is singular and no angle differential or Dirac inverse is defined. Distinguish a removable chart choice on a regular circle from a genuinely invalid inverse at q zero. Adding a tiny denominator would obscure both issues rather than resolve them.

## Interpret singular states and reference drawings honestly

The desired radius control is always positive, so the origin cannot lie on the physical surface. At q zero, the code returns no Dirac geometry, no inverse, no constrained field and no reduced leaf dimension. The ambient canonical field remains available because J is defined there.

The constraint-gradient rank at the origin can be one when momentum is nonzero and zero when momentum is also zero. A fixed nominal count of two constraints is not a substitute for checking regularity.

The reference circle is drawn with trigonometric samples solely to show the target position set. It is not an exact rational trajectory or a proof of constraint preservation. Structural ranks, zeros, forces and brackets use exact rational arithmetic, while the displayed principal angle and circle drawing are explicitly numerical geometry aids.

## Apply the structure checks to learned constrained models

A learned bracket should satisfy more than one good-looking trajectory. Check antisymmetry, constraint annihilation, rank on the regular domain and the Jacobi identity. Then check the selected Hamiltonian field, admissible initial data and the distinction between local angle charts and ambient coordinates.

Conversely, a network trained only to keep energy constant can preserve the wrong constraint leaf. A tangent velocity at one sample can miss a secondary momentum constraint. Reporting one aggregated matrix norm across mixed physical units can hide which identity actually failed.

This example is a second-class pair, not a local gauge redundancy. First-class constraints, gauge fixing, quantization and operator-ordering questions require additional constructions. The circle bracket is an entry point to those topics, not a claim that their full analysis has been implemented.

## Exercises with explained solutions

1. **A missing condition.** At the physical free state above, compare the canonical rates of the two constraints.

   **Solution:** $\dot\phi_1=q\cdot p/m=0$ but $\dot\phi_2=p^2/m=1$ J. Position tangency at this instant does not preserve the compatible momentum condition. The Dirac field makes both rates zero.

2. **Momentum bracket sign.** At $q=(R,0)$, $p=(0,P_t)$, obtain the normal momentum rate from the p-p bracket.

   **Solution:** $\{p_x,p_y\}_D=-P_t/R$. Multiplication by $\partial H/\partial p_y=P_t/m$ gives $-P_t^2/(mR)$ N, inward. The opposite sign would violate the circle consistency condition.

3. **An invalid momentum leaf.** For target radius one, $q=(1,0)$ and $p=(1,1)$, is a zero Dirac constraint rate a repair certificate?

   **Solution:** No. Phi two equals 1 Js and stays at that nonzero value. The velocity is $(0,1)$ m/s, which differs from p/m. The state lies on a regular wrong leaf, and no physical reaction force is reported for the intended zero surface.

4. **External torque and energy.** With the valid unit state and $F=(0,1)$ N, determine angular-momentum rate and reaction power.

   **Solution:** Torque is $q_xF_y-q_yF_x=1$ N m. The reaction remains $(-1,0)$ N while velocity is $(0,1)$ m/s, so its power is zero. Total H is conserved although L changes.

5. **A chart boundary.** At $q=(-1,0)$, is the Dirac bracket singular because the principal angle has a jump?

   **Solution:** No. $q^2=1$, so C is invertible and the reduced leaf has dimension two. A different local chart gives $\{\theta,L\}_D=1$. Only the derivative of the chosen principal branch is unavailable at its cut.

6. **A true singularity.** If q is zero but p is nonzero, can the target radius be used in the denominator to define a replacement bracket?

   **Solution:** No. C is determined by the actual q and has zero determinant. Replacing its s by the desired $R^2$ would not be its inverse and would generally lose the Casimir identities. The origin is outside the positive-radius constraint surface; initialization must be fixed rather than disguised.

## Reading and convention checks

Brian P. Dolan's [Constrained Dynamics in the Hamiltonian formalism, definition (12) and sections 3 and 5](https://arxiv.org/html/2112.12216v1) provides the constraint-inverse construction. Hagen Kleinert and Sergei V. Shabanov's [Proper Dirac Quantization of Free Particle on D-Dimensional Sphere, equations (5)–(9)](https://arxiv.org/pdf/quant-ph/9702006) gives a second reference for the bracket convention; this lesson uses only its classical structure. Component signs are derived here from the stated definition and verified by Casimir, Jacobi and inward-force checks. The quantum conclusions of that paper are not implemented or inferred from this classical example.
