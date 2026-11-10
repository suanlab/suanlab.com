## Two moving objects can contain two simpler motions

Connect two masses with a spring and displace only one. Both objects generally move, and neither displacement is a single sinusoid. Nevertheless, suitable combinations of their coordinates can evolve as independent oscillators. Those combinations are normal coordinates; the associated physical displacement patterns are normal modes.

The useful question is more than “what are the eigenvalues?” We want to turn measured initial positions and velocities into mode amplitudes, evolve those amplitudes, reconstruct actual motion and verify energy. When masses differ, the mass matrix determines the appropriate geometry. Ordinary Euclidean projection can give incorrect initial amplitudes.

## Specify the physical system and units

Let $q_1,q_2$ denote displacement from equilibrium. Masses $m_1,m_2>0$ have units kg. Each mass has an anchoring spring, with stiffnesses $k_1,k_2\ge0$ in N/m. A coupling spring has stiffness $\kappa\ge0$ in N/m. Position is in m, velocity in m/s and energy in J. Use

$$
M=\begin{pmatrix}m_1&0\\0&m_2\end{pmatrix},\qquad
K=\begin{pmatrix}k_1+\kappa&-\kappa\\-\kappa&k_2+\kappa\end{pmatrix},\qquad
M\ddot q+Kq=0.
$$

There is no damping or external drive. The spring law is linear; the coordinates can also describe a small-oscillation approximation to another mechanical system. Nonlinear large-angle pendulums require additional terms.

## Derive the coupling signs from forces

If $q_2>q_1$, the coupling spring pulls mass 1 toward increasing displacement and mass 2 toward decreasing displacement. Therefore

$$
m_1\ddot q_1=-k_1q_1+\kappa(q_2-q_1),\qquad
m_2\ddot q_2=-k_2q_2-\kappa(q_2-q_1).
$$

The coupling forces cancel in the total-force sum. Anchoring forces remain, so total momentum generally is not conserved. When both anchoring stiffnesses vanish, the center of mass has zero acceleration. These are consequences of the chosen physical boundary conditions.

## Why the ordinary eigenproblem needs care

Substituting $q=\phi\cos(\omega t)$ gives the generalized eigenproblem

$$
K\phi=\omega^2M\phi.
$$

The matrix $M^{-1}K$ is generally not symmetric when the masses differ. Applying symmetric-matrix reasoning directly to its Euclidean eigenvectors is unsafe. The physical system still has real nonnegative squared frequencies because $M$ is positive definite and $K$ is positive semidefinite.

In particular, the stiffness energy is $(k_1q_1^2+k_2q_2^2+\kappa(q_2-q_1)^2)/2\ge0$. The conclusion uses this physical assumption, rather than claiming that every real matrix has real eigenvalues.

## Whiten the mass geometry

Set $y=M^{1/2}q$. Multiplying the motion equation by $M^{-1/2}$ gives

$$
\ddot y+Dy=0,\qquad D=M^{-1/2}KM^{-1/2}.
$$

The matrix $D$ is symmetric. Choose orthonormal eigenvectors $U$ so that $U^TU=I$ and $U^TDU=\Lambda$. Define $\Phi=M^{-1/2}U$. Its columns are physical mode shapes and satisfy

$$
\Phi^TM\Phi=I,\qquad \Phi^TK\Phi=\Lambda.
$$

Mass normalization gives mode vectors units kg⁻¹ᐟ² and normal coordinates units kg¹ᐟ² m. Setting a matrix numerically equal to the identity does not make every underlying physical coordinate dimensionless.

## Mode shape, sign and normal coordinate are different objects

A column $\phi_j$ gives relative physical displacements. Its components can have opposite signs: a single undamped mode can move the masses in opposite directions. Multiplying a mode shape and its coordinate by $-1$ leaves physical motion unchanged.

Normal coordinates are

$$
a=\Phi^TMq,\qquad q=\Phi a,\qquad \ddot a+\Lambda a=0.
$$

The transformation can be read as a concept diagram:

$$
(q_0,v_0)\ \xrightarrow{\ \Phi^TM\ }\ (a_0,\dot a_0)
\ \xrightarrow{\ \text{independent evolution}\ }\ (a(t),\dot a(t))
\ \xrightarrow{\ \Phi\ }\ (q(t),v(t)).
$$

## Project both position and velocity

Initial positions alone do not specify a second-order system. Compute

$$
a_j(0)=\phi_j^TMq_0,\qquad \dot a_j(0)=\phi_j^TMv_0.
$$

The mass factors are essential for unequal masses. An equivalent physical-space projector is $P_j=\phi_j\phi_j^TM$. For simple modes, $P_j^2=P_j$, $P_1P_2=0$ and $P_1+P_2=I$. These projectors need not be symmetric in ordinary coordinates; they are self-adjoint in the mass metric: $MP_j=P_j^TM$.

This provides a useful check that a coordinate transformation has been applied in both directions consistently.

## Evolve oscillatory and zero-frequency coordinates

For $\lambda_j=\omega_j^2>0$,

$$
a_j(t)=a_j(0)\cos(\omega_jt)+\dot a_j(0)t\operatorname{sinc}(\omega_jt),
\qquad \operatorname{sinc}(z)=\frac{\sin z}{z},\quad\operatorname{sinc}(0)=1.
$$

This expression is continuous at zero frequency. At $\lambda_j=0$ the coordinate is $a_j(0)+\dot a_j(0)t$, with constant velocity. A zero mode can therefore describe drift. Treating it as a constant would discard part of the initial data; replacing its frequency by a small positive number would invent a restoring force.

The implementation identifies structural zeros from nonnegative stiffness products. It does not classify a small positive eigenvalue as zero using a tolerance.

## Reconstruct the actual trajectory

Add each mode's position and velocity contributions:

$$
q(t)=\sum_j\phi_ja_j(t),\qquad v(t)=\sum_j\phi_j\dot a_j(t).
$$

The contribution from a mode is a vector, not the trajectory of one physical mass. Both modes can contribute to both masses. The normal-coordinate rates must be differentiated correctly before reconstructing velocity.

The physical diagram shows the two masses and their coupling, displacement signs and the selected time. Read actual positions, normal coordinates and reconstruction separately in the coordinate plots, including their different units.

## Energy separates in normal coordinates

The physical energy is

$$
E=\frac12v^TMv+\frac12q^TKq
=\frac12\sum_j\left[\dot a_j^2+\lambda_ja_j^2\right].
$$

Each modal energy is constant for this linear undamped system. Energy assigned to a particular physical mass can still vary; that is a different partition. Include the coupling spring energy when computing the physical total.

For a zero mode, only its kinetic term remains. A displaced static zero mode can have zero energy and nonzero displacement. If total energy is zero, normalized energy fractions are unavailable; assigning equal fractions would add information the experiment does not contain.

## Repeated frequencies do not select a unique basis

If the coupling vanishes and $k_1/m_1=k_2/m_2$, both frequencies are equal. Any orthonormal rotation within the whitened repeated eigenspace gives another valid mode basis. After mass normalization, reconstruction gives the same actual initial-value solution.

Individual modal energy entries can change with this basis choice. Their sum remains the physical total. When comparing numerical libraries, compare the repeated subspace and reconstructed motion, rather than demanding the same signed eigenvector or the same per-mode energy labels. The educational calculation chooses a reproducible coordinate basis in this case and labels it as a choice.

## Worked example with unequal masses

Take $m_1=1$ kg, $m_2=2$ kg, $k_1=k_2=\kappa=1$ N/m. The characteristic equation is

$$
\det(K-\lambda M)=(2-\lambda)(2-2\lambda)-1
=2\lambda^2-6\lambda+3=0.
$$

Hence $\lambda_\mp=(3\mp\sqrt3)/2$. The angular frequencies are approximately 0.796225 and 1.538189 rad/s. From the first row, the displacement ratio is $\phi_2/\phi_1=2-\lambda$: approximately 1.366025 for the lower mode and −0.366025 for the upper mode. The masses move together in one shape and oppositely in the other.

Starting with $q_0=(1,0)$ m and $v_0=0$, total energy is 1 J: 0.5 J in the first anchoring spring and 0.5 J in the coupling spring. The mass-normalized modal energies are $(2-\sqrt3)/4\approx0.066987$ J and $(2+\sqrt3)/4\approx0.933013$ J. Setting the second mass initially at equilibrium does not eliminate its later motion.

## Worked example with translation

Set $k_1=k_2=0$, $m_1=1$, $m_2=2$, $\kappa=1$ in SI units. One squared frequency is zero, and the other is $\kappa(1/m_1+1/m_2)=1.5$ s⁻². Starting at $q_0=(1,1)$ m and $v_0=(0.5,0.5)$ m/s excites only translation.

At 12 s both displacements are 7 m and both velocities remain 0.5 m/s. The coupling spring stays unextended, and energy is $(m_1+m_2)(0.5)^2/2=0.375$ J. A zero frequency does not imply zero velocity or zero kinetic energy.

## Six exercises with explained solutions

### Exercise 1: Build the force matrix

Let $k_1=2$, $k_2=0$, $\kappa=1$ N/m. Write $K$ and the force at $q=(1,0)$ m.

**Solution:** $K=\begin{pmatrix}3&-1\\-1&1\end{pmatrix}$ N/m. Thus $-Kq=(-3,1)$ N. The coupling contributes equal and opposite forces, while the first anchoring spring contributes −2 N.

### Exercise 2: Choose the projection geometry

For unequal masses, should a mode coefficient be computed as $\phi_j^Tq_0$?

**Solution:** For the mass-normalized basis used here it is $\phi_j^TMq_0$. The inverse relation follows from $\Phi^TM\Phi=I$. Omitting $M$ generally prevents correct reconstruction of the measured initial displacement.

### Exercise 3: Handle a zero mode

A normal coordinate has $\lambda=0$, $a(0)=2$ and $\dot a(0)=0.5$ in its stated units. Find its value at 4 s.

**Solution:** The normal equation is $\ddot a=0$, so $a(4)=2+0.5\times4=4$. Its kinetic modal energy is $0.5(0.5)^2=0.125$ J. The units of the normal coordinate include the square root of mass.

### Exercise 4: Count constants

How many scalar initial values are needed for two coupled masses?

**Solution:** Four: two positions and two velocities. The mode transformation produces two initial normal coordinates and two initial rates. Knowing only two mode amplitudes with an assumed common phase would describe a restricted family of motions.

### Exercise 5: Compare repeated modes

Two solvers return different mode vectors and different individual modal energies at a repeated frequency. Is one necessarily wrong?

**Solution:** No. A rotation in the repeated eigenspace can redistribute energy labels. Check mass orthogonality, the eigenvalue equation, reconstructed initial data, actual trajectories and total energy. A basis convention is needed before comparing individual entries.

### Exercise 6: Account for the coupling spring

In the unequal-mass example, why is the initial energy not 0.5 J?

**Solution:** The first anchoring spring stores 0.5 J and the coupling spring stores another 0.5 J because $q_1-q_2=1$ m. The total is 1 J. Dropping the coupling term would make subsequent apparent energy changes a bookkeeping error.

## Read numerical evidence and connect it to AI

The calculation supports masses from 0.25 to 4 kg in 0.25 kg increments and stiffnesses from 0 to 4 N/m in 1/16 N/m increments. Initial positions and velocities lie between −4 and 4 in SI units. Observation duration is at most 128 s; an actual point query can be evaluated up to 512 s. These are explicit experiment bounds; the theory above is more general.

Drawing intervals adapt so that the fastest modal phase advances by at most 0.125 rad per interval. They sample analytic normal evolution rather than integrate a time-stepping method. Floating-point eigensystem and energy residuals are reported as diagnostics.

A learned dynamics model can be tested on unseen masses, coupling strengths and initial data. Validate physical reconstruction, forces and energy, not just a latent-coordinate plot. A learned mode label may rotate inside a repeated eigenspace. Directly comparing latent coordinates can therefore punish an equivalent physical representation.

## Primary sources and continuation

For the matrix formulation and translation mode, see [MIT 8.03SC lecture 4, pp. 1–5](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/b4f512207b1e991ffdfa8df5f519b6eb_MIT8_03SCF16_Lec4.pdf). Mass-weighted normal coordinates and repeated eigenspaces are discussed in [MIT 8.03SC chapter 3, sections 3.3.3–3.4.3](https://www.ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/29afe7e96aadefce1bd80486771aeae9_MIT8_03SCF16_Text_Ch3.pdf). The next chapter examines beats and energy exchange between physical masses, including the coupling spring and the limits of an envelope picture.
