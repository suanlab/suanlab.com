## 1. A conserved total does not describe the traffic

A bank account balance can stay fixed while transfers move money between accounts. Likewise, a scalar lattice can conserve total energy while one site loses energy and another gains it. The useful question is not only “is the sum constant?” but “which boundary currents explain each local change?”

Review [scalar-field action and boundary variation](/book/online/physics-math-for-ai/scalar-field-action-boundary-variation/). Keep its conservative, uniform-coefficient spatial model. Time remains continuous. We will calculate instantaneous rates, not advance a numerical trajectory.

## 2. Fix the model and the orientation first

Write $q_i$ for displacement, $v_i$ for velocity and $h$ for site mass as well as spacing in the chosen nondimensionalization. For an oriented internal edge $i\to j$, set

$$
\Delta q_{ij}=q_j-q_i,\qquad k=\frac{c^2}{h},\qquad E_{ij}=\frac{k}{2}(\Delta q_{ij})^2.
$$

The total energy is

$$
H=\sum_i\left[\frac h2v_i^2+hV(q_i)\right]+\sum_eE_e,
\qquad V(q)=\frac a2q^2+\frac b4q^4.
$$

A positive current points from the listed left site to the listed right site. The periodic wrap edge is $N-1\to0$. For fixed-zero boundaries, exterior endpoint values and velocities are zero and are not extra degrees of freedom. The zero-flux model has no exterior edges.

## 3. Where Noether's current enters

For a local Lagrangian, let an infinitesimal transformation be $\delta\phi=\epsilon X$ and suppose its Lagrangian variation is a divergence, $\delta\mathcal L=\epsilon\partial_\mu F^\mu$. The Euler–Lagrange identity then yields, on solutions,

$$
j^\mu=\frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}X-F^\mu,
\qquad \partial_\mu j^\mu=0.
$$

This is a local statement. Conservation of an integral additionally requires appropriate boundary fluxes. We use this framework below, then derive the finite-grid balances explicitly instead of assuming that every continuum symmetry survives discretization.

## 4. Three transformations give different quantities

For $\mathcal L=\phi_t^2/2-c^2\phi_x^2/2-V(\phi)$, define $\pi=\phi_t$. Time translation gives energy density and flux

$$
e=\frac12\phi_t^2+\frac{c^2}{2}\phi_x^2+V,
\qquad J_E=-c^2\phi_t\phi_x.
$$

Uniform field-value shifts, $\delta\phi=\epsilon$, are a symmetry when $V$ is constant and the boundaries permit them. Their density is $\pi$, with current $J_Q=-c^2\phi_x$. With a nonconstant potential the corresponding identity has a source:

$$
\partial_t\pi+\partial_xJ_Q=-V'(\phi).
$$

A spatial-label transformation with convention $\delta\phi=+\epsilon\phi_x$ instead has

$$
p=\pi\phi_x,\qquad J_P=-c^2\phi_x^2-\mathcal L,
\qquad \partial_tp+\partial_xJ_P=0.
$$

The sign convention matters: translating a profile to the right using $\phi(x-\epsilon)$ uses the opposite generator. Do not identify $\int\pi\,dx$ with $\int\pi\phi_x\,dx$. A symmetry of the action also does not automatically mean physically different field values are gauge-equivalent.

## 5. Read the finite-volume statement

If $\partial_t\rho+\partial_xJ=s$, a fixed interval satisfies

$$
\frac{d}{dt}\int_{x_L}^{x_R}\rho\,dx
=J(x_L)-J(x_R)+\int_{x_L}^{x_R}s\,dx.
$$

An outward flux is a loss for the interval. Periodicity can cancel two endpoint fluxes; a fixed wall can break a transformation even when its energy flux vanishes. “No external energy work” is not the same claim as conservation of every Noether charge.

## 6. How much edge energy belongs to a site?

The energy of an edge is shared by two sites. Choose $0\le\alpha\le1$: allocate fraction $\alpha$ to its listed left site and $1-\alpha$ to its right site. On a periodic chain,

$$
\mathcal E_i^{(\alpha)}=\frac h2v_i^2+hV(q_i)
+\alpha E_{i,i+1}+(1-\alpha)E_{i-1,i}.
$$

The sum is $H$ for every $\alpha$. The same $\alpha$ is used on internal edges and held constant during the time derivative. A time-dependent allocation would require additional $\dot\alpha E_e$ terms. This is a declared allocation of the finite model's energy, not a unique measured continuum density. In the fixed-zero model we allocate each exterior-edge energy fully to the adjacent active site. That boundary rule is separate from the internal-edge split.

## 7. Derive the matching energy current

The edge force on its left site is $k\Delta q_{ij}$ and

$$
\dot E_{ij}=k\Delta q_{ij}(v_j-v_i).
$$

The left site's contribution from work and its allocated edge energy is

$$
k\Delta q_{ij}v_i+\alpha\dot E_{ij}
=k\Delta q_{ij}[(1-\alpha)v_i+\alpha v_j].
$$

Therefore the consistent current is

$$
J_{ij}^{(\alpha)}=-k\Delta q_{ij}[(1-\alpha)v_i+\alpha v_j].
$$

It enters the left site's balance with a minus sign and the right site's with a plus sign. Keeping an old current while changing the energy allocation would create an artificial residual.

## 8. Work through three sites

Take the periodic state $h=c=1$, $a=b=0$, $q=(1,0,-1)$ and $v=(1,0,0)$. With the symmetric split $\alpha=1/2$:

| Quantity | Site or edge 0 | Site or edge 1 | Site or edge 2 |
|---|---:|---:|---:|
| Site energy $\mathcal E_i$ | $7/4$ | $1/2$ | $5/4$ |
| Oriented energy current $J_{01},J_{12},J_{20}$ | $1/2$ | $0$ | $-1$ |
| Site energy derivative | $-3/2$ | $1/2$ | $1$ |

Site zero loses $3/2$ units of energy per unit time: $1/2$ flows toward site one and one flows toward site two against the wrap-edge orientation. The energies sum to $7/2$ and the derivatives sum to zero. A zero current on edge $1\to2$ does not imply that both adjacent sites have constant energy.

## 9. Check a subregion, not just the whole ring

For the region containing sites zero and one, the energy is $9/4$ and its derivative is $-1$. The wrap-edge current entering the region is $-1$ and the outgoing current on edge $1\to2$ is zero. Thus inflow minus outflow is $-1$.

For the full ring there are no cut edges, so its derivative is zero. Adding site identities gives the balance of any chosen set; checking only the whole-ring sum could hide equal and opposite local mistakes.

## 10. Change the allocation without changing the dynamics

For the same state:

| Share $\alpha$ | Site energies | Oriented currents | Site energy derivatives |
|---|---|---|---|
| $0$ | $(5/2,1/2,1/2)$ | $(1,0,0)$ | $(-1,1,0)$ |
| $1/2$ | $(7/4,1/2,5/4)$ | $(1/2,0,-1)$ | $(-3/2,1/2,1)$ |
| $1$ | $(1,1/2,2)$ | $(0,0,-2)$ | $(-2,0,2)$ |

The difference between two currents is

$$
J_{ij}^{(\alpha)}-J_{ij}^{(\beta)}
=-(\alpha-\beta)\dot E_{ij}.
$$

No force, acceleration or total energy changed. This finite bookkeeping freedom should not be confused with a proof about all relativistic stress-tensor improvements. It does show why a learned local current cannot be identified from a conserved global sum alone.

## 11. Fixed boundaries store energy too

An exterior fixed-zero edge attached to $q_i$ stores $kq_i^2/2$. Its derivative is $kq_iv_i$, cancelling the wall-force power $-kq_iv_i$ in the adjacent active site's energy balance. Under our full-to-adjacent-site allocation, the exterior energy transport current is zero.

For $q=(1,0,-1)$, $v=(1,0,0)$ and the preceding coefficients with fixed-zero boundaries, the site energies are $(5/4,1/2,3/4)$ and their derivatives are $(-1/2,1/2,0)$. An omitted boundary-edge energy would give the wrong local balance even though the wall itself does no work.

## 12. A zero derivative at one instant is not a symmetry

Let $Q=h\sum_iv_i$ be the canonical charge for uniform field-value shifts. Its site charge $Q_i=hv_i$ has oriented current $C_{ij}=-k\Delta q_{ij}$ and source $s_i=-hV'(q_i)$. Summing gives

$$
\dot Q=\text{boundary inflow}-h\sum_iV'(q_i).
$$

In the fixed-zero example above, $\dot Q=0$ at the selected state but $\ddot Q=-1$. A uniform shift does not preserve the fixed endpoint values. With a nonconstant potential, even an exactly stationary equilibrium can have constant $Q$ without the action possessing uniform-shift symmetry. Distinguish a trajectory observation from a symmetry valid for all allowed states.

## 13. What a periodic spatial stencil does conserve

On a periodic grid define

$$
(Dq)_i=\frac{q_{i+1}-q_{i-1}}{2h},\qquad P_h=h\sum_iv_i(Dq)_i.
$$

This is not $Q$. In the first three-site state, $Q=1$ but $P_h=1/2$. The matrix $D$ is skew-symmetric. When $b=0$, the uniform quadratic stiffness commutes with $D$, and $\dot P_h=0$ exactly. The associated finite-dimensional symmetry is generated by $\delta q=\epsilon Dq$, $\delta v=\epsilon Dv$.

Its exponential is not automatically a literal spatial translation of arbitrary lattice data. Negative quadratic stiffness can still possess this symmetry; a conserved quantity alone does not establish stability.

## 14. A cyclic symmetry with a nonzero stencil defect

For the quartic model,

$$
\dot P_h=-hb\sum_iq_i^3(Dq)_i.
$$

Choose $N=3$, $h=c=b=1$, $a=0$, $q=(1,2,0)$ and $v=(0,0,0)$. Then $Dq=(1,-1/2,-1/2)$, the acceleration is $(-1,-11,3)$, and $\dot P_h=3$ even though $P_h=0$.

The cyclic permutation $q\mapsto(2,0,1)$ still preserves energy $29/4$. A finite cyclic group is not a continuous spatial-translation group. The continuum identity $\phi^3\phi_x=\partial_x(\phi^4)/4$ has no automatic exact counterpart for this centered difference. Grid-refinement claims require a smooth family of fields and consistency assumptions; shrinking $h$ for the same three arbitrary values does not supply them.

## 15. Use the balances in AI research

A global energy penalty can miss local transport errors. A local residual can still be meaningless if predicted energy, flux, orientation and boundary allocation use different conventions. Specify the representation before comparing models.

When data include forcing, damping or time-dependent parameters, add their source terms before calling a residual a conservation failure. When a discretization breaks a continuum symmetry, report its defect rather than forcing a false exact invariant. Separate continuum identity, finite-model identity and time-integrator error.

## 16. Explained checks and further reading

**Can a site lose energy while total energy stays fixed?** Yes: in the symmetric three-site example site zero loses $3/2$, while the others gain $1/2$ and one.

**Does $\alpha$ change the trajectory?** No. It changes energy allocation and its matching current; the same Hamiltonian generates the same forces.

**Does zero wall work preserve every charge?** No. The fixed-zero example has zero exterior energy current but lacks uniform-field-shift symmetry.

**Does cyclic permutation symmetry force $\dot P_h=0$?** No. The quartic counterexample gives three; the central stencil is not an exact continuous spatial derivative.

**Is a field-value shift a gauge identification?** Not merely because it preserves an action. Gauge equivalence requires a separate specification of redundant descriptions and observables.

For the general current/divergence framework, boundary integrals and energy-momentum tensor, see [David Tong, Classical Field Theory, sections 1.3.1–1.3.2](https://davidtong.org/teaching/quantum-field-theory/qfthtml/S1). The allocated lattice energies, finite differences and numerical counterexamples above are explicit calculations for this textbook's stated model.
