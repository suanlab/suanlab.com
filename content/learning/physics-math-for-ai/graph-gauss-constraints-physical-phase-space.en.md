## 1. A gauge orbit is not yet a physical phase space

The previous chapter compared link configurations under allowed changes of local frame. It did not introduce a momentum for each link or impose a dynamical constraint. Those missing ingredients matter: removing a redundant coordinate is different from requiring an electric field to satisfy Gauss's law.

Think of a pendulum. Its angle is a configuration; its angle and angular momentum specify a phase-space point. Our graph replaces one pendulum by a circular coordinate on every edge. Local frame choices introduce redundancy, while prescribed charges restrict which momentum configurations are allowed.

This chapter constructs a finite **classical** model. It has compact U(1) link coordinates, real electric momenta, fixed interior charges and explicitly declared boundary anchors. It has no dynamical matter, quantum electric spectrum, continuum limit or numerical time integration. A graph drawing is an abstract layout. These limits let us inspect the constraint and its reduction exactly.

## 2. Circular coordinates and their conjugate momenta

For each stored edge e, introduce a phase and a real momentum:

$$
U_e=e^{iA_e},\qquad A_e\in\mathbb R/(2\pi\mathbb Z),\qquad E_e\in\mathbb R.
$$

The unreduced phase space is the cotangent bundle of a product of circles. On a local angular chart, our Poisson convention is

$$
\{A_e,E_f\}=\delta_{ef},\qquad \{A_e,A_f\}=\{E_e,E_f\}=0.
$$

The dimension is twice the number of edges. The global quantity is U, not one globally continuous real-valued angle. Crossing an angle branch cut does not make the physical link discontinuous. Globally one can write the corresponding relation as `{Ue,Ef}=i δef Ue`.

“Electric” names E in this declared canonical normalization. Factors and signs relating a continuum electric field to a canonical momentum depend on conventions and additional terms. Here E is real; an integer electric spectrum would require a separate quantization.

## 3. Store an orientation before writing a divergence

For an edge stored from i to j, put +1 in row i and −1 in row j of an incidence matrix B. All other entries are zero. A self-loop contributes zero because its two endpoint contributions cancel.

For the triangle `0→1, 1→2, 2→0`,

$$
B=\begin{pmatrix}1&0&-1\\-1&1&0\\0&-1&1\end{pmatrix},
\qquad BE=\begin{pmatrix}E_0-E_2\\E_1-E_0\\E_2-E_1\end{pmatrix}.
$$

A positive E is an oriented flow variable. Reversing a stored edge requires reversing the sign of its E and conjugating its U when describing the same state. A negative value is meaningful, not invalid input.

Every column sums to zero. Summing divergence over a complete connected component therefore gives zero, even with parallel edges or self-loops. This elementary identity will determine whether prescribed charges are possible.

## 4. Gauss's law and the generator of frame changes

Let I be the non-anchor sites. Restrict B to those rows, producing BI. The constraint and its generator are

$$
G_i=(BE)_i-\rho_i=0\quad(i\in I),\qquad
G[\alpha]=\sum_{i\in I}\alpha_iG_i.
$$

Charges are fixed parameters, so differentiating a charge with respect to A or E gives zero. Extend α by zero on anchors. Hamiltonian brackets with this generator give

$$
\delta A=B_I^T\alpha,\qquad \delta E=0,\qquad
U_{ij}'=g_iU_{ij}g_j^*,\quad g_i=e^{i\alpha_i}.
$$

Thus a frame change moves angles while preserving electric momenta. It cannot repair an electric field that violates Gauss's law. All G depend only on E, so their mutual Poisson brackets vanish: they are Abelian first-class constraints, with possible dependencies between rows.

Choosing a temporal or tree gauge does not erase a constraint equation. Gauge fixing selects a representation; Gauss's law selects an allowed momentum level.

## 5. An anchor permits boundary charge exchange

An anchor fixes the external phase reference. Our allowed transformations are identity there. We impose Gauss targets only at sites in I; anchor charge entries are `null`, not numerical zeros. The anchor divergence is calculated from E and represents the boundary source or sink required by that state.

For a connected component with anchors, summing its incidence rows gives

$$
\sum_{i\in I}(BE)_i=-\sum_{a\in\mathcal A}(BE)_a.
$$

On the constraint surface, the left side equals the prescribed interior charge. A nonzero interior charge can therefore be balanced by anchor divergence. Do not demand neutrality of the interior merely because the complete incidence matrix has zero column sums.

Conversely, a component without anchors has no such boundary freedom. Its prescribed charges must sum to zero. A disconnected graph can have zero total charge and still be inconsistent because its components cannot exchange flux.

## 6. Count independent constraints before counting pairs

Let N be the number of sites, M the number of edges, a the number of anchors and C0 the number of components containing no anchor. Then

$$
r=\operatorname{rank}B_I=N-a-C_0,\qquad k=M-r.
$$

Why? On an unanchored component, the sum of incidence rows is zero and that is its only row dependency. On an anchored component, deleting at least one row removes that dependency; its remaining rows are independent. Self-loops add zero columns and do not raise rank.

The full allowed group has dimension N−a. Its componentwise constant phases act trivially on links on each unanchored component, giving a stabilizer dimension C0. Its **effective** orbit dimension is therefore r. Subtracting the group dimension without accounting for that stabilizer would overcount removed variables.

The exact model checks the formula against rational row reduction. A formula is not accepted merely because the implementation printed it.

## 7. Empty levels have no physical dimension to report

First ask whether `BI E=ρI` has a solution. If it does, the constraint surface and its gauge quotient have dimensions

$$
\dim\mathcal C_\rho=2M-r,\qquad
\dim\mathcal P_\rho=2M-2r=2k.
$$

One independent first-class constraint restricts one momentum direction; quotienting its effective gauge flow removes one coordinate direction. This explains the factor of two. It is not the static orbit dimension from the preceding chapter.

If the charge level is inconsistent, the model reports the physical dimension as absent. It does not attach a positive dimension to an empty space. Separately, a compatible charge level may contain many valid states while the currently entered E fails to belong to it. The interface must distinguish **level compatibility** from **input admissibility**.

These statements describe this pure-link classical model. Adding canonical matter introduces its own momenta, charges and stabilizers; one cannot reuse the count without deriving that enlarged system.

## 8. Use an anchor-compatible forest as a coordinate choice

Start with the declared spanning forest of the original graph. Retain edges without joining two anchored subtrees. Each anchored subtree is rooted at its anchor; an unanchored component uses its declared computational root. There are r retained edges and k remaining edges.

The fixing phases h set retained tree links to one. Every remaining edge supplies either a rooted closed product or an open product between anchors. This is the same coordinate construction developed in the orbit chapter, now applied to the configuration part of a canonical model.

Do not confuse the original full spanning forest with the retained fixing forest. With two anchored endpoints of a three-site chain, the original forest has two edges but only one can be fixed by the allowed group. The remaining open coordinate is observable relative to the two external references.

## 9. A signed path matrix supplies canonical coordinates

For each remaining edge a, record its complete rooted traversal. Add +1 for each forward occurrence of an edge and −1 for each reverse occurrence, including repetitions. These rows form P. Backtracking cancels algebraically even if an edge is highlighted in the picture.

$$
Q_a=\prod_e U_e^{P_{ae}},\qquad \theta_a=(PA)_a\pmod{2\pi},\qquad
B_IP^T=0.
$$

The non-forest columns of P form the identity matrix: the corresponding edge occurs once forward and the other remaining edges do not occur. Define πa as E on that edge. Then, on local angular charts,

$$
\{\theta_a,\pi_b\}=\delta_{ab},\qquad
\{\theta_a,G_i\}=0.
$$

Globally use `{Qa,πb}=i δab Qa`. The columns of K=Pᵀ span the homogeneous electric constraint kernel. These are complete cycle/open-path coordinates for this graph and boundary group, not an arbitrary incomplete list of measured loops.

## 10. Solve the charged level as an affine space

Choose the unique forest-supported particular solution E0 with every non-forest electric momentum zero. It exists when the charge level is compatible. Every allowed electric field is

$$
E=E^{(0)}+K\pi,\qquad B_IE^{(0)}=\rho_I,\qquad B_IK=0.
$$

This separates a charge-determined part from free cycle or boundary-to-boundary electric flow. The producer solves charges through subtree balances; the independent reference instead solves a rational linear system. Agreement between those methods checks both orientation signs and boundary treatment.

For an invalid input, keeping its non-forest values π produces a constraint-compatible **lift**. That lift is not the same physical state, is not a gauge transformation and is generally not the closest field in Euclidean distance. A different forest can choose a different correction. An orthogonal projection would require a separately declared metric and optimization problem.

## 11. Worked neutral triangle: one pair remains

Take the oriented triangle above, no anchors, U=(i,1,1), E=(1,1,1) and zero charges. Choose forest edges0,1. All divergences vanish. Here r=2, k=1, P=(1,1,1), E0=0 and π=E2=1.

The invariant phase is Q=U0U1U2=i. With unit electric weights and one unit-weight triangle potential,

$$
H=\tfrac12(E_0^2+E_1^2+E_2^2)+1-\operatorname{Re}Q
 =\tfrac32+1=\tfrac52.
$$

In reduced coordinates, `H=3π²/2+1−cos θ`. The physical phase space has dimension2, consisting of one circular coordinate and its real conjugate momentum. A zero value of the electric divergence did not remove that cycle flow.

Now enter E=(1,0,0) with the same zero charge target. Its divergences are (1,−1,0), so the input is invalid. Keeping its non-forest value π=0 lifts it to E=(0,0,0). This changes the electric field; no local frame choice could have done that.

## 12. Fixed charges change the affine origin, not the rank

Keep E=(1,0,0), but prescribe ρ=(1,−1,0). The same field now satisfies Gauss's law. With forest0,1, its particular field is E0=(1,0,0) and π=E2=0. The reduced phase-space dimension remains2: changing a compatible right-hand side does not change the incidence rank.

Choose instead forest1,2 and root1 while keeping the original links, electric field, weights and potential path. The new particular field is (0,−1,−1), and the new momentum is E0=1. Adding Kπ recovers (1,0,0).

The reduced kinetic polynomial changes its coefficients because the coordinate chart changed. The physical Hamiltonian evaluated on the same field is unchanged. Defining a new potential by “one term per new chord” would generally change the model; our potential terms retain their explicitly stored paths when the forest changes.

A target such as ρ=(1,0,0) is different: the unanchored component has nonzero total charge, so no electric field solves it. A small least-squares residual is not a physical state of that inconsistent exact level.

## 13. Worked anchored chain: interior charge reaches the boundary

Use edges `0→1,1→2`, anchors0,2 and interior charge ρ1=1. Let E=(1,2), U=(i,1). Its divergence is (1,1,−2); only the middle entry has a prescribed target. The anchor values balance it automatically.

The retained forest is edge0, the remaining edge is1, and

$$
P=(1,1),\quad K=(1,1)^T,\quad E^{(0)}=(-1,0),\quad
\pi=E_1=2,\quad E=(-1+\pi,\pi).
$$

There is one independent constraint and one effective gauge direction. Starting from four phase-space dimensions leaves a two-dimensional physical phase space. Without anchors, a neutral open chain would have no remaining pair.

Choose the anchored open-path potential with weight2. Then

$$
H=\tfrac12(-1+\pi)^2+\tfrac12\pi^2+2(1-\cos\theta).
$$

At Q=i and π=2, H=9/2. This open potential uses external endpoint references. It is an allowed invariant of the anchored model, not a claim that every open product is a Maxwell magnetic observable.

## 14. Declare the Hamiltonian independently of the forest

The model permits positive electric weights we and nonnegative path weights κt:

$$
H(A,E)=\tfrac12\sum_e w_eE_e^2+
\sum_t\kappa_t\left(1-\operatorname{Re}W_t\right),\qquad
W_t=\prod_e U_e^{s_{te}}.
$$

Each stored walk must connect and either close or run between anchors. Its signed count vector st satisfies BI st=0. A lone open edge ending at an unanchored site is rejected. A connected closed walk may backtrack; its complete product and signed counts are evaluated exactly.

For fixed charge, the reduced electric energy has metric KᵀWK, linear term KᵀWE0 and constant `(E0)ᵀWE0/2`. Positive weights and independent columns of K make the metric positive definite when at least one pair remains. The phase potential is expressed through Q using its non-forest exponents. This is a coordinate transformation of a declared Hamiltonian, not a rule that every graph approximates electromagnetism.

## 15. Exact instantaneous flow and constraint preservation

Differentiating the declared Hamiltonian gives

$$
\dot A_e=w_eE_e,\qquad
\dot E_e=-\sum_t\kappa_t s_{te}\operatorname{Im}W_t,\qquad
\dot U_e=iU_e\dot A_e.
$$

Since BI st=0 for every allowed term, `BI Ė=0`. The prescribed interior Gauss level is tangent to the exact vector field. Anchor divergence can change when an open anchored potential exchanges boundary charge; that does not violate the interior constraint.

For the neutral triangle at Q=i, π=1, the rates are θdot=3 and πdot=−1. Each electric momentum has rate−1, so the differences defining divergence remain unchanged. For the charged anchored chain above, θdot=3 and πdot=−2; both edge momenta have rate−2 while the middle divergence rate is zero.

Zero Gauss rate also preserves an initial violation; it does not repair it. These are instantaneous derivatives. A finite-step algorithm still needs separate checks for circle normalization, energy behavior, symplectic structure and error. For example, an additive Euler update to U generally leaves the unit circle. Exact derivatives alone do not certify an accurate trajectory.

## 16. Pathologies are useful tests

A self-loop gives a zero incidence column but retains a circular coordinate and electric momentum. Parallel links can supply a genuine cycle even with only two incident sites. An isolated unanchored site imposes a zero-charge compatibility condition while contributing no effective gauge direction. With all sites anchored, there are no imposed Gauss rows and no allowed local gauge directions: all M canonical pairs remain.

A zero-length edge list, zero remaining pairs and an empty constraint level are three different cases. Never replace a missing result by zero simply to fill a metric card.

Fractional Cayley parameters encode circle phases exactly, including a separate −1 point. We use exact integer/fraction arithmetic for residuals and independent comparisons. Drawing coordinates are approximate and do not decide whether a constraint is exactly satisfied.

## 17. Connections to AI research

A learned Hamiltonian can preserve a linear Gauss level structurally if every phase feature has signed exponent s with BI s=0. This is stronger than checking a few gauge-transformed samples. It also exposes an important modeling choice: a boundary feature valid for anchored inputs may be invalid under a larger gauge group.

Null-space parameterization `E=E0+Kπ` makes the constraint exact when the level is compatible. It differs from penalizing a residual in a loss and from projecting an arbitrary prediction with a chosen norm. Compare those methods using constraint residual, approximation error, conditioning and boundary assumptions separately.

Use the same physical Hamiltonian when testing coordinate or forest invariance. Check both frame covariance and Gauss admissibility; an energy or phase invariant can agree while a state violates the charge constraint. For learned time evolution, additionally test residual drift and convergence with step size. None of these tests by itself proves generalization or a continuum physical interpretation.

## 18. Questions with explanations

### A. Why does one independent Gauss condition remove two phase-space dimensions after reduction?

**Answer:** It restricts one momentum direction on the level set. Its effective gauge flow changes a conjugate coordinate direction, which is removed by the quotient. Dependent Gauss rows do not remove extra dimensions. For this model a compatible level has dimension2(M−r) after reduction.

### B. Is zero total charge enough on two disconnected unanchored components?

**Answer:** No. Charges (1,0,0) on one triangle and (−1,0,0) on another have zero global sum, but each component violates its own neutrality condition. There is no connecting edge to carry flux between them, so the exact level is empty.

### C. Can a gauge transformation fix the invalid neutral-triangle input E=(1,0,0)?

**Answer:** No. A gauge transformation changes A and leaves E fixed. The charge-compatible lift E=(0,0,0) at the same non-forest momentum is a different field, not another representation of the old field. The interface must not label the lift as an equivalent state.

### D. Why are the anchor charge targets null?

**Answer:** This boundary model does not prescribe their Gauss values. It computes the required anchor divergences from E. In the charged chain E=(1,2), the interior charge1 is balanced by anchor divergences1 and−2. Writing zero targets at those anchors would specify a different,more constrained problem.

### E. Does changing the spanning forest change the Hamiltonian?

**Answer:** It changes the coordinate chart,particular field and reduced coefficients. It does not change a Hamiltonian whose electric weights and actual potential paths are held fixed. Replacing the potential list with new chord-based terms can change the Hamiltonian and must not be presented as a coordinate-invariance test.

### F. Why does a self-loop retain a physical pair?

**Answer:** Its incidence column is zero,so its angle is unchanged by vertex gauge transformations and its electric momentum contributes no site divergence. It is already an invariant coordinate with an unconstrained conjugate momentum. A chosen self-loop potential may still exert a force on that momentum.

### G. Does an exact zero Gauss rate certify a numerical simulator?

**Answer:** No. It proves tangency of this exact instantaneous Hamiltonian vector field. A discrete integrator can still distort circle coordinates,energy or symplectic structure. Integration requires separate algorithmic and step-size evidence; this experiment does not integrate time.

### Sources and scope

The canonical constraint background is discussed in [Tong, Gauge Theory §2.2.1](https://davidtong.org/pdfs/teaching/gauge-theory/gauge2.pdf). [Kaplan and Stryker, arXiv:1806.08797v2 §I](https://arxiv.org/html/1806.08797) treats quantum periodic lattice constraints. The arbitrary classical graph,boundary conventions and worked reductions above are derived for this teaching model and checked independently; the quantum and continuum theories have additional assumptions.
