## 1. Separate a complicated shape into simple motions

A vibrating string can look complicated because several spatial patterns are superposed. A normal-mode basis changes the question from “what is every site doing?” to “how much of each independent pattern is present?” For a linear system near an equilibrium, each modal coordinate obeys its own scalar equation. This is also the origin of many spectral representations used in scientific machine learning.

Review [scalar-field action and boundary variation](/book/online/physics-math-for-ai/scalar-field-action-boundary-variation/) and [local currents and Noether transformations](/book/online/physics-math-for-ai/field-noether-current-boundary-balance/). Here we study the same spatial discretizations, keeping time continuous. Three distinctions matter throughout: an equilibrium versus a frozen reference, a physical wavenumber versus a lattice index, and fixed-mode convergence versus accuracy across the entire spectrum.

## 2. Specify the grid before choosing a transform

Let $L$ be interval length and $N$ the number of active values. In the stated nondimensional model, the mass matrix is $M=hI$. For edges joining active sites or fixed exterior values, use

$$
U(q)=\frac{c^2}{2h}\sum_{(i,j)}(q_j-q_i)^2
+h\sum_i\left(\frac a2q_i^2+\frac b4q_i^4\right),\qquad
H=\frac h2\sum_i v_i^2+U(q).
$$

| Boundary | Active positions | Spacing | Exterior treatment |
| --- | --- | --- | --- |
| Periodic | $x_j=jh$ | $h=L/N$ | The final edge closes to site zero |
| Fixed zero | $x_j=(j+1)h$ | $h=L/(N+1)$ | Two fixed zero values are not active coordinates |
| Zero flux | $x_j=(j+1/2)h$ | $h=L/N$ | Cell centers with $N-1$ interior edges |

Changing the boundary while holding $L,N$ fixed can change $h$. A node-centered Neumann scheme would be a different discretization; its transform and endpoint weights must be derived again.

## 3. Linearize around a declared reference

Set $q_i=\bar q+\eta_i$ at active sites and take reference velocity zero. The local curvature is

$$
\mu^2=a+3b\bar q^2.
$$

The notation $\mu^2$ does not assume nonnegativity. Writing $g=\nabla U(\bar q)$ and $K=\nabla^2U(\bar q)$ gives

$$
h\ddot\eta=-g-K\eta+O(\eta^2).
$$

Only when $g=0$ is the reference a static equilibrium of this finite problem. Otherwise the constant forcing must remain. The Hessian can still be diagonalized, but its eigenvalues describe a frozen local operator, not oscillations around a stationary solution.

For periodic or cell-centered zero-flux boundaries, a uniform reference is an equilibrium when $a\bar q+b\bar q^3=0$. Fixed-zero walls add an endpoint acceleration $-c^2\bar q/h^2$. Thus a bulk minimum at nonzero $\bar q$ is not automatically a finite-domain equilibrium.

## 4. The mass inner product determines normalization

Let $D$ be the dimensionless positive second-difference matrix built from the edge energy. Then

$$
A=M^{-1}K=\frac{c^2}{h^2}D+\mu^2I,
\qquad Ku_\alpha=\lambda_\alpha Mu_\alpha.
$$

Choose $u_\alpha^TMu_\beta=\delta_{\alpha\beta}$. A Euclidean unit eigenvector must therefore be divided by $\sqrt h$. Forgetting this factor changes modal coordinates, norms and energies even when the eigenvalues look correct.

The matrices here are symmetric because all active sites have the same mass. This does not justify replacing every generalized eigenproblem by an ordinary symmetric one. Nonuniform masses require the appropriate mass transformation or generalized solver. In a repeated eigenspace, many orthonormal bases are valid; individual coefficients depend on that basis, while the subspace norm is invariant.

## 5. Derive the periodic symbol from a plane wave

Insert $u_j=e^{ikjh}$ into the periodic stencil:

$$
2u_j-u_{j-1}-u_{j+1}=4\sin^2(kh/2)u_j.
$$

Consequently,

$$
\lambda_h(k)=\mu^2+\frac{4c^2}{h^2}\sin^2(kh/2),
\qquad k_m=\frac{2\pi m}{L}.
$$

At equilibrium, a linear modal coordinate satisfies $\ddot Q_m+\lambda_hQ_m=0$. Oscillation frequency is $\omega_h=\sqrt{\lambda_h}$ only for positive $\lambda_h$.

Samples cannot distinguish $k$ from $k+2\pi r/h$ for integer $r$. This is an exact sampling identity, not a numerical precision problem. Positive and negative physical wavenumbers also produce the same eigenvalue; a real representation must account for both directions without duplicating degrees of freedom.

## 6. Count the real periodic modes correctly

One convenient mass-normalized real basis is

$$
u_0(j)=\frac1{\sqrt{Nh}},\quad
u_{m,c}(j)=\sqrt{\frac2{Nh}}\cos\frac{2\pi mj}{N},\quad
u_{m,s}(j)=\sqrt{\frac2{Nh}}\sin\frac{2\pi mj}{N}.
$$

The paired modes use $1\le m\le\lfloor(N-1)/2\rfloor$. For even $N$, add one Nyquist mode

$$
u_{N/2}(j)=\frac{(-1)^j}{\sqrt{Nh}}.
$$

There is no independent Nyquist sine vector: its samples vanish. The constant mode also has no independent sine partner. The resulting dimension is exactly $N$: $1+2(N-1)/2$ for odd $N$, or $1+2(N/2-1)+1$ for even $N$.

A sine and cosine pair has the same eigenvalue but generally different coefficients for a specified state. Do not sort degenerate numerical eigenvectors and assume their orientations will match this chosen convention.

## 7. Fixed walls and cell-centered zero flux need different bases

For $N$ interior fixed-zero sites, the mass-normalized sine basis is

$$
u_m(j)=\sqrt{\frac2L}\sin\frac{\pi m(j+1)}{N+1},\quad
k_m=\frac{\pi m}{L},\quad 1\le m\le N.
$$

For the cell-centered zero-flux grid, use the constant vector $1/\sqrt L$ and

$$
u_m(j)=\sqrt{\frac2L}\cos\frac{\pi m(j+1/2)}N,
\quad 1\le m\le N-1.
$$

Both use the same stencil dispersion formula evaluated at their permitted $k_m$. Their endpoint rows differ: Dirichlet has diagonal two, while the cell-centered Neumann matrix has endpoint diagonal one. This changes the basis and the presence of a zero spatial mode.

[Gilbert Strang's discrete cosine transform paper](https://math.mit.edu/~gs/papers/dct.pdf) explains the connection between boundary placement and symmetric second-difference eigenvectors. The formulas above are derived for the particular edge energy and mass convention specified here; “use a cosine transform” is not a complete boundary specification.

## 8. Project, reconstruct and check energy

For the chosen real basis, define

$$
Q_\alpha=h\sum_j u_\alpha(j)\eta_j,
\qquad V_\alpha=h\sum_j u_\alpha(j)v_j,
\qquad \eta_j=\sum_\alpha u_\alpha(j)Q_\alpha.
$$

Completeness gives the mass-weighted Parseval identities

$$
\sum_\alpha Q_\alpha^2=h\sum_j\eta_j^2,
\qquad \sum_\alpha V_\alpha^2=h\sum_jv_j^2.
$$

The frozen quadratic energy is

$$
H_2=\frac12\sum_\alpha(V_\alpha^2+\lambda_\alpha Q_\alpha^2).
$$

Check reconstruction, both Parseval identities and the energy sum. Agreement of the energy alone can hide a sign or basis-order error. If $b\ne0$, $H_2$ is generally not the full nonlinear energy. If the reference is not an equilibrium, the linear term in the energy expansion also matters.

## 9. Work through a four-site state by hand

Take periodic $N=L=4$, $h=c=1$, $a=b=\bar q=0$ and

$$
q=(1,0,-1,0),\qquad v=(1,0,0,0).
$$

The constant, cosine, sine and Nyquist eigenvalues are $(0,2,2,4)$. In that basis,

$$
Q=(0,\sqrt2,0,0),\qquad
V=(1/2,1/\sqrt2,0,1/2).
$$

| Component | Kinetic contribution | Potential contribution |
| --- | --- | --- |
| Constant | $1/8$ | $0$ |
| Cosine, $m=1$ | $1/4$ | $2$ |
| Sine, $m=1$ | $0$ | $0$ |
| Nyquist | $1/8$ | $0$ |

Thus $K=1/2$, $U=2$ and $H=5/2$. The shape uses only one displacement mode, but the localized velocity excites three modal velocities. A displacement plot alone does not specify the state. The constant velocity component also produces drift, even though the other excited coordinates oscillate.

## 10. Positive, zero and negative curvature are different dynamics

For a stationary reference and the homogeneous linearized equation:

- $\lambda>0$: $Q=A\cos(\sqrt\lambda t)+B\sin(\sqrt\lambda t)$.
- $\lambda=0$: $Q=A+Bt$; this is not an oscillator with a finite period.
- $\lambda<0$: $Q=Ae^{\sqrt{-\lambda}t}+Be^{-\sqrt{-\lambda}t}$.

A zero-curvature direction can drift without changing its kinetic energy. Conserved energy therefore need not bound positions when the energy is only semidefinite. An indefinite quadratic energy can remain constant while an unstable coordinate grows.

These statements concern the constant linearized operator. Higher-order terms can change the dynamics of a zero-curvature nonlinear mode. For a nonstationary reference, the displayed frequency or growth scale is a property of the frozen homogeneous operator; it does not remove the reference forcing or prove an observed growth trajectory.

## 11. A boundary can stabilize a negative local curvature

Use three fixed-zero interior sites, $c=2$, $a=-1$, $b=\bar q=0$. At $L=4$, $h=1$ and

$$
A=\begin{pmatrix}7&-4&0\\-4&7&-4\\0&-4&7\end{pmatrix},
\qquad \lambda_{\min}=7-4\sqrt2>0.
$$

The restoring effect of the spatial gradients and walls outweighs the negative local curvature. With the same coefficients and number of sites but $L=8$, $h=2$ and

$$
\lambda_{\min}=1-\sqrt2<0.
$$

The model now has an unstable mode. The continuum lowest-mode criterion is $\mu^2+c^2(\pi/L)^2>0$, but the finite-grid eigenvalue must be checked separately. A local potential test, a finite-domain stability result and a continuum threshold are three different claims.

## 12. Distinguish phase speed from symbol group speed

For a positive-frequency periodic branch with $k>0$,

$$
v_{\mathrm{ph}}=\frac{\omega_h(k)}k,\qquad
v_{\mathrm g}=\frac{d\omega_h}{dk}
=\frac{c^2\sin(kh)}{h\omega_h(k)}.
$$

For the massless positive branch $0<kh\le\pi$, these reduce to

$$
v_{\mathrm{ph}}=c\frac{\sin(kh/2)}{kh/2},\qquad
v_{\mathrm g}=c\cos(kh/2).
$$

At Nyquist, phase speed is $2c/\pi$ but symbol group speed is zero. A rapidly varying lattice pattern need not transport an envelope at speed $c$.

A finite sine or cosine mode is a standing pattern. Interpreting the derivative as packet transport requires a sufficiently broad domain, a narrow packet spectrum and the appropriate branch. The calculation does not supply a packet evolution. Phase speed at $k=0$ or at a nonoscillatory mode is left undefined rather than obtained by dividing by zero.

## 13. Refine the grid while keeping the physical problem fixed

Fix $L,c,a,b,\bar q$ and a physical mode index $m$. The corresponding $k$ remains fixed. Periodic and cell-centered grids refine $N$ to $2N$; an interior Dirichlet grid refines $N$ to $2(N+1)-1$. All three halve $h$.

For $x=kh/2$ in $[0,\pi/2]$,

$$
0\le\lambda_{\mathrm{cont}}-\lambda_h
=c^2k^2\left[1-\left(\frac{\sin x}{x}\right)^2\right]
\le\frac{c^2k^4h^2}{12},
\qquad \lambda_{\mathrm{cont}}=\mu^2+c^2k^2.
$$

The bound follows by combining $\sin x\le x$ with $\sin x\ge x-x^3/6$ on this interval. Expansion gives leading eigenvalue error $c^2k^4h^2/12$. Its ratio between successive refinements approaches four; it is not exactly four at arbitrary resolution.

Frequency error needs extra care near zero curvature, where a square root is poorly conditioned. Convergence of these operator eigenvalues also does not establish convergence of a time integrator or nonlinear solution.

## 14. Fixed-mode convergence is not uniform high-frequency accuracy

If you choose the Nyquist mode anew on every grid, then $k=\pi/h$ changes as the grid changes. The relative spatial-symbol error remains

$$
1-\frac4{\pi^2},
$$

rather than approaching zero. Such a sequence is not the fixed-$k$ experiment of the previous section. A new grid adds newly resolvable frequencies; it does not make every frequency near its own cutoff continuum-like.

In the refinement calculation, an initially selected Nyquist index is kept fixed. After doubling $N$, that same physical index is no longer Nyquist. Also keep the reference's continuum admissibility separate: nonzero constant active values with fixed-zero ghosts develop a boundary jump as $h$ decreases. Their frozen Hessian spectrum can be compared, but that reference sequence is not a smooth constant field satisfying the continuum walls.

## 15. Measure the nonlinear remainder rather than assuming it away

For the quartic potential, the exact finite-energy expansion is

$$
H(q,v)=U(\bar q)+g^T\eta+H_2(\eta,v)
+hb\sum_j\left(\bar q\eta_j^3+\frac{\eta_j^4}{4}\right).
$$

At periodic $N=L=4$, $a=-1$, $b=\bar q=c=1$, choose $q=(2,1,0,1)$ and $v=(1,0,0,0)$. The reference is an equilibrium with $\mu^2=2$, but

$$
U(\bar q)=-1,\quad g^T\eta=0,\quad H_2=9/2,
\quad R=1/2,\quad H=4.
$$

The finite perturbation is not automatically small just because a modal basis exists. With $a=0,b=\bar q=1$, the same uniform reference has acceleration $-1$ at every site and is nonstationary. With fixed-zero walls, even the bulk double-well minimum has endpoint forces. Check both the remainder and the reference force before treating modes as independent physical oscillators.

## 16. Read numerical evidence at the right level

Some quantities are rational: spacing, exact energy components and the energy-expansion residual. Most sine/cosine coefficients, eigenvalues and frequencies are irrational and are reported through outward enclosures. A narrow interval containing zero is not an exact-zero certificate. Algebraically recognized zero eigenvalues are distinguished from unresolved signs.

The calculation checks mass orthogonality, modal energy, Parseval identities, reconstruction and the selected eigen-equation. Independent references construct the energy matrix from edge incidences and diagonalize that matrix at high precision instead of using the production dispersion formula. Their high-precision comparisons complement, rather than replace, the directed-arithmetic bounds.

None of these checks advances time. A study using a time integrator must additionally test step-size dependence, phase error, stability and energy drift. Near degeneracies, compare eigenspaces or declared basis conventions instead of demanding identical vectors from unrelated solvers.

## 17. Use the distinction in AI research

A spectral neural operator may learn a relationship between coefficients, but coefficients are tied to a domain, grid, boundary and normalization. A periodic FFT representation can introduce a discontinuity when applied to fixed-wall data. Changing resolution also changes the meaning of a raw frequency-array position unless physical wavenumbers are tracked.

A useful experiment holds $L$, the physical band and reference conditions fixed, then measures both low-mode prediction error and error near the grid cutoff. Include reconstruction, mass norms and energy/remainder diagnostics. For nonlinear products, unresolved harmonics can alias back into the retained spectrum; a de-aliasing rule must match the polynomial degree and discretization rather than being applied as an unexplained universal recipe.

[Stamatopoulos's lattice-dispersion study](https://arxiv.org/abs/1201.3368) illustrates why dispersion depends on the spatial stencil. It concerns three-dimensional scalar-field simulations; the one-dimensional bounds and worked examples here are derived separately. Numerical agreement on one grid does not establish resolution transfer or a new physical law.

## 18. Predict, calculate and explain

1. **Does the four-site displacement excite every modal velocity?** Displacement and velocity are separate data. Here $Q=(0,\sqrt2,0,0)$ while $V=(1/2,1/\sqrt2,0,1/2)$; three velocities are nonzero.
2. **Does zero curvature imply a stationary coordinate?** No. The homogeneous linear equation is $\ddot Q=0$, so $Q=A+Bt$. Nonzero $B$ causes drift.
3. **Does negative local curvature prove finite-domain instability?** No. The three-site fixed-wall example has $7-4\sqrt2>0$ at $L=4$, but $1-\sqrt2<0$ at $L=8$.
4. **Does a Nyquist wave have zero phase speed?** No. On the massless positive branch its phase speed is $2c/\pi$ and symbol group speed is zero.
5. **Does an observed error ratio near four certify all modes?** No. The second-order expansion fixes $k$ and the physical problem. Re-selecting Nyquist on every grid violates that setup.
6. **Does a positive frozen Hessian justify independent nonlinear oscillators?** No. First verify equilibrium and a sufficiently small nonlinear remainder. The nonstationary-reference and finite-amplitude double-well examples fail different parts of that inference.

For a reproducible comparison, record boundary placement, $L,N,h$, reference force, basis normalization, physical mode index, eigenvalue sign and the exact energy decomposition. State separately which of the finite algebra, spatial limit, time evolution and nonlinear approximation your evidence actually tests.
