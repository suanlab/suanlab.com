## 1. From a list of coordinates to a field

A vibrating string has a displacement at every position. Write it as $\phi(x,t)$: $x$ labels a place, while $\phi$ is the quantity that changes. A temperature map is also a field, but it obeys a different physical evolution law. A field is a type of description, not a promise of wave motion or quantum behavior.

Start with one real field on $0\le x\le L$. Before drawing a moving surface, specify its units, boundary conditions, initial displacement and initial velocity. The examples below use dimensionless coordinates and parameters obtained after choosing reference scales.

## 2. A local action and its assumptions

Our teaching model is

$$
S[\phi]=\int_{t_0}^{t_1}\!dt\int_0^L\!dx\,\mathcal L,
\qquad \mathcal L=\frac12\phi_t^2-\frac{c^2}{2}\phi_x^2-V(\phi),
\qquad V(\phi)=\frac a2\phi^2+\frac b4\phi^4.
$$

Here $c>0$, $b\ge0$, and $a$ is a signed coefficient. It is not automatically a physical mass squared. This is a local, conservative, real scalar model with no forcing or damping. A trained neural field is not necessarily governed by this action.

The action combines an entire history, not one spatial snapshot. Its units before nondimensionalization differ from those of energy.

## 3. What does varying a field mean?

Compare $\phi$ with $\phi+\epsilon\eta$. The test function $\eta(x,t)$ must respect the allowed boundary variations. Fix the field at the two time endpoints, so $\eta(x,t_0)=\eta(x,t_1)=0$. Differentiating at $\epsilon=0$ gives

$$
\delta S=\int dt\,dx\,[\phi_t\eta_t-c^2\phi_x\eta_x-V'(\phi)\eta].
$$

This is a directional derivative of a functional. Changing one slider is only one possible direction; stationarity requires the derivative to vanish for every admissible direction.

## 4. Keep the boundary terms

Integration by parts yields

$$
\delta S=\int dt\,dx\,[-\phi_{tt}+c^2\phi_{xx}-V'(\phi)]\eta
+\left[\int dx\,\phi_t\eta\right]_{t_0}^{t_1}
-\int dt\,[c^2\phi_x\eta]_{0}^{L}.
$$

The time term vanishes by the endpoint condition. At a fixed spatial boundary $\eta=0$. At periodic boundaries the two spatial terms cancel when both field and variation are periodic. For independently free endpoints, the natural boundary condition is $\phi_x=0$. These are different experiments; one cannot discard a boundary term without stating why.

## 5. The equation follows from all test directions

For smooth solutions and the specified boundary conditions,

$$
\phi_{tt}=c^2\phi_{xx}-a\phi-b\phi^3.
$$

The first term communicates neighboring displacements, the second is a linear local response, and the third is nonlinear. The signs follow from the variation; they are not selected to make an animation look stable.

## 6. Stationary does not mean smallest

An extremum of a static potential and stationarity of a dynamical action are different statements. For the free quadratic model the second variation is

$$
\delta^2S=\int dt\,dx\,[\eta_t^2-c^2\eta_x^2-a\eta^2].
$$

Allowed rapidly varying time directions can make this positive; rapidly varying spatial directions can make it negative. Thus a stationary history can be a saddle. Minimizing an arbitrary squared equation residual is another optimization problem, with its own boundary and identifiability questions.

## 7. A spatial grid defines a finite model

For a periodic grid with $N\ge3$, spacing $h=L/N$, and $q_{i+N}=q_i$, define

$$
L_h=h\sum_i\left[\frac{v_i^2}{2}-V(q_i)\right]
-\frac{c^2}{2h}\sum_i(q_{i+1}-q_i)^2.
$$

Time remains continuous. The kinetic mass matrix is $hI$, not $I$ unless $h=1$. Differentiating this finite Lagrangian gives

$$
\dot q_i=v_i,\qquad
\dot v_i=\frac{c^2}{h^2}(q_{i-1}-2q_i+q_{i+1})-aq_i-bq_i^3.
$$

A spatial discretization is not yet a time integrator or a convergence proof.

Boundary choice also changes the grid geometry. For homogeneous Dirichlet boundaries, use $N$ interior sites, fixed zero values outside them, and $L=(N+1)h$; include the two edges joining interior sites to those fixed endpoints. For a zero-flux Neumann discretization, use $N$ cell centers at $(i+1/2)h$ on $L=Nh$ and only the $N-1$ interior edges. This specifies a finite variational model, not an automatic accuracy guarantee for every continuum boundary approximation.

With $a=b=0$ and all $q_i=1$, the periodic and zero-flux models have zero acceleration. The fixed-zero model instead has endpoint-interior accelerations $-c^2/h^2$. Zero boundary value and zero boundary flux must not share a control labeled merely “zero boundary”.

## 8. Work through three sites exactly

Take $N=3$, $h=c=1$, $a=b=0$, $q=(1,0,-1)$ and $v=(1,0,0)$. The edge differences are $(-1,-1,2)$, so the gradient energy is $3$ and kinetic energy is $1/2$. The total is $H_h=7/2$.

The acceleration is $(-3,0,3)$. Its sum is zero, although the individual sites accelerate. Neighbor interaction redistributes motion; it does not imply that each coordinate is conserved.

## 9. Independently check a force sign

Let $U$ denote the spatial potential energy. Perturb only $q_0$ in the preceding example:

$$
U(q+\epsilon e_0)=3+3\epsilon+\epsilon^2.
$$

The derivative is $3$. Dividing its negative by the mass $h=1$ gives acceleration $-3$. At $\epsilon=1/10$, the two energies are $331/100$ and $271/100$; their centered difference gives $3$ exactly. For a quartic potential a finite difference generally has a nonzero truncation error, so the same check must retain its step-size dependence.

For this specific quartic lattice energy, the finite-step error can be removed exactly. Write $e_i$ for a coordinate direction and take a nonzero amplitude perturbation $\delta$:

$$
\frac{\partial U}{\partial q_i}
=\frac{U(q+\delta e_i)-U(q-\delta e_i)}{2\delta}-hbq_i\delta^2,
$$

$$
\frac{\partial^2 U}{\partial q_i^2}
=\frac{U(q+\delta e_i)+U(q-\delta e_i)-2U(q)}{\delta^2}-\frac{hb\delta^2}{2}.
$$

Expand $(q_i\pm\delta)^4$ to verify both correction terms. The neighboring gradient energy is quadratic, so it contributes no higher-order error here. Mixed coordinate differences isolate the bilinear neighbor coupling; nonadjacent coordinates have zero mixed derivative in this model. The amplitude step $\delta$ and spatial spacing $h$ play different roles.

This gives an independent way to check a force or Hessian obtained by symbolic differentiation or automatic differentiation. The exact identities depend on the stated polynomial energy. Applying the same correction to an arbitrary learned potential would be unjustified.

## 10. Energy conservation belongs to the specified dynamics

The finite model has

$$
H_h=\frac h2\sum_i v_i^2+U(q),\qquad
\frac{dH_h}{dt}=\sum_i\frac{\partial U}{\partial q_i}v_i+h\sum_i v_i\dot v_i=0.
$$

In our three-site state these terms are $3$ and $-3$. Their cancellation is an independent numerical diagnostic. A time-stepping algorithm can still introduce energy error; conservation of the continuous-time model does not guarantee conservation by an implementation.

In the continuum, $e=\phi_t^2/2+c^2\phi_x^2/2+V$ and $j=-c^2\phi_t\phi_x$ satisfy $\partial_te+\partial_xj=0$. Consequently the energy change on the interval is $j(0)-j(L)$: boundary flux matters.

## 11. Symmetry, zero modes and stability

With $a=b=0$ and periodic boundaries, adding the same constant to every $q_i$ leaves the potential unchanged. The uniform mode has zero restoring force and $h\sum_i v_i$ is conserved. A fixed boundary generally breaks that shift symmetry. This sum is the canonical charge for uniform shifts of the field value. It is not automatically the momentum associated with translations of the spatial label x; in the continuum these are different transformations.

A uniform configuration $q_i=Q$ has acceleration $-aQ-bQ^3$. For $a=-1,b=1$, equilibria are $Q=0,\pm1$; local potential curvature is $-1$ at zero and $2$ at either nonzero equilibrium. This is a statement about this potential, not evidence of a thermodynamic phase transition in a finite three-site demonstration.

A lower bound on the local potential is also different from stability of the whole finite problem. With $a<0,b=0$, the local potential has no lower bound. Yet for three interior Dirichlet sites, $h=1,c=2,a=-1$, the total potential Hessian has diagonal $7$ and adjacent off-diagonal entries $-4$. Its leading principal minors are $7,33,119$, so it is positive definite. Boundary stiffness stabilizes this particular finite quadratic problem; this does not prove stability for arbitrary interval lengths or continuum limits.

## 12. What a visual experiment should expose

Display displacement, velocity, edge gradients, local forces, energy terms and boundary choice separately. Let the learner compare a uniform shift with a nonuniform disturbance. A paused picture cannot determine velocity: identical displacements can have different futures.

Use a spatial strip or a surface only as a representation of $\phi$; plot the field value on the vertical axis rather than suggesting an extra physical dimension. Provide a readable table and keyboard controls alongside motion. A residual or energy-error panel must name the discretization and time step used.

## 13. Questions with explanations

**Can the field equation determine a unique evolution from displacement alone?** Generally no: this second-order-in-time model also needs initial velocity and compatible boundary conditions.

**Does a constant displacement have zero acceleration?** Only when $aQ+bQ^3=0$. Its spatial derivative vanishes, but the local potential can still exert a force.

**Does an energy-conserving numerical curve prove continuum accuracy?** No. One must also test spatial and temporal refinement, the intended boundary problem and observables relevant to the research claim.

**Is a classical scalar field already a quantum field?** No. Quantization, states, observables and regularization require additional constructions and assumptions.

## 14. Connection to AI research and further reading

A field model motivates structured surrogate dynamics and variational objectives. Keep three claims separate: matching sampled trajectories, satisfying a chosen discretized equation, and approximating the intended continuum problem. For inverse problems, several potentials or boundary inputs may explain the same sparse observations.

For the general classical field action, boundary variation and Euler–Lagrange framework, consult [David Tong, Classical Field Theory, sections 1.1 and 1.3](https://davidtong.org/teaching/quantum-field-theory/qfthtml/S1). The periodic three-site calculations above are worked examples for this textbook.
