## Are the potential wave equations enough?

If all four potential components satisfy their wave equations, do all Maxwell equations follow? The separated equations are a powerful method when the Lorenz representation and compatible initial and boundary data are included. Checking the wave equations alone is weaker. We will construct a solution of the scalar and vector wave equations whose assigned sources are conserved, yet whose fields fail Gauss law for those sources. The purpose is to identify the missing constraints explicitly, not to reject the potential formulation.

## Fix notation and local assumptions

Assume smooth potentials, constant positive epsilon and mu, and c squared equal to one over their product. Use the normalized coordinates and potential scales from the previous unit. A normalized angular frequency is distinct from a physical frequency in radians per second. The longitudinal examples are constructed to expose constraints on the potential representation. A nonzero longitudinal electric field in these examples should not be labelled a source-free transverse radiation wave. Its physical interpretation depends on the sources and initial and boundary conditions actually supplied.

## The physical residuals associated with G

Define the wave operator and Lorenz combination as below. Substituting the potential definitions into Maxwell equations gives explicit residuals. When the separated wave equations hold, the remaining terms are minus the time derivative of G and the gradient of G. Faraday law and zero magnetic divergence are potential identities, while the equations containing charge and current need these additional checks. The Gauss residual has units of electric field per metre; the Ampere residual here has units of magnetic field per metre. Do not compare their raw magnitudes without scaling.

$$
\Box=\Delta-c^{-2}\partial_t^2,\quad G=\nabla\cdot\mathbf A+c^{-2}\partial_tV,\quad \nabla\cdot\mathbf E-\rho/\epsilon=-\Box V-\rho/\epsilon-\partial_tG,\quad \nabla\times\mathbf B-c^{-2}\partial_t\mathbf E-\mu\mathbf J=-\Box\mathbf A-\mu\mathbf J+\nabla G.
$$

## Source conservation controls the wave equation for G

Take the divergence of the vector wave equation and the scaled time derivative of the scalar equation. Their sum gives the wave equation for G. Conserved sources make it homogeneous. A homogeneous wave equation still has many nonzero solutions, however. Source conservation therefore does not select G equal to zero by itself. This matters for a learned solver: adding a charge-conservation loss does not automatically supply the missing initial gauge data. Identify which equation enforces an evolution law and which conditions select the desired solution of that law.

$$
\Box\mathbf A=-\mu\mathbf J,\quad \Box V=-\rho/\epsilon\quad\Longrightarrow\quad \Box G=-\mu(\nabla\cdot\mathbf J+\partial_t\rho).
$$

## Two initial data and appropriate boundary conditions

G obeys a second-order evolution equation. In a well-posed initial-boundary problem, zero initial G, zero initial time derivative, and compatible boundary conditions select the zero solution. Checking one point does not establish either initial function over the whole region. Even checking G over the complete initial surface misses its time derivative. These conditions are a sufficient way to maintain the Lorenz representation under the stated uniqueness assumptions. They are not a claim that every physically valid representation must have zero G. A constant nonzero G will provide a useful qualification later.

$$
G(\mathbf x,0)=0,\qquad \partial_tG(\mathbf x,0)=0,\qquad \text{compatible boundary data}.
$$

## Specify longitudinal harmonic potentials

Use the normalized wave operator, with wave speed one in the coordinate variables. Let the vector potential point along z and share the scalar potential phase. Both normalized k and omega are nonnegative. At k=0 the potentials have no spatial dependence and are spatially uniform; at omega=0 they have no time dependence and are static. When both vanish, the potentials are constant in space and time. Calling every member a travelling physical wave would be misleading. We deliberately choose the two amplitudes independently so that the consequences of a missing constraint can be observed. The same formulas then include compatible forced modes, incompatible source requests, and pure-gauge resonant cases.

$$
v=v_0\cos\phi,\qquad \mathbf a=(0,0,a_0\cos\phi),\qquad \phi=k\zeta-\omega\tau,\qquad D=k^2-\omega^2.
$$

## Compare the coefficients of the actual field and G

Define Q for the electric-field amplitude and C for the Lorenz amplitude. Their terms look similar but exchange the roles of frequency and wavenumber. B is zero because the vector potential has only a z component depending on z and time. C equal to zero establishes the Lorenz condition for the family. Q equal to zero makes its electric field vanish. A sine node at a particular query point establishes neither global statement. Keep the coefficients, instantaneous values, and family-level judgments visible as different quantities.

$$
Q=kv_0-\omega a_0,\quad C=\omega v_0-ka_0,\quad \mathbf e=(0,0,Q\sin\phi),\quad \mathbf b=0,\quad \widetilde G=C\sin\phi.
$$

## Sources assigned by the separated potential equations

Applying the normalized wave operator multiplies each cosine potential by minus D. The separated equations therefore assign the sources below. At this stage they are sources defined by those wave equations; we have not yet checked whether the actual fields require the same charge and current. Their continuity residual is D times C times sine of the phase. In particular, D equal to zero makes both assigned sources vanish, regardless of C. This is precisely why a source-conservation check alone can fail to expose a missing Lorenz constraint.

$$
\widetilde\rho_{\rm wave}=Dv_0\cos\phi,\quad \widetilde j_{{\rm wave},z}=Da_0\cos\phi,\quad \partial_{\tau}\widetilde\rho_{\rm wave}+\partial_{\zeta}\widetilde j_{{\rm wave},z}=DC\sin\phi.
$$

## Sources required by the actual fields

Compute charge from the divergence of the actual electric field, and current from curl b minus the time derivative of e. These field-implied sources satisfy continuity with each other, but can differ from the wave-assigned sources. The differences are the Gauss and Ampere residuals for the assigned request. A clear table should give these two source sets separate columns. Replacing the requested sources silently with the field-implied ones would conceal the incompatibility rather than solve the original problem. The distinction is equally important in inverse models that infer sources from predicted fields.

$$
\widetilde\rho_{\rm field}=kQ\cos\phi,\quad \widetilde j_{{\rm field},z}=\omega Q\cos\phi,\quad R_G=\omega C\cos\phi=-\partial_{\tau}\widetilde G,\quad R_{A,z}=kC\cos\phi=\partial_{\zeta}\widetilde G.
$$

## A resonant counterexample that is not a vacuum field solution

Set k and omega to one, scalar amplitude to one, and vector amplitude to zero. D vanishes, so the separated potential equations hold for zero sources, and those sources conserve charge. Nevertheless Q and C are both one. The electric field is nonzero in general and its divergence is nonzero. At phase zero the field value itself is zero while the Gauss residual equals one. Thus even a zero instantaneous field value does not establish a satisfied derivative constraint. This is a concrete counterexample to treating the separated potential wave equations as a complete vacuum verification.

$$
k=\omega=v_0=1,\quad a_0=0:\quad D=0,\quad Q=C=1,\quad \widetilde\rho_{\rm wave}=\widetilde j_{{\rm wave},z}=0,\quad R_G=R_{A,z}=\cos\phi.
$$

## A compatible forced example away from resonance

Choose k equal to two, omega equal to one, scalar amplitude one, and vector amplitude one half. C vanishes while Q and D are nonzero. Both constructions give the same nonzero charge and current, so the local Maxwell equations hold for that supplied source pair. Zero B and nonzero longitudinal E are not automatically impossible; the declared sources matter. This mode is a forced longitudinal teaching family rather than source-free radiation. Comparing it with the resonant counterexample prevents the lesson from becoming an indiscriminate rejection of longitudinal potential components.

$$
k=2,\ \omega=1,\ v_0=1,\ a_0=\tfrac12:\quad C=0,\quad Q=\tfrac32,\quad D=3,\quad \widetilde\rho=3\cos\phi,\quad \widetilde j_z=\tfrac32\cos\phi.
$$

## A free potential wave with only initial G equal to zero

Choose zero scalar potential and a standing longitudinal vector potential. Its free wave equation holds and the assigned sources vanish. Initially the vector potential and G are zero, but the electric field comes from the time derivative and is generally nonzero. That initial electric field can violate vacuum Gauss law. This example exposes the missing second initial constraint particularly clearly: setting the initial value of A does not determine its initial velocity. A numerical solver can reproduce the potential wave equation accurately while faithfully evolving incompatible physical initial data.

$$
v=0,\quad a_z=a\sin(k\zeta)\sin(k\tau),\quad e_z=-ak\sin(k\zeta)\cos(k\tau),\quad \widetilde G=ak\cos(k\zeta)\sin(k\tau).
$$

## Locate the missing initial time derivative

The missing initial time derivative is generally nonzero. At a=k=1 and zeta zero, initial G is zero, its time derivative is one, and the Gauss residual is minus one. The electric field value at that point is again zero. These values illustrate why constraints involving spatial derivatives cannot be inferred from a pointwise field value. Compatible initial electric-field data and suitable boundary data must be checked as part of the same problem. A boundary that injects nonzero G can also defeat otherwise appropriate zero initial gauge data.

$$
\widetilde G(\zeta,0)=0,\qquad \partial_{\tau}\widetilde G(\zeta,0)=ak^2\cos(k\zeta),\qquad R_G(\zeta,0)=-ak^2\cos(k\zeta).
$$

## Understand a constant nonzero G

Let the scalar potential be linear in normalized time and let the vector potential vanish. Then e and b are zero, while G is the constant gamma. Both free potential wave equations and vacuum Maxwell equations hold. Rejecting every representation with nonzero G would incorrectly reject this valid field description. The residual identities depend on derivatives of G. Zero G is a standard sufficient representation choice, not a necessary condition for every physically valid set of potentials. Report Lorenz compliance separately from compatibility with the requested physical sources.

$$
v=\gamma\tau,\quad \mathbf a=0:\quad \mathbf e=\mathbf b=0,\quad \widetilde G=\gamma,\quad \partial_{\tau}\widetilde G=0,\quad \nabla_{\xi}\widetilde G=0.
$$

## Evaluate near resonance and small coefficients

Near resonance, factoring the difference of squares can preserve information lost by direct subtraction. Q and C introduce a second issue: each product may round before the two products are subtracted. Compensated products help retain a small contrast between the exact binary inputs. Compare them with an independently generated high-precision reference, and distinguish an unrepresentable result from exact zero. No formula can recover a distinction already removed when the inputs were rounded to the same value. Numerical tolerances used in comparisons should not redefine a physically nonzero coefficient as identically zero.

$$
D=(k-\omega)(k+\omega),\qquad Q=kv_0-\omega a_0,\qquad C=\omega v_0-ka_0.
$$

## Separate the questions in an AI verification

For learned potentials, record the potential wave equations, source continuity, G, both initial gauge data, initial field constraints, boundary conditions, and the actual field Maxwell residuals. Collapsing these into one raw number can hide a decisive defect through different units or scales. Show normalized residuals alongside SI observations. An identity built into a prescribed family is not an independent numerical certificate. The background potential equations are discussed in [Feynman II §18–6](https://www.feynmanlectures.caltech.edu/II_18.html); the harmonic and initial-data counterexamples above are explicit derivations. Global topology, quantum gauges, and general initial-boundary existence theory remain further topics.

## Research extension: constraint energy and boundary influx

### From a constraint value to a propagation problem

Use the normalized coordinates and potential scales already defined, with a single spatial coordinate zeta and normalized speed one. Write g for the normalized Lorenz combination and s for the continuity defect of the sources assigned by the separated potential equations. Smooth potentials satisfying those equations obey the driven constraint wave equation. Its derivatives are exactly the Gauss and longitudinal Ampere residuals for that assigned request. A nonzero s is not a physically permitted creation of charge in Maxwell theory; it diagnoses an inconsistent source request or numerical approximation.

$$
g=\partial_\zeta a_z+\partial_\tau v,\quad s=\partial_\tau\widetilde\rho+\partial_\zeta\widetilde j_z,\quad g_{\tau\tau}-g_{\zeta\zeta}=s,\quad R_G=-g_\tau,\quad R_A=g_\zeta.
$$

### Derive a constraint energy

Multiply the wave equation by g_tau and integrate in space. Integrating the spatial term once by parts produces the boundary flux and the energy identity below. This positive functional measures derivatives of a representation constraint; it is not electric-plus-magnetic field energy in joules. Its two components directly measure the squared physical-equation defects in normalized units. A small value cannot determine a spatially constant g. Smoothness, the spatial domain and boundary data are part of the statement.

$$
\mathcal E_g=\tfrac12\int_0^\ell(g_\tau^2+g_\zeta^2)\,d\zeta,\qquad\frac{d\mathcal E_g}{d\tau}=\int_0^\ell g_\tau s\,d\zeta+[g_\tau g_\zeta]_0^\ell.
$$

### Which characteristic enters at which boundary?

For a right-moving function f(tau−zeta), w_minus is nonzero and w_plus is zero. Thus w_minus at the left edge and w_plus at the right edge are incoming data. The other two are outgoing. The boundary contribution is incoming squared amplitude minus outgoing squared amplitude, divided by four. Periodic boundaries cancel it. Homogeneous Dirichlet g also gives zero flux when the boundary data are compatible in time; zero incoming characteristics make it nonpositive. Zero initial data do not prevent a boundary from injecting a later constraint wave.

$$
w_\pm=g_\tau\pm g_\zeta,\qquad[g_\tau g_\zeta]_0^\ell=\tfrac14\left(w_+^2(\ell)+w_-^2(0)-w_-^2(\ell)-w_+^2(0)\right).
$$

### A bound includes the complete source history

With no positive boundary contribution, Cauchy–Schwarz bounds the volume term by sqrt(2 energy) times the L2 norm of s. Integrating the inequality gives the bound below; at zero energy it follows by a positive regularization and a limit. A small instantaneous continuity defect is insufficient to bound a long evolution without its time integral. The estimate has no damping term and does not promise decay. Discrete solvers need a corresponding discrete integration-by-parts identity and stable boundary implementation; a continuum estimate alone is not their certification.

$$
\sqrt{\mathcal E_g(\tau)}\le\sqrt{\mathcal E_g(0)}+\frac1{\sqrt2}\int_0^\tau\|s(\cdot,u)\|_{L^2}\,du\qquad\text{if the boundary contribution is nonpositive}.
$$

### The spatial mean is a separate degree of freedom

On a periodic interval the mean obeys a forced ordinary differential equation. The energy controls g minus its mean by the periodic Poincare inequality, but cannot see a constant mean. State the mean and its initial velocity separately. With zero source defect and zero energy, g is constant in both space and time. To select the Lorenz representation g=0, its initial constant must also be zero. Constant nonzero g can still describe valid Maxwell fields because the physical residuals involve derivatives. Representation compliance and physical compatibility remain separate judgments.

$$
\overline g=\ell^{-1}\int_0^\ell g\,d\zeta,\quad\overline g_{\tau\tau}=\overline s\ \text{(periodic)},\qquad\|g-\overline g\|_{L^2}\le\frac{\ell}{2\pi}\|g_\zeta\|_{L^2}.
$$

### Undamped initial errors persist

On a 2pi-periodic interval take a positive integer k. The standing mode has zero initial g but a generally nonzero initial derivative, and its constraint energy is constant. In the existing initial-data lab, g=a k cos(k zeta) sin(k tau), so A=a k and the energy is length times a squared times k to the fourth divided by four. A wave-equation loss and conserved assigned sources can therefore coexist with persistent Maxwell residuals. Increasing spatial resolution does not remove an incompatible continuum initial mode.

$$
g=A\cos(k\zeta)\sin(k\tau),\quad\ell=2\pi,\quad k\in\mathbb N_{>0}:\qquad g(\zeta,0)=0,\quad g_\tau(\zeta,0)=Ak\cos(k\zeta),\quad\mathcal E_g=\ell A^2k^2/4.
$$

### Build a boundary-injection counterexample from potentials

Let r_plus=max(r,0). The fourth-power potential below is sufficiently smooth for the displayed derivatives and solves the free potential wave equation. Both potentials and initial velocities vanish on the interval at tau=0. Assigned sources and their defect are zero. Nevertheless the left boundary sends in a front: g=e_z is nonzero behind it, and Gauss/Ampere residuals equal −3(tau−zeta) squared there. Before the front crosses the right boundary, direct integration gives the displayed growing energy; all of its growth comes from the left boundary. It is a potential-wave solution with incompatible vacuum field constraints, not a vacuum Maxwell radiation wave.

$$
v=\tfrac14(\tau-\zeta)_+^4,\quad a_z=0,\quad g=e_z=(\tau-\zeta)_+^3,\quad s=0:\qquad\mathcal E_g=9\tau^5/5,\quad\dot{\mathcal E}_g=9\tau^4\quad(0\le\tau\le\ell).
$$

### A small uniform source defect accumulates

On a periodic interval, choose the spatially uniform scalar potential below. The electric and magnetic fields are zero, while the separated scalar equation assigns charge delta times tau and zero current. Its continuity defect is the constant delta. The Gauss mismatch grows linearly; the constraint grows quadratically. The source-history bound is saturated in this example. The table uses length one and delta=0.1; it compares that source-driven example with the distinct boundary-front example for times up to one. The two energies are constraint diagnostics, not physical field energies and not two solutions of the same boundary problem.

$$
v=\delta\tau^3/6,\quad a_z=0,\quad\widetilde\rho_{\rm wave}=\delta\tau,\quad\widetilde j_{\rm wave}=0,\quad s=\delta,\quad g=\delta\tau^2/2,\quad R_G=-\delta\tau,\quad R_A=0,\quad\mathcal E_g=\ell\delta^2\tau^2/2.
$$

### Zero energy does not fix the Lorenz constant

The final exact family has a scalar potential linear in time, zero fields and constant nonzero g. Its constraint energy and both Maxwell residuals vanish. Rejecting it as a physical failure would confuse a representation choice with the field equations. For research verification, report g and its mean, both initial constraint functions, assigned-source continuity history, incoming boundary data and derivative residuals separately. The supplementary energy model adds no source-injection or boundary-wave controls to the existing prescribed-family lab.

$$
v=\gamma\tau,\quad a_z=0:\qquad g=\gamma,\quad\mathcal E_g=0,\quad R_G=R_A=e_z=b_z=0.
$$

| tau | uniform constraint energy | uniform Gauss defect magnitude | front constraint energy | front influx |
| --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| 0.25000000 | 0.00031250 | 0.02500000 | 0.00175781 | 0.03515625 |
| 0.50000000 | 0.00125000 | 0.05000000 | 0.05625000 | 0.56250000 |
| 1.00000000 | 0.00500000 | 0.10000000 | 1.80000000 | 9.00000000 |

```python
from fractions import Fraction as F
length, delta = F(1), F(1, 10)
for tau in [F(0), F(1, 4), F(1, 2), F(1)]:
    uniform_energy = length*delta**2*tau**2/2
    gauss_defect = abs(delta*tau)
    # Incoming polynomial front; this formula requires 0 <= tau <= length.
    incoming_energy = 9*tau**5/5
    incoming_power = 9*tau**4
    values = [tau, uniform_energy, gauss_defect, incoming_energy, incoming_power]
    print(" ".join(f"{float(x):.8f}" for x in values))
```

[Feynman II §18–6](https://www.feynmanlectures.caltech.edu/II_18.html) supplies the classical potential and gauge background. The finite-grid exactness/projection and the one-dimensional constraint energy examples here are derived explicitly under the stated domains and boundary assumptions.

![Persistent initial constraint, injected front and accumulated source error](/assets/learning/electromagnetism-lorenz-constraint-energy.en.svg)

### Research questions with explanations

**Can zero initial constraint data fail later with conserved assigned sources?** **Answer:** Yes. Incoming boundary data can inject a nonzero constraint wave. The polynomial front gives an explicit potential-wave example.

**What does zero constraint energy establish on a connected space-time region?** **Answer:** Both derivatives of g vanish, so g is constant. Its value still needs a separate initial mean or representation condition.

**Does a small instantaneous continuity loss guarantee a small long-time defect?** **Answer:** No. The estimate contains the time integral of the source norm and boundary influx. A constant small defect accumulates in the uniform example.

## Exercises with explanations

### Exercise 1

Do zero sources and a homogeneous wave equation imply G=0?

**Answer:** No. A homogeneous wave equation admits nonzero solutions selected by initial or boundary data. Maintaining zero G requires the appropriate two initial data and compatible boundary conditions.

### Exercise 2

For k=omega=v0=1, a0=0, phase zero, find E and the Gauss residual.

**Answer:** The normalized field is zero but the Gauss residual is one. Free potential wave equations and conserved assigned sources are insufficient for vacuum compatibility.

### Exercise 3

For k=2, omega=1 and v0=1, choose a0 for C=0 and calculate sources.

**Answer:** Choose a0=1/2. Q=3/2 and D=3 give normalized charge 3cos(phi) and current 1.5cos(phi), matching both constructions.

### Exercise 4

In the standing-wave counterexample at a=k=1, zeta=tau=0, calculate both initial gauge data.

**Answer:** G is zero and its normalized time derivative is one. The Gauss residual is minus one, exposing the missing initial derivative constraint.

### Exercise 5

Must a constant normalized G=2 violate vacuum Maxwell equations?

**Answer:** No. Take v=2tau and a=0. Both fields and both derivatives of G vanish. The representation is non-Lorenz but describes a valid vacuum field.

### Exercise 6

Convert normalized charge density 3 using L=1m and U=2V.

**Answer:** Use rho=(epsilon U/L^2) times normalized rho, giving 6epsilon C/m^3. Specify the medium permittivity rather than attaching charge units directly to the number three.
