## Where does accumulating charge come from?

If charge increases inside a region, more charge must enter through its boundary than leave. This intuition becomes the continuity equation. When the electric field changes in time, conduction current alone is insufficient for the magnetic curl law. Maxwell displacement current connects the two relationships.

We calculate an electric field that grows around the origin, then separately test charge conservation, conduction and displacement currents, and the Gauss constraint.

## Units of conduction and displacement current

Conduction current density J describes charge transport. The quantity ε∂tE also has units A/m², but it is not the same description of particle flow. Displacement current is defined from the changing electric field. In this constant-coefficient model, their sum supplies the magnetic curl.

Free and bound charges, polarization, magnetization, and constitutive laws for D and H require a separate material description. Here we test consistency between prescribed fields and sources.

$$
\mathbf J_{\mathrm{disp}}=\epsilon\partial_t\mathbf E,\qquad\nabla\times\mathbf B=\mu(\mathbf J+\mathbf J_{\mathrm{disp}}).
$$

## Derive charge conservation

The divergence of a curl is zero. Take divergence of the Ampere–Maxwell equation, interchange space and time derivatives for smooth fields, and use Gauss law to obtain continuity. This derivation uses the Gauss constraint too. Omitting displacement leaves ∇·J=0, which generally conflicts with changing charge.

$$
0=\nabla\cdot\mathbf J+\epsilon\partial_t(\nabla\cdot\mathbf E)=\nabla\cdot\mathbf J+\partial_t\rho.
$$

## Charge balance across a closed boundary

Use outward normals on a fixed volume V. Positive outward current decreases enclosed charge, explaining the minus sign. Charge Q is in C, current in A, and time-integrated current in C. Moving control volumes require the appropriate transport theorem.

$$
\frac{dQ_V}{dt}=-\oint_{\partial V}\mathbf J\cdot d\mathbf S,\qquad Q_V(t_1)-Q_V(t_0)=-\int_{t_0}^{t_1}\oint_{\partial V}\mathbf J\cdot d\mathbf S\,dt.
$$

## A prescribed radial field on a bounded domain

Prescribe E=a(t)r and B=0 inside a coordinate box. The units of a are V/m²; a₁ has units V/(m²·s). Each electric component is proportional to its corresponding coordinate. Prescribing a uniform charge density and compatible boundary current produces a consistent family.

This is a manufactured solution on a bounded domain supplied by boundary currents, not a finite isolated apparatus in infinite space. It does not solve carrier dynamics or a material response.

$$
a(t)=a_0+a_1t,\quad\mathbf E=a(t)(x,y,z),\quad\mathbf B=0,\quad\rho=3\epsilon a(t),\quad\mathbf J=-\epsilon a_1(x,y,z).
$$

## Visual intuition for accumulation

For a>0 the electric arrows point outward and grow with distance from the origin. If a₁>0 they lengthen over time. Conduction current then points inward, supplying the increasing charge density. Displacement-current arrows point outward and cancel conduction current.

Setting conduction current to zero would prevent retaining this same changing electric field with B=0. This illustrates why choosing fields and assigning unrelated sources afterward can fail.

## Check each local relationship

Differentiating each coordinate component gives ∇·E=3a, while both curls vanish. The sum of J and ε∂tE is zero, consistent with B=0 in Ampere–Maxwell. The charge-density time derivative is 3εa₁ and current divergence is −3εa₁. These test separate relationships; one cancellation does not verify every law.

$$
\begin{aligned}\nabla\cdot\mathbf E&=3a=\rho/\epsilon,&\mathbf J+\epsilon\partial_t\mathbf E&=0,\\\partial_t\rho&=3\epsilon a_1,&\nabla\cdot\mathbf J&=-3\epsilon a_1.\end{aligned}
$$

## Zero field at the origin

At the origin E=J=0, yet charge density and its time derivative can be nonzero. Divergence measures surrounding variation, not arrow length at one point. Avoid inferring zero density from a zero field value.

For neural fields, fitting sampled values and fitting derivatives or conservation relationships are separate validation tasks.

## Sphere flux and enclosed charge

On a sphere of radius R centered at the origin, E·n=aR. Multiplying by surface area gives electric flux. Multiplying uniform density by sphere volume gives enclosed charge. Conduction and displacement currents through that sphere are equal and opposite. Here Iout denotes outward conduction current.

$$
\Phi_E=4\pi aR^3,\quad Q=4\pi\epsilon aR^3,\quad I_{\mathrm{out}}=-4\pi\epsilon a_1R^3,\quad I_{\mathrm{disp,out}}=4\pi\epsilon a_1R^3,\quad\dot Q=-I_{\mathrm{out}}.
$$

## A complete numerical example

Choose normalized ε=1, a₀=1, a₁=1/2, t=2, and R=1. Then a=2. At (1,0,0), E=(2,0,0), J=(−1/2,0,0), and displacement current density is (+1/2,0,0). Charge density is 6, its time derivative is 3/2, and current divergence is −3/2.

Electric flux is 8π, enclosed charge is 8π, outward conduction current is −2π, and outward displacement current is +2π. Between t=0 and 2, charge increases by 4π. Integrating constant inward current 2π over two seconds also gives 4π. Distinguish this normalized calculation from default SI coefficients.

## Conservation can pass while Gauss law fails

Keep the fields and current, but add a time-independent offset δρ to charge density. Its time derivative is unchanged, so continuity still passes. Evolution equations are also unchanged. Gauss law fails because electric divergence no longer matches density. This changes the prescribed source; it is a negative control, not a gauge transformation.

Adding δρ=0.3 to the preceding example gives density 6.3 and enclosed charge 8.4π, while electric flux remains 8π. This is not a complete Maxwell solution. Display the initial constraint violation explicitly.

$$
g=\nabla\cdot\mathbf E-\rho/\epsilon=-\delta\rho/\epsilon,\qquad \epsilon\Phi_E-Q=-\delta\rho\,\frac{4\pi R^3}{3}.
$$

## Why initial constraints matter

Under constant coefficients and sufficient smoothness, Ampere–Maxwell plus continuity gives zero time derivative of g. Existing error can persist; it is not automatically corrected to zero. The initial condition g=0 is necessary. Faraday similarly propagates the magnetic divergence constraint. Boundaries and discretization require additional care in numerical solvers.

$$
\partial_tg=\partial_t\left(\nabla\cdot\mathbf E-\rho/\epsilon\right)=0,\qquad\partial_t(\nabla\cdot\mathbf B)=0.
$$

## Small values and different computation paths

Adding 10⁻¹⁶ to density 3 may still store the rounded value 3. Obtaining zero residual by subtracting rounded values differs from prescribing exactly zero offset. The model separates the stable defect derived from the offset from a residual formed from rounded endpoints.

Finite-time charge change likewise uses a₁Δt rather than subtracting two large charges. For a very small sphere, flux can remain representable while charge multiplied by ε falls outside arithmetic range. Do not substitute a physical zero. At R=0, flux and charge are zero, but an area-normalized mean is undefined.

$$
\Delta Q=4\pi\epsilon a_1R^3(t_1-t_0)=-I_{\mathrm{out}}(t_1-t_0).
$$

## Design checks for an AI model

A generative model trained only on conservation may miss a constant density offset. Check initial Gauss constraints, evolution, charge balance, and boundary data separately, with negative controls. If charge is defined structurally from electric divergence, passing that relationship is not independent validation.

Separate references sum actual three-dimensional sphere normals against E and J, and compare time differences of charge and electric field with spatial differences of current. This symmetric example is a starting point, not an accuracy guarantee for general meshes, boundaries, or difficult fields.

## Exercises with explanations

Specify the current normal before calculating.

### Exercise 1: Current sign

For a₁>0, what is the sign of outward conduction current? **Answer:** Negative. Inward J increases enclosed charge. Outward displacement current is positive.

### Exercise 2: Density at the origin

For a=2 and ε=1, find E and ρ at the origin. **Answer:** E=0 but ρ=6. Spatial variation around the point gives divergence 6.

### Exercise 3: Charge balance

For a₁=1/2, ε=R=1, and Δt=2, find ΔQ. **Answer:** 4π. Negating outward current −2π and multiplying by duration 2 gives the same result.

### Exercise 4: A density offset

For δρ=0.3 and ε=1, find the local Gauss residual. **Answer:** −0.3. Continuity can still pass, so conservation alone misses this defect.

### Exercise 5: Zero current and electrostatics

If a₁=0 but a₀ is nonzero, do field and charge vanish? **Answer:** No. Conduction current, displacement current, and charge rate vanish, but static E=a₀r and ρ=3εa₀ remain.

### Exercise 6: Set the radius to zero

Can flux divided by area define a mean field at R=0? **Answer:** No; the ratio is 0/0. Distinguish a limiting sequence of shrinking spheres from exactly zero-area geometry.

## References and next connections

Compare §§13.1–13.3 of [MIT 8.02T Chapter 13](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/13e4e0069a9ffca5ae3f909083695c2b_chapte13em_waves.pdf). Later induction, radiation, and electromagnetic energy lessons extend boundaries, materials, and more complex time-dependent fields.
