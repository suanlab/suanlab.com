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

## Research extension: continuity defects and the cost of Gauss correction

### Track two discrete obligations separately

Consider a longitudinal electric field on N≥3 equally spaced periodic edges. E_j labels the outgoing edge from node j to node j+1; indices wrap modulo N. Charge density lives at the nodes and current on the edges. There is no longitudinal magnetic curl term in this restricted update. Define g with charge-density units; it differs by a factor ε from the original radial lesson’s field-divergence residual. Substitution of the Ampere step into this definition proves the next identity term by term. A consistent continuity update keeps an existing g unchanged, while a persistent defect r accumulates a Gauss error.

$$
(GE)_j=\frac{E_j-E_{j-1}}{\Delta z},\quad g=\epsilon GE-\rho,\quad [G]={\rm m}^{-1},\quad[g]={\rm C\,m}^{-3}.
$$

$$
E_{\rm raw}^{n+1}=E^n-\frac{\Delta t}{\epsilon}J^{n+1/2},\quad r=\frac{\rho^{n+1}-\rho^n}{\Delta t}+GJ^{n+1/2},\quad g_{\rm raw}^{n+1}=g^n-\Delta t\,r.
$$

### Check whether the requested source is representable

Every outgoing edge occurs once positively and once negatively in the sum of GE. Therefore periodic Gauss law requires zero total cell charge. A_c is the constant cross-sectional area. The constant component of the residual cannot be changed by any periodic E; Cauchy–Schwarz gives the lower bound below. This is a solvability condition for this closed periodic model. The original radial solution is on a bounded domain with boundary current and is not subject to this periodic zero-charge assumption. Silently subtracting the mean charge introduces a different physical source or neutralizing background.

$$
\mathbf1^TGE=0\ \Longrightarrow\ \mathbf1^T\rho=0,\quad Q_{\rm cell}=A_c\Delta z\,\mathbf1^T\rho,\quad\|\epsilon GE-\rho\|_2\ge\sqrt N\,|\overline\rho|.
$$

### Derive the smallest compatible field correction

For a zero-mean defect, seek the smallest Euclidean change on this uniform grid. Use the divided constraint GδE=−g/ε and multiplier p/ε to obtain δE=−Gᵀp/ε; imposing the constraint yields GGᵀp=g. The matrix has the constant null vector, so fix the mean of p to zero and solve only compatible data. In physical units p has units C/m and δE has units V/m. To prove minimality, any other feasible correction is δE+h with Gh=0. Its cross term is zero because δE lies in the range of Gᵀ; its squared norm therefore adds ||h||². Different meshes or physical weights require the corresponding adjoint and metric.

$$
\min_{\delta E}\frac12\|\delta E\|_2^2\quad\text{subject to}\quad\epsilon G\delta E=-g:\qquad GG^Tp=g,\quad\mathbf1^Tp=0,\quad\delta E=-\frac{G^Tp}{\epsilon},\quad E_c=E_{\rm raw}+\delta E.
$$

### A field correction changes physics

The mean E is preserved because G times a constant edge field is zero. But pointwise E, forces on charges and energy generally change: this is not a gauge transformation. Correcting one snapshot does not retroactively make the same prescribed current obey the previous Ampere step. If the previous Gauss defect was zero, reinterpreting that single corrected step with the effective current below restores discrete continuity as well. This explicitly changes the current; it is not evidence that an unchanged source history satisfies all Maxwell equations.

$$
\overline{E_c}=\overline{E_{\rm raw}},\quad J_{\rm eff}=J-\frac{\epsilon\delta E}{\Delta t},\quad g^n=0\ \Longrightarrow\ \frac{\rho^{n+1}-\rho^n}{\Delta t}+GJ_{\rm eff}=0.
$$

### Understand when energy falls and when it rises

For zero target charge, the corrected E is in ker G and the correction lies in range Gᵀ. Orthogonality gives an exact energy decomposition. For a nonzero compatible charge target, the constraint is affine: the cross term need not vanish. Expansion of the square gives the general energy change below, whose sign can be either positive or negative. Thus “Gauss cleaning always removes energy” is not a valid statement for arbitrary nonzero target charge.

$$
U(E)=\frac{\epsilon A_c\Delta z}{2}\|E\|_2^2,\qquad\rho=0:\quad E_c\perp\delta E,\quad U(E_{\rm raw})=U(E_c)+U(\delta E).
$$

$$
U(E_c)-U(E_{\rm raw})=\epsilon A_c\Delta z\left(E_{\rm raw}^T\delta E+\frac12\|\delta E\|_2^2\right).
$$

### Work a four-edge example in declared scales

Let E∗ be a positive field scale, ρ∗=εE∗/Δz and U∗=εA_cΔzE∗². Hats denote fields and charge divided by these scales; G_hat=Δz G. For four edges the explicit dimensionless pseudoinverse below annihilates constants. Multiplying by G_hat G_hatᵀ gives the projector onto zero-mean node vectors, which verifies rather than assumes its role. Use p_hat=(G_hat G_hatᵀ)^+ g_hat and E_c_hat=E_raw_hat−G_hatᵀ p_hat. The chosen source has zero sum for every α. For α=0 the correction removes all of the raw field; for α=1 it changes nothing; for α=2 it doubles the field and quadruples the energy. The Python cell computes the projection, checks the resulting Gauss law and prints the table. These are supplementary grid calculations, not new controls in the radial lab.

$$
\widehat G=\Delta zG,\quad(\widehat G\widehat G^T)^+=\frac1{16}\begin{pmatrix}5&-1&-3&-1\\-1&5&-1&-3\\-3&-1&5&-1\\-1&-3&-1&5\end{pmatrix},\quad (\widehat G\widehat G^T)(\widehat G\widehat G^T)^+=I-\frac{\mathbf1\mathbf1^T}{4}.
$$

$$
\widehat E_{\rm raw}=(1,0,-1,0)^T,\quad\widehat\rho=\alpha(1,-1,-1,1)^T,\quad\widehat E_c=\alpha\widehat E_{\rm raw},\quad\frac{U(E_{\rm raw})}{U_*}=1,\quad\frac{U(E_c)}{U_*}=\alpha^2,\quad\frac{\|g\|_2}{\rho_*}=2|1-\alpha|.
$$

| alpha | raw defect / rho∗ | raw U / U∗ | corrected U / U∗ |
| --- | --- | --- | --- |
| 0.00000000 | 2.00000000 | 1.00000000 | 0.00000000 |
| 0.50000000 | 1.00000000 | 1.00000000 | 0.25000000 |
| 1.00000000 | 0.00000000 | 1.00000000 | 1.00000000 |
| 2.00000000 | 2.00000000 | 1.00000000 | 4.00000000 |

```python
from math import sqrt
M = [[5,-1,-3,-1],[-1,5,-1,-3],
     [-3,-1,5,-1],[-1,-3,-1,5]]
E = [1.0, 0.0, -1.0, 0.0]
div = lambda x: [x[j]-x[(j-1)%4] for j in range(4)]
for alpha in [0.0, 0.5, 1.0, 2.0]:
    rho = [alpha*x for x in div(E)]
    assert sum(rho) == 0, "Incompatible periodic source"
    g = [d-r for d,r in zip(div(E),rho)]
    p = [sum(m*x for m,x in zip(row,g))/16 for row in M]
    Ec = [E[j]-(p[j]-p[(j+1)%4]) for j in range(4)]
    assert div(Ec) == rho
    raw = sum(x*x for x in E)/2
    corrected = sum(x*x for x in Ec)/2
    print(f"{alpha:.8f} {sqrt(sum(x*x for x in g)):.8f} {raw:.8f} {corrected:.8f}")
```

The existing radial lab gives an independent contrast: a fixed charge offset leaves continuity and total current unchanged but preserves a Gauss defect over time. Its boundary-fed geometry is different from the periodic projection. [Gkeyll’s Maxwell documentation](https://gkeyll.readthedocs.io/en/latest/dev/maxwell-eigensystem.html) discusses divergence-control approaches as numerical context. The finite projection, compatibility condition, effective-current identity and energy examples here follow from the displayed matrices; they are not a benchmark claim about that solver.

![Continuity error, corrected edge fields and electric energy](/assets/learning/electromagnetism-periodic-gauss-repair.en.svg)

### Research questions with explanations

**Can periodic field cleaning fix a nonzero total charge?** **Answer:** No. Every periodic divergence has zero sum. Altering the mean source or adding a background changes the physical problem.

**Does the smallest Gauss correction always lower energy?** **Answer:** No. The zero-target orthogonal identity is special. At alpha=2 in this affine example energy rises from U∗ to 4 U∗.

**Does corrected Gauss law certify the original source history?** **Answer:** No. A single snapshot correction changes the field. Keeping a compatible Ampere step generally requires the explicitly changed effective current.

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
