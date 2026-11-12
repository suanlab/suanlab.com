## Follow the energy, not just the field arrows

A changing magnetic field can store more energy inside a region even when no charged particle crosses its boundary. What supplies that energy? To answer, draw a fixed region and count three things: the energy stored inside, energy crossing the boundary, and work done on matter inside. A water tank gives a useful bookkeeping analogy, but electromagnetic energy flux is a vector field, not a stream of material particles.

This lesson reuses the bounded prescribed induction field. Review dot and cross products, divergence, the divergence theorem, and time derivatives first. We will connect a local equation in W/m³ to an integrated equation in W. The key skill is reading the sign of every term before trying to make a residual vanish.

## State the assumptions before defining energy

Use constant positive scalar permittivity ε and permeability μ, with a medium at rest. In this teaching model the field energy density and Poynting vector take the following forms. The familiar vacuum formulas follow by using the vacuum coefficients.

$$
u_E=\frac{\epsilon|\mathbf E|^2}{2},\qquad u_B=\frac{|\mathbf B|^2}{2\mu},\qquad u=u_E+u_B,\qquad \mathbf S=\frac{\mathbf E\times\mathbf B}{\mu}.
$$

Here u has units J/m³ and S has units W/m². These formulas do not automatically describe energy stored in a dispersive, hysteretic, nonlinear, or moving material. Such systems need constitutive and mechanical terms. Calling every occurrence of J·E “heat” would also be too strong: it is field-to-matter power density, whose eventual destination depends on the matter model.

## Derive the local balance

Dot Ampere–Maxwell with E and Faraday with B/μ. The vector identity below converts their curl terms into a divergence. Constant coefficients let us identify εE·∂tE and B·∂tB/μ as time derivatives of the two energy densities.

$$
\nabla\cdot(\mathbf E\times\mathbf B)=\mathbf B\cdot(\nabla\times\mathbf E)-\mathbf E\cdot(\nabla\times\mathbf B),\qquad \partial_tu+\nabla\cdot\mathbf S+\mathbf J\cdot\mathbf E=0.
$$

If J·E is positive, the field transfers energy to matter. If it is negative, matter or an imposed source supplies field energy. Positive divergence means a net outward flow locally. Thus increasing field storage needs inward flow, negative field-to-matter work, or both. No single term must vanish by itself.

## Integrate over a fixed region

Let the normal n point outward. For a fixed volume, differentiating the volume integral introduces no moving-boundary transport term. The divergence theorem gives the integral statement.

$$
U=\int_Vu\,dV,\qquad P_{\rm out}=\oint_{\partial V}\mathbf S\cdot\mathbf n\,dA,\qquad W_{\rm matter}=\int_V\mathbf J\cdot\mathbf E\,dV,\qquad \frac{dU}{dt}+P_{\rm out}+W_{\rm matter}=0.
$$

U is in J; its derivative, Pout, and Wmatter are in W. Wmatter here names an instantaneous power, not an accumulated work in J. Negative Pout means energy enters. A moving control volume requires an additional transport calculation and is outside this fixed-cylinder example.

## Reuse the prescribed induction fields

Inside the supported bounded region, prescribe Bz=b(t) and a circulating E. The current required by Ampere–Maxwell opposes the displacement current. The magnetic field is spatially uniform, so its curl is zero.

$$
b=b_0+b_1t+\tfrac12b_2t^2,\quad \mathbf B=(0,0,b),\quad \mathbf E=(b'y/2,-b'x/2,0),\quad \mathbf J=(-\epsilon b_2y/2,\epsilon b_2x/2,0).
$$

These fields require prescribed sources and boundary data. They are not a solution for an isolated finite solenoid, a complete circuit, or the mechanical response of a conductor. Using them is useful because each energy term can be derived without hiding a numerical PDE solver.

## Work out storage and flow at a point

Write r²=x²+y². Squaring E gives b′²r²/4. Taking E×B gives an inward radial flux when bb′>0. Reversing the sign of b without changing b′ can reverse that flow.

$$
u_E=\frac{\epsilon b'^2r^2}{8},\quad u_B=\frac{b^2}{2\mu},\quad \mathbf S=-\frac{bb'}{2\mu}(x,y,0),\quad \mathbf J\cdot\mathbf E=-\frac{\epsilon b'b_2r^2}{4}.
$$

For b′>0 the electric arrows circulate clockwise, but the Poynting arrows point radially inward when b is also positive. The two arrow diagrams represent different quantities. A cross product changes both direction and units; rotating an E arrow visually does not supply a quantitative energy flux.

## Read the two cancellations separately

The magnetic storage rate cancels the divergence of S. The electric storage rate cancels source work. Keeping these pairings visible prevents a large term from hiding a small missing contribution.

$$
\partial_tu_B=\frac{bb'}{\mu},\quad \nabla\cdot\mathbf S=-\frac{bb'}{\mu},\qquad \partial_tu_E=\frac{\epsilon b'b_2r^2}{4},\quad \mathbf J\cdot\mathbf E=-\partial_tu_E.
$$

If b₂=0, electric storage is constant and J·E is zero. Changing magnetic storage can still be supplied through the boundary. Zero local source work does not imply zero electromagnetic energy transfer.

## Calculate the fixed cylinder

Choose radius R and height h with its axis along z. The flux through the top and bottom is zero because Sz=0. Integrating the radial energy density requires the volume element 2πrh dr. The side surface has area 2πRh and an outward radial normal.

$$
U_E=\frac{\epsilon\pi hR^4b'^2}{16},\quad U_B=\frac{\pi R^2hb^2}{2\mu},\quad P_{\rm out}=-\frac{\pi R^2hbb'}{\mu},\quad W_{\rm matter}=-\frac{\epsilon\pi hR^4b'b_2}{8}.
$$

Differentiate UE and UB and compare with the two powers. The R⁴ scaling of electric storage differs from the R² scaling of magnetic storage. Doubling R therefore does more than multiply every quantity by the same volume factor.

## A worked example with normalized coefficients

To see the arithmetic, choose ε=1 F/m, μ=1 H/m, b₀=1 T, b₁=0.5 T/s, b₂=0.2 T/s², and t=1 s. These are pedagogical coefficients, not the vacuum constants. Then b=1.6 T and b′=0.7 T/s. At r=1 m, uE=0.06125 J/m³, uB=1.28 J/m³, the electric storage rate is 0.035 W/m³, the magnetic rate is 1.12 W/m³, divergence is −1.12 W/m³, and J·E=−0.035 W/m³.

For R=h=1 m, UE=0.0962113 J and UB=4.021239 J. Their rates are 0.0549779 W and 3.518584 W. Pout=−3.518584 W and Wmatter=−0.0549779 W, so both pairs close. Distinguish the local point values from the integrated cylinder values.

## Remove the required current deliberately

Keep E and B but set conduction current to zero. Faraday still holds, while Ampere–Maxwell generally fails if b₂≠0. The energy residual becomes the electric storage rate. At r=0 it happens to vanish, so one successful probe cannot certify the entire field family.

$$
R_{\rm energy}^{J=0}=\partial_tu_E=\frac{\epsilon b'b_2r^2}{4}.
$$

With physical SI coefficients and microtesla-scale b, the magnetic rate can be about 7.125×10⁻⁷ W/m³ while the missing electric term is about 3.683×10⁻²⁶ W/m³. Ordinary floating-point addition can hide the latter and report a zero total residual. The implementation displays paired components and stable grouping alongside ordinary summation. A rearranged analytic identity still needs an independent reference; it is not a numerical proof by itself.

## Why an energy residual is insufficient

A static E=(0,x,0), B=J=0 has zero energy time derivative, zero Poynting vector, and zero source work. Its energy residual is zero, but curl E has z component 1, violating Faraday. Energy balance is a consequence of the complete compatible equations; it is not a replacement for every equation.

For an AI field predictor, examine vector Maxwell residuals, source consistency, boundary data, energy components, and independent integral observations. Normalizing only by a dominant magnetic term can erase an important electric defect from a training objective.

## Numerical meaning and independent checks

The Python reference generator calculates energy from vector dot products and S from an actual cross product. It compares five-point time and space differences, radial volume quadrature, and oriented side-surface quadrature over 81 selected conditions. A separate 180-digit calculation checks the small SI defect. These measured discrepancies are not rigorous error bounds.

The TypeScript family marks lost nonzero intermediates unavailable rather than replacing them with physical zero. Empty cylinders have exact zero integrals; a nonzero cylinder whose volume cannot be represented is a different case. The supported coordinates and coefficients are explicit bounds, not silently clamped inputs.

## Read a storage-and-flow diagram

Draw the cylinder as a box in an energy-accounting diagram. Attach boundary arrows to its surface and source-work arrows inside. Storage changes belong inside the box; Pout and Wmatter are signed outgoing channels. For the normalized example, the following powers make the diagram quantitative.

| Channel | Power in W | Interpretation |
| --- | ---: | --- |
| Magnetic storage change | +3.518584 | Magnetic energy increases |
| Outward boundary flow | −3.518584 | Energy enters through the side |
| Electric storage change | +0.0549779 | Electric energy increases |
| Field-to-matter work | −0.0549779 | The prescribed source supplies energy |

Start by removing b₂ and observe which pair vanishes. Then restore b₂ and remove only J: the source-work arrow disappears while electric storage still grows. This comparison connects the missing term to a physical accounting failure rather than merely a red residual label.

## Research extension: power cannot identify every current component

### Ask what a power observation actually determines

Assume the electric field is known. A power-density measurement gives one scalar projection of a three-component current, not the whole current vector. The formula below separates the determined parallel part from an arbitrary perpendicular part. At E=0 the projection formula is undefined: power is zero for every J, so the datum determines even less. In a learned field model, a zero energy residual can therefore coexist with an incorrect transverse current. Initial constraints, continuity, vector Maxwell equations and boundary data remain distinct tests.

$$
p=\mathbf J\cdot\mathbf E,\quad[p]={\rm W/m^3},\qquad\mathbf E\ne0:\quad\mathbf J=\frac{p}{|\mathbf E|^2}\mathbf E+\mathbf J_\perp,\quad\mathbf E\cdot\mathbf J_\perp=0.
$$

### Build an energy-invisible negative control

In the bounded induction family above E_z=0. Add an axial current q(x,y,t) that is independent of z while keeping E, B and charge unchanged. Its divergence is zero and it does no work against E, so continuity and the entire energy accounting are unchanged. Yet the Ampere residual changes by −μq along z. Thus these are deliberately incompatible source predictions when q is nonzero, not alternative Maxwell solutions for the same fields. Boundary currents and supports belong to the stated bounded model; periodic or isolated-system assumptions are not added silently.

$$
\delta\mathbf J=q(x,y,t)\mathbf e_z:\quad\nabla\cdot\delta\mathbf J=0,\quad\mathbf E\cdot\delta\mathbf J=0,\quad\delta\mathbf r_A=-\mu q\mathbf e_z,\quad\mathbf r_A=\nabla\times\mathbf B-\mu(\mathbf J+\epsilon\partial_t\mathbf E).
$$

### Separate dissipative and transverse tensor response

For a new local identification example, assume a real, instantaneous, spatially homogeneous planar law J=ΣE, with no imposed relation between its four coefficients. Split Σ into a symmetric S and an antisymmetric A. A ninety-degree rotation of E is perpendicular to E, so its work is zero. Equivalently, the scalar EᵀAE equals its negative after transposition. Every power experiment is blind to h, even if arbitrarily many field directions are available. A Hall-like response provides physical motivation, but this algebra does not claim quantization, a measured material or a full Drude parameter relation. Additional constitutive priors can reduce ambiguity; that information comes from the priors.

$$
\mathbf J=\Sigma\mathbf E,\quad \Sigma=S+A,\quad S=\begin{pmatrix}a&b\\b&d\end{pmatrix},\quad A=\begin{pmatrix}0&-h\\h&0\end{pmatrix},\quad [a]=[b]=[d]=[h]={\rm S/m}.
$$

$$
\mathbf E^TA\mathbf E=0,\qquad p=\mathbf E^TS\mathbf E=aE_x^2+2bE_xE_y+dE_y^2.
$$

### Test passivity for every direction

In this restricted instantaneous resistive model, nonnegative power for every field is equivalent to S being positive semidefinite. If a>0, complete the square: p=a(E_x+bE_y/a)²+(d−b²/a)E_y². If a=0, positivity requires b=0 and d≥0. This explains the determinant condition including its degenerate boundary. Positive powers on a few probes are insufficient. For S=(1,2;2,1) S/m, the fields (1,0), (0,1) and (1,1) V/m give powers 1, 1 and 6 W/m³, yet (1,−1) gives −2 W/m³. General J·E can be negative when sources or reversible storage supply energy; do not apply this resistive passivity test to every material or every prescribed current.

$$
p\ge0\ \text{for all planar }\mathbf E\quad\Longleftrightarrow\quad S\succeq0\quad\Longleftrightarrow\quad a\ge0,\ d\ge0,\ ad-b^2\ge0.
$$

### Design calibrated probes and expose their blind column

Use exactly known field components with E∗>0 and dimensionless δ. The third probe has magnitude E∗sqrt(1+δ²), so these are not equal-amplitude directions. P_i is the actual power density; p_i=P_i/E∗² has conductivity units. The h column is identically zero. The three-column symmetric block has determinant −2δ: it is invertible for δ≠0, while δ=0 duplicates the first probe and also loses b. With independent equal-variance errors of variance v_p in p_i, error propagation yields the variance below without requiring Gaussian noise. As δ approaches zero the inverse becomes ill conditioned. Unequal or correlated errors and uncertain E∗ need their own propagation; the plotted curve holds absolute p-error variance fixed.

$$
\mathbf E_1=E_*(1,0),\quad\mathbf E_2=E_*(0,1),\quad\mathbf E_3=E_*(1,\delta),\quad p_i=\frac{P_i}{E_*^2},\quad\begin{pmatrix}p_1\\p_2\\p_3\end{pmatrix}=\begin{pmatrix}1&0&0&0\\0&0&1&0\\1&2\delta&\delta^2&0\end{pmatrix}\begin{pmatrix}a\\b\\d\\h\end{pmatrix}.
$$

$$
\delta\ne0:\quad a=p_1,\quad d=p_2,\quad b=\frac{p_3-p_1-\delta^2p_2}{2\delta},\qquad\operatorname{Var}(\widehat b)=v_p\frac{2+\delta^4}{4\delta^2}\quad\text{(independent equal-variance }p_i\text{ errors)}.
$$

### Add a different observation to identify h

After recovering b, one calibrated transverse-current observation at E=E∗e_x identifies h. Changing h changes a physical current vector, rather than a gauge reference, despite leaving power unchanged. The table uses a=2, b=1 and d=3 S/m, whose symmetric determinant is 5 (S/m)². It varies h in units of 1 S/m: J_y changes while J_x and power stay fixed. In the displayed matrix h_bar is h divided by 1 S/m. The constitutive example is separate from the original manufactured induction solution; substituting arbitrary ΣE into that solution does not establish Maxwell compatibility.

$$
\mathbf E=E_*\mathbf e_x:\quad J_y=(b+h)E_*,\quad h=J_y/E_*-b,\qquad (\Sigma_{h_2}-\Sigma_{h_1})\mathbf E=(h_2-h_1)(-E_y,E_x)^T.
$$

$$
S=\begin{pmatrix}2&1\\1&3\end{pmatrix}{\rm S/m},\quad E_*=1\ {\rm V/m}:\quad\bar h=\frac{h}{1\ {\rm S/m}},\quad\Sigma_h=\begin{pmatrix}2&1-\bar h\\1+\bar h&3\end{pmatrix}\ {\rm S/m},\quad\mathbf J=(2,1+\bar h)^T\ {\rm A/m^2},\quad p=2\ {\rm W/m^3}.
$$

| h in units of S/m | Jx [A/m2] | Jy [A/m2] | power [W/m3] |
| --- | --- | --- | --- |
| -2.00000000 | 2.00000000 | -1.00000000 | 2.00000000 |
| 0.00000000 | 2.00000000 | 1.00000000 | 2.00000000 |
| 1.00000000 | 2.00000000 | 2.00000000 | 2.00000000 |
| 3.00000000 | 2.00000000 | 4.00000000 | 2.00000000 |

```python
# Entries and h in S/m; E=(1,0) V/m; J in A/m^2.
a, b, d = 2.0, 1.0, 3.0
E = [1.0, 0.0]
for h in [-2.0, 0.0, 1.0, 3.0]:
    Sigma = [[a,b-h],[b+h,d]]
    J = [sum(x*y for x,y in zip(row,E)) for row in Sigma]
    power = sum(x*y for x,y in zip(E,J))
    print(f"{h:.8f} {J[0]:.8f} {J[1]:.8f} {power:.8f}")
```

[Tong’s classical Hall discussion, §1.2.2](https://arxiv.org/html/1606.06687#S1.SS2.SSS2), supplies conductivity-tensor context. Our unconstrained real tensor family, calibrated probe inverse, passivity counterexample and uncertainty calculation are derived here. They do not use a microscopic Drude relation to infer the hidden coefficient. The existing induction-energy lab supplies the manufactured E and required J for the separate axial-current negative control; no new conductivity-tensor control is implied.

![Hidden transverse current, passivity and probe conditioning](/assets/learning/electromagnetism-power-invisible-current.en.svg)

### Research questions with explanations

**Does unchanged energy balance mean the added axial current is Maxwell-compatible?** **Answer:** No. The energy-invisible axial current changes the vector Ampere residual by −μq e_z while leaving continuity and J·E unchanged.

**Can enough power-only directions reveal the antisymmetric coefficient?** **Answer:** No, within this unconstrained real instantaneous tensor family. Every EᵀAE vanishes. A constitutive prior or another measurement supplies different information.

**Do three positive measured powers certify passivity?** **Answer:** No. Recover S and test its positive-semidefinite condition; the displayed indefinite tensor gives three positive probe values and a negative-power direction.

## Exercises with explanations

### Exercise 1: Interpret a sign

Pout=−3 W and Wmatter=1 W. What is dU/dt? **Answer:** 2 W. Three watts enter, one watt goes to matter, and two watts increase field storage.

### Exercise 2: Remove acceleration

Set b₂=0 while bb′>0. Must field energy be constant? **Answer:** No. Electric storage is constant, but magnetic storage grows through inward boundary flux.

### Exercise 3: Change the cylinder radius

Double R with all other quantities fixed. How do UE and UB change? **Answer:** UE multiplies by16 and UB by4 because the electric field grows with radius.

### Exercise 4: Omit current

At r=1 with normalized ε=1, b′=0.7 and b₂=0.2, what energy defect remains if J=0? **Answer:** 0.035 W/m³. Faraday alone does not supply the missing source term.

### Exercise 5: Certify a model from one residual

Does the static field E=(0,x,0), B=J=0 become a Maxwell solution because energy balance passes? **Answer:** No. Curl E is nonzero while ∂tB=0, so Faraday fails.

### Exercise 6: A rounded zero

Why might a total SI energy residual be zero for the omitted-current model? **Answer:** A small electric rate is lost when added to a much larger magnetic rate before cancellation. Inspect components and compare stable grouping with independent high-precision arithmetic.

## References and next connection

The balance and sign conventions follow [Feynman II Chapter27, §§27–2–27–3](https://www.feynmanlectures.caltech.edu/II_27.html) and [Haus–Melcher Chapter11, §§11.1–11.4](https://ocw.mit.edu/courses/res-6-001-electromagnetic-fields-and-energy-spring-2008/0f814aee1c72517df7d7e7e96b7365e1_11.pdf). Next, apply the same bookkeeping to transverse pulses and distinguish finite-region energy from energy per transverse area.
