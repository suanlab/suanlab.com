## A material boundary changes the question

Place two dielectric materials on opposite sides of a flat interface. An electric-field arrow can change its normal length when it crosses that interface, even if there is no free charge on the surface. This is a physical response, not automatically an error in a simulation. The right question is which component of which field must be continuous, and which surface charge accounts for a jump.

We start with stationary, isotropic, electrically linear materials. Each half-space has a constant positive permittivity. Review normal and tangential components, polarization, Gauss law, potential gradients, and oriented line integrals. The flat, nondecaying fields need remote sources or prescribed boundaries; this is not an isolated finite device.

## Define the normal and the three fields

Medium1 occupies z<0 and medium2 occupies z>0. Fix n=+z, from medium1 to medium2. Use ε₀ as the vacuum reference and distinguish electric field E, polarization P, and displacement D.

$$
\mathbf D=\epsilon_0\mathbf E+\mathbf P=\epsilon\mathbf E,\qquad \mathbf P=(\epsilon-\epsilon_0)\mathbf E.
$$

E is measured in V/m; D and P are in C/m². D is a macroscopic variable that puts polarization into the constitutive description. It is not a second kind of electric force. Do not compare the numerical lengths of E and D arrows without accounting for their different units.

## Integrate a pillbox across the interface

Draw a thin box straddling z=0. The upper face has outward normal+z and the lower face has outward normal−z. The side contributions cancel for our uniform tangential fields. Gauss law for D counts free charge; Gauss law for ε₀E counts all charge.

$$
\mathbf n\cdot(\mathbf D_2-\mathbf D_1)=\sigma_{\rm free},\qquad \epsilon_0\mathbf n\cdot(\mathbf E_2-\mathbf E_1)=\sigma_{\rm free}+\sigma_{\rm bound}.
$$

Multiplying either equation by the face area gives charge in C. In a general nonuniform configuration the vanishing-height argument makes side flux negligible. Here each side's cancellation can be checked directly at positive height because fields are uniform within each material.

## Locate the polarization charge

Uniform P has no bulk divergence in either open half-space. At the interface its discontinuity produces a surface distribution. The outward normal of medium1 is+n, while that of medium2 is−n.

$$
\rho_{\rm bound}=-\nabla\cdot\mathbf P,\qquad \sigma_{\rm bound}=\mathbf n\cdot(\mathbf P_1-\mathbf P_2).
$$

The sign is easy to reverse accidentally. Think of summing the two material surfaces at the same interface. “No free charge” does not mean “no total surface charge.” Polarization can still produce the normal E jump.

## Match the tangential field

In electrostatics, integrate E around a narrow rectangle crossing the interface. Shrinking its height leaves the two tangential edges. Their circulation must vanish, so tangential E is continuous. Equivalently, a regular electrostatic potential can be chosen continuous across this ideal interface, without an imposed dipole sheet.

$$
\mathbf n\times(\mathbf E_2-\mathbf E_1)=0,\qquad V_2(x,y,0)=V_1(x,y,0).
$$

Normal E continuity is not the corresponding condition. Normal D is continuous only when free surface charge is zero. Tangential D can jump when the permittivity changes even though tangential E matches.

## Specify a solvable family

Let d be the normal D in medium1, σf the specified free surface charge, and Et a common tangential electric component. Prescribe the normal D in medium2 using the charge jump.

$$
\mathbf E_1=(E_t,0,d/\epsilon_1),\qquad \mathbf E_2=(E_t,0,(d+\sigma_f)/\epsilon_2),\qquad \mathbf n=(0,0,1).
$$

Both bulk free and bound charge densities vanish. Surface charge and remote boundary sources supply the required field configuration. We use positive coefficients as a mathematical teaching family, not a claim that arbitrary constants describe every passive material at every frequency.

## Recover the potential on both sides

For each region, integrating the constant field gives a linear potential. Use one common additive reference. A change of reference shifts both potentials together and leaves all fields and jumps unchanged.

$$
V_j(x,z)=-E_t x-E_{jz}z+V_{\rm ref}.
$$

At z=0 the normal terms vanish, so the two potentials agree. The slope in z changes when En changes. A continuous function can have a discontinuous derivative: potential continuity does not require equal normal E.

## Work an example with physical units

Choose ε₁=ε₀, ε₂=4ε₀, d=2 nC/m², σf=0, and Et=1 V/m. Then E1z≈225.882 V/m and E2z≈56.4705 V/m. Normal D remains2 nC/m². P1z=0 and P2z=1.5 nC/m², giving σbound=−1.5 nC/m².

For a face area of0.6 m², the upper and lower D fluxes are+1.2 nC and−1.2 nC. Their sum is zero free charge. The ε₀E flux is−0.9 nC, equal to bound charge in that patch. At x=0.3 m on the interface, both potentials are−0.3 V if Vref=0.

| Quantity | Medium1 | Medium2 |
| --- | ---: | ---: |
| Normal E, V/m | 225.882 | 56.4705 |
| Normal D, nC/m² | 2 | 2 |
| Normal P, nC/m² | 0 | 1.5 |

The smaller E in medium2 and the unchanged D are compatible descriptions of one interface.

## Add free surface charge

Keep the preceding parameters but set σf=1 nC/m². Now D2z=3 nC/m² and E2z≈84.7057 V/m. Polarization gives σbound=−2.25 nC/m², so total surface charge is−1.25 nC/m².

A positive free charge can coexist with negative total charge because the polarization contribution is larger and negative. Reporting only one of these charges without naming it can make a correct flux appear inconsistent.

## Test an incorrect continuity rule

Suppose a model forces equal normal E while keeping unequal ε. Its required free charge changes. In our family with ε₂=2ε₁ and d=2 nC/m², equal normal E requires σfree=2 nC/m². If the requested free charge was zero, the model has changed the source problem.

If that nonzero free charge is explicitly allowed, equal En can be valid. The implementation distinguishes the specified source from the source required by the predicted fields. A failed source request is not proof that no Maxwell configuration could support those fields.

## Deliberately break tangential matching

Add a tangential jump δ to E2x while keeping E1x fixed. For a rectangle traversed along+x below the interface and−x above it, the normal is−y. The vertical edge integrals cancel.

$$
\oint\mathbf E\cdot d\boldsymbol\ell=-\delta L,\qquad V_2(x,0)-V_1(x,0)=-\delta x.
$$

For δ=0.3 V/m and L=0.6 m, circulation is−0.18 V. The potential jump at x=0.3 m is−0.09 V. One additive gauge constant cannot remove an x-dependent mismatch. This is an electrostatic negative control. Its static energy balance with B=J=0 could still pass, as the preceding energy lesson warned.

## Read limits without inventing an interface average

At z=0, the model returns the two side limits and no single averaged E or D. That is intentional: the value on an ideal mathematical surface does not replace the boundary jump conditions. A graph should show the two traces separately.

For positive box height, face fluxes and loop edges have ordinary geometric meanings. At exactly zero height, coincident faces and an interface path no longer define the same measurement. The API reports the collapsed measurement unavailable while keeping its labelled two-sided limit. A tangential-jump counterexample can have nonzero limiting circulation even as height approaches zero.

## Preserve small jumps numerically

Subtracting rounded face values can lose a small free charge on a large normal background. Cancel the common background algebraically first.

$$
\Delta E_n=\frac{\sigma_f}{\epsilon_2}+\frac{d(\epsilon_1-\epsilon_2)}{\epsilon_1\epsilon_2}.
$$

With equal vacuum coefficients, d=10⁻⁹ C/m² and σf=10⁻³⁰ C/m², rounded D2−D1 can be zero while the prescribed analytic jump is nonzero. Label this stable parameter-derived quantity separately from rounded face subtraction. Nonzero products lost to arithmetic are unavailable, not physical zero. Exact zero area also makes area-normalized charge undefined.

## Independent checks and AI field models

Independent Python references evaluate vector surface and edge integrals over243 static conditions. Two180-digit references examine tiny source and material jumps. These checks supplement the analytic formulas; shared jump identities are not independent certificates or universal error bounds.

For a learned field, test bulk residuals, constitutive relations, the declared normal, free and bound charges, interface traces, and potential differences. Training on only bulk points can miss a surface defect of measure zero. Smoothing a legitimate interface jump into an average may solve a different material problem.

## Research extension: resolve a dipole sheet before taking its limit

### Change the source model explicitly

Potential continuity above assumes an ordinary interface without a prescribed dipole sheet. To examine that assumption, resolve a new vacuum layer of thickness ℓ with an externally prescribed polarization moment per area p_s pointing along +z. It is not obtained by substituting the original positive linear dielectric law into D=0. The polarization is constant inside the layer and zero outside. Its negative divergence gives a negative lower sheet and a positive upper sheet. The unit of p_s is C/m, whereas each surface charge has units C/m². Reversing p_s reverses both charges and the potential jump.

$$
P_z^{(\ell)}=\frac{p_s}{\ell}\mathbf1_{(-\ell/2,\ell/2)}(z),\quad\rho_b^{(\ell)}=\frac{p_s}{\ell}\left[\delta(z-\ell/2)-\delta(z+\ell/2)\right],\quad [p_s]={\rm C/m},\quad\ell>0.
$$

### Find the field and potential before taking a limit

With no free charge, no imposed normal background and D_z=0, the field is opposite to the polarization inside and vanishes outside. Superposing the fields of the two charge sheets gives the same result. Choose the lower exterior potential as zero and integrate −E along z. At every positive thickness the potential is continuous, including at both sheet locations. The exterior-to-exterior potential difference is independent of thickness at fixed p_s. A common additive potential reference changes neither this difference nor the field.

$$
D_z=0,\quad E_z^{(\ell)}=-\frac{P_z^{(\ell)}}{\epsilon_0},\qquad E_z=-\frac{p_s}{\epsilon_0\ell}\ \text{inside},\quad E_z=0\ \text{outside}.
$$

$$
V^{(\ell)}(z)=\frac{p_s}{\epsilon_0}\begin{cases}0,&z\le-\ell/2,\\(z+\ell/2)/\ell,&|z|<\ell/2,\\1,&z\ge\ell/2,\end{cases}\qquad \Delta V=-\int E_z\,dz=\frac{p_s}{\epsilon_0}.
$$

### Distinguish zero total charge from zero dipole moment

The two charges cancel in total, but their first spatial moment does not. Thus a pillbox outside the entire layer sees zero net charge even when traversing the layer gives a nonzero voltage. In the ideal limit, normal exterior field traces alone do not encode this additional dipole-source datum. It is a different boundary model from the original regular interface, not an error to repair by assigning unrelated potential references to the two sides.

$$
\int\rho_b^{(\ell)}dz=0,\qquad\int z\rho_b^{(\ell)}dz=p_s,\qquad \sigma_-=-p_s/\ell,\quad\sigma_+=p_s/\ell.
$$

### Define the singular limit through measurements

Do not read a delta derivative as an ordinary point value. Pair the charge distribution with a smooth test function φ. The centered difference tends to φ′(0), which identifies −p_s δ′. Taylor expansion gives the displayed error bound for C³ test functions near the layer; compactly supported smooth functions give the usual distributional interpretation. The sign follows from δ′ acting as minus differentiation. Correspondingly E tends to −p_s δ/ε₀ and V has the two traces 0 and p_s/ε₀. The value assigned at the mathematical surface does not replace the traces or define an interface average.

$$
\langle\rho_b^{(\ell)},\varphi\rangle=\frac{p_s}{\ell}[\varphi(\ell/2)-\varphi(-\ell/2)]\longrightarrow p_s\varphi'(0)=\langle-p_s\delta',\varphi\rangle,\quad\left|\langle\rho_b^{(\ell)},\varphi\rangle-p_s\varphi'(0)\right|\le\frac{|p_s|\ell^2}{24}\sup_{|z|\le\ell/2}|\varphi'''(z)|.
$$

### Inspect the energy of the regularized layer

Integrate the vacuum electric-field energy density over z to obtain energy per transverse area. For fixed nonzero p_s, the field grows like 1/ℓ and its energy like 1/ℓ. A finite voltage jump therefore does not imply a finite field-energy cost in this idealization. This is the electric-field contribution; microscopic polarization energy and the preparation mechanism are not modeled. The divergence is not merely an artifact of choosing a rectangular profile. Cauchy–Schwarz gives the lower bound for any square-integrable field confined to a thickness ℓ with the same voltage drop. Equality holds for the constant interior field used here.

$$
U_A^{(\ell)}=\frac{\epsilon_0}{2}\int(E_z^{(\ell)})^2dz=\frac{p_s^2}{2\epsilon_0\ell},\qquad [U_A]={\rm J/m^2}.
$$

$$
\left(\int_{-\ell/2}^{\ell/2}E_zdz\right)^2\le\ell\int_{-\ell/2}^{\ell/2}E_z^2dz\quad\Longrightarrow\quad U_A\ge\frac{\epsilon_0(\Delta V)^2}{2\ell}.
$$

### State which physical quantity is held fixed

If the sheet charge magnitude σ is fixed instead, then p_s=σℓ shrinks and both the voltage and field energy tend to zero. These limits are different experiments. For p_s=0 and positive thickness the fields and energy are exactly zero; ℓ=0 is outside the finite-layer formula, rather than a denominator to replace by a small number. Use the positive reference scales below to compare the two limits in the table. For example, taking V_scale=1 V and ℓ_ref=1 mm gives p_ref=ε₀×1 V and U_ref≈4.427×10⁻⁹ J/m². Reducing thickness by four at fixed p_s multiplies energy by four; at fixed σ it divides energy by four and also divides voltage by four. No continuum validity at arbitrarily small physical thickness is certified.

$$
\sigma\ \text{fixed},\ p_s=\sigma\ell:\quad\Delta V=\frac{\sigma\ell}{\epsilon_0}\to0,\quad U_A=\frac{\sigma^2\ell}{2\epsilon_0}\to0;\qquad p_s\ne0\ \text{fixed}:\quad\Delta V=\frac{p_s}{\epsilon_0},\quad U_A\to\infty.
$$

$$
q=\frac{\ell}{\ell_{\rm ref}},\quad p_{\rm ref}=\epsilon_0V_{\rm scale},\quad\sigma_{\rm ref}=\frac{p_{\rm ref}}{\ell_{\rm ref}},\quad U_{\rm ref}=\frac{p_{\rm ref}^2}{2\epsilon_0\ell_{\rm ref}}:\quad\begin{array}{c|ccc}&\sigma/\sigma_{\rm ref}&\Delta V/V_{\rm scale}&U_A/U_{\rm ref}\\p_s=p_{\rm ref}&1/q&1&1/q\\\sigma=\sigma_{\rm ref}&1&q&q\end{array}.
$$

| ell / ell_ref | sigma at fixed moment / sigma_ref | fixed-moment voltage / V_scale | fixed-moment energy / U_ref | fixed-charge energy / U_ref |
| --- | --- | --- | --- | --- |
| 1.00000000 | 1.00000000 | 1.00000000 | 1.00000000 | 1.00000000 |
| 0.25000000 | 4.00000000 | 1.00000000 | 4.00000000 | 0.25000000 |
| 0.06250000 | 16.00000000 | 1.00000000 | 16.00000000 | 0.06250000 |
| 0.01562500 | 64.00000000 | 1.00000000 | 64.00000000 | 0.01562500 |

```python
from fractions import Fraction
# Dimensionless reference units: epsilon0=1, ell_ref=1, p_ref=1.
# The common reference energy per area is 1/2.
for q in [Fraction(1),Fraction(1,4),Fraction(1,16),Fraction(1,64)]:
    sigma = 1/q
    E = -sigma
    jump = -E*q
    energy_fixed_moment = E*E*q/2
    energy_fixed_charge = q/2
    print(f"{float(q):.8f} {float(sigma):.8f} {float(jump):.8f} "
          f"{float(2*energy_fixed_moment):.8f} {float(2*energy_fixed_charge):.8f}")
```

[Haus–Melcher §4.5, “Potential of Uniform Dipole Layer”](https://www.mit.edu/course/6/6.013_book/www/chapter4/4.5.html), provides ideal dipole-layer context. The resolved planar profile, weak-error estimate, fixed-moment versus fixed-charge energy scaling and shape bound are derived here. The existing regular-interface lab retains its no-dipole-sheet source contract and common zero potential reference; it does not expose a new layer-thickness or dipole control.

![Resolved dipole potentials, concentrated fields and two energy limits](/assets/learning/electromagnetism-dipole-layer-limit.en.svg)

### Research questions with explanations

**Does zero enclosed charge force a zero voltage jump?** **Answer:** No. The two sheets have zero total charge but nonzero dipole moment per area, giving a nonzero exterior potential difference.

**Does a fixed finite voltage jump have finite energy as thickness vanishes?** **Answer:** Not in this prescribed vacuum-layer field-energy model. The lower bound scales as 1/ℓ for a fixed nonzero voltage.

**Can a common gauge reference remove the ideal dipole jump?** **Answer:** No. It shifts both exterior potentials equally. Adding a dipole source changes the boundary problem; it is not a reference change.

## Exercises with explanations

### Exercise 1: No free charge

If ε₂=4ε₁ and σfree=0, must E2n=E1n? **Answer:** No. D2n=D1n, so E2n=E1n/4 in this family.

### Exercise 2: Bound charge sign

For P1n=0 and P2n=1.5 nC/m² with n=+z, find σbound. **Answer:** −1.5 nC/m². The two materials contribute opposite outward normals.

### Exercise 3: Which flux counts free charge?

Should ε₀E flux equal only free charge? **Answer:** No. D flux counts free charge; ε₀E flux counts free plus bound charge.

### Exercise 4: A tangential mismatch

Find circulation for δ=0.2 V/m and L=0.5 m using the stated traversal. **Answer:** −0.1 V. The sign follows the lower+x and upper−x edges.

### Exercise 5: Equal normal fields

Does equal En across unequal ε necessarily violate Maxwell? **Answer:** No. It requires a particular free surface charge. Compare that source with the requested one.

### Exercise 6: A rounded charge jump

Why can rounded D2−D1 be zero for nonzero prescribed σfree? **Answer:** Adding a tiny charge to a larger background may round away the increment. Use a stable analytic jump and independent high-precision references, and state its provenance.

## References and next connection

Compare the macroscopic conventions with [Haus–Melcher §6.2](https://web.mit.edu/6.013_book/www/chapter6/6.2.html) and [§6.5](https://web.mit.edu/6.013_book/www/chapter6/6.5.html). Next, match traveling E and H at the same interface and connect reflection amplitudes to energy flux.
