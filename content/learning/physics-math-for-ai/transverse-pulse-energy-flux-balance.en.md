## Watch a pulse cross a measuring window

Imagine two light pulses moving in opposite directions through a fixed window. An electric-field plot can show a large peak where they overlap. Does that peak mean energy has been created? Does zero magnetic field at one instant mean there is no electromagnetic energy? Answer these questions by plotting electric storage, magnetic storage, and signed energy flux separately.

This lesson continues the smooth transverse Gaussian pulse. Review initial E/B compatibility, squared norms, Gaussian integrals, and the fixed-volume Poynting balance. The wave has infinite transverse extent: every integral along z is an energy per transverse area, not the finite total energy of a three-dimensional antenna.

## Fix the medium and the initial data

Use constant positive ε and μ, with c=1/√(εμ). Let G be a Gaussian with signed electric amplitude A and width σ. Split its initial electric profile into right- and left-moving components using β∈[0,1].

$$
G(\xi)=A\exp\!\left(-\frac{\xi^2}{2\sigma^2}\right),\quad f=\beta G(z-ct),\quad g=(1-\beta)G(z+ct),\quad E_x=f+g,\quad cB_y=f-g.
$$

At t=0 all choices of β have the same electric profile G. Their magnetic profiles differ. β=1 gives a right-moving pulse, β=0 a left-moving pulse, and β=1/2 an initially zero magnetic field with symmetric later splitting. These are different initial electromagnetic states.

## Separate electric and magnetic storage

Substitute the fields into the constant-coefficient energy formula. Since 1/(μc²)=ε, the magnetic term can be expressed using f−g. Both terms are nonnegative even when the signed field amplitude is negative.

$$
u_E=\tfrac12\epsilon(f+g)^2,\qquad u_B=\tfrac12\epsilon(f-g)^2,\qquad u=\epsilon(f^2+g^2).
$$

The +2fg term in electric storage and −2fg term in magnetic storage cancel in total energy. They do not disappear from the individual electric and magnetic plots. For an equal split at t=0, B=0 and all energy is electric at that instant. This is not an absence of field energy.

## The flux is a signed difference

The cross product points along z. Its sign tells us which direction dominates. There is no mixed term in the total flux either.

$$
S_z=\frac{E_xB_y}{\mu}=\epsilon c(f^2-g^2),\qquad |S_z|\le cu.
$$

If f²=g², local net flux is zero even though both traveling components can carry energy in opposite directions. This is analogous to equal opposing traffic counts producing zero net traffic; it does not mean every vehicle is stationary. The analogy describes accounting, not a particle interpretation of the field.

## A conditional energy-flow ratio

Where u>0, the signed ratio S/u has velocity units and lies between −c and c. A single right-moving component gives +c; a single left-moving component gives −c.

$$
v_{\rm energy}=\frac{S_z}{u}=c\frac{f^2-g^2}{f^2+g^2},\qquad u>0.
$$

This local energy-flow ratio is not automatically a particle velocity, a group velocity in a general medium, or an information velocity. At exact zero amplitude u=0, so the ratio is undefined. If absolute energy underflows numerically, distinguish a model-derived analytic ratio from a quotient of available observed numbers.

## Check the local conservation law

The right-moving component satisfies ft=−cfz, and the left-moving component satisfies gt=+cgz. Differentiating the total energy and flux shows the cancellation directly. This source-free family has J=0.

$$
\partial_tu=2\epsilon c(-ff_z+gg_z),\qquad \partial_zS_z=2\epsilon c(ff_z-gg_z),\qquad \partial_tu+\partial_zS_z=0.
$$

This calculation is an analytic consistency check using the same prescribed family. Independent finite differences and actual-field quadrature provide additional evidence. A shared derivative identity alone does not validate every implementation detail or every Maxwell equation.

## Draw a fixed slab and its outward normals

Choose a≤z≤b. The outward normal is −z at the left face and +z at the right face. Let UA be stored energy per transverse area. The two face fluxes therefore enter with different signs.

$$
U_A(t)=\int_a^b u(z,t)\,dz,\qquad P_{{\rm out},A}=S_z(b,t)-S_z(a,t),\qquad \frac{dU_A}{dt}+P_{{\rm out},A}=0.
$$

UA is in J/m² and the boundary power per area is in W/m². Multiplying both by a fixed transverse area gives J and W. We keep a and b fixed while the pulse travels. Moving the window along with the pulse would change the transport calculation.

## Calculate a finite window

With normalized ε=μ=1, A=1, σ=1, β=1, c=1, and t=0.3, choose a=−1 and b=1. Integrating the actual field energy gives UA≈1.42842866 J/m². The boundary values give Pout,A≈0.42810687 W/m² and the storage rate is about−0.42810687 W/m².

The energy in the window is decreasing because more leaves on the right than enters on the left. The entire pulse has not lost energy: part of it is outside the window. Always state whether “energy decreases” refers to a fixed region or the whole modeled axis.

## Integrate the entire axis

Integrating a squared Gaussian gives σ√π. Translation does not change this integral. The whole-axis energy per transverse area is time independent for the compatible family.

$$
U_{A,\mathbb R}=\epsilon A^2\sigma\sqrt\pi\,[\beta^2+(1-\beta)^2].
$$

For normalized unit parameters, β=1 gives √π≈1.77245385 J/m², while β=1/2 gives √π/2≈0.88622693 J/m². The same initial E does not imply equal initial energy because initial B changes. Comparing the two is not a redistribution experiment with the same energy budget.

## A local split-pulse example

For the equal split at z=1, t=1 with normalized unit parameters, f=0.5 and g≈0.06766764. Thus Ex≈0.56766764 V/m and cBy≈0.43233236 V/m. The total density is about0.25457891 J/m³ and flux is about0.24542109 W/m². Their ratio is about0.96402758 m/s, below c=1 m/s.

At t=0 and the same z, E≈0.60653066 V/m, B=0, u≈0.18393972 J/m³, and S=0. The later local increase does not violate conservation: energy has entered that neighborhood. A point value is not the energy of a material object that must remain attached to the same position.

## Distinguish actual and reference magnetic fields

For a negative control, keep a traveling E but force actual B=0 and J=0. The actual magnetic energy and Poynting flux are zero; the compatible B reference must not be substituted into those actual quantities. The scalar wave equation for E still holds, but first-order Maxwell relations and generally energy balance fail.

The actual zero-B model also has a different whole-axis energy. For the two Gaussian components its electric-only integral is the following expression; it need not be constant while the components overlap.

$$
U^{B=0}_{A,\mathbb R}=\frac{\epsilon A^2\sigma\sqrt\pi}{2}\left[\beta^2+(1-\beta)^2+2\beta(1-\beta)e^{-(ct/\sigma)^2}\right].
$$

For β=1 this integral happens to be constant, yet the local energy residual is generally nonzero because the profile travels with S=0. Whole-axis conservation alone is weaker than local conservation. At A=0 the nominal negative control reduces to the valid zero field; at selected peak points a sampled residual can also vanish without validating a nonzero family.

## Retain tiny fluxes without inventing tails

For β=1/2, computing f²−g² by subtracting rounded squares can erase a small nonzero flux. Use the stable Gaussian difference already derived in the preceding pulse lesson and multiply it by the Gaussian sum.

$$
S_z=\frac{\epsilon c}{4}(G_+-G_-)(G_++G_-),\qquad G_\pm=G(z\mp ct).
$$

At z=1, t=10⁻¹⁸ with normalized unit parameters, the flux is about3.67879441×10⁻¹⁹ W/m², while direct square subtraction may give0. Use expm1 for small exponent differences. A Gaussian tail below the representable range is unavailable, not a compactly supported physical zero. A=10⁻²⁰⁰ can leave representable fields while their squared energies are unavailable.

## Integrate thin windows carefully

Subtracting two nearly equal Gaussian antiderivatives can lose a thin slab's energy. Integrate in a unit coordinate while retaining the real physical width outside the integral. The implementation uses composite eight-node Gauss quadrature on subintervals no wider than σ/2. Stable differences of squared Gaussian endpoints provide a separate boundary formula.

$$
\int_a^b u(z)\,dz=(b-a)\int_0^1u\bigl(a+(b-a)s\bigr)\,ds.
$$

Independent 180-digit references check nine thin slabs and a tiny balanced flux. Actual-field SciPy quadrature and five-point derivatives check54 selected pulse conditions. These are observed checks, not a certified quadrature bound. A zero-width interval has exact zero integral; a positive width with lost arithmetic is a different state.

## Use the lesson to evaluate learned fields

A field-generating AI model may match an electric snapshot while predicting the wrong magnetic state and energy budget. Compare initial E and B, electric and magnetic storage, signed boundary flow, finite-region storage change, and first-order Maxwell relations. Do not reward a zero total-energy loss if the model merely hides local defects or confuses the compatible reference with its actual output.

## Research extension: local zero-flux observers in vacuum

### Declare a vacuum calculation at one event

For this supplement use vacuum, with c₀ the physical vacuum light speed and ε₀μ₀c₀²=1. Alternatively use consistently normalized Lorentz-invariant vacuum units. A material wave speed 1/sqrt(εμ) from an arbitrary stationary medium is not a replacement for c₀ in a Lorentz transformation. Take transverse counterpropagating components f and g at one event. They may have either sign. The energy and flux formulas below retain both directions, and the field momentum density is S/c₀². This local calculation does not assign a finite total energy to the plane pulse’s infinite transverse extent.

$$
\mathbf E=(f+g,0,0),\quad c_0\mathbf B=(0,f-g,0),\quad u=\epsilon_0(f^2+g^2),\quad s=\frac{S_z}{c_0}=\epsilon_0(f^2-g^2),\quad \mathbf p_{\rm field}=\frac{\mathbf S}{c_0^2}.
$$

### Transform fields and coordinates together

Let an inertial observer move along +z at v, with α=v/c₀ distinct from the pulse’s splitting fraction β. The primed fields are measured at the same physical event expressed in primed coordinates. Using z×y=−x and z×x=y fixes the two field-transform signs. Form half the sum and difference of E′ and c₀B′ to obtain f′ and g′. Rapidity η makes their opposite Doppler factors especially clear. Transforming a whole lab-time curve also requires transforming its event coordinates; a fixed laboratory slab is not the same simultaneous region for the moving observer.

$$
\alpha=\frac v{c_0},\quad|\alpha|<1,\quad\gamma=(1-\alpha^2)^{-1/2},\quad t'=\gamma(t-vz/c_0^2),\quad z'=\gamma(z-vt),\quad E_x'=\gamma(E_x-vB_y),\quad c_0B_y'=\gamma(c_0B_y-\alpha E_x).
$$

$$
\eta=\operatorname{artanh}\alpha,\quad f'=\gamma(1-\alpha)f=e^{-\eta}f,\quad g'=\gamma(1+\alpha)g=e^{\eta}g.
$$

### Energy density and energy flux mix through stress

Substitute f′ and g′ into the quadratic energy formulas. The resulting two-by-two block T uses indices ct,z: its diagonal entries are u because the longitudinal momentum flux is also u for these transverse fields. Applying Λ to both tensor indices gives the expanded formulas below. This is not the transformation of an ordinary two-component vector. The pressure contribution is why dividing momentum density by energy density does not directly supply the observer velocity that makes Poynting flux zero. Squaring the expanded formulas proves the next invariant; the field form gives an independent algebraic route.

$$
T=\begin{pmatrix}u&s\\s&u\end{pmatrix},\quad\Lambda=\gamma\begin{pmatrix}1&-\alpha\\-\alpha&1\end{pmatrix},\quad T'=\Lambda T\Lambda^T,\quad\begin{aligned}u'&=\gamma^2[(1+\alpha^2)u-2\alpha s],\\s'&=\gamma^2[(1+\alpha^2)s-2\alpha u].\end{aligned}
$$

$$
u'^2-s'^2=u^2-s^2=4\epsilon_0^2f^2g^2=\frac{\epsilon_0^2}{4}(E_x^2-c_0^2B_y^2)^2.
$$

### Find a local zero-flux observer when one exists

When f and g are both nonzero, s′=0 means |f′|=|g′|. Taking the ratio of the Doppler factors gives α∗ and η∗ below; substituting back gives u∗. The expression using s/(u+sqrt(u²−s²)) avoids the removable division by s that would occur in (u−sqrt(u²−s²))/s at a balanced state. It still requires available nonzero energy and a trustworthy invariant. Write the laboratory energy-flow ratio as w=S_z/(c₀u). Its relation to α∗ shows that these are different quantities except in special cases such as balance. For f=1 and g=1/4 in a common field unit, α∗=3/5, while w=15/17 and u∗/(ε₀ field-unit²)=1/2.

$$
fg\ne0:\quad\alpha_* =\frac{|f|-|g|}{|f|+|g|}=\frac{s}{u+\sqrt{u^2-s^2}},\quad\eta_* =\frac12\log\left|\frac f g\right|,\quad s'=0,\quad u_*=2\epsilon_0|fg|.
$$

$$
w=\frac{S_z}{c_0u}=\frac{s}{u}=\frac{2\alpha_*}{1+\alpha_*^2},\qquad fg\ne0:\quad |\alpha_*|<1,\quad |w|<1.
$$

### Keep magnetic, electric, null and zero states separate

If f and g have the same sign, their equal transformed magnitudes give B′=0. With opposite signs the zero-flux frame has E′=0 instead. If only one travelling component is nonzero, every finite rapidity keeps a nonzero flux of that sign: the required boost lies at |α|=1 and is not an inertial observer. This nonzero one-way state is a null electromagnetic field: the invariant u²−s² vanishes, while E and B do not. If both vanish, fields are zero in every frame; neither w nor a preferred normalized frame is defined. A nonzero u∗ is a local field invariant here, not a photon rest mass or an assertion of a global rest frame. In an overlapping pulse, α∗ generally changes from event to event, so the construction supplies a locally selected family of inertial observers, not one observer field that is necessarily inertial everywhere.

$$
\begin{cases}fg>0:&\text{a longitudinal zero-flux frame has }B_y'=0,\\fg<0:&\text{a longitudinal zero-flux frame has }E_x'=0,\\fg=0,\ u>0:&\text{no subluminal longitudinal zero-flux frame},\\f=g=0:&\text{zero fields in every frame; normalized ratios undefined}.\end{cases}
$$

### Treat a nearly one-way field without inventing a light-speed observer

For f≠0 set r=|g/f|. The limit r→0 through positive values has valid but increasingly large rapidity; the exact r=0 case has no subluminal zero-flux observer. Finite precision can round α∗ to 1 before the small counterpropagating component vanishes. Compute from signed f,g and logarithmic amplitude ratios when appropriate, track arithmetic availability, and never repair the result by silently clipping a physical speed. Squaring rounded u and s can also erase the invariant; 2ε₀|fg| avoids that particular cancellation. The cell includes the exact one-way case and reports its zero-flux observer and energy there as unavailable. Other rows use signed vacuum components f=1, g=r≥0 and evaluate transformed fields through rapidity.

$$
r=|g/f|>0:\quad 1-\alpha_* =\frac{2r}{1+r},\quad\frac{u_*}{u}=\frac{2r}{1+r^2},\quad\eta_*=-\frac12\log r,\qquad r\to0^+:\quad\alpha_*\to1,\quad u_*/u\to0.
$$

| r = g/f | lab S / (c0 u) | zero-flux alpha | zero-flux u / energy unit |
| --- | --- | --- | --- |
| 0 | 1 | unavailable | unavailable |
| 1e-06 | 0.999999999998 | 0.999998000002 | 2e-06 |
| 0.1 | 0.980198019802 | 0.818181818182 | 0.2 |
| 1 | 0 | -0 | 2 |

```python
from math import log, exp, tanh
# Vacuum amplitudes f=1, g=r, in one fixed field unit.
# Energies are divided by epsilon0 times that unit squared.
for r in [0.0, 1e-6, 0.1, 1.0]:
    u, s = 1+r*r, 1-r*r
    w = s/u
    if r == 0:
        print(f"{r:.12g} {w:.12g} unavailable unavailable")
        continue
    eta = -0.5*log(r)
    alpha = tanh(eta)
    fp, gp = exp(-eta), exp(eta)*r
    up, sp = fp*fp+gp*gp, fp*fp-gp*gp
    assert abs(sp) < 1e-12
    print(f"{r:.12g} {w:.12g} {alpha:.12g} {up:.12g}")
```

[Feynman II §26–3](https://www.feynmanlectures.caltech.edu/II_26.html) gives the vacuum field transformations, and [§27–6](https://www.feynmanlectures.caltech.edu/II_27.html) gives field momentum. The transverse stress-block algebra, local classification and near-null diagnostics here follow directly from our stated fields. The existing Gaussian lab can supply vacuum event data and laboratory energy/flux; it does not add a moving-observer control or reinterpret a material’s wave speed as vacuum c₀.

![Vacuum zero-flux frame, observer-dependent energy and near-null limits](/assets/learning/electromagnetism-vacuum-zero-flux-frame.en.svg)

### Research questions with explanations

**Is S/u the speed of the observer who measures zero flux?** **Answer:** Generally no. In the stated vacuum transverse family, S/(c₀u)=2α∗/(1+α∗²), rather than α∗.

**Does a one-way nonzero wave have a finite zero-flux inertial frame?** **Answer:** No. The limiting boost has |α|=1. An arbitrarily small nonzero opposite component and an exact zero component are different cases.

**Can the same formula use the wave speed of a dielectric?** **Answer:** No. This is a vacuum Lorentz calculation using c₀. Moving media require additional constitutive and material energy–momentum treatment.

## Exercises with explanations

### Exercise 1: Same E, different energy

Compare β=1 and β=1/2 at t=0 with the same A and σ. **Answer:** Initial E is identical, but the one-way state's whole-axis energy per area is twice as large because initial B is nonzero.

### Exercise 2: Equal opposing components

If f=g≠0, find u and S. **Answer:** u=2εf²>0 and S=0. Zero net flux does not imply zero stored energy.

### Exercise 3: Read the left boundary

If S(a)=2 W/m² and S(b)=5 W/m², find the storage rate. **Answer:** Pout,A=3 W/m² and dUA/dt=−3 W/m². The left face has outward normal−z.

### Exercise 4: Zero amplitude

What is S/u when A=0? **Answer:** Undefined, since both numerator and denominator are0. Assigning velocity0 would hide the missing normalization.

### Exercise 5: Traveling E with B=0

Does time-independent whole-axis electric energy certify this field? **Answer:** No. For a one-way E profile the local storage changes, S=0, and Maxwell compatibility generally fails.

### Exercise 6: Tiny energy

Can an analytic energy-flow ratio be meaningful when absolute energies underflow? **Answer:** Yes, as a separately labelled property of the compatible family. It is not a quotient of available double-precision energy measurements.

## References and next connections

Use [Feynman II Chapter27](https://www.feynmanlectures.caltech.edu/II_27.html) for local energy bookkeeping and [Haus–Melcher Chapter11](https://ocw.mit.edu/courses/res-6-001-electromagnetic-fields-and-energy-spring-2008/0f814aee1c72517df7d7e7e96b7365e1_11.pdf) for field storage and power flow. The explicit pulse calculations above follow our stated Gaussian initial data. Later lessons add material interfaces, where constitutive assumptions and boundary conditions become central.
