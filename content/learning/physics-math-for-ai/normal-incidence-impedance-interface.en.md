## A wave reaches a material boundary

A wave approaching a new material usually produces a reflected wave as well as a transmitted one. Matching only the electric-field amplitude is insufficient: the magnetic field carries an independent boundary requirement and determines the energy flux. An AI model can look smooth in E while violating that requirement.

We study normal incidence at one abrupt plane interface. Review the previous dielectric boundary lesson, transverse E/B compatibility, harmonic phases, and Poynting flow. Our media are stationary, isotropic, lossless, and nondispersive with constant positive ε and μ. Oblique incidence, absorbing media, coatings, finite slabs, and general frequency-dependent response require additional models.

## Separate wave speed from wave impedance

Speed sets how phase moves. Impedance relates the electric field to magnetic intensity H in a single traveling plane wave. They depend on different combinations of material coefficients.

$$
c_j=\frac1{\sqrt{\epsilon_j\mu_j}},\qquad Z_j=\sqrt{\frac{\mu_j}{\epsilon_j}},\qquad \omega=c_1k_1=c_2k_2.
$$

H is in A/m, B=μH is in T, and Z is in ohms. Here μ may differ between the media. Do not extend a nonmagnetic refractive-index reflection formula to unequal μ without rederiving its assumptions.

## Orient the three waves

Medium1 occupies z<0 and medium2 z>0. Let the incident E point along x and travel toward+z. The reflected E uses the same signed x basis but travels toward−z, so its H points with the opposite sign. The transmitted wave travels toward+z.

$$
\begin{aligned}
E_i&=A\cos(k_1z-\omega t),&H_i&=E_i/Z_1,\\
E_r&=rA\cos(k_1z+\omega t),&H_r&=-E_r/Z_1,\\
E_t&=\tau A\cos(k_2z-\omega t),&H_t&=E_t/Z_2.
\end{aligned}
$$

All H components above use the signed y basis. r can be negative, indicating a phase reversal of reflected E in this convention. We use τ for transmission amplitude to keep it distinct from time t.

## Match both tangential fields

Assume no imposed free surface-current sheet and no singular magnetic source. At z=0 all waves share the temporal factor. Tangential E and H continuity give two equations for the two unknown amplitudes.

$$
1+r=\tau,\qquad \frac{1-r}{Z_1}=\frac{\tau}{Z_2}.
$$

The minus sign in1−r comes from the reversed direction of reflected H. Using1+r for both E and H would mix the propagation conventions and generally produce a wrong interface solution. Normal D and B are zero in this transverse geometry, so those jump conditions are automatically satisfied.

## Solve for amplitude coefficients

Solving the two boundary equations gives the familiar impedance result. The same frequency appears on both sides, while the wavenumbers change with speed.

$$
r=\frac{Z_2-Z_1}{Z_2+Z_1},\qquad \tau=\frac{2Z_2}{Z_1+Z_2},\qquad 1-r=\frac{2Z_1}{Z_1+Z_2}.
$$

The formulas apply to the stated real, positive-coefficient normal-incidence family. They are not a general solution for arbitrary material interfaces. In particular, an abrupt interface and a gradual transition across many wavelengths can reflect differently.

## Convert amplitudes into power fractions

For nonzero incident amplitude and positive frequency, a single traveling wave has cycle-mean power per area A²/(2Z). Reflection returns power toward−z. Define R using its positive magnitude and T using transmitted forward power.

$$
R=r^2,\qquad T=\frac{Z_1}{Z_2}\tau^2,\qquad R+T=1.
$$

T is not simply τ². In normalized media with Z1=1 ohm and Z2=2 ohms, r=1/3 and τ=4/3, yet R=1/9 and T=8/9. A transmitted electric amplitude larger than the incident amplitude does not create energy: the changed impedance changes H and therefore E×H.

## Keep the power sign visible

The reflected cycle-mean power is negative in the signed z convention. The transmitted power is positive. A region containing only the source-free interface has equal net power entering and leaving.

$$
\langle S_i\rangle=\frac{A^2}{2Z_1},\qquad \langle S_r\rangle=-\frac{r^2A^2}{2Z_1},\qquad \langle S_t\rangle=\frac{\tau^2A^2}{2Z_2}.
$$

These values are in W/m². Infinite harmonic plane waves do not have a finite total three-dimensional energy. Work with local density, flux, or a clearly defined finite region rather than calling the cycle-mean flux “total energy.”

## Work an SI example

Use ε1=ε₀, μ1=μ₀, ε2=4ε₀, μ2=μ₀, and A=1 V/m. Then Z1≈376.730 ohms and Z2≈188.365 ohms. The result is r=−1/3, τ=2/3, R=1/9, T=8/9. Reflected E has the opposite sign at the interface.

The cycle-mean incident flux is about1.327209 mW/m², reflected signed flux−0.147468 mW/m², and transmitted flux1.179742 mW/m². Add the signed incident and reflected values to obtain the transmitted value. The model uses a rounded teaching μ₀ and derives ε₀ consistently from the SI light speed.

## Match impedance without matching speed

If ε2=2ε1 and μ2=2μ1, the impedance stays equal while speed halves. There is no reflected wave in this ideal family, but transmitted wavelength halves at the common frequency.

$$
Z_2=Z_1\ \Rightarrow\ r=0,\quad \tau=1;\qquad c_2=c_1/2\ \Rightarrow\ k_2=2k_1.
$$

Tangential H matches, while tangential B can change because B=μH. In this example B2=2B1 at the interface. The boundary condition is not unconditional continuity of tangential B. A refractive-index-only formula with an implicit equal-μ assumption would incorrectly predict reflection here.

## Check actual fields, not only coefficients

In medium1, the actual E and H are the sums of incident and reflected fields. Their instantaneous Poynting product is E1H1. Opposite propagation signs make the mixed terms cancel in that product, even though electric and magnetic storage separately show interference.

The independent reference generator integrates the actual signed Poynting product over phase and solves the E/H boundary equations as a linear system. Merely evaluating an algebraic R+T identity inside the same implementation would not provide independent validation.

## Force no reflection and inspect the missing source

For a negative control, set actual r=0 and τ=1 despite unequal impedances. E remains continuous. H generally does not. With normal n=+z, the magnetic jump requires a free surface-current sheet along x.

$$
\mathbf K_f=\mathbf n\times(\mathbf H_2-\mathbf H_1),\qquad K_{fx}=H_{1y}-H_{2y},\qquad S_1-S_2=K_{fx}E_x.
$$

This contradicts the no-sheet source request. It is not proof that these bulk fields could never belong to a Maxwell solution: an explicitly supplied current sheet can account for the boundary power. Positive sheet work absorbs field energy; negative work supplies it. No constitutive law or driving circuit for that sheet is solved here.

For Z2=Z1/2, the forced transparent state needs negative sheet work. Its transmitted mean power is twice the incident mean power; the missing power would have to come from an active source. For Z2=2Z1, the required sheet absorbs half the incident mean power.

## Preserve nearly matched coefficients

Subtracting rounded square roots can give an inaccurate small r or even zero. Rationalize the difference and retain rounding errors in the two products forming the numerator.

$$
r=\frac{\mu_2\epsilon_1-\mu_1\epsilon_2}{\left(\sqrt{\mu_2\epsilon_1}+\sqrt{\mu_1\epsilon_2}\right)^2}.
$$

For normalized coefficients ε1=1−δ, ε2=μ1=1, μ2=1+δ, with δ=2⁻⁵², the numerator is−δ². Direct products or impedance square roots can round the contrast away. Compensated product arithmetic retains r of order10⁻³² and a positive reflectance of order10⁻⁶⁴. Seven180-digit references check near matches, exact matches, and strong contrasts.

T can round to1 while a very small R remains nonzero. That does not justify reporting zero reflection. Conversely, if the input coefficients themselves already rounded to identical values, no arithmetic rearrangement can reconstruct an unrecorded contrast.

## Evaluate a stable field basis

Near strong reflection,1+r or1−r can also lose relative accuracy. Use their direct positive impedance formulas and a standing-wave basis. Let C=cos(k1z), S=sin(k1z), U=cos(ωt), and V=sin(ωt).

$$
E_1=A[(1+r)CU+(1-r)SV],\qquad H_1=\frac{A}{Z_1}[(1-r)CU+(1+r)SV].
$$

In the matched branch, compute1+r using τ and1−r using2Z1/(Z1+Z2). The forced-zero-reflection branch instead uses its actual coefficients. Spatial and temporal derivatives follow the same field representation; they remain analytic diagnostics that need independent differences for verification.

## Handle interfaces and degenerate observations explicitly

At z=0, return two side limits rather than an arbitrary averaged field. H may be continuous while B is not. The two regional formulas are one-sided extensions, not physical claims that both materials occupy the same point.

At A=0, fields and incident power are zero. Formal medium-dependent coefficients remain defined, but measured power ratios are0/0 and unavailable. At k1=0, the family is static and has no oscillation cycle, so cycle averages are not reported. Nonzero absolute energies lost to arithmetic are also unavailable. Finite phase bounds are checked explicitly rather than silently clipping coordinates or wrapping a badly resolved phase.

## Connect boundary evidence to AI research

Bulk Maxwell residuals can pass in both open half-spaces while the interface source request fails. Validate boundary E/H traces, material coefficients, reflected H sign, actual signed power, and any declared surface current. Keep the compatible reflection solution separate from the actual forced-control output.

The references include81 independent boundary solves and phase integrals,96 spacetime field/derivative comparisons, and high-precision cancellation cases. These selected numerical observations are not rigorous error bounds or proof of all material behavior.

## Research extension: passive sheets and coherent absorption

### Close the surface-current law

Keep the original lossless half-spaces and normal incidence, but add an electric sheet with surface current K_x=Y_s E_s. There is no magnetic sheet, so tangential E remains continuous. The real positive port admittances Y_j=1/Z_j and the sheet admittance have units S, unlike a bulk conductivity in S/m. Write Y_s=G+iX with G≥0 and real sheet susceptance X. Use peak phasors with exp(−iωt), ω>0; cycle powers carry a factor 1/2. This is a frequency-local constitutive model, not a full dispersion or causality certificate for a material.

$$
Y_j=1/Z_j>0,\quad Y_s=G+iX,\quad G\ge0,\quad[Y_j]=[Y_s]={\rm S},\qquad E_s=E_1=E_2,\quad H_1-H_2=K_x=Y_sE_s.
$$

### Derive scattering from both boundary conditions

For a unit incident electric amplitude from the left, tangential E gives t=1+r. The reversed magnetic sign of the reflected wave gives the second boundary equation. Solving it yields r and t below. Passive G and positive port admittances keep the real part of the denominator positive, so the driven boundary solution exists. At G=X=0 recover the original no-sheet Fresnel solution. A purely reactive sheet G=0 can reflect but absorbs no cycle-mean power. At very large G the sheet field tends to zero and reflection tends to −1; an increasingly good conductor is not an increasingly good absorber.

$$
\mathcal D=Y_1+Y_2+Y_s,\quad t=1+r,\quad Y_1(1-r)-Y_2t=Y_st,\quad r=\frac{Y_1-Y_2-Y_s}{\mathcal D},\quad t=\frac{2Y_1}{\mathcal D}.
$$

$$
R=|r|^2,\quad T=\frac{Y_2}{Y_1}|t|^2,\quad\mathcal A=\frac{G}{Y_1}|t|^2=\frac{4Y_1G}{(Y_1+Y_2+G)^2+X^2},\quad R+T+\mathcal A=1.
$$

### Prove the one-beam absorption bound

The average sheet work is G|E_s|²/2. Dividing reflected, transmitted and absorbed powers by the nonzero incident power gives R, T and the absorbed fraction. The displayed nonnegative square proves the joint optimum G=Y₁+Y₂ and X=0. For equal media the maximum is one half. This bound assumes one incident beam, a zero-thickness electric sheet, no magnetic response and no backing reflector. It is not a universal bound for every absorber. Unequal media have a different one-sided bound Y₁/(Y₁+Y₂). Zero input has zero power but no observed normalized absorption fraction.

$$
a=Y_1+Y_2:\quad\mathcal A\le\frac{4Y_1G}{(a+G)^2}\le\frac{Y_1}{a},\quad\frac{Y_1}{a}-\frac{4Y_1G}{(a+G)^2}=\frac{Y_1(G-a)^2}{a(a+G)^2},\quad\mathcal A_{\max}=\frac{Y_1}{a}\ \text{at }G=a,\ X=0.
$$

### Allow coherent inputs from both sides

Let a_L and a_R be incoming electric phasors and b_L and b_R outgoing phasors, all in the same signed x basis. Solve E_s=a_L+b_L=a_R+b_R together with the current jump to obtain the following expressions. Input power is the sum of incoming traveling-channel powers; it is not the square of a summed electric amplitude. The actual sheet field does depend on the coherent sum, so relative phase controls absorption. The same local power balance accounts for that dependence without creating energy.

$$
E_s=\frac{2(Y_1a_L+Y_2a_R)}{\mathcal D},\quad b_L=E_s-a_L,\quad b_R=E_s-a_R,\quad P_{\rm in}=\frac{Y_1|a_L|^2+Y_2|a_R|^2}{2},\quad P_{\rm in}-P_{\rm out}=\frac G2|E_s|^2.
$$

### Normalize ports before discussing scattering modes

Multiplying each channel amplitude by sqrt(Y_j) puts power into a Euclidean norm. With v=(sqrt(Y₁),sqrt(Y₂)), the matrix has one mode along v and a perpendicular mode with eigenvalue −1. The latter has zero sheet field and never dissipates in this electric sheet. Direct multiplication gives the Hermitian absorption operator below, including reactive X. Its nonzero eigenvalue reaches one only at G=Y₁+Y₂, X=0. Then equal nonzero electric input phasors give no outgoing wave: coherent perfect absorption. In unequal media those equal electric amplitudes carry unequal powers. The general dark condition is Y₁a_L+Y₂a_R=0, not unconditional equal-and-opposite electric amplitudes.

$$
\widehat a=\begin{pmatrix}\sqrt{Y_1}a_L\\\sqrt{Y_2}a_R\end{pmatrix},\quad\widehat b=\mathcal S\widehat a,\quad v=\begin{pmatrix}\sqrt{Y_1}\\\sqrt{Y_2}\end{pmatrix},\quad\mathcal S=\frac{2vv^T}{\mathcal D}-I,\quad P_{\rm in}=\frac12\|\widehat a\|^2,\quad P_{\rm out}=\frac12\|\widehat b\|^2.
$$

$$
\mathcal S v=\frac{a-Y_s}{a+Y_s}v,\quad\mathcal S v_\perp=-v_\perp,\quad I-\mathcal S^\dagger\mathcal S=\frac{4aG}{|a+Y_s|^2}\frac{vv^T}{a},\qquad 0\le\frac{4aG}{|a+Y_s|^2}\le1.
$$

$$
G=a,\ X=0,\ a_L=a_R\ne0:\quad b_L=b_R=0;\qquad Y_1a_L+Y_2a_R=0:\quad E_s=0,\quad P_{\rm absorbed}=0.
$$

### Separate a fixed phase from an incoherent ensemble

For symmetric media at the critical conductance, equal input magnitudes with phase difference φ absorb cos²(φ/2) of the incoming power. In-phase inputs are fully absorbed; opposite-phase inputs have E_s=0 and no absorption. The code solves the boundary field for four exact phase choices and prints the outgoing and absorbed powers. Averaging repeated monochromatic experiments with equal mean-squared electric amplitudes gives the coherence expression below. It is a ratio of averaged powers, not generally an average of fluctuating normalized ratios. Uniform random relative phase gives zero coherence and mean absorption one half. This reconciles the one-beam limit with coherent perfect absorption; broadband dynamics and frequency-dependent sheet models require separate treatment.

$$
Y_1=Y_2=Y,\ G=2Y,\ X=0:\quad r=-\tfrac12,\ t=\tfrac12,\quad a_L=A,\ a_R=Ae^{i\phi},\quad A\ne0:\quad\mathcal A_{\rm two}=\cos^2(\phi/2).
$$

$$
\langle|a_L|^2\rangle=\langle|a_R|^2\rangle=I_0>0,\quad\gamma_{12}=\frac{\langle a_La_R^*\rangle}{I_0},\quad\frac{\langle P_{\rm absorbed}\rangle}{\langle P_{\rm in}\rangle}=\frac{1+\operatorname{Re}\gamma_{12}}{2}\quad\text{(symmetric critical sheet)}.
$$

| phase / pi | left outgoing / input | right outgoing / input | absorbed / input | power residual |
| --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 0.00000000 | 1.00000000 | 0.00000000 |
| 0.50000000 | 0.25000000 | 0.25000000 | 0.50000000 | -0.00000000 |
| 1.00000000 | 0.50000000 | 0.50000000 | 0.00000000 | 0.00000000 |
| 1.50000000 | 0.25000000 | 0.25000000 | 0.50000000 | -0.00000000 |

```python
# Peak phasors: Y1=Y2=1, G=2, X=0; power unit Y*E_ref^2.
# Exact quarter-turn phasors avoid treating sin(pi) roundoff as a physical field.
for phase_over_pi, right in [(0.0,1), (0.5,1j), (1.0,-1), (1.5,-1j)]:
    left = 1
    Es = 2*(left+right)/4
    out_left, out_right = Es-left, Es-right
    incoming = (abs(left)**2+abs(right)**2)/2
    L, R = abs(out_left)**2/2, abs(out_right)**2/2
    absorbed = 2*abs(Es)**2/2
    assert abs(incoming-L-R-absorbed) < 1e-14
    print(f"{phase_over_pi:.8f} {L/incoming:.8f} {R/incoming:.8f} "
          f"{absorbed/incoming:.8f} {(incoming-L-R-absorbed):.8f}")
```

[Chong and colleagues, PRL 105, 053901 (2010)](https://www.eng.yale.edu/caolab/papers/PRL10.pdf), supplies the scattering-zero and coherent-illumination context. Our zero-thickness electric-sheet model, passivity bound, unequal-port normalization and phase/coherence examples are derived from the displayed boundary conditions, not adopted as a benchmark result for that paper’s slab. The native no-sheet Fresnel limit and forced-transparent source accounting remain available as separate comparisons.

![One-beam absorption bound, coherent phase and reactive mismatch](/assets/learning/electromagnetism-coherent-sheet-absorption.en.svg)

### Research questions with explanations

**Does the symmetric one-beam 50 percent limit rule out complete absorption?** **Answer:** No. Two coherent incident channels can occupy the fully absorbed mode. The one-beam bound has different illumination assumptions.

**Are equal incoming powers always the perfectly absorbed state?** **Answer:** No. At the critical sheet the electric phasors must match; unequal port admittances then imply unequal incoming powers.

**Does making the sheet conductance arbitrarily large maximize absorption?** **Answer:** No. It drives the sheet field to zero and approaches a reflector. Maximum absorption needs the matched finite conductance and zero reactive part.

## Exercises with explanations

### Exercise 1: Greater electric amplitude

If Z2=2Z1, calculate τ and T. **Answer:** τ=4/3 but T=8/9. The impedance factor converts amplitude squared into power fraction.

### Exercise 2: Reflected H sign

Why is Hr=−Er/Z1? **Answer:** Reflected propagation is toward−z. Keeping the same x and y component bases reverses the E-to-H relation.

### Exercise 3: Same impedance, different speed

Double both ε and μ in medium2. Is there reflection? **Answer:** No in this ideal model. Z stays fixed, while speed and wavelength halve.

### Exercise 4: Which magnetic field is continuous?

If tangential H matches but μ2=2μ1, must tangential B match? **Answer:** No. B2=2B1 for the same H.

### Exercise 5: A missing sheet

Is forced r=0, τ=1 across unequal impedances always impossible? **Answer:** It violates the no-sheet request, but a required surface current can supply or absorb the boundary power. Its driving law needs a separate model.

### Exercise 6: No incident power

At A=0, can one measure R as reflected power divided by incident power? **Answer:** No. That ratio is undefined. A formal scattering coefficient describes the media and does not become a measured0/0 quotient.

## References and next connections

Compare the boundary matching procedure and its nonmagnetic caveat with [Feynman II Chapter33](https://www.feynmanlectures.caltech.edu/II_33.html). [Haus–Melcher §9.2](https://web.mit.edu/6.013_book/www/chapter9/9.2.html) distinguishes magnetic flux density from magnetic intensity and magnetization. Our general positive-μ impedance formulas follow the stated macroscopic E/H conditions. Later work adds gauge freedom, circuit/sensor coupling, and effective material models.
