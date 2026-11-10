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
