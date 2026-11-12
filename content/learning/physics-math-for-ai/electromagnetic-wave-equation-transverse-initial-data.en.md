## Same initial electric field, different futures

If you know the electric-field shape perfectly at one instant, do you know its future? Generally you also need initial magnetic field or compatible time-change information. The same initial Gaussian E can propagate one way or split in two. We first study field compatibility and propagation; quantitative energy-flow analysis comes later.

The preceding sinusoidal plane wave was one exact solution. Smooth nonperiodic pulses now connect the wave equation, first-order Maxwell relations, and initial data. The goal is to avoid solving a scalar wave equation and attaching an unrelated magnetic field afterward.

## Constant coefficients and transverse assumptions

The positive coefficients ε and μ are constant, with c=1/√(εμ), in a vacuum-form SI teaching model. Default ε is derived from the rounded teaching μ and SI c; arbitrary coefficient choices do not constitute a complete real-material model.

E has only an x component, B only a y component, and spatial variation is only along z. These fields are transverse to the propagation axis. This does not assert that every electromagnetic field is transverse: near sources, in waveguides, and at material boundaries, other components and constraints matter.

$$
\mathbf E=(E_x(z,t),0,0),\qquad\mathbf B=(0,B_y(z,t),0),\qquad c=\frac{1}{\sqrt{\epsilon\mu}}.
$$

## Start from the two first-order evolution laws

With these components both divergences vanish structurally. Faraday and Ampere–Maxwell reduce to the following first-order equations. Each field restricts the spatial and temporal variation of the other. Check their signs directly from curl components.

The initial z derivative of B determines the initial time derivative of E, and the initial z derivative of E determines the initial time derivative of B. You cannot prescribe all four initial fields and rates independently.

$$
\partial_z E_x=-\partial_tB_y,\qquad-\partial_zB_y=\frac{1}{c^2}\partial_tE_x.
$$

## Derive the wave equation and state its limits

Differentiate the first equation in time and use the z derivative of the second to derive the E wave equation. B satisfies the same form. This proves that first-order Maxwell compatibility implies the second-order wave equations; it does not prove an unconditional converse.

Differentiation can lose the original first-order relation between the fields. Just as Gauss constraints and initial or boundary data need checking, small componentwise wave residuals cannot replace full Maxwell verification.

$$
\partial_t^2E_x=c^2\partial_z^2E_x,\qquad\partial_t^2B_y=c^2\partial_z^2B_y.
$$

## Two profiles moving in opposite directions

The profile f(z−ct) translates along +z and g(z+ct) along −z. Adding them produces E, while B requires a relative sign that depends on propagation direction. That sign makes both first-order equations hold.

With fixed field axes, a sign change in B reverses its component direction. Where the profiles overlap, the single-direction relation B=E/c cannot be imposed on the total sum. Apply it to each travelling component before superposing.

$$
E_x=f(z-ct)+g(z+ct),\qquad B_y=\frac{f(z-ct)-g(z+ct)}{c}.
$$

## Split one Gaussian initial profile

Amplitude A is in V/m, width σ in m, and β is dimensionless between zero and one. Define the travelling components to share one initial shape. β=1 gives right-only propagation, β=0 left-only, and β=1/2 symmetric splitting.

A Gaussian becomes very small far away but has no exactly zero mathematical tail. This plane field is also infinite in x and y. Longitudinal localization does not make it a finite three-dimensional antenna field or a sharp causal wavefront.

$$
G(\xi)=A\exp\left(-\frac{\xi^2}{2\sigma^2}\right),\qquad f=\beta G,\qquad g=(1-\beta)G.
$$

## Read compatible initial data

At t=0 all three cases have E=G(z), but their B and ∂tE differ as shown below. One electric-field snapshot cannot identify propagation direction.

| β | Initial E | Initial B | Initial ∂tE |
| --- | --- | --- | --- |
| 1 | G | G/c | −cG′ |
| 1/2 | G | 0 | 0 |
| 0 | G | −G/c | +cG′ |

Even for the symmetric split with initial B=0, initial ∂tB=−G′ is generally nonzero. Replacing zero initial B with zero B for all time changes the model.

$$
E_x(z,0)=G(z),\quad B_y(z,0)=\frac{(2\beta-1)G(z)}{c},\quad\partial_tE_x(z,0)=c(1-2\beta)G'(z).
$$

## Differentiate at one point

Use normalized c=1, A=σ=1 at z=1 and t=0. Then G=e⁻¹ᐟ²≈0.60653066 and G′=−G. For right propagation, E=B=G, ∂zE=∂zB=−G, and ∂tE=∂tB=+G. Check both first-order signs.

The symmetric split has the same E and ∂zE but B=∂zB=∂tE=0, while ∂tB=+G. Comparing initial field values alone misses this change. Do not confuse the normalized numerical B with B obtained using default SI coefficients.

## Values after the split develops

With normalized c=A=σ=1, β=1/2, t=2, and z=2, the right-moving component is at its center and the left component is four units away. Their tails still overlap, so E is not exactly one half. Retain the small contribution shown below.

For a sketch, comparing E and cB in the same units is convenient, while original B should be read separately. Overlapping pulses can have different E and cB magnitudes; a large E does not always imply B in a fixed single-direction ratio.

$$
E_x=\tfrac12(1+e^{-8})\approx0.50016773,\qquad cB_y=\tfrac12(1-e^{-8})\approx0.49983227.
$$

## Zero B at the center with nonzero curl

At z=0 in the symmetric split, B is zero at every time. This does not mean B vanishes throughout space. It has opposite signs on the two sides and can have a nonzero spatial slope at the center.

For normalized c=A=σ=1 at t=1, central E=e⁻¹ᐟ², B=0, ∂tE=−e⁻¹ᐟ², and ∂zB=+e⁻¹ᐟ². Thus the x component of curl B, −∂zB, matches changing E. Distinguishing a point value from surrounding variation was also essential in the earlier charge-density example.

## A wrong magnetic field that passes wave equations

Keep the right-moving E=G(z−ct) but replace B by zero everywhere. E still satisfies its scalar wave equation, and zero B satisfies its own wave equation. Both divergences vanish too. Nevertheless, generic nonzero ∂zE violates Faraday, and the required magnetic curl for ∂tE is missing.

This is an explicit negative control, not a complete source-free Maxwell solution. At a pulse center the derivatives can happen to vanish, giving zero sampled residual. Test multiple positions and times. The complete zero field A=0 remains a valid special case.

$$
B_y=0:\qquad r_F=\partial_zE_x,\qquad r_A=-\frac{\partial_tE_x}{c^2}.
$$

## Numerical cancellation between two pulses

For β=1/2 at a very small time, G(z−ct) and G(z+ct) can round to the same floating-point value. Direct subtraction may suggest B=0 although their mathematical difference is nonzero. Compute the exponent difference first and use expm1 to retain it.

The function expm1(x) evaluates exp(x)−1 accurately for small x. Avoid multiplying a huge sinh by a tiny exponential, which can overflow at an intermediate step. The expression below covers zct≥0; the opposite-sign branch uses the other Gaussian as reference. Exact symmetry zeros at z=0 or t=0 differ from numerical loss.

$$
d=\frac{2zct}{\sigma^2},\qquad G(z-ct)-G(z+ct)=G(z-ct)\big[-\operatorname{expm1}(-d)\big]\quad(d\ge0).
$$

## Separate zero amplitude, tails and supported range

If A=0 the fields are exactly zero. If a nonzero Gaussian tail rounds to zero outside arithmetic range, the model reports it unavailable rather than declaring a mathematical zero field. An exact symmetric B=0 can sometimes remain known even when the E tail is unavailable. Track availability separately for each quantity.

The pure model supports A∈[−4,4] V/m, σ∈[0.1,4] m, z∈[−4,4] m, t∈[−4,4] s, and |ct|≤8 m. With default SI c, seconds exceed the displacement budget, so nanosecond-scale times are needed. Inputs are not silently clamped or wrapped into another pulse. This is a declared computational domain, not a limitation of the physical law.

## Checks for an AI field model

Training only E values and wave residuals can miss an incompatible B. Check initial E and B, first-order evolution, divergence, and boundary data separately. Loss terms have different units, so document nondimensionalization and weighting. Distinguish a pointwise residual from a family-wide classification and confirm that negative controls actually fail.

Derivatives in the product model are diagnostics derived from the same analytic family. Independent references use separate Python five-point spatial and temporal differences, actual-field line integrals and flux rates, and160-digit mpmath calculations. Do not confuse differentiation error, cancellation, or tail representation limits with mathematical defects or physical zeros. Energy, Poynting flow, and material boundaries remain subsequent original topics.

## Research extension: infer initial magnetic fields from electric snapshots

### Turn initial-state inference into a two-number problem

Suppose the source-free fields are smooth finite Fourier polynomials on a periodic interval of length L_z with constant ε and μ. Assume spatial sampling actually resolves each retained mode. For one real standing-mode quadrature write E and cB below, so a and b both have units V/m. The other spatial quadrature is treated analogously. Substitution into the first-order Maxwell equations gives a rotation, with a_dot=−ωb and b_dot=ωa. This is continuous propagation; it is distinct from the preceding staggered-grid update. The periodic geometry is a supplementary inverse problem, not the native Gaussian pulse’s boundary model.

$$
E_x=a(t)\cos(kz),\quad cB_y=b(t)\sin(kz),\quad k=\frac{2\pi n}{L_z}>0,\quad\omega=ck,\quad\frac{d}{dt}\binom a b=\omega\begin{pmatrix}0&-1\\1&0\end{pmatrix}\binom a b.
$$

$$
\binom{a(T)}{b(T)}=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}\binom{a_0}{b_0},\quad\phi=\omega T,\quad a(T)^2+b(T)^2=a_0^2+b_0^2.
$$

### Identify exactly when two electric snapshots suffice

If both a(0) and a(T) are measured, their observation matrix has determinant −sin φ. At φ=mπ, every possible b₀ gives the same second electric amplitude (−1)^m a₀. This is genuine non-identifiability even with exact data and a perfect Maxwell model. Away from these times the inverse exists. Adding a small positive denominator at a blind time invents a choice; it does not recover the missing datum. Here “snapshot” means a spatially resolved Fourier coefficient, not one electric-field value at an arbitrary point.

$$
\binom{a(0)}{a(T)}=O_T\binom{a_0}{b_0},\quad O_T=\begin{pmatrix}1&0\\\cos\phi&-\sin\phi\end{pmatrix},\quad\det O_T=-\sin\phi,\quad\operatorname{rank}O_T=\begin{cases}2,&\sin\phi\ne0,\\1,&\sin\phi=0.\end{cases}
$$

### Propagate errors before interpreting a reconstruction

Let y₀ and y_T include zero-mean measurement errors. Algebra gives the exact error expression below, and taking its variance gives the following formula. No Gaussian distribution is required. If a₀ is known exactly, error in the later snapshot is amplified by 1/|sin φ|. If both errors are independent with equal variance, the standard-deviation gain is sqrt(1+cos²φ)/|sin φ|. Correlated errors require the covariance term: sharing an uncertain initial snapshot across several later residuals does not produce independent residual noise. A well-conditioned formula also assumes that c, k and the measurement times are known; parameter and clock errors are additional sources.

$$
\widehat b_0=\frac{y_0\cos\phi-y_T}{\sin\phi},\quad y_0=a_0+\eta_0,\quad y_T=a_0\cos\phi-b_0\sin\phi+\eta_T,\quad\widehat b_0-b_0=\frac{\eta_0\cos\phi-\eta_T}{\sin\phi}.
$$

$$
\operatorname{Var}(\widehat b_0)=\frac{v_0\cos^2\phi+v_T-2v_{0T}\cos\phi}{\sin^2\phi},\quad v_0=\operatorname{Var}\eta_0,\quad v_T=\operatorname{Var}\eta_T,\quad v_{0T}=\operatorname{Cov}(\eta_0,\eta_T).
$$

### Add informative times, rather than duplicate blind times

For this paragraph take a₀ as exactly known and later errors as independent with common variance v_eta. Then z_i=s_i b₀−η_i. Differentiating the sum of squared residuals yields the scalar least-squares solution and its variance when D>0. Two times whose phases differ by π/2 have D=sin²φ+cos²φ=1 for this particular mode. Repeating only blind times leaves D=0. A common quarter-period interval is mode dependent; do not promise D=1 for every Fourier mode from the same two global times. Weighted or correlated noise requires the corresponding measurement covariance, not this equal-variance shortcut.

$$
a_0\ \text{known}:\quad z_i=a_0\cos\phi_i-y_i,\quad s_i=\sin\phi_i,\quad D=\sum_i s_i^2,\quad\widehat b_0=\frac{\sum_i s_i z_i}{D},\quad\operatorname{Var}(\widehat b_0)=\frac{v_\eta}{D}\quad(D>0).
$$

### Distinguish regularization from new information

Adding λ b² to the declared least-squares objective selects the ridge estimate below. With exact noiseless data it is biased toward zero by D/(D+λ). At D=0 it returns zero for every possible true b₀. That is a prior preference, not an observation. For a finite nonzero band n=1,…,N_b, the proposed single later time T_* avoids all exact temporal blind modes. However its worst known-a₀ noise gain grows approximately as 2N_b/π. Increasing bandwidth therefore removes the uniform conditioning benefit; no infinite-mode stability theorem follows. The mean magnetic mode remains unobserved at every time because spatial derivatives annihilate it.

$$
\widehat b_{\lambda}=\frac{\sum_i s_i z_i}{D+\lambda},\quad\lambda>0,\quad\eta_i=0:\quad\widehat b_{\lambda}=\frac{D}{D+\lambda}b_0,\qquad D=0:\quad\widehat b_{\lambda}=0\ \text{for every true }b_0.
$$

$$
1\le n\le N_b,\quad T_*={\pi\over2\omega_{N_b}}:\quad\phi_n={n\pi\over2N_b},\quad\min_n|\sin\phi_n|=\sin{\pi\over2N_b}>0,\quad\max_n {1\over|\sin\phi_n|}=\csc{\pi\over2N_b}.
$$

### Separate temporal blindness from spatial blindness

A temporal inversion formula does not by itself justify treating aliased spatial samples as separate Fourier coefficients. At N equally spaced points, modes n and n+N have the same cosine samples at the initial time. State a resolved finite band, or analyze the full space–time sampling operator, before inferring amplitudes. Additional times can distinguish some spatial aliases, but that is a different inverse problem. The existing Gaussian family gives a different, point-observation counterexample: at z=0 all right fractions β have the same E at every time, yet different B. An off-center electric measurement can distinguish that family. This does not replace full Fourier-coefficient observations or establish a general point-sensor theorem. In the cell below a₀=0.6 and b₀=0.8 V/m, and only the later amplitude receives a fixed +0.01 V/m error. Exact designed phases are declared as rational multiples of π; integer multiples are reported unavailable rather than using the floating approximation sin(π) as information. Nonzero nearby phases remain available but ill conditioned.

$$
\partial_t\overline E_x=\partial_t\overline B_y=0,\qquad E_x(0,t)=A e^{-(ct)^2/(2\sigma_G^2)},\qquad cB_y(0,t)=(2\beta-1)E_x(0,t)\quad\text{(Gaussian family)}.
$$

| phase / pi | absolute sine | later-noise gain | estimated b0 [V/m] |
| --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | unavailable | unavailable |
| 0.16666667 | 0.50000000 | 2.00000000 | 0.78000000 |
| 0.50000000 | 1.00000000 | 1.00000000 | 0.79000000 |
| 1.00000000 | 0.00000000 | unavailable | unavailable |

```python
from fractions import Fraction
from math import sin, cos, pi
# Fractions declare exact design phases in units of pi.
# A phase equal to an integer*pi is mathematically blind.
a0, b0, eta = 0.6, 0.8, 0.01
for p in [Fraction(0), Fraction(1,6), Fraction(1,2), Fraction(1)]:
    if p.denominator == 1:
        print(f"{float(p):.8f} 0.00000000 unavailable unavailable")
        continue
    c, s = cos(pi*float(p)), sin(pi*float(p))
    y = a0*c-b0*s+eta  # a0 known exactly; noise on later snapshot
    estimate = (a0*c-y)/s
    print(f"{float(p):.8f} {abs(s):.8f} {1/abs(s):.8f} {estimate:.8f}")
```

[Feynman II §20–1](https://www.feynmanlectures.caltech.edu/II_20.html) supplies the counterpropagating-wave background. [Boyd’s observability notes](https://see.stanford.edu/materials/lsoeldsee263/19-observ.pdf) connect observation-matrix nullspaces and least-squares state estimation. The finite-mode timing, noise, ridge and bandwidth formulas above are derived directly here. The native Gaussian example tests the stated center-point ambiguity without claiming a new periodic Fourier-inversion control.

![Electric-only timing ambiguity, noise and finite-band conditioning](/assets/learning/electromagnetism-electric-snapshot-observability.en.svg)

### Research questions with explanations

**Can exact Maxwell physics identify b0 at a blind sampling time?** **Answer:** No. When sin φ=0 the electric observation matrix has a nontrivial nullspace even with noiseless data.

**Does repeating an uninformative time improve identifiability?** **Answer:** No. Its s_i is zero and adds nothing to D. Choose an informative time or a different physical measurement.

**Does a finite-band schedule prove stable recovery of all modes?** **Answer:** No. Its worst noise gain grows with bandwidth, and the mean magnetic mode remains invisible to electric evolution.

## Exercises with explanations

Write initial values, time derivatives, and spatial derivatives separately.

### Exercise 1: Same initial E

If initial E=G and B=G/c, what is β in this family? **Answer:** β=1, giving right propagation and initial ∂tE=−cG′. The same E with B=0 corresponds to β=1/2.

### Exercise 2: Initially zero B

For β=1/2, does initial B=0 imply initial ∂tB=0? **Answer:** Generally no. The derivative is −G′ and can be nonzero away from z=0. Distinguish initial value from initial rate.

### Exercise 3: Passing both wave equations

Is a travelling E paired with B=0 sufficient if both wave residuals vanish? **Answer:** No. Check Faraday ∂zE=−∂tB and Ampere −∂zB=∂tE/c² separately.

### Exercise 4: Zero B at the center

In a symmetric split, if B at z=0 is always zero, is curl B there also zero? **Answer:** No. Opposite B values on the two sides can produce nonzero ∂zB, supporting the central ∂tE.

### Exercise 5: A small pulse difference

For β=1/2, c=σ=A=z=1 and t=10⁻¹⁸, is B exactly zero if direct subtraction gives zero? **Answer:** No. Stable evaluation and high precision retain cB≈e⁻¹ᐟ²×10⁻¹⁸. Distinguish this from exact symmetry at t=0.

### Exercise 6: Width and curvature

If A stays fixed and σ is halved, what happens to the initial central second spatial derivative of E? **Answer:** Ezz(0,0)=−A/σ², so its magnitude grows fourfold. In the compatible family Ett=c²Ezz changes with it.

## References and next connections

Compare §20–1, equations20.24–20.25 of [Feynman Lectures II Chapter 20](https://www.feynmanlectures.caltech.edu/II_20.html), with axes relabeled to z propagation, x electric field, and y magnetic field. The following energy and Poynting-flow topic will connect propagation to quantitative energy transport.
