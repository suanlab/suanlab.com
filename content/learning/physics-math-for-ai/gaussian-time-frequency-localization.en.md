## Define spread before discussing uncertainty

A narrow pulse and a narrow frequency spectrum compete, but the word narrow needs a precise definition. We use second moments of normalized squared magnitudes, centered at their respective means. These energy densities are mathematical distributions over time and angular frequency; they are not noisy estimates of an unknown signal.

Choose the complex Gaussian

$$
g(t)=\exp\!\left(-\frac{(t-\tau)^2}{4\sigma^2}\right)e^{i\Omega t},\qquad
G(\omega)=\int g(t)e^{-i\omega t}\,dt.
$$

The denominator 4σ² is intentional. Squaring the magnitude produces exp(−(t−τ)²/(2σ²)), so the normalized time-energy density has mean τ and standard deviation σ. This parameterization differs from a Gaussian amplitude written with denominator 2σ².

## Derive the frequency width

Translation gives a phase factor, modulation shifts frequency, and completing the square gives a frequency magnitude proportional to exp(−σ²(ω−Ω)²). After squaring and normalizing,

$$
p_t(t)=\frac{|g(t)|^2}{\int |g|^2},\quad
p_\omega(\omega)=\frac{|G(\omega)|^2}{\int |G|^2},\quad
\Delta t=\sigma,\quad\Delta\omega=\frac1{2\sigma}.
$$

Consequently ΔtΔω=1/2. For σ=1 the widths are one and one half. For σ=0.25 they are one quarter and two: a pulse four times narrower has a spectrum four times wider. Both have the same product.

## Centers and widths are different

Moving τ translates the blue time density. Moving Ω translates the orange angular-frequency density. Neither operation changes the centered variance. The overlay uses different physical coordinates for the two curves; their apparent intersection has no coordinate-independent meaning. Only the finite interval [−8,8] is drawn, so visible area may exclude tails. Analytic widths are computed over the whole real line.

## Units change the constant

For frequency in cycles per unit time, f=ω/(2π). Therefore Δf=Δω/(2π) and ΔtΔf=1/(4π), not one half. If time is seconds, angular frequency is radians per second and ordinary frequency is Hz. A factor-of-2π mistake can make two equivalent formulas appear inconsistent.

## Prove the centered inequality under explicit regularity

Take a nonzero Schwartz signal and remove its time and frequency means by translation and modulation. Call the centered result h. Rapid decay removes the boundary term in integration by parts:

$$
\operatorname{Re}\int t h'(t)\overline{h(t)}\,dt
=-\frac12\|h\|_2^2,
\qquad\frac12\|h\|_2^2\le\|th\|_2\|h'\|_2.
$$

Cauchy–Schwarz proves the second inequality. Plancherel and the derivative transform identify ‖h′‖₂/‖h‖₂ with Δω; the other ratio is Δt. Thus ΔtΔω≥1/2 in angular units. Equality requires h′=−cth with a positive real c, giving a Gaussian envelope. Translation, modulation and a constant complex factor preserve equality. Extension beyond Schwartz functions needs appropriate moment, weak-derivative and boundary control, not just finite energy.

## A chirp keeps the time width but broadens frequency

Consider the separate family

$$
g_\kappa(t)=e^{-t^2/(4\sigma^2)}e^{i\kappa t^2/2},\qquad
(\Delta\omega)^2=\frac1{4\sigma^2}+\kappa^2\sigma^2,
\qquad\Delta t=\sigma.
$$

The chirp changes phase but not |g|². Differentiate g explicitly: its derivative is (−t/(2σ²)+iκt)g. Integrating its squared magnitude and applying Plancherel proves the displayed frequency variance. Here covariance means Re ∫t ḡ(−ig′)/‖g‖² for these centered functions, the symmetrized multiplication/derivative covariance; no positive joint time-frequency density is assumed. It equals κσ², so Δt²Δω²−Cov²=1/4 even when the simpler width product is larger than 1/2.

| κ at σ=1 | Time standard deviation | Angular-frequency standard deviation | Product |
|---|---|---|---|
| 0 | 1 | .5 | .5 |
| .5 | 1 | .707107 | .707107 |
| 1 | 1 | 1.118034 | 1.118034 |
| 2 | 1 | 2.061553 | 2.061553 |

![A chirp leaves the Gaussian time-energy density unchanged while broadening its angular-frequency density; only the unchirped curve attains width product one half.](/assets/learning/harmonic-chirp-widths-en.svg)

The code integrates time and derivative energies independently of fitting a plotted spectrum. Chirp is an additional mathematical example, not a new control in the live Gaussian equality lab. The two density panels use different physical coordinates and units.

**Exercise:** Does the same time-energy envelope determine the frequency spread? **Explanation:** No. Quadratic phase changes the derivative energy and frequency covariance while leaving that envelope unchanged.

```python
from math import exp, sqrt

def simpson(f,a,b,n=20000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))/3

sigma=1.0
energy=lambda t:exp(-t*t/(2*sigma*sigma))
norm=simpson(energy,-12,12)
time_variance=simpson(lambda t:t*t*energy(t),-12,12)/norm
for chirp in [0,.5,1,2]:
    # |g'|^2 from g=exp(-t^2/4sigma^2) exp(i chirp*t^2/2), not a spectral-width fit.
    derivative_energy=simpson(lambda t:t*t*(1/(4*sigma**4)+chirp**2)*energy(t),-12,12)
    frequency_variance=derivative_energy/norm
    covariance=chirp*time_variance
    determinant=time_variance*frequency_variance-covariance**2
    assert abs(determinant-.25)<1e-12
    print('chirp, time sd, angular-frequency sd, product:',chirp,
          round(sqrt(time_variance),6),round(sqrt(frequency_variance),6),
          round(sqrt(time_variance*frequency_variance),6))
```

## Exercise and scope

Set σ=0.5, τ=1 and Ω=3. Predict widths 0.5 and one, centers one and three, and product one half. Then shift both centers and confirm unchanged widths.

For suitable nonzero finite-energy signals with the necessary moment and derivative regularity, Fourier uncertainty gives ΔtΔω≥1/2. The Gaussian realizes equality. This experiment is an equality example, not a numerical proof for all signals. The same mathematical tradeoff helps reason about localized AI features; it is not by itself a claim about quantum measurement or statistical estimation error.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
