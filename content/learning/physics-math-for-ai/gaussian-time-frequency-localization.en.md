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

## Exercise and scope

Set σ=0.5, τ=1 and Ω=3. Predict widths 0.5 and one, centers one and three, and product one half. Then shift both centers and confirm unchanged widths.

For suitable nonzero finite-energy signals with the necessary moment and derivative regularity, Fourier uncertainty gives ΔtΔω≥1/2. The Gaussian realizes equality. This experiment is an equality example, not a numerical proof for all signals. The same mathematical tradeoff helps reason about localized AI features; it is not by itself a claim about quantum measurement or statistical estimation error.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
