## A gradient flow descends an energy on distributions

Consider free energy F(ρ)=∫x²ρ/2+T∫ρlogρ, with diffusion temperature T≥0. Its variational derivative gives velocity v=−x−T∂ₓlogρ. Combined with continuity, this produces the Ornstein–Uhlenbeck Fokker–Planck equation:

$$
\partial_t\rho=\partial_x(x\rho)+T\partial_{xx}\rho.
$$

For Gaussian initial data, mean and variance solve m′=−m and (σ²)′=−2σ²+2T. Thus m(t)=m₀e⁻ᵗ and σ²(t)=σ₀²e⁻²ᵗ+T(1−e⁻²ᵗ). The implementation uses a stable exponential remainder near time zero.

## A variational time step, not a density mixture

The Jordan–Kinderlehrer–Otto minimizing-movement step solves

$$
\rho_{k+1}=\arg\min_\rho
\left[\frac{W_2^2(\rho,\rho_k)}{2h}+F(\rho)\right].
$$

For a Gaussian previous state, Gaussian candidates attain the global minimum here. For fixed mean and variance, a Gaussian minimizes negative entropy; the moment lower bound on W₂² is attained by the Gaussian affine coupling. Both observations justify this reduction rather than merely assuming a convenient family.

Up to an additive constant, Gaussian free energy is (m²+σ²)/2−Tlogσ. Minimizing the proximal objective gives

$$
m^+=\frac{m}{1+h},\qquad
(1+h)(\sigma^+)^2-\sigma\sigma^+-hT=0.
$$

The positive root supplies the next standard deviation. This is not implicit Euler applied directly to the variance equation. Parameterization and Wasserstein geometry matter.

## Compare at the same physical time

Blue plots exact continuous free energy at times kh. Orange plots actual repeated JKO values at those times. A finite variational step is not exact time integration. As h decreases at fixed total time, the time-discretization discrepancy can be examined separately from energy decrease.

With m₀=σ₀=T=1,h=0.5 and two steps, σ remains one and JKO mean is 4/9. Continuous mean at time one is e⁻¹≈0.367879. Their W₂ discrepancy is approximately 0.076565. With T=0 both mean and standard deviation become 4/9, giving discrepancy approximately 0.108279.

## Movement is paid by energy decrease

Using the previous law as a competitor yields

$$
F(\rho_{k+1})+W_2^2(\rho_{k+1},\rho_k)/(2h)\le F(\rho_k).
$$

Summing bounds accumulated movement cost by total energy drop. The table reports each movement cost and the readout compares the accumulated quantity with the actual drop. Proximal stationarity residuals are also checked separately.

## Justify the Gaussian reduction and bound a time error

For any coupling of laws with means m,m₀ and standard deviations σ,σ₀, Cauchy–Schwarz bounds their centered covariance by σσ₀. Expanding E|X−Y|² gives W₂²≥(m−m₀)²+(σ−σ₀)². For a density ρ with these moments and finite entropy, nonnegativity of KL(ρ||Gaussian(m,σ²)) shows that the Gaussian has the smallest ∫ρlogρ. The affine Gaussian coupling attains the distance lower bound. Thus both lower bounds are attained together, proving the stated global Gaussian reduction; positive diffusion multiplies the entropy bound by T≥0.

For T=0, the mean and standard deviation both multiply by 1/(1+h) at each JKO step. At physical time t=nh, use h−h²/2≤log(1+h)≤h to obtain

$$
W_2(\rho_n,\rho(t))
\le\sqrt{m_0^2+\sigma_0^2}\,e^{-t}\operatorname{expm1}(th/2).
$$

Indeed δ=t−nlog(1+h) lies in [0,th/2], and the actual error is √(m₀²+σ₀²)e⁻ᵗ(expm1 δ). This proves a first-order fixed-time error bound for this zero-temperature Gaussian model, rather than inferring an order from one plot. For positive T the same mean bound holds, but the scale needs its own analysis.

| h | n, t=1 | Compared states |
|---|---:|---|
| 1/2 | 2 | (1+h)⁻ⁿ versus e⁻¹ |
| 1/4 | 4 | (1+h)⁻ⁿ versus e⁻¹ |
| 1/8 | 8 | (1+h)⁻ⁿ versus e⁻¹ |

```python
from fractions import Fraction as F
from math import exp, expm1, sqrt
for h in (F(1, 2), F(1, 4), F(1, 8)):
    n = int(1/h)
    state = F(1)
    movement = F(0)
    for _ in range(n):
        next_state = state/(1+h)
        movement += (next_state-state)**2/h  # two equal coordinates.
        state = next_state
    drop = 1-state*state
    error = sqrt(2)*(float(state)-exp(-1))
    bound = sqrt(2)*exp(-1)*expm1(float(h)/2)
    assert movement <= drop
    assert 0 <= error <= bound
    print(h, n, "W2 error/bound:", error, bound, "energy drop:", drop)
```

For tiny means at equilibrium scale, the total absolute free energies can round to the same number even though movement is positive. The lab evaluates the energy drop through mean products and a logarithmic scale remainder, preserving a mean of 10⁻²⁰ without turning a rounded difference of large energies into the dissipation certificate. The coefficient T−σ₀² also includes the low part of the rounded square, retaining the leading term near equilibrium. The absolute energies remain separately displayed; finite-precision state and scale rounding are still numerical limits.

![At fixed physical time one, zero-temperature Gaussian JKO errors decrease under refinement and remain below the proved bound.](/assets/learning/transport-jko-error-en.svg)

## Exercise and explained answer

Set m₀=0,σ₀=T=1. The standard Gaussian is equilibrium: all states and energies remain unchanged and movement cost is zero. With T=0, concentration toward a Dirac law occurs only as time grows; positive finite-time standard deviation should not be replaced by a zero-width density.

Explain the difference between a free-energy gradient flow, a shortest path between fixed endpoint laws and a linear mixture of densities. All evolve distributions, but they solve different mathematical problems.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
