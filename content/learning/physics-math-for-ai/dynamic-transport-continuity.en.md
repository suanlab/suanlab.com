## Replace a static coupling with a moving density

For finite-second-moment probability densities, dynamic quadratic transport minimizes kinetic action subject to mass conservation:

$$
\inf_{\rho,v}\int_0^1\int_{\mathbb R}\rho_t(x)v_t(x)^2\,dx\,dt,
\qquad \partial_t\rho+\partial_x(\rho v)=0
$$

With fixed endpoints, the minimum is W₂² under this convention without a factor one-half. The continuity equation alone is a feasibility condition, not a minimum-action certificate. Many clocks and detours can preserve the same mass and endpoints.

## An exactly computable Gaussian path

Let endpoints be N(m₀,σ₀²) and N(m₁,σ₁²), with positive standard deviations. Use a progress parameter s(t) and interpolate m=(1−s)m₀+sm₁ and σ=(1−s)σ₀+sσ₁. A particle with fixed standardized coordinate z follows x=m+σz. Its velocity at physical position x is

$$
v_t(x)=\dot m+\frac{\dot\sigma}{\sigma}(x-m).
$$

Interpolating standard deviations, rather than variances, corresponds to the monotone Gaussian transport map. The density is normalized on the whole real line; the finite drawing window does not truncate the analytic probability model.

## Check density change against flux divergence

Writing ρ for the Gaussian density,

$$
\partial_t\rho=\rho\left[-\frac{\dot\sigma}{\sigma}
+\frac{(x-m)\dot m}{\sigma^2}
+\frac{(x-m)^2\dot\sigma}{\sigma^3}\right].
$$

The code independently computes ∂ₓ(ρv) from ρₓ=−(x−m)ρ/σ² and vₓ=σ̇/σ. Their sum cancels analytically; the displayed residual exposes floating-point roundoff rather than forcing it to zero.

Blue is density and orange is flux ρv. They are different quantities in these dimensionless coordinates. Negative flux can represent motion to the left, while density remains nonnegative.

## A different clock increases action

Gaussian moments give ∫ρv²dx=ṁ²+σ̇². Thus the total action is

$$
\bigl((m_1-m_0)^2+(\sigma_1-\sigma_0)^2\bigr)
\int_0^1(s'(t))^2dt.
$$

For s=t the clock integral is one and action equals endpoint W₂². For s=t² it is 4/3. The second path visits the same Gaussian family and satisfies continuity, but runs at nonconstant speed and pays more action. Endpoint agreement does not establish dynamic optimality.

## Prove the action lower bound using moments and the clock

Assume sufficient regularity to differentiate moments and integrate the flux by parts with vanishing boundary terms. For a density with positive standard deviation σ, the functions 1 and (x−m)/σ are orthonormal in L2(ρ). Continuity gives ṁ=∫ρv and σ̇=∫ρv(x−m)/σ. Bessel's inequality therefore yields

$$
\int\rho v^2dx\ge\dot m^2+\dot\sigma^2,\qquad
\int_0^1\!\int\rho v^2dxdt
\ge(m_1-m_0)^2+(\sigma_1-\sigma_0)^2.
$$

The second step applies Cauchy–Schwarz in time to both derivatives. The affine Gaussian velocity attains the moment inequality; linear m and σ attain the time inequality. This proves global minimal action for the Gaussian endpoints within the stated admissible regularity class, beyond merely comparing two Gaussian clocks.

For the displayed Gaussian family put D²=(Δm)²+(Δσ)². Any absolutely continuous clock with square-integrable derivative and s(0)=0,s(1)=1 satisfies the exact identity

$$
A-D^2=D^2\int_0^1(s'(t)-1)^2dt\ge0.
$$

When D>0, equality requires unit clock speed almost everywhere. Identical endpoints have D=0 and are a separate degenerate case. For s=t+αt(1−t), |α|≤1 keeps the clock nondecreasing, and the excess factor is α²/3.

| α | Action / D² |
|---|---:|
| −1 | 4/3 |
| 0 | 1 |
| 1 | 4/3 |

```python
from fractions import Fraction as F
D2 = F(17, 4)
for alpha in (F(-1), F(0), F(1)):
    speed = (1+alpha, -2*alpha)  # polynomial coefficients of s'(t).
    squared = [F(0)]*3
    for i, x in enumerate(speed):
        for j, y in enumerate(speed): squared[i+j] += x*y
    integral = sum(x/F(i+1) for i, x in enumerate(squared))
    assert integral == 1+alpha*alpha/3
    assert integral >= 1
    print(alpha, integral, "action:", D2*integral)
```

The rational polynomial integral checks a continuum of clock choices through the derived formula, with three executed examples. It adds no clock control to the original lab. Conservation, sufficient moment regularity and optimal action remain distinct requirements.

![Three clocks follow the same Gaussian path but have different integrated action, with constant clock speed minimizing the action.](/assets/learning/transport-clock-en.svg)

## Exercise and explained answer

Use m₀=−1,m₁=1,σ₀=0.5,σ₁=1. The endpoint squared distance is 4.25. At t=0.5 with the linear clock, m=0,σ=0.75 and integrated instantaneous kinetic energy is 4.25. The nonlinear clock has total action 17/3≈5.666667.

At query x=0 compare the separately computed derivatives and their sum. Then set identical endpoints: velocity, flux and action vanish while the density remains a normalized Gaussian. Explain why conservation, endpoint matching and action minimization require separate checks in learned velocity-field models.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/), including the dynamic transport formulation.
