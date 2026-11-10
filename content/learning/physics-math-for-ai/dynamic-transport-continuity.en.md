## Replace a static coupling with a moving density

For finite-second-moment probability densities, dynamic quadratic transport minimizes kinetic action subject to mass conservation:

$$
\inf_{\rho,v}\int_0^1\int_{\mathbb R}\rho_t(x)v_t(x)^2\,dx\,dt,
\qquad \partial_t\rho+\partial_x(\rho v)=0.
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

## Exercise and explained answer

Use m₀=−1,m₁=1,σ₀=0.5,σ₁=1. The endpoint squared distance is 4.25. At t=0.5 with the linear clock, m=0,σ=0.75 and integrated instantaneous kinetic energy is 4.25. The nonlinear clock has total action 17/3≈5.666667.

At query x=0 compare the separately computed derivatives and their sum. Then set identical endpoints: velocity, flux and action vanish while the density remains a normalized Gaussian. Explain why conservation, endpoint matching and action minimization require separate checks in learned velocity-field models.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/), including the dynamic transport formulation.
