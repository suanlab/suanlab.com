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

## Exercise and explained answer

Set m₀=0,σ₀=T=1. The standard Gaussian is equilibrium: all states and energies remain unchanged and movement cost is zero. With T=0, concentration toward a Dirac law occurs only as time grows; positive finite-time standard deviation should not be replaced by a zero-width density.

Explain the difference between a free-energy gradient flow, a shortest path between fixed endpoint laws and a linear mixture of densities. All evolve distributions, but they solve different mathematical problems.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
