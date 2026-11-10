## A still histogram can hide a circulating current

The experiments below compare eighteen independently checked preparations. Exact rational quantities, directed enclosures and display approximations are distinguished.

Imagine taking a photograph of a crowded circular walkway every second. The number of people in each region might stay constant while everyone keeps walking around the center. A stationary density has the same limitation: it describes occupancy at a time, not all movement between times. To study that movement we need a probability current and a rule governing trajectories.

The previous OU chapter gave a stationary one-dimensional Gaussian whose probability-flow velocity was zero. We now add a direction in which probability can circulate without changing its density. We will distinguish drift, diffusive current, total current and deterministic probability flow; then compare time correlations and numerical bias.

## Specify a two-dimensional model and its initialization

Work on the whole plane with independent standard Brownian coordinates. Let gamma and D be positive constants, omega a signed angular rate, and R the counterclockwise quarter-turn matrix:

$$
dX_t=AX_t\,dt+\sqrt{2D}\,dW_t,\qquad
A=-\gamma I+\omega R,\qquad
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
$$

Both coordinates represent positions. Positive omega points upward at the point(1,0), and leftward at(0,1). We choose the stationary initialization independently of future Brownian increments:

$$
X_0\sim\mathcal N(0,vI),\qquad v=\frac D\gamma,
\qquad p_\infty(x)=\frac1{2\pi v}
\exp\left(-\frac{|x|^2}{2v}\right).
$$

Gamma and omega have units s⁻¹, with radians dimensionless. D has units m²/s, each coordinate variance v has units m², and density on the plane has units m⁻². The density is positive everywhere and integrates to one on the entire plane. A finite diagram is a crop, not a reflecting container or a renormalized domain.

The assumptions gamma>0 and D>0 are essential to this preparation. Removing confinement, removing noise or using an anisotropic initialization requires a separate treatment. We do not infer a thermal bath, temperature, heat or entropy production from these coefficients.

## Derive stationarity rather than assuming that motion stops

For a linear SDE, the covariance equation is A Sigma+Sigma A transpose+2DI. Substituting Sigma=vI makes the rotation cancel because R transpose=−R:

$$
A(vI)+(vI)A^T+2DI=(-2\gamma v+2D)I=0.
$$

The mean remains zero. Since the linear solution is Gaussian, these statements preserve the entire stated law. This is stronger than observing a zero derivative of one moment in an unspecified non-Gaussian family.

The drift contains two effects. The radial part pulls positions inward; the rotational part moves them along circles. Diffusion continually spreads probability. A stationary law balances those effects at the ensemble level while the stochastic process keeps fluctuating.

The [nonreversible Langevin study by Duncan, Lelièvre and Pavliotis](https://link.springer.com/article/10.1007/s10955-016-1491-2), especially Sections1.3 and3.4, motivates invariant-density-preserving perturbations. We use our own coefficient scales, rotation convention and direct Gaussian calculations here. Their displayed vector(x2,−x1) has the opposite orientation to our R x.

## Separate drift current from diffusion current

Probability conservation and the constant-diffusion current are

$$
\partial_t p=-\nabla\!\cdot J,\qquad
J=(Ax)p-D\nabla p,\qquad
\nabla\log p_\infty=-\frac{x}{v}.
$$

At the stationary law the inward drift current and outward diffusive current cancel their radial parts. The total current is

$$
J_{\rm drift}=(-\gamma x+\omega Rx)p_\infty,
\qquad J_{\rm diff}=\gamma xp_\infty,
\qquad J=\omega Rxp_\infty.
$$

At gamma=D=omega=1 and x=(1,0)m, density is exp(−1/2)/(2pi)m⁻², approximately0.0965324m⁻². Drift current is density times(−1,1)m/s; diffusion current is density times(1,0)m/s; their sum is density times(0,1)m/s. The cancellation is exact, even if separately rounded entries leave a small numerical residue.

Current components have units m⁻¹s⁻¹. Dividing a current by density gives a probability-transport velocity with units m/s. Neither current nor that velocity is the ordinary instantaneous velocity of a Brownian sample path.

## Check divergence and distinguish normal from tangential flux

The rotational vector has zero divergence and is perpendicular to the radial density gradient. Therefore

$$
\nabla\!\cdot J=0,\qquad x\cdot J=0,
\qquad x_1J_2-x_2J_1=\omega |x|^2p_\infty(x).
$$

Zero divergence means no net local accumulation; it does not say each component of J is zero. At the origin the current vanishes, but one zero probe does not establish a zero current everywhere. Only omega=0 gives that conclusion for this family.

On a circle, the outward normal is radial. Its integrated normal probability flux is zero. For a counterclockwise contour, the tangential line integral is a different quantity:

$$
\oint_{|x|=r}J\cdot d\ell=2\pi\omega r^2p_\infty(r),
\qquad
\oint_{|x|=r}\frac{J}{p_\infty}\cdot d\ell=2\pi\omega r^2.
$$

Zero net normal current does not forbid individual stochastic paths from crossing the circle. Inward and outward transport can balance. This circle is not a reflecting wall; the deterministic probability-flow ODE preserves radius, while the SDE need not.

The first has units s⁻¹; the second has units m²/s. Neither is probability leaking out of the circle. A line crossing measurement uses current dotted with a normal and integrated over line length, whereas circulation follows a tangent. Naming the direction of the measurement prevents an incorrect mass-balance interpretation.

## Explain what stationary density cannot identify

At fixed gamma and D, changing omega changes the generator and the current but preserves the same stationary density. Occupancy photographs alone cannot identify omega. At a single stationary density, v identifies only the ratio D/gamma, not an absolute relaxation time.

This is a comparison between different specified SDEs. Once a particular drift, diffusion coefficient and density are given, the current formula supplies their current. We are not claiming that an arbitrary extra current can be assigned to one unchanged SDE.

Likewise, observing a current at a point is not enough to identify a full field without assumptions. In this linear family, calibrated locations and density/current measurements impose particular constraints; in a general model, many functions remain possible. Distinguish the assumed model class from the data that support it.

## Compare the same marginals with different two-time laws

The exact linear flow matrix is an exponentially damped rotation. Fix covariance order as C(t)=E[X_t X_0 transpose], rather than its transpose:

$$
e^{tA}=e^{-\gamma t}\operatorname{Rot}(\omega t),\qquad
C_{\rm SDE}(t)=v e^{-\gamma t}\operatorname{Rot}(\omega t).
$$

The probability-flow ODE at stationarity uses u=J/p=omega R x. Starting that deterministic ODE from the same random Gaussian gives

$$
\dot Y_t=\omega RY_t,\qquad
C_{\rm ODE}(t)=v\operatorname{Rot}(\omega t).
$$

Both one-time densities remain the same Gaussian. The SDE loses memory through damping and fresh noise; the ODE preserves each realization's radius. At omega=0, this reduces to the frozen ODE versus fluctuating SDE comparison from the previous chapter.

Off-diagonal covariance entries can be negative. They are signed covariances, not probabilities to clip into[0,1]. The sign also depends on the stated order of the two times and coordinates. Reading only a covariance magnitude hides rotational direction.

## Sampling times introduce an additional identification limit

Suppose measurements occur only at multiples of delta>0. Replacing omega by omega+2pi k/delta for integer k adds whole rotations between observations. The sampled Gaussian transition means and innovation covariances are then the same. Their continuously observed paths are different.

This is an exact analytical alias within the stated linear family. The numerical controls accept rational radians; a rational approximation to pi is not exactly pi and must not be labeled an exact alias preset. For example, at interval2s the exact alias increment is pi rad/s. The accompanying symbolic check uses the analytical whole-turn identity separately. Later controls for turns or explicit alias families would need their own exact representation.

## Ask which time-average observable improves

For a stationary zero-mean observable f, its long-time average has asymptotic variance given by twice the integrated time covariance when the covariance is integrable. For the linear coordinate and centered radial observable, Gaussian covariance calculations yield

$$
\sigma^2_{x_1}=\frac{2v\gamma}{\gamma^2+\omega^2},\qquad
\sigma^2_{|x|^2-2v}=\frac{4v^2}{\gamma}.
$$

The first decreases as the rotation magnitude increases. The second does not. In this isotropic example the drift eigenvalues have real part−gamma for every omega, so a variance improvement is not an increased radial confinement rate. A benefit for one observable is not a universal mixing improvement.

The probability-flow ODE makes the distinction sharper. Its radius is constant, so averaging a centered radial observable over time does not remove randomness in the initial radius. Sharing one-time density with an ergodic SDE does not give the ODE the same time-average behavior.

## Include the numerical price of rotation

Euler–Maruyama has matrix multiplier B=I+hA. Starting with an isotropic Gaussian keeps its covariance isotropic, and the exact discrete variance recurrence is

$$
q=(1-\gamma h)^2+(\omega h)^2,\qquad
v^E_{n+1}=qv^E_n+2Dh.
$$

These are exact moments of the discrete model, not Monte Carlo estimates. Every step remains a valid Gaussian law even if q exceeds one. Zero mean can hide an expanding mean mode and increasing variance.

For positive h, strict contraction requires h<2gamma/(gamma²+omega²). When that holds, the unique discrete invariant coordinate variance is

$$
v_E^*=\frac{2D}{2\gamma-h(\gamma^2+\omega^2)}.
$$

It exceeds D/gamma. Increasing rotation at fixed h can therefore increase stationary bias or cross into instability while the continuous-time linear observable's asymptotic variance decreases. At gamma=omega=1 and h=1, B is a quarter-turn, q=1 and each step adds2D to variance. At h=0 the method is the identity, so a divided invariant-variance expression is unavailable.

Compare physical time T, numerical step h=T/N, angular rate and computation cost separately. A better continuous generator does not automatically give a better finite-step implementation.

## Preserve signs and very small positive density

The prepared model uses exact rational inputs and Euler moments with directed256bit pi, exponential, logarithm and trigonometric intervals. It computes the simplified total current directly, preserving exact cancellation of radial parts. Scalar density is mathematically positive for all accepted parameters, even when a linear enclosure is too wide to resolve its tiny magnitude.

For a concentrated Gaussian probed far into its tail, the density interval can include zero while a separate logarithm accurately locates the nonzero density. This is an unresolved linear observation, not proof of zero density. Current component logarithms are unavailable when their exact coefficient is zero; otherwise they combine log density with the logarithm of that coefficient's absolute value. Structural zero and positive density below binary64 range must remain distinct.

Numerical logs use dimensionless ratios to1m⁻² for density and1/(m·s) for current-component magnitude. Changing length units shifts these readouts without changing probability.

## Experiments

| Preparation | Question | Readout to compare |
| --- | --- | --- |
| Zero, positive and negative rotation | Does the same density identify direction? | Separate currents and signed off-diagonal covariances |
| Origin versus off-origin probe | Does a zero probe imply no current anywhere? | Local zero and whole-field classification |
| Stable, marginal and expanding Euler | Is a valid Gaussian necessarily accurate? | Amplification, variance error and invariant variance |
| Rescaled clock | Does the same width set a time scale? | Density, physical currents and paired-time covariance |
| Tiny positive field and concentrated tail | Is a small display an exact zero? | Rational coefficients, enclosures and separate logarithms |

Interaction bounds are gamma,D in(0,2], omega in[−4,4]rad/s, probe coordinates in[−4,4]m, T in[0,4]s and N from1 to64, with bounded integer numerators and positive denominators. They limit numerical work, not course length or research scope. The reader uses native workers, exact linear-field samples and view-only record selection. Changing preparation terminates the previous calculation.

## Worked exercises

### 1. Find the stationary current at a probe

Use gamma=D=omega=1 and x=(1,0). Find drift, diffusive and total current velocities, and explain why stationarity does not make the last one zero.

**Explanation.** Dividing by density gives drift(−1,1), diffusive(1,0) and total(0,1)m/s. The total is tangent to a circle and divergence-free with this radial density. Probability moves without local accumulation, so the unchanged histogram is consistent with a nonzero current.

### 2. Interpret a zero at the origin

At x=(0,0), the current is zero for every omega in this family. Does this measurement establish zero stationary current everywhere?

**Explanation.** No. R x vanishes at the origin independently of omega. At a nonzero point the current is nonzero whenever omega is nonzero. The local probe, a field identity and identification of a parameter are three different statements.

### 3. Compare two definitions of flux

For gamma=D=omega=1 and radius1m, calculate normal flux through the circle, tangential current integral and velocity circulation.

**Explanation.** Normal flux is zero because J is tangent. The tangential current integral is exp(−1/2)s⁻¹, while velocity circulation is2pi m²/s. Their different units already show that they cannot all mean probability escaping per second.

### 4. Expose a valid but marginal Euler model

With gamma=D=omega=1, stationary initial coordinate variance1m² and one step h=1s, find the Euler variance and compare it with the exact stationary value.

**Explanation.** q=1, so Euler variance becomes3m² while the exact value remains1m². The law is still a normalized Gaussian. Its deterministic mean multiplier is a quarter-turn rather than a contraction; added noise makes the variance grow. Clipping probabilities is neither needed nor a correction for this error.

### 5. Separate an observable benefit from a universal claim

Set gamma=D=1. Compare asymptotic variances at omega=0 and omega=1 for f=x1 and f=|x|²−2.

**Explanation.** For x1 the values are2 and1 in units m²s. For the radial observable both values are4 in units m⁴s. Rotation helps the first observable while leaving the second unchanged. It also tightens the Euler stability step from h<2 to h<1 second.

### 6. State what discrete sampling misses

In the analytical family, compare omega=0 and omega=pi rad/s when observing only at multiples of two seconds. Are the continuous currents the same? Can rational control input355/113rad/s be declared exactly equivalent?

**Explanation.** At those observation times the exact rotation matrices are both the identity, and the sampled Gaussian kernels agree. The continuous currents differ. The rational355/113 is only an approximation to pi, so its numerical kernel must not be declared exactly aliased. The identity belongs to the analytical parameterization, with an explicitly stated sampling interval.
