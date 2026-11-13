## A bowl-shaped energy can define a probability cloud

A finite state list always has a finite Boltzmann sum at finite nonzero temperature. A continuous state space can extend infinitely far. Before drawing a probability cloud, we must ask whether its total mass is finite. A finite picture with a pleasant-looking peak does not settle that question.

Imagine a ball in a two-dimensional bowl. Steeper directions restrict its fluctuations more strongly, a linear force moves the center, and temperature changes the spread. This analogy explains the geometry of an equilibrium density; it does not specify a trajectory, friction coefficient or relaxation time.

We will use a symmetric quadratic energy on the entire plane. The connection between Gaussian precision and inverse covariance agrees with [StatLect's definition of a precision matrix](https://www.statlect.com/glossary/precision-matrix). The normalization, source responses, coordinate changes and numerical boundary checks below are derived for the stated preparation. A local quadratic approximation to a general AI model will require additional assumptions.

## Specify the reference measure and units

Let x=(x1,x2) be dimensionless real coordinates, and use Lebesgue area dx=dx1 dx2 as the reference measure. Temperature is strictly positive. The energy and density are

$$
E(x)=\frac12x^{\mathsf T}Ax-h^{\mathsf T}x+c,\qquad
\tau=k_BT>0,\qquad
p(x)=\frac{e^{-E(x)/\tau}}{Z},\qquad
Z=\int_{\mathbb R^2}e^{-E(x)/\tau}\,dx.
$$

A is a real symmetric 2×2 matrix. Its entries, h and c have units of joules because x is dimensionless. The ratio E/τ is dimensionless. Specifying dimensionless coordinates also gives a definite convention for the numerical density and differential entropy. Physical coordinates with units require a reference length or volume convention before taking logarithms of density values.

The reference measure is part of the model, just as the base weights were part of a finite Boltzmann law. We will not replace the full plane by the plotting window without saying so. We also distinguish energy curvature A from Gaussian precision: temperature will supply the missing scale between them.

## Test normalizability before taking an inverse

Write A=[[a,b],[b,d]]. For this positive-temperature quadratic model on the full plane,

$$
Z<\infty\quad\Longleftrightarrow\quad A\succ0
\quad\Longleftrightarrow\quad a>0\ \text{and}\ \det A=ad-b^2>0.
$$

Positive definiteness means every nonzero displacement eventually raises the quadratic energy. The linear source can move the center but cannot defeat positive quadratic growth at infinity. The two strict inequalities are an exact mathematical classification, not a rule to round a small eigenvalue to zero.

If a direction has negative curvature, the energy tends to −∞ quadratically and the exponential weight grows without bound along that direction. If a positive-semidefinite matrix has a null direction, a zero source component leaves infinite flat volume; a nonzero source component tilts that direction so the weight grows at one end. Neither case gives a proper density relative to full-plane area.

| Preparation | Behavior on the full plane | Valid operation |
| --- | --- | --- |
| A positive definite, T>0 | Finite normalizing integral | Compute Gaussian mean and covariance |
| A positive semidefinite with a null direction | Unconfined flat or linear direction | Report non-normalizability on R² |
| A has negative curvature | Exponentially increasing weight in some direction | Report non-normalizability on R² |
| A singular Gaussian on a stated lower-dimensional support | Different reference measure and support | Define that separate model explicitly |

A pseudoinverse does not turn the second row into the first. Adding a diagonal regularizer can create a different positive-definite Hamiltonian; it is a modeling change whose effect must be reported.

## Complete the square and integrate

Assume A is positive definite. Solve for the minimum rather than guessing the center:

$$
\mu=A^{-1}h,\qquad E_{\min}=c-\frac12h^{\mathsf T}A^{-1}h,\qquad
E(x)=E_{\min}+\frac12(x-\mu)^{\mathsf T}A(x-\mu).
$$

An orthonormal change to the eigenvectors of A has absolute Jacobian one. In those coordinates the two eigenvalues λ1,λ2 are positive and the integral is the product of two one-dimensional Gaussian integrals. Consequently,

$$
Z=\frac{2\pi\tau}{\sqrt{\det A}}e^{-E_{\min}/\tau},\qquad
\ln Z=\ln(2\pi\tau)-\frac12\ln\det A-\frac{E_{\min}}{\tau}.
$$

The factor 2πτ/√det A is dimensionless under our coordinate convention: the determinant has units J². The common minimum-energy factor cancels when the density is normalized. A large energy offset can therefore make Z extremely small or large without changing the probability cloud.

The split logarithms use numerical values in the same declared energy unit; their combined shape term is the logarithm of the dimensionless ratio 2πτ/√det A.

Keeping log Z is useful, but it does not make every subsequent calculation automatically stable. Subtracting enormous energy terms to recover a small entropy can still lose information. We will compute the shape contribution and exact rational energy terms separately.

## Read mean, covariance and precision geometrically

The normalized law is Gaussian, with

$$
\mathbb E[x]=\mu,\qquad
\Sigma=\operatorname{Cov}(x)=\tau A^{-1},\qquad
\Lambda=\Sigma^{-1}=A/\tau.
$$

Precision is inverse covariance, not the unscaled energy matrix. At fixed A, doubling T doubles covariance and halves precision. A positive linear source moves the mean through A inverse, so the other coordinate can shift too when the energy has a cross term.

Contours of equal density are ellipses centered at μ. Their axes follow the eigenvectors of A, and their squared widths scale as τ/λj. A steep energy direction has a narrow probability width. A broad direction corresponds to a small positive curvature, not necessarily an unstable negative direction.

A useful visual places the energy contours and probability contours side by side. Changing h moves the center; changing temperature at fixed A expands both probability axes; changing the cross term tilts the ellipse. The contour orientation shows correlation in this equilibrium law, not a temporal arrow or a learned causal direction.

## Work through a correlated two-dimensional example

At one base preparation, choose A/τ=[[2,1],[1,2]], h=0 and c=0. Then

$$
\Sigma=\frac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix},\qquad
\mu=\begin{pmatrix}0\\0\end{pmatrix},\qquad
\operatorname{Corr}(x_1,x_2)=-\frac12,\qquad
Z=\frac{2\pi}{\sqrt3}.
$$

The same-sign direction (1,1) has energy curvature 3τ and covariance eigenvalue 1/3. The opposite-sign direction (1,−1) has curvature τ and covariance eigenvalue 1. The probability ellipse is therefore wider along the opposite-sign diagonal. Positive cross energy penalizes simultaneous positive or simultaneous negative displacements, explaining the negative correlation.

Now apply h/τ=(1,0) at that base preparation, keeping A fixed. Matrix inversion gives

$$
\mu=\begin{pmatrix}2/3\\-1/3\end{pmatrix},\qquad
E_{\min}=-\tau/3,\qquad U=\mathbb E[E]=2\tau/3.
$$

The covariance does not change. The source moves the cloud to a new minimum without changing the bowl's curvature. Comparing these two preparations separates center displacement from fluctuation width.

The ratios A/τ and h/τ above specify one preparation. They are not instructions to change A and h whenever T changes. For a temperature derivative, we will hold the Hamiltonian coefficients fixed.

## Derive energy, entropy and heat capacity

The centered quadratic energy has mean τ/2 per real quadratic coordinate. With two coordinates,

$$
U=E_{\min}+\tau,\qquad
F=-\tau\ln Z=E_{\min}-\tau\left[\ln(2\pi\tau)-\frac12\ln\det A\right],\qquad
\frac{C}{k_B}=1.
$$

The heat capacity follows by differentiating U with respect to T at fixed A,h,c. The differential entropy relative to our area measure is

$$
H_x=-\int p(x)\ln p(x)\,dx
=1+\ln(2\pi\tau)-\frac12\ln\det A
=\frac12\ln\!\left[(2\pi e)^2\det\Sigma\right].
$$

For this reference convention, the corresponding classical entropy is kB H_x and satisfies U−F=τH_x. Differential entropy can be negative. It is not the Shannon entropy of a finite list of cell labels, and its numerical value changes under a coordinate rescaling.

Holding A/τ fixed while changing T means A is changing as well. If h/τ is also held fixed, the density shape can stay unchanged, but the derivative is a path through different Hamiltonians. It is not the fixed-Hamiltonian heat capacity above. Before comparing an experiment to a response formula, state which sliders are physically fixed.

## Distinguish a density height from a cell probability

The maximum density occurs at μ:

$$
p(\mu)=\frac{\sqrt{\det A}}{2\pi\tau},\qquad
\Pr(x\in B)=\int_Bp(x)\,dx.
$$

The first quantity is a height; the second is a probability mass. In the ideal isotropic preparation A/τ=100 I, covariance is 0.01 I and the peak density is 100/(2π), approximately 15.9155. This does not violate a probability bound. The total integral is one.

For a square of side 0.01 centered at μ, peak density times area is about 0.00159155. It is a small-cell approximation to the probability, not the density value itself. In this example the exponent changes by at most 0.0025 over that square, so the exact cell probability lies between the peak-area product times exp(−0.0025) and the peak-area product. This gives a concrete error bound instead of treating a pixel value as an exact cell mass.

A visualization should identify whether color encodes density, energy or cell mass. Renormalizing colored pixels to sum to one changes their meaning. A finite plotting window can omit tail mass even when the underlying full-plane Gaussian is proper.

## Transform coordinates with the absolute Jacobian

Let x=Cz+d for an invertible real matrix C. Probability assigned to a region must stay the same, so

$$
p_z(z)=p_x(Cz+d)|\det C|,\qquad
\mu_z=C^{-1}(\mu_x-d),\qquad
\Sigma_z=C^{-1}\Sigma_xC^{-\mathsf T}.
$$

The same unnormalized energy composed with this coordinate map has a different normalizing integral relative to dz:

$$
\ln Z_z=\ln Z_x-\ln|\det C|,\qquad
H_z=H_x-\ln|\det C|,\qquad
F_z=F_x+\tau\ln|\det C|.
$$

These changes do not describe a new probability measure on physical outcomes. They describe the same measure with a different reference volume and coordinate labels. The absolute value matters when the map reverses orientation.

The calculator implements diagonal C with two nonzero scales and an independent translation. For C=diag(−2,3), the absolute Jacobian is 6. The transformed peak density is six times larger and differential entropy decreases by ln 6. The cross covariance changes sign because it is divided by −6. Ignoring the Jacobian would lose normalization even if the transformed mean and ellipse looked plausible.

This connects to finite reference weights: a normalized law does not determine reference-dependent normalizers and entropy numbers without specifying how outcomes are measured.

## Separate a source response from covariance

For the quadratic family at fixed A and positive τ,

$$
\frac{\partial\mu}{\partial h}=A^{-1}=\Sigma/\tau,\qquad
\frac{\partial\mu}{\partial(h/\tau)}=\Sigma.
$$

The first derivative is with respect to a dimensional energy source; the second is with respect to a dimensionless source. They differ by τ. In the correlated example, increasing h1 moves the second mean negatively because the off-diagonal entry of A inverse is negative.

In AI, a Gaussian approximation often comes from a local Hessian of a negative log density or an objective. If that objective is already dimensionless, its Hessian is a precision only under the stated scaling and reference convention. An energy Hessian in joules must be divided by τ. Confusing these scales changes predicted uncertainty.

A positive Hessian at a local minimum also does not prove that a general density is globally normalizable. For example, with a positive energy scale E_*, the energy E(x,y)/E_*=(x²+y²)/2−x⁴ has positive-definite Hessian at the origin but falls to −∞ for large |x|. A local Gaussian can approximate the neighborhood while the full-plane Boltzmann integral diverges. The exact quadratic test cannot be silently extended to every nonlinear energy-based model.

## Treat a boundary or a plotting window as a different preparation

For A=[[1,1],[1,1+δ]], exact positive δ gives a positive determinant, δ=0 gives a flat direction, and negative δ gives negative curvature. An arbitrary eigenvalue tolerance can merge these physically different cases. The numerical model tests the exact binary inputs rather than an epsilon-adjusted matrix.

A singular Gaussian can be defined on a line with its own lower-dimensional measure. A positive-semidefinite energy on all of R² instead leaves infinite unconfined volume. Replacing A inverse by a pseudoinverse does not resolve this difference. Likewise, a bounded box makes a different truncated model normalizable; a finite grid normalized inside that box is not evidence that the original full-plane integral is finite.

The model rejects nonpositive temperatures. For the positive-definite Hamiltonian here, negative temperature makes the weight increase along the unbounded high-energy directions. At T→0 from above, covariance tends to zero and the probability measure converges weakly to a point at μ. That limiting point mass is not a density relative to the original two-dimensional area measure.

Along positive temperatures with fixed A,h,c, differential entropy tends to −∞ and C/kB stays one. Although U tends to E_min, its positive-temperature slope remains kB. A finite gapped spectrum instead suppresses excited probabilities exponentially and can have vanishing heat capacity. The classical continuous approximation does not derive quantum low-temperature behavior or a general third-law statement.

## Read exact decisions and directed numerical observations

The implementation preserves exact binary rational products and sums of the supplied stiffnesses, sources, offset and temperature scale. It decides the determinant sign before any inverse or plotting approximation. Means, covariance, precision and affine coordinate transformations are rational quantities computed before projection to the display range.

For normalization and entropy, π is enclosed by Machin's alternating arctangent identity with outward-rounded terms and an explicit remainder bound. Logarithms also have directed bounds. Square roots use relative mantissa precision, so an extremely small positive determinant does not turn into a zero root just because a fixed absolute quantum would be too coarse.

Precision adapts from 128 to 8192 bits. Resolved nonzero observations have a certified sign and relative interval width at most 2^-48. A nonzero value outside binary64 range has a null linear value and retained sign; a precision ceiling remains unresolved. Neither state is mathematical non-normalizability. An overflowing floating-point determinant is not an infinite mathematical determinant, and an underflowed covariance is not an exact constraint.

The independent references classify the actual binary inputs with exact Fraction arithmetic and compute the Gaussian identities with high-precision Decimal arithmetic. They include positive and negative boundary cases, common offsets, very small stiffness and temperature, density above one, means outside display range and orientation-reversing coordinate maps. These numerical checks validate the stated mathematical model; renderer, sampler and physical-apparatus claims need their own evidence.

## Research extension: a stable Gaussian sampler can have the wrong covariance

A bounded-looking cloud does not prove that a sampler has the desired equilibrium spread. An update can forget its starting point and settle into a perfectly well-defined Gaussian with the wrong width. Normalizing a target density, stabilizing an iteration and preserving that target are three separate questions.

Keep the original positive-definite quadratic Gibbs law, center at its mean and divide energy by τ=kBT. Add an independent Gaussian-noise Euler sampling algorithm for these dimensionless coordinates. Its step η is an algorithm parameter; without a declared mobility or relaxation scale we do not identify it with physical seconds. The reader still plots static Gaussian preparations and has no new step-size control.

P is precision, distinct from the energy-curvature matrix A. With the original dimensionless x coordinates, P and the following algorithm step are dimensionless. The target normalizes for every positive-definite P independently of the algorithm choice.

$$
y=x-\mu,\quad P=A/\tau\succ0,\qquad\pi(y)\propto e^{-y^{\mathsf T}Py/2},\quad\Sigma=P^{-1}.
$$

This specifies the whole sampling update, including independent noise. It is the unadjusted Euler/Langevin step for the standardized quadratic target. There is no accept/reject correction in this algorithm.

$$
y_{k+1}=(I-\eta P)y_k+\sqrt{2\eta}\,\xi_k,\qquad\xi_k\overset{\rm iid}{\sim}\mathcal N(0,I),\quad\eta>0.
$$

An orthonormal rotation keeps isotropic Gaussian noise standard. Each eigenmode is therefore a scalar autoregression. Inspecting these two multipliers is enough to derive the exact stability condition in this quadratic case.

$$
P=Q\operatorname{diag}(\lambda_i)Q^{\mathsf T}:\quad z=Q^{\mathsf T}y,\qquad z_{i,k+1}=(1-\eta\lambda_i)z_{i,k}+\sqrt{2\eta}\,\zeta_{i,k}.
$$

Independence removes the noise cross term. A stationary finite variance must balance the retained old variance against the newly added 2η. Stability of the deterministic mean is not yet a check that this balance equals the target variance.

$$
v_{i,k+1}=(1-\eta\lambda_i)^2v_{i,k}+2\eta.
$$

This strict condition makes every eigenmode forget its initial state and gives a finite Gaussian stationary covariance. It concerns the chosen numerical chain; violating it does not make the original positive-definite target non-normalizable.

$$
|1-\eta\lambda_i|<1\ \text{for every }i\quad\Longleftrightarrow\quad0<\eta<2/\lambda_{\max}.
$$

Solve the scalar variance balance inside the stability interval. Every positive step inflates every mode relative to its target. Running longer removes the transient but does not remove this stationary discretization bias.

$$
v_i^{(\eta)}=\frac{2\eta}{1-(1-\eta\lambda_i)^2}=\frac1{\lambda_i(1-\eta\lambda_i/2)}>\lambda_i^{-1}.
$$

The matrix form is a discrete Lyapunov equation. Check stability before taking the displayed inverse; outside the interval its algebraic inverse need not be a covariance. A direct rational linear-system solve independently verifies the finite examples.

$$
\Sigma_\eta=(P-\tfrac\eta2P^2)^{-1},\qquad\Sigma_\eta=(I-\eta P)\Sigma_\eta(I-\eta P)^{\mathsf T}+2\eta I.
$$

Compare the stationary Gaussian with the intended target in a stated direction. Each variance ratio rᵢ is dimensionless, and the divergence is in nats. A visually settled cloud can have positive KL; a covariance from the wrong stationary law is not a thermal response of the target.

$$
D(\mathcal N(0,\Sigma_\eta)\Vert\mathcal N(0,P^{-1}))=\tfrac12\sum_i[r_i-1-\log r_i],\qquad r_i=\lambda_i v_i^{(\eta)}.
$$

At the boundary the stiff mode alternates sign but accumulates noise variance indefinitely. From zero variance, v₂,k=k. The target stiff-mode variance remains 1/4. A dash in the table means no finite stationary covariance for this update, not a missing target distribution.

$$
\lambda=(1,4),\quad\eta=1/2:\qquad z_{2,k+1}=-z_{2,k}+\zeta_{2,k},\qquad v_{2,k+1}=v_{2,k}+1.
$$

For comparison this exact linear Gaussian transition preserves the target for any η>0. Multiply the old target covariance by the decay factors and add the specified independent noise covariance to check it. It is a special quadratic formula, not an exact transition for arbitrary nonlinear energies or a physical clock supplied by a density.

$$
y_{k+1}=e^{-\eta P}y_k+\nu_k,\quad\nu_k\sim\mathcal N\!\left(0,P^{-1}[I-e^{-2\eta P}]\right):\qquad\operatorname{Cov}_{\pi}(y_{k+1})=P^{-1}.
$$

### Read the comparison

The upper panel shows how Euler stationary variances inflate for eigenvalues 1 and 4 as η approaches 0.5. The lower panel compares one-standard-deviation covariance ellipses at η=0.45 after a 45-degree eigenvector rotation. They are Gaussian geometry, not sampled paths or confidence regions for a fitted mean.

![A positive-definite Gaussian target remains valid while a stable Euler sampler inflates its covariance and reaches a noisy stability boundary.](/assets/learning/boltzmann-research-gaussian-sampler-bias.en.svg)

### Reproduce the example

Use precision eigenvalues 1 and 4. The spectral radius is max |1−ηλ|. Euler columns refer to its stationary law only when η<1/2. The exact-linear-transition column retains the target stiff variance 1/4 at every displayed step. Calculations are theoretical moments, not empirical estimates from simulated samples.

| Step eta | Spectral radius | Euler covariance trace | Euler stiff variance | Euler-target KL | Exact stiff variance |
| --- | --- | --- | --- | --- | --- |
| 0.05000000 | 0.95000000 | 1.30341880 | 0.27777778 | 0.00303691 | 0.25000000 |
| 0.20000000 | 0.80000000 | 1.52777778 | 0.41666667 | 0.08079582 | 0.25000000 |
| 0.45000000 | 0.80000000 | 3.79032258 | 2.50000000 | 3.36642262 | 0.25000000 |
| 0.50000000 | 1.00000000 | -- | -- | -- | 0.25000000 |

```python
from fractions import Fraction as F
from math import log
lam = [F(1),F(4)]
for step in [F(1,20),F(1,5),F(9,20),F(1,2)]:
    radius = max(abs(1-step*l) for l in lam)
    trace = stiff = divergence = "--"
    if step < F(1,2):
        v = [1/(l*(1-step*l/2)) for l in lam]
        ratios = [float(l*a) for l,a in zip(lam,v)]
        kl = sum(r-1-log(r) for r in ratios)/2
        trace = f"{float(sum(v)):.8f}"
        stiff = f"{float(v[1]):.8f}"
        divergence = f"{kl:.8f}"
    print(f"{float(step):.8f} {float(radius):.8f} {trace} "
          f"{stiff} {divergence} {0.25:.8f}")
```

### Three research questions with explanations

#### 1. Does a normalizable target imply a stable sampling step?

**Explanation.** No. Positive-definite precision normalizes the target. The chosen Euler chain additionally requires η<2/λmax. The two conditions answer different questions and must be tested separately.

#### 2. Will more iterations remove the inflated covariance?

**Explanation.** No. Within the stable interval, longer runs approach Ση, which differs from the target for every positive step. Reducing transients and reducing discretization bias are distinct tasks. Independent sample count also does not change that invariant law.

#### 3. Why can the exact linear transition take a larger step?

**Explanation.** Its exponential decay and noise covariance are matched to preserve the Gaussian target exactly. Euler uses a first-order decay with different added noise. The exact quadratic construction does not provide an exact general sampler for nonlinear energies.

### Source and research boundary

The Gaussian exact transition and unadjusted Langevin bias agree with [Wibisono, examples 1–2](https://proceedings.mlr.press/v75/wibisono18a/wibisono18a.pdf). The two-dimensional table, stability witness and rational Lyapunov checks are computed for the stated precision matrices. No physical relaxation scale or arbitrary-energy accuracy guarantee is inferred.

## Exercises with explained answers

1. **Which matrix normalizes?** Compare [[1,1],[1,1+δ]] for positive, zero and negative δ. **Answer:** Its first leading minor is positive and its determinant is δ. Positive δ gives a proper full-plane Gaussian, zero leaves an unconfined direction, and negative δ gives negative curvature. A pseudoinverse or an eigenvalue cutoff changes the question rather than normalizing the original density.

2. **Which way is the ellipse wide?** Use A/τ=[[2,1],[1,2]], h=0. **Answer:** The same-sign direction has curvature 3τ and covariance eigenvalue 1/3; the opposite-sign direction has curvature τ and covariance eigenvalue 1. The wider diagonal is therefore the opposite-sign direction, consistent with correlation −1/2.

3. **Is peak density 15.9 invalid?** Use A/τ=100 I. **Answer:** The peak is a density height, not a probability. A square of side 0.01 centered at the mean has a peak-area estimate about 0.00159155; its exact mass is bounded by that value and that value times exp(−0.0025). Integrating all space still gives one.

4. **What changes when temperature doubles?** Keep A,h,c fixed. **Answer:** The mean stays A inverse times h, covariance doubles, precision halves, U rises by the original τ, and differential entropy rises by ln 2. Heat capacity remains kB. Keeping A/τ fixed would change A and follow a different Hamiltonian path.

5. **What happens under x1=−2z1+1 and x2=3z2−2?** **Answer:** The absolute Jacobian is 6. Divide the centered mean coordinates by −2 and 3, divide covariance entries by the corresponding products of scales, multiply peak density by 6, and subtract ln 6 from differential entropy. Region probabilities remain unchanged with the Jacobian included.

6. **Does a zero-looking covariance prove an exact constraint?** A positive-definite model reports a null linear covariance with positive sign. **Answer:** The true covariance is positive but outside the display range; it is not a singular distribution. The model can still have representable mean energy and C/kB=1. Exact constraints require a stated support and reference measure, not deletion of an underflowed matrix entry.

## Connect the Gaussian geometry to the experiment

Start with **Baseline correlated ellipse**, then compare **Linear source shifts the center**. The plotted axes subtract the mean, so the ellipse remains centered even when μ changes. Read the selected-coordinate mean table to see the shift to (2/3,−1/3). Its unchanged covariance shows that a linear source moved the center without changing the quadratic confinement.

| Experiment | What to compare | What to explain |
| --- | --- | --- |
| Strictly positive small determinant → Flat direction → Negative curvature | Read the exact normalization decision before the picture | A thin proper law, a flat nonnormalizable law and an unstable law are different |
| Density height greater than one | Compare p(max) with the probability-integral example above | A density height is not a cell mass |
| Orientation reversal and absolute Jacobian | Switch Original coordinates x and Transformed coordinates z | Covariance and entropy numbers change with coordinates while region probabilities agree with the Jacobian |

The viewer renders centered relative probability contours, rather than a separate energy-contour panel or a plot of absolute positions. L is a plotting scale and r labels relative-density contours p/p(max)=exp(−r²/2); neither is a probability assigned to one pixel. Absolute means and density peaks remain in their own tables. If a valid ellipse is too thin to draw or outside the plotting range, read that notice and the exact normalization decision instead of interpreting an absent cloud as nonnormalizability.

To test the temperature response, keep all stiffnesses, sources and the common offset fixed while doubling positive T. The mean stays fixed, covariance doubles and C/kB remains1. Do not keep A/τ fixed at the same time: that changes the Hamiltonian. The model supports diagonal coordinate scales and translations; arbitrary rotation matrices in the derivation are not additional controls. A zero scale or nonpositive temperature is rejected rather than repaired by a pseudoinverse or a finite plotting box.

Commit each scalar field with Enter or by moving focus away from it.
