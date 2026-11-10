## Test four laws on one field

A field changing across space and a field changing in time are connected descriptions. Maxwell equations relate both changes to charge and current. We will differentiate one plane wave, then integrate that same field around rectangles and through a closed box. This connects local laws to measurements over extended regions.

Review vector components, divergence, curl, line integrals, and flux first. The goal is to explain what each law compares, beyond recalling its symbols.

## Units and model boundaries

Coordinates are in m, time in s, electric field in V/m, magnetic field in T, charge density in C/m³, and current density in A/m². The positive constants ε and μ have units F/m and H/m. This lesson uses a constant-coefficient vacuum-form teaching model; arbitrary coefficient choices are not a complete material description.

The default derives ε from the teaching value μ=4π×10⁻⁷ H/m and the SI speed of light. This does not claim that μ is exact in modern SI. Do not combine the preceding rounded Coulomb constant with an independently chosen exact permittivity. Hand calculations below explicitly use normalized coefficients ε=μ=1.

$$
c=\frac{1}{\sqrt{\epsilon\mu}},\qquad \epsilon_{\mathrm{teach}}=\frac{1}{\mu_{\mathrm{teach}}(299792458\ \mathrm{m/s})^2}.
$$

## The four local equations

Divergence is the limiting outward flux per volume. Curl is the limiting circulation per oriented area. Here J is conduction current density; ε∂tE is displayed separately. These equations assume smooth fields and constant coefficients.

$$
\begin{aligned}\nabla\cdot\mathbf E&=\rho/\epsilon,&\nabla\cdot\mathbf B&=0,\\\nabla\times\mathbf E&=-\partial_t\mathbf B,&\nabla\times\mathbf B&=\mu\mathbf J+\mu\epsilon\partial_t\mathbf E.\end{aligned}
$$

## Integral laws and types of surface

The first two laws use a closed surface ∂V with outward normals. The last two use a fixed open surface S with boundary C. The right-hand rule ties the traversal of C to the normal of S. Reversing a curve requires reversing the corresponding surface orientation. Motional electromotive force from moving conductors is outside this fixed-path model.

$$
\begin{aligned}\oint_{\partial V}\mathbf E\cdot d\mathbf S&=Q_V/\epsilon,&\oint_{\partial V}\mathbf B\cdot d\mathbf S&=0,\\\oint_C\mathbf E\cdot d\boldsymbol\ell&=-\frac{d}{dt}\int_S\mathbf B\cdot d\mathbf S,&\oint_C\mathbf B\cdot d\boldsymbol\ell&=\mu I_{\mathrm{cond}}+\mu\epsilon\frac{d}{dt}\int_S\mathbf E\cdot d\mathbf S.\end{aligned}
$$

## A plane wave you can sketch

Draw z horizontally and sketch the sinusoidal Ex and By traces. Both vary with the same phase. The direction parameter d=+1 gives propagation along +z; d=−1 gives −z. Check the relationship between the two field directions and propagation. An infinite plane wave is an ideal family for testing the laws, not the field of a finite antenna.

$$
\phi=d\kappa z-\omega t+\phi_0,\quad\mathbf E=(A\cos\phi,0,0),\quad\mathbf B=(0,dA\cos\phi/c,0),\quad d\in\{-1,1\},\quad\omega=c\kappa.
$$

## Pair spatial and temporal derivatives

Ex is independent of x, so its divergence contribution is zero. By is independent of y, so magnetic divergence is also zero. Variation along z still produces curl. Directly compute the following components and check both signs and coefficients. With A=1, κ=1, ε=μ=1, d=1, z=t=0, and φ₀=π/6, Ex=By=√3/2, the y component of curl E is −1/2, and ∂tBy is +1/2.

$$
\begin{aligned}(\nabla\times\mathbf E)_y&=-dA\kappa\sin\phi,&\partial_t B_y&=dA\kappa\sin\phi,\\(\nabla\times\mathbf B)_x&=A\kappa\sin\phi/c,&\mu\epsilon\partial_t E_x&=A\kappa\sin\phi/c.\end{aligned}
$$

## Orient the Faraday rectangle

In the xz plane traverse the bottom edge along +x, then the right edge along +z. The normal is −y. Vertical edges contribute no electric line integral, leaving the difference between bottom and top Ex. A positive By gives negative flux through this oriented surface. Distinguish the sign of flux from the sign of its time derivative. Magnetic flux is in Wb; electric circulation is in V.

$$
\oint_C\mathbf E\cdot d\boldsymbol\ell=L_x[E_x(z_0)-E_x(z_1)],\qquad\Phi_B=-L_x\int_{z_0}^{z_1}B_y(z,t)\,dz.
$$

## Orient the Ampere–Maxwell rectangle

Traversing +y then +z in the yz plane gives a +x normal. Conduction current is zero in this example, but changing electric flux supports magnetic circulation. Zero conduction current does not imply zero magnetic field. Magnetic circulation has units T·m, distinct from magnetic flux.

$$
\oint_C\mathbf B\cdot d\boldsymbol\ell=L_y[B_y(z_0)-B_y(z_1)]=\mu\epsilon\frac{d}{dt}\left(L_y\int_{z_0}^{z_1}E_x\,dz\right).
$$

## A worked pair of loops

Use normalized ε=μ=1, A=κ=1, φ₀=t=0, d=1, z₀=0, z₁=π/2, and Lx=Ly=1. Electric circulation is 1, magnetic flux is −1, and minus its time derivative is 1. Magnetic circulation is 1, electric flux is 1, and its time derivative times με is 1. These are values at this instant; flux and its time derivative are different quantities.

Floating-point cos(π/2) may not be exactly zero. Treat this representation error explicitly rather than inventing a physical zero threshold.

## Check divergence with a closed box

The −x and +x faces measure the same Ex with opposite normals, so their fluxes cancel. The −y and +y faces similarly cancel By; the remaining contributions vanish. Nonzero flux through one face can coexist with zero total closed flux. Net enclosed charge follows from the closed sum.

A box of zero thickness cannot support a flux-per-volume inference. Summing opposite analytic face values is an analytic cancellation, distinct from independently integrating each face.

## A negative control that passes divergence

Change the frequency to ω=σκc while retaining the magnetic amplitude ratio. With σ=1.2 both divergences remain zero, but evolution generally fails. At phase π/6 with normalized coefficients and A=κ=d=1, the Faraday residual is +0.1 and the Ampere residual is −0.1.

At phase zero, sinφ=0, so even this invalid family has zero local residual at that sample. One passing sample does not establish a solution over the domain. The zero field A=0 and the uniform static field κ=0 are valid degenerate cases.

## Thin surfaces and numerical availability

Subtracting nearby cosine values can hide a small circulation in rounding. Midpoint phase and sinc preserve the actual width. Defining sinc(0)=1 extends a removable coordinate singularity; it does not discard a small physical surface.

Exactly zero width or height gives zero flux and circulation, but no area-normalized mean. A nonzero quantity lost outside arithmetic range is unavailable. The teaching model explicitly excludes temporal phase |ωt|>10⁶ rad; it does not silently clamp or wrap inputs.

$$
\cos\phi(z_0)-\cos\phi(z_1)=d\kappa h\sin\phi(z_m)\operatorname{sinc}(\kappa h/2),\quad h=z_1-z_0,\quad z_m=(z_0+z_1)/2,\quad\operatorname{sinc}(u)=\sin(u)/u.
$$

## Use and verification in AI research

Neural field training can include divergence and curl residuals. Their units and scales differ, so state nondimensionalization, weights, and sampling explicitly. A pointwise residual, a domain-wide equation, boundary conditions, initial conditions, and energy or error checks provide different evidence.

Comparing analytic formulas from the same family is not independent proof. Separate Python references use five-point spatial/time differences and time differences of SciPy surface integrals. Observed discrepancies describe selected cases, not certified bounds. Radiation, induction, materials, energy flow, and wave packets are developed in later lessons.

## Exercises with explanations

Write signs and units before calculating, then compare with the explanations.

### Exercise 1: Propagation direction

For d=−1, A>0, and φ=0, which way do the fields point? **Answer:** E points +x and B points −y. Their cross product points −z, consistent with propagation.

### Exercise 2: Reverse orientation

What must change when reversing a Faraday traversal? **Answer:** Reverse the line-integral sign and the associated surface normal together. Flux and its derivative reverse, preserving the law.

### Exercise 3: One face versus a closed surface

If electric flux through the +x face is 3, is enclosed charge 3ε? **Answer:** Other faces must be included. In this plane wave the −x face contributes −3, giving zero closed flux.

### Exercise 4: A pointwise residual trap

For σ=1.2 a sample at phase zero has zero residual. Is it a solution? **Answer:** No. Other phases generally fail. With A=κ=d=1 and normalized coefficients, phase π/6 gives residuals +0.1 and −0.1.

### Exercise 5: Zero wavenumber

Must B vanish when κ=0 but A is nonzero? **Answer:** No. Since ω=0, both fields may be uniform and static. Zero derivatives and sources satisfy all four equations in this model.

### Exercise 6: A very thin loop

If two endpoint cosines round to the same value, is circulation physically zero? **Answer:** Generally no. Retain the nonzero width and use the stable midpoint identity or an independent integral of the derivative. Distinguish exact zero area from numerical loss.

## References and next connections

Compare §§13.3–13.4 of [MIT 8.02T Chapter 13](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/13e4e0069a9ffca5ae3f909083695c2b_chapte13em_waves.pdf). The next lesson connects accumulating charge, conduction and displacement currents, and initial constraints.
