## A lower concentration peak does not mean less material

Imagine releasing a smooth cloud of dissolved material into a very long tube. The cloud spreads and its peak becomes lower. A uniform background flow can also carry the cloud downstream. These are different changes: advection moves its mean, while diffusion increases its variance. Neither process destroys material in the model considered here. Looking only at the highest concentration can therefore produce a false conclusion about mass loss.

We construct a Gaussian concentration field with a finite, explicitly specified mass and physical tube area. This connects familiar Gaussian probability formulas to a conservation law without treating concentration as a probability or a cloud center as a particle trajectory. Every plotted density, flux, integral and moment needs its own definition and units.

## Prerequisites, assumptions and physical units

Review derivatives, Gaussian integrals, mean and variance, and finite-window conservation. The tube is an infinite straight line with constant area. Diffusivity is constant and nonnegative; the prescribed advection velocity is spatially uniform. There are no reactions, sources, walls, variable coefficients or feedback from concentration to flow. Initial standard deviation is strictly positive, so time zero is an ordinary smooth field rather than a singular point source.

| Quantity | Unit | Meaning |
| --- | --- | --- |
| M | kg | total material mass |
| c | kg/m³ | mass concentration |
| area | m² | physical tube cross-section |
| U | m/s | prescribed advecting velocity |
| D | m²/s | diffusivity |
| j | kg/(m² s) | signed total mass flux |

$$
M=A\int_{-\infty}^{\infty}c(x,t)\,dx,\qquad[c]=\mathrm{kg/m^3},\qquad[j]=\mathrm{kg/(m^2s)}.
$$

These variables describe a dilute scalar teaching model. They do not automatically supply the density, pressure or momentum equations of a mixture.

## Derive the equation from advection and Fick flux

Fick's constitutive law assigns diffusive flux opposite to the concentration gradient. Advection contributes velocity times concentration. Their signed sum enters continuity. Because U and D are constant, expanding the flux derivative gives the usual advection-diffusion equation. The sign convention matters: in our convention a positive U moves the mean toward increasing x.

$$
j_{\rm adv}=Uc,\qquad j_{\rm diff}=-Dc_x,\qquad c_t+j_x=0,\qquad c_t+Uc_x=Dc_{xx}.
$$

Flux is not a speed. Dividing it by concentration can produce a local ratio where concentration is positive, but that ratio should not silently be assigned as the deterministic trajectory of every molecule. Diffusion describes spreading, including microscopic fluctuations that this continuum field does not explicitly simulate.

## Construct the smooth Gaussian family

Let the initial cloud have total mass M, mean mu-zero and standard deviation sigma-zero. Its mean translates uniformly and its variance grows linearly. Use variance, rather than standard deviation, in the growth law. Keeping the normalization factor makes spreading lower the peak without changing the integral. The area converts a mass-per-length profile into a physical volume concentration.

$$
\mu(t)=\mu_0+Ut,\qquad v(t)=\sigma_0^2+2Dt,\qquad c(x,t)=\frac{M}{A\sqrt{2\pi v(t)}}\exp\!\left[-\frac{(x-\mu(t))^2}{2v(t)}\right].
$$

The Gaussian has no sharp finite edge. Its positive tails are part of the field, not a plotting defect to remove and renormalize. A limited display interval is an observation window, not the physical boundary of the infinite tube.

## Differentiate the actual field

Write displacement from the current mean as delta. Differentiating the exponential gives the spatial gradient. A second derivative includes both a positive displacement-squared term and a negative normalization-scale term. The time derivative must account for both mean motion and variance growth. Ignoring the changing normalization factor would break mass conservation even if the profile broadened visually.

$$
\delta=x-\mu,\quad c_x=-\frac{\delta}{v}c,\quad c_{xx}=\left(\frac{\delta^2}{v^2}-\frac1v\right)c,\quad c_t=\left[\frac{U\delta}{v}+D\left(\frac{\delta^2}{v^2}-\frac1v\right)\right]c.
$$

Substitution verifies the PDE on this family. A residual computed from these shared formulas is an analytic consistency diagnostic. Independent differences of actual concentrations and independent spatial quadrature provide stronger checks of an implementation's signs, units and normalization.

## Interpret conserved mass and normalized moments

Multiply concentration by area and integrate over the entire line to recover total mass. With positive M, divide the first and central second mass moments by M. The resulting mean and variance are exactly the declared Gaussian shape parameters. They are global observations, not a statement that all material is located at the mean.

$$
A\int c\,dx=M,\qquad\frac{A}{M}\int xc\,dx=\mu,\qquad\frac{A}{M}\int(x-\mu)^2c\,dx=v\quad(M>0).
$$

The mean can move while the shape spreads, and the mass can remain constant while any particular finite window gains or loses material. Keep these scopes separate before comparing a network prediction with a conservation target.

## A hand calculation with a finite mass

Choose two kilograms of material, unit area, initial standard deviation half a metre, initial mean zero, advection speed half a metre per second and diffusivity one eighth square metre per second. After one second, the variance has grown from one quarter to one half square metre, and the mean has moved to half a metre. Peak concentration becomes two divided by the square root of pi kilograms per cubic metre. Total mass remains two kilograms.

$$
M=2,\ A=1,\ \sigma_0=0.5,\ \mu_0=0,\ U=0.5,\ D=0.125,\ t=1:\quad\mu=0.5\mathrm{\ m},\quad v=0.5\mathrm{\ m^2},\quad c_{\max}=\frac2{\sqrt\pi}\mathrm{\ kg/m^3}.
$$

At the current mean, diffusive flux is zero by symmetry. Advective flux there is still positive because U and concentration are positive. Zero diffusive flux does not imply zero total flux.

## Compare height, width and direction of flux

The peak is inversely proportional to standard deviation. Doubling standard deviation halves peak height at fixed mass and area; it quadruples variance. On the left side of a symmetric cloud the diffusive flux points left, and on the right it points right. A sufficiently large positive advection can make the total flux positive even on the left side.

$$
c_{\max}(t)=\frac{M}{A\sqrt{2\pi v}},\qquad j_{\rm diff}=\frac{D(x-\mu)}v c,\qquad j=\left(U+\frac{D(x-\mu)}v\right)c.
$$

Do not use a single color or arrow to hide the decomposition. A net flux can become small through cancellation while each contribution remains nonzero. Its sign concerns transport across an oriented surface, not the sign of concentration, which stays nonnegative.

## Measure the mass in a finite physical window

For fixed left and right positions, integrate concentration across the actual interval and multiply by area. The Gaussian formula can be written with a standard-normal CDF difference. This is a mathematical identity, not a requirement to implement the difference directly. In a far tail both CDF values may round to one, making their floating-point difference zero even when the interval contains nonzero material.

$$
M_W=A\int_a^b c\,dx=M\left[\Phi\!\left(\frac{b-\mu}{\sqrt v}\right)-\Phi\!\left(\frac{a-\mu}{\sqrt v}\right)\right].
$$

We evaluate the physical-width integral directly with log-scaled quadrature. The physical endpoints and their difference are preserved, even when standardized endpoints round together. Comparing two quadrature orders is a diagnostic, not a rigorous error certificate.

## Verify the window budget independently

The outward normal is negative on the left and positive on the right. Multiply the signed flux difference by tube area to obtain kilograms per second. Integrating the actual time derivative of concentration gives the storage rate. This can be checked separately from the two face fluxes and from finite differences of stored mass.

$$
\frac{dM_W}{dt}+A[j(b,t)-j(a,t)]=0.
$$

If the cloud is spreading out of a window around its mean, mass in that window can fall while total mass remains conserved. If the cloud is being advected into the window, it can rise. Omitting the boundary flux would create a false conservation failure. Assuming that every finite window contains all of M would create a different error.

## Separate exact limiting controls

At zero D, variance remains fixed and the Gaussian translates without broadening. At zero U, the mean remains fixed while variance grows. At positive D and time zero, the initial field is unchanged but its time derivative need not vanish: evolution begins immediately. These controls probe different terms of the equation.

$$
D=0:\ c(x,t)=c(x-Ut,0),\qquad U=0:\ \mu=\mu_0,\qquad\frac{dv}{dt}=2D.
$$

Neither control introduces a wall or a reaction. The initial positive standard deviation is retained; replacing it with zero would create a singular initial condition outside this implementation's domain.

## Zero mass has no mass-normalized mean

If M is exactly zero, concentration and all physical fluxes are zero. There is no mass by which to normalize moments. The parameterized shape can still have a declared center and variance, but these are not the mean and variance of observed material. Reporting zero as the “measured mean” would manufacture an observation rather than mark its absence.

$$
M=0:\quad c=j=0,\qquad\frac{A\int xc\,dx}{M}\ \text{is undefined}.
$$

This distinction is useful for data pipelines too: an empty observation and a nonempty observation centered at zero are different states. Preserve the physical rest control and the unavailable normalized channel separately.

## Retain far tails and logarithmic observations

For positive M, evaluate concentration by combining normalization and exponential decay before converting back to an ordinary number. A bare Gaussian exponential can underflow although the scaled concentration is representable. Conversely concentration itself may be outside arithmetic range while its derivative, multiplied by a large spatial factor, remains representable. Losing one channel does not justify erasing all others.

$$
\ell=\log\!\left(\frac c{c_*}\right)=\log\!\left(\frac{M}{A\sqrt{2\pi v}\,c_*}\right)-\frac{(x-\mu)^2}{2v},\qquad c_*=1\mathrm{\ kg/m^3}.
$$

The reference c-star makes the logarithm dimensionless. A finite log concentration with an unavailable ordinary value means a nonzero positive tail, not a physical zero. Likewise a finite log interval fraction can remain useful when the fraction itself underflows. Keep log mass readings referenced to one kilogram if they are displayed. Zero mass has no finite logarithm and must be handled as a separate exact case.

## Preserve small spreading below rounded equality

For tiny positive D or time, adding the variance increment to the initial variance can round back to the initial number. The increment can still be representable. Avoid subtracting nearly equal square roots to obtain the standard-deviation change. Rational identities also preserve a peak change that a rounded peak ratio might erase.

$$
\Delta v=2Dt,\qquad\Delta\sigma=\frac{\Delta v}{\sqrt{v_0+\Delta v}+\sqrt{v_0}},\qquad v_0=\sigma_0^2.
$$

With eta equal to relative variance growth, the peak ratio follows from the same normalization factor. The stable expression below retains its negative change even when the rounded square root equals one.

$$
\eta=\frac{\Delta v}{v_0},\qquad\frac{c_{\max}(t)-c_{\max}(0)}{c_{\max}(0)}=-\frac{\eta}{\sqrt{1+\eta}\,[\sqrt{1+\eta}+1]}.
$$

If a nonzero increment cannot be represented even after stable scaling, mark that channel unavailable. Do not replace it with the exactly zero-diffusivity control. Rounded mean motion and a stable mean displacement also need distinct labels when a small translation is lost in addition.

## Experiment sequence and AI research use

Start with zero advection and inspect left/right diffusive fluxes. Increase time, compare peak concentration with variance, and verify total mass using a sufficiently broad physical integral or an independent full-domain calculation. Then activate advection and separate mean translation from spreading. Track a fixed window as the cloud enters and leaves it. Compare zero diffusivity, zero mass, far-tail windows and tiny positive variance changes.

For a learned transport model, report errors in concentration, flux, total mass, mean and variance separately. A peak-fitting loss can reward the wrong width; a finite display window can hide missing tail mass; and normalizing every predicted frame can conceal an incorrect PDE flux. Hold out advection speeds and diffusivities, preserve actual area and units, and use boundary budgets before interpreting local mass changes. This Gaussian field is a controlled benchmark for constant coefficients, not evidence of correct reactions, walls, mixture dynamics or individual molecular paths.

## Research extension: diffusion can hide a persistent spatial error

Suppose two transport predictions become hard to distinguish as both clouds broaden. Has their center error disappeared? Compare **two full-line Gaussian fields**, not two cropped pictures. Use the same tube area and positive masses, and normalize each concentration by its own conserved mass. The resulting p and q are probability densities along the tube, not physical volume concentrations. In this section q denotes the second probability density; it is unrelated to the heat flux symbol in the preceding chapter.

$$
p=\frac{Ac_p}{M_p},\qquad q=\frac{Ac_q}{M_q},\qquad\int_{\mathbb R}p\,dx=\int_{\mathbb R}q\,dx=1,\qquad[p]=[q]=\mathrm{m^{-1}},\quad M_p,M_q>0.
$$

Give the two clouds the same constant U and D, but allow different initial means and positive variances. Both means translate by the same amount, so their separation delta remains constant. Both variances receive the same increment. This shared evolution is essential to the comparison; a reference frozen at its initial shape is a different problem.

$$
p=\mathcal N(\mu_p,v_p),\quad q=\mathcal N(\mu_q,v_q),\quad\mu_i=\mu_{i0}+Ut,\quad v_i=v_{i0}+2Dt>0,\qquad\delta=\mu_p-\mu_q=\mu_{p0}-\mu_{q0}.
$$

Define the relative entropy K, also called KL divergence. The ratio p/q is dimensionless even though each density has units per metre. Insert their Gaussian logarithms, then use the first two moments under p to obtain the closed expression. The variance contribution is nonnegative because r - 1 - log r is nonnegative, and the center contribution is visibly nonnegative. K is dimensionless and is not automatically the thermodynamic entropy of the dissolved material.

$$
K(p\Vert q)=\int_{\mathbb R}p\log(p/q)\,dx=\frac12\left[\frac{v_p}{v_q}-1-\log\frac{v_p}{v_q}+\frac{\delta^2}{v_q}\right]\ge0.
$$

The spatial score difference measures how the two log profiles slope differently. Write x = mu-p + y: its value is -y/vp + (y + delta)/vq. Under p, y has mean zero and second moment vp. Squaring and averaging gives the relative Fisher information below. Its inverse-square-length units matter when it is multiplied by diffusivity.

$$
I_{\rm rel}=\int_{\mathbb R}p\left[\partial_x\log(p/q)\right]^2dx=\frac{(v_p-v_q)^2}{v_pv_q^2}+\frac{\delta^2}{v_q^2},\qquad[I_{\rm rel}]=\mathrm{m^{-2}}.
$$

Differentiate the integral defining K, using the PDE for **both** densities. Each density retains total integral one. Integrate by parts on the full line; the Gaussian tails make the boundary terms vanish. Contributions from their common uniform drift cancel, and diffusion leaves minus D times the score square. This identity uses a moving reference q with the same diffusion. It is not the derivative formula for an arbitrary fixed target, different drift or different diffusivity.

$$
p_t+Up_x=Dp_{xx},\quad q_t+Uq_x=Dq_{xx}\quad\Longrightarrow\quad\frac{d}{dt}K(p\Vert q)=-D\int_{\mathbb R}p\left[\partial_x\log(p/q)\right]^2dx.
$$

Substitute the Gaussian relative Fisher expression to see why K decreases. At zero diffusivity it stays constant while both clouds translate. With positive diffusivity it falls unless the normalized profiles coincide. A decreasing measure is useful evidence for this particular notion of distinguishability, but its definition alone does not bound every other error in the same units or with a time-independent constant.

$$
\dot K=-D\left[\frac{(v_p-v_q)^2}{v_pv_q^2}+\frac{\delta^2}{v_q^2}\right]\le0,\qquad[\dot K]=\mathrm{s^{-1}}.
$$

Compare a distance with spatial units: the least expected squared displacement over all joint couplings with these two marginals. Expanding that cost gives delta² + vp + vq minus twice their covariance. Cauchy–Schwarz bounds covariance by the product of standard deviations. The coupling X = mu-p + sqrt(vp) Z and Y = mu-q + sqrt(vq) Z, using the same standard normal Z, attains the lower bound. This proves the displayed optimal quadratic transport cost for this one-dimensional Gaussian pair.

$$
W_2^2(p,q):=\inf_{(X,Y)}\mathbb E[(X-Y)^2]=\delta^2+(\sqrt{v_p}-\sqrt{v_q})^2,\qquad[W_2^2]=\mathrm{m^2}.
$$

Now give the clouds equal initial variance and a nonzero center separation. Diffusion increases their shared width, so the same displacement occupies a smaller fraction of that width and K tends to zero. Nevertheless their centers stay delta metres apart, and the optimal squared transport cost remains delta² square metres. There is no contradiction: the distributions spread on an unbounded line and the comparison metrics measure different errors. A time-independent entropy-to-distance bound cannot be assumed without checking its reference-dependent constants.

$$
v_p=v_q=v_0+2Dt:\qquad K=\frac{\delta^2}{2(v_0+2Dt)}\longrightarrow0\ (D>0),\qquad W_2^2=\delta^2,\qquad\mu_p-\mu_q=\delta.
$$

A different negative control reveals what normalization removes. Keep exactly the same shape but use two kilograms in one cloud and one kilogram in the other. Their normalized densities agree and K is zero, while physical concentrations and masses differ. Thus a normalized shape loss must be accompanied by a separate physical mass budget. If either mass is exactly zero, that cloud has no mass-normalized density here; do not assign zero KL to an undefined comparison.

$$
M_p=2\mathrm{\ kg},\quad M_q=1\mathrm{\ kg},\quad p=q:\qquad K=0,\qquad c_p=2c_q,\qquad M_p-M_q=1\mathrm{\ kg}.
$$

For nearly equal variances, subtracting r from its logarithmic approximation can lose the small variance contribution. Form epsilon from the variance difference, use log1p, and, when needed, evaluate the convergent series with a stated truncation error. log1p alone does not remove the final subtraction. For a truncation after n = N and |epsilon| < 1, the absolute remainder is at most |epsilon| to power N+1 divided by (N+1)(1-|epsilon|). The table below uses exactly equal variances, so this contribution is exactly zero and only the center term remains.

$$
\epsilon=\frac{v_p-v_q}{v_q},\quad\frac{v_p}{v_q}-1-\log\frac{v_p}{v_q}=\epsilon-\operatorname{log1p}(\epsilon)=\sum_{n=2}^{\infty}\frac{(-1)^n\epsilon^n}{n}\quad(|\epsilon|<1).
$$

For the table, both initial variances are 1 m², D = 0.5 m²/s, and delta = 1 m; any common U gives the same entries. Columns are time in s, common variance in m², dimensionless KL, relative Fisher information in 1/m², and squared transport cost in m². Every positive-mass cloud retains its own mass throughout. The table does not evolve a second cloud inside the existing single-cloud interactive controls.

| t | variance | KL | relative Fisher | W2 squared |
| --- | --- | --- | --- | --- |
| 0.00000000 | 1.00000000 | 0.50000000 | 1.00000000 | 1.00000000 |
| 1.00000000 | 2.00000000 | 0.25000000 | 0.25000000 | 1.00000000 |
| 4.00000000 | 5.00000000 | 0.10000000 | 0.04000000 | 1.00000000 |
| 9.00000000 | 10.00000000 | 0.05000000 | 0.01000000 | 1.00000000 |

```python
# SI teaching scales: v0=1 m^2, D=0.5 m^2/s, delta=1 m.
for t in (0.0, 1.0, 4.0, 9.0):
    variance = 1+t
    relative_entropy = 1/(2*variance)
    fisher = 1/(variance*variance)
    transport_cost = 1.0  # m^2, from the constant center separation
    print(" ".join(f"{x:.8f}" for x in (t, variance, relative_entropy, fisher, transport_cost)))
```

The [Hirata–Nemoto–Yoshida paper](https://www.mdpi.com/1099-4300/14/8/1469) proves a moving-reference relative de Bruijn identity with a half-Laplacian convention. Here the common drift and physical D are stated explicitly, and the Gaussian and transport expressions are derived separately. The first two figure panels use initial centers 0 and 1 m; the last is a separate equal-center mass control. The plotted finite interval is not renormalized. Normalized information divergence, physical entropy, conserved mass and spatial error remain distinct observations.

![Gaussian broadening reduces relative entropy while center separation and mass errors can remain](/assets/learning/continuum-gaussian-relative-information-distance.en.svg)

### Research questions with explanations

**Does KL tending to zero imply that cloud centers converge?** **Answer:** No. Equal-variance clouds can retain a fixed mean separation as diffusion broadens them. Their squared transport cost remains the squared separation.

**May the moving-reference dissipation identity be used for a fixed target?** **Answer:** Not without a new derivation. Both p and q obey the same evolution in this identity; freezing q removes terms used in its proof.

**Does zero normalized KL certify the correct physical mass?** **Answer:** No. Separate positive masses can normalize to the same shape. Check actual mass and flux budgets, and keep zero-mass normalized comparisons undefined.

## Exercises with explanations

### Exercise 1: variance or standard deviation

For diffusivity one eighth square metre per second, what grows by one quarter square metre each second?

**Answer:** Variance, because its growth rate is twice diffusivity. Standard deviation is the square root of variance and generally does not grow linearly in time.

### Exercise 2: a lower peak

Standard deviation doubles at fixed mass and tube area. What happens to peak concentration and total mass?

**Answer:** Peak concentration halves while total mass stays fixed. The cloud is wider. A lower peak alone is not evidence of reaction or mass loss.

### Exercise 3: symmetry and total flux

At the current mean, diffusive flux is zero. Must total flux also be zero?

**Answer:** No. Total flux includes U times concentration. It is nonzero there when advection and concentration are nonzero, even though the diffusive contribution vanishes by symmetry.

### Exercise 4: the empty cloud

With zero total mass, may the declared Gaussian center be reported as a measured mass-normalized mean?

**Answer:** No. The physical field is zero and normalization would divide by zero. Keep a declared shape parameter separate from an unavailable mean of material.

### Exercise 5: subtracting tail CDF values

Two standard-normal CDF values both round to one. Does their zero difference prove that no material lies between the physical endpoints?

**Answer:** No. Evaluate the interval directly or retain a stable log integral. Keep the original physical width; do not erase the tail or renormalize the remaining visible profile.

### Exercise 6: local mass change

Mass in a fixed window decreases while a full-domain reference retains the same total mass. What must you check before declaring a conservation failure?

**Answer:** Check the oriented total boundary flux, including advection and diffusion and multiplying by actual area. Outflow can explain the storage change. Also distinguish quadrature error and unavailable tiny channels from physical absence.

## Sources and scope

[Gilbert Strang's MIT OCW heat and convection-diffusion notes](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf) discuss Gaussian spreading and the fundamental solution. [MIT's diffusion illustration](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec09/heat/vorticity.html) motivates varying a small diffusivity. Here mass normalization, physical tube area, flux signs and arithmetic controls are derived explicitly. The illustration's initial-width parameter uses a different convention from our standard deviation; its old Java applet is not used as verification.
