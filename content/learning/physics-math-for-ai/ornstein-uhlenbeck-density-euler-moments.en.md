## From a probability vector to a probability law on the line

The experiments below use independently checked moment models. Exact rational values, directed enclosures and display approximations are distinguished.

In the preceding master equation chapters, probability occupied a finite set of boxes. Now imagine repeating a noisy relaxation experiment many times and recording a real position in each run. A density describes how that ensemble spreads over the line. The height of a density curve is not the probability of one exact position; area over an interval gives a probability.

The earlier stochastic-process chapters introduced OU paths, their coupling to a numerical method, and local Gaussian flux. Here we ask a different set of questions: which moment statements are exact, when does a numerical method preserve a probability law while misrepresenting its dynamics, and what changes when the initial law is a point mass? Those questions are useful before extending to spatially discretized Fokker–Planck equations.

## Specify the process, initialization and units

Use the one-dimensional Itô equation on the whole real line:

$$
dX_t=-\gamma X_t\,dt+\sqrt{2D}\,dW_t,
\qquad \gamma\geq0,\quad D\geq0,
\qquad X_0\sim\mathcal N(m_0,v_0),\quad v_0\geq0.
$$

The initial variable is independent of future Brownian increments. At v0=0, the notation means a point mass at m0. Gamma and D are constant, independently chosen coefficients. A harmonic thermal model needs additional assumptions relating mobility, stiffness, temperature and diffusion; they are not inferred here.

| Quantity | Interpretation | Unit |
| --- | --- | --- |
| X, m0, m(t) | Position or ensemble mean | m |
| v0, v(t) | Ensemble variance | m² |
| gamma | Linear relaxation rate | s⁻¹ |
| D | Diffusion coefficient | m²/s |
| W_t | Standard Brownian coordinate | s¹ᐟ² |
| p(x,t) | Density with respect to position | m⁻¹ |
| J(x,t) | Probability crossing a position per time | s⁻¹ |

For a short step h, the noise has variance 2Dh, rather than amplitude 2Dh. Independent increments scale in standard deviation as the square root of time. Gamma times h is dimensionless; D divided by gamma has units of variance. These checks catch both a missing square root and a factor-of-two error.

## Derive the density equation and its boundary assumption

For a smooth test function phi, Itô's formula gives expected rate E[−gamma X phi'(X)+D phi''(X)]. Transferring the spatial derivatives onto a sufficiently regular density gives

$$
\partial_t p=\gamma\partial_x(xp)+D\partial_{xx}p
             =-\partial_xJ,\qquad
J=-\gamma xp-D\partial_xp.
$$

The drift term transports mass toward zero. Diffusion spreads mass down a density gradient. They are currents of probability; neither alone is a velocity trajectory or a measurement of heat.

Integrating between a and b gives the rate of mass change there as J(a,t)−J(b,t). Whole-line normalization follows when the boundary currents vanish at infinity. In the positive-variance Gaussian cases below, polynomial factors times Gaussian tails have that property. A finite plot window does not impose reflecting walls and does not contain all Gaussian probability. Never renormalize a crop to make the visible area one.

The [MIT OU notes, Appendix A](https://ocw.mit.edu/courses/18-366-random-walks-and-diffusion-fall-2006/0a8bdd9c04730a715ad5d54640a3702a_lec14.pdf) provide a reference for linear confinement. We derive this chapter's independently initialized Gaussian moments directly.

## Solve one linear SDE and obtain an ensemble

Multiplication by exp(gamma t) turns the drift into a product derivative. Integrating yields

$$
X_t=e^{-\gamma t}X_0+
\sqrt{2D}\int_0^t e^{-\gamma(t-s)}\,dW_s.
$$

The first term carries the initial random variable forward. The stochastic integral is an independent centered Gaussian. Its variance is the integral of the squared deterministic kernel, multiplied by 2D. Consequently,

$$
m(t)=m_0e^{-\gamma t},\qquad
v(t)=v_0e^{-2\gamma t}+
\frac D\gamma(1-e^{-2\gamma t})\quad(\gamma>0).
$$

At gamma=0, do not divide by gamma: m(t)=m0 and v(t)=v0+2Dt. At zero time the original law is recovered. At D=0, randomness already present in X0 still contracts; setting the forcing noise to zero does not necessarily make every realization identical.

For gamma=1/s, D=1/2m²/s, m0=2m and v0=1/4m², the law at one second has mean approximately 0.735759m and variance 0.466166m². The narrowing mean and increasing variance are compatible: the center moves toward zero while an initially narrow ensemble broadens toward its stationary width.

## Obtain the same moments without guessing a density

Taking expectation of the SDE gives the mean equation. Applying Itô's formula to X² introduces the quadratic-variation contribution 2D. Subtracting the derivative of the squared mean gives

$$
\dot m=-\gamma m,\qquad
\dot v=2D-2\gamma v.
$$

This derivation independently checks the moment formulas and the noise convention. Gaussianity follows from the explicit linear solution and the Gaussian initial law, not merely from knowing two moments. For a non-Gaussian initialization, these moment equations still hold when its second moment is finite, but two moments do not determine the entire law.

If v is positive, the Gaussian density and score are

$$
p(x,t)=\frac{e^{-(x-m)^2/(2v)}}{\sqrt{2\pi v}},
\qquad \partial_x\log p=-\frac{x-m}{v}.
$$

If v is zero, that formula is unavailable. The law is a Dirac measure, not a very tall finite Gaussian. Its mass remains one even though there is no ordinary density height or score to plot.

## Separate stationarity, a moment balance and an instantaneous identity

When gamma>0 and D>0, the unique stationary probability law is a centered Gaussian with variance D/gamma. When gamma>0 and D=0, it is the point mass at zero. When gamma=0 and D>0, Brownian spreading has no stationary probability law on the whole line. A constant density is not normalizable there. When both coefficients vanish, every initial law is stationary.

A Gaussian with the stationary variance but nonzero mean is not stationary: its center still moves. Likewise, at t=0 the time-evolution map is the identity for every initialization; that fact does not prove stationarity. Stationarity requires the entire law to remain invariant for all future times.

For gamma>0, calculate variance deviation directly:

$$
v(t)-\frac D\gamma=
\left(v_0-\frac D\gamma\right)e^{-2\gamma t}.
$$

Subtracting two rounded variance readouts may erase a small nonzero deviation. The prepared model encloses the right-hand expression separately. Exact stationarity, a positive quantity below display resolution, and an interval whose sign is unresolved are different observations.

## Derive Euler–Maruyama as a different probability model

Choose a final physical time T and N equal numerical steps h=T/N. With independent standard Gaussian innovations,

$$
X_{n+1}^{E}=(1-\gamma h)X_n^{E}+\sqrt{2Dh}\,Z_n,
\qquad Z_n\sim\mathcal N(0,1).
$$

Let a=1−gamma h. Independence gives exact moments of this discrete model:

$$
m_{n+1}^{E}=a m_n^{E},\qquad
v_{n+1}^{E}=a^2v_n^{E}+2Dh.
$$

These values are not samples and do not have Monte Carlo error. They are exact rational quantities when the input coefficients and time are rational. They approximate the continuous-time OU moments at the corresponding physical times.

Every step still defines a Gaussian or point-mass probability law, even if a is negative or has magnitude above one. That differs from the previous finite-state Euler matrix, whose negative entries can destroy probability positivity. Here the failure can be an expanding or oscillating mean and strongly biased variance while normalization remains correct.

## Distinguish numerical stability from correct stationary statistics

For a positive step, strict mean contraction requires 0<gamma h<2. In that regime the discrete chain has a unique invariant law with variance

$$
v_E^*=\frac{2Dh}{1-(1-\gamma h)^2}
      =\frac{D}{\gamma(1-\gamma h/2)}.
$$

For D>0 this exceeds D/gamma. A stable numerical method can therefore produce the wrong stationary width. At gamma h=1, Euler erases the initial mean in one step, whereas exact OU relaxation does not reach zero from a nonzero mean at finite time. At gamma h=2, a=−1; positive diffusion makes the variance increase by 2Dh every step. Above two, the mean mode expands unless its initial mean happens to be zero.

If D=0 and |a|=1, the variance need not grow; at a=−1, two-step oscillations and invariant symmetric laws are possible. If h=0, every Euler step is the identity, including its noise variance. Do not infer strict contraction or a unique invariant law by dividing a zero numerator by a zero denominator.

## Connect the score to probability flow without identifying paths

For positive v, writing J=p times a velocity gives the probability-flow field

$$
u(x,t)=-\gamma x-D\partial_x\log p
       =-\gamma x+\frac Dv(x-m).
$$

This scalar constant-diffusion specialization motivates the probability-flow construction in [Song et al., Section 4.3 and Appendix D](https://arxiv.org/html/2011.13456v2). Learning a high-dimensional score and controlling solver error are further problems; this exact Gaussian calculation does neither.

At a stationary Gaussian, u is zero everywhere. A deterministic flow starting from that Gaussian can leave each coordinate fixed while preserving the same one-time law as the OU SDE. The SDE still has time correlations that decay. Under the stationary initialization and gamma,D>0,

$$
\operatorname{Cov}(X_0,X_t)_{\mathrm{SDE}}
  =\frac D\gamma e^{-\gamma t},\qquad
\operatorname{Cov}(X_0,X_t)_{\mathrm{frozen\ flow}}
  =\frac D\gamma.
$$

Matching marginal density at every time does not match the joint law across times. A moving animation of density is not evidence that the displayed velocity is a particular particle's stochastic velocity.

## Treat the singular initial endpoint explicitly

With v0=0 and D>0, positive times have a smooth Gaussian density, but t=0 does not. As t approaches zero, v(t) behaves like 2Dt and the coefficient D/v behaves like 1/(2t). The score-based velocity is singular at that endpoint.

An ordinary deterministic map of a single fixed point remains a point. It cannot turn a deterministic initial point into a spread of positions through a unique regular flow. To use the smooth Gaussian probability-flow representation, start at a specified positive epsilon with the law at epsilon. For this OU example, the affine map from epsilon to t is m(t)+sqrt(v(t)/v(epsilon)) times the centered initial coordinate; it transports that Gaussian correctly. It does not establish a regular point-to-Gaussian flow from zero time.

With D=0, an atomic law instead follows the deterministic drift. Its absence of an ordinary score does not prevent the original drift ODE from existing. Density-based reasoning and measure-valued evolution have different availability conditions.

## Experiments and what each comparison means

| Preparation | Change | Read before concluding |
| --- | --- | --- |
| Baseline and fine Euler | Hold T fixed; increase N | Both signed moment errors, not one sampled path |
| Stationary Gaussian | Set m0=0 and v0=D/gamma | Exact moments stay fixed; Euler variance can drift |
| Correct width, moving center | Keep v0=D/gamma, change m0 | Constant variance does not imply stationary law |
| Mean reset, boundary, expansion | Choose gamma h=1,2,3 | Probability validity and dynamical fidelity differ |
| Brownian and identity | Set gamma=0; then D=0 | Handle the divided formula and stationary family |
| Atomic and nearly atomic starts | Set v0=0; vary D and T | Undefined density at an atom versus positive width |

The preparation accepts exact rational coefficients within bounded interaction ranges: gamma up to 5/s, D up to 2m²/s, mean between −5 and 5m, variance up to 5m², T up to 10s, and N from 1 to 128. These bound numerical work, not the course's scope. A later spatial experiment must additionally specify its grid, boundary condition, conservation law and positivity restriction. It must not silently reuse this whole-line Gaussian solution as a finite-domain solver.

## Worked exercises

### 1. Verify a freely spreading law

Set gamma=0, D=1/2m²/s, m0=2m, v0=1/4m² and T=1s. Find the final mean and variance. Does Euler need a small step to reproduce those two moments?

**Explanation.** The mean is 2m and variance is 5/4m². Here a=1 and every Euler variance increment is 2Dh; summing N increments gives 2DT exactly for any positive integer N. This agreement concerns the law at the recorded times. Coupled path error is a separate question requiring a specified common noise construction.

### 2. Diagnose a stable stationary bias

Use gamma=1/s, D=1/2m²/s and h=1/2s. Compare continuous and discrete invariant variances.

**Explanation.** The exact value is 1/2m². Euler has a=1/2 and invariant variance (1/2)/(1−1/4)=2/3m². The mean contracts, the law is normalized, and the stationary variance is still too large. Starting Euler at the exact stationary Gaussian does not remove this bias.

### 3. Keep a failed approximation instead of clipping it

Take gamma=1/s, D=1/2m²/s, m0=2m, v0=0, T=2s and N=1. Compute the Euler law and compare the exact mean.

**Explanation.** Euler has a=−1, mean −2m and variance 2m². It is a valid Gaussian, so no negative probability needs clipping. The exact mean is 2exp(−2)m, about 0.270671m, and its variance is (1−exp(−4))/2m². The large discrepancy is a dynamical error, not a failure of Gaussian normalization.

### 4. Determine when a density exists

Start from m0=2m and v0=0. Compare D=0 at T=1s, D>0 at T=0, and D>0 at T=1s.

**Explanation.** The first two are point masses. The first moves deterministically; the second is the unchanged singular initialization. Only the third has positive variance and an ordinary Gaussian density and score. Replacing the first two by a narrow plotting curve would change the specified law.

### 5. Show that equal marginals do not identify dynamics

Start the gamma=1/s, D=1/2m²/s SDE at its stationary law. Compare its lag-one-second covariance with the stationary probability-flow ODE.

**Explanation.** Both one-time laws have mean zero and variance 1/2m². The SDE covariance is exp(−1)/2m², about 0.183940m². The frozen deterministic flow has covariance 1/2m². The difference is observable in paired measurements at two times despite identical histograms at either time.

### 6. Explain why zero mean is insufficient

Suppose gamma h=3 and the Euler initialization has m0=0, v0=1/4m², with D=1/2m²/s and gamma=1/s. Is the discretization reliable because its mean remains zero?

**Explanation.** The multiplier is −2, so the initial zero mean hides an expanding mean mode. The next variance is four times 1/4 plus 3, giving 4m². A centered curve can broaden rapidly while its mean looks correct. Check the variance recurrence, coefficient regime and exact reference together.
