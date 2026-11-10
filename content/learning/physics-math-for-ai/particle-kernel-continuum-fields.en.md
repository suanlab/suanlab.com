## From point particles to continuous fields

A river looks smooth from a distance even though its constituents move individually. To connect these descriptions, specify an observation scale. Imagine spreading each point mass over a small neighbourhood, rather than increasing the number of particles. The resulting curve may be smooth without any collisions having occurred. We construct such a field explicitly, then distinguish its exact kinematic properties from additional assumptions needed for a fluid model.

## Prerequisites and line-density units

Review definite integrals, the chain rule, weighted averages and moving control volumes. This experiment places masses on a line; its density is mass per length, not mass per volume. A three-dimensional gas would require a three-dimensional kernel or a specified transverse area. Keep this distinction when comparing the next unit’s cubic observation volume. Normalization of a kernel means its spatial integral is one, not that its value is one everywhere.

$$
[m_i]=\mathrm{kg},\quad [W_h]=\mathrm{m^{-1}},\quad [\rho_h]=\mathrm{kg/m},\quad [j_h]=\mathrm{kg/s}.
$$

## Define the observation kernel first

Choose a symmetric compact kernel supported within distance h of each particle. Here h is a support half-width; it is neither a grid spacing nor a standard deviation. The coefficient below normalizes its integral. Both the value and first derivative join continuously to zero at the support edge. The second derivative jumps there. This matters when a learned differential model requests derivatives beyond those supported by the chosen representation.

$$
W_h(s)=\frac{15}{16h}\left(1-\frac{s^2}{h^2}\right)^2\mathbf1_{|s|<h},\qquad \int_{\mathbb R}W_h(s)\,ds=1.
$$

## The same particles and their trajectories

Each particle follows a straight trajectory with constant velocity. There are no interparticle forces, collisions or external accelerations. Particles may pass through one another; the experiment does not replace that crossing with an elastic collision. This deliberate simplification lets us inspect averaging without conflating it with relaxation. Track the same initial particles when changing time, rather than drawing unrelated random samples that would obscure the origin of the field.

$$
x_i(t)=X_i+v_it,\qquad \dot x_i=v_i,\qquad \dot v_i=0.
$$

## Average density and mass flux separately

Form density by summing mass-weighted kernels and mass flux by also weighting each contribution with velocity. Their ratio defines the observed velocity wherever density is positive. Averaging particle velocities without the mass and spatial weights generally gives another quantity. For example, coincident masses one and three moving at minus and plus one have weighted velocity one half, rather than the unweighted value zero.

$$
\rho_h=\sum_i m_iW_h(x-x_i),\qquad j_h=\sum_i m_iv_iW_h(x-x_i),\qquad u_h=j_h/\rho_h\quad(\rho_h>0).
$$

## Conservation follows from the chain rule

With fixed h, time enters a kernel only through the particle trajectory. The chain rule supplies a factor minus velocity, which cancels the spatial derivative of the corresponding mass flux. Thus even a single smoothed particle satisfies continuity. That exact identity is useful for checking the implementation, but it cannot establish a large particle population, local equilibrium or a valid constitutive law. Those are different physical questions.

$$
\partial_t\rho_h=-\sum_i m_iv_iW_h^{\prime}(x-x_i),\qquad \partial_xj_h=\sum_i m_iv_iW_h^{\prime}(x-x_i),\qquad \partial_t\rho_h+\partial_xj_h=0.
$$

## The second moment lost by mean velocity

Momentum transport depends on a second velocity moment. Squaring an average is not the same operation as averaging a square. The difference is a nonnegative weighted velocity dispersion. Computing it from centred velocities avoids subtracting two potentially similar large moments. In this ballistic example it is a kinematic dispersion contribution. Interpreting it as thermodynamic pressure or temperature would require physical assumptions and dimensional information that the experiment has not supplied.

$$
Q_h=\sum_i m_iv_i^2W_h=\rho_hu_h^2+S_h,\qquad S_h=\sum_i m_i(v_i-u_h)^2W_h\ge0.
$$

## Ballistic momentum balance and closure

Constant particle velocities also give a momentum balance involving Q. Predicting only density and mean velocity, then setting dispersion to zero, can remove genuine momentum transport. This is a small concrete example of a closure problem: lower moments do not generally determine higher ones. It does not derive Newtonian viscosity or a Navier–Stokes stress. Collisional transport and constitutive assumptions belong to subsequent models with additional physical information.

$$
\partial_tj_h+\partial_xQ_h=0,\qquad \partial_tj_h+\partial_x(\rho_hu_h^2)=-\partial_xS_h.
$$

## Worked example: two streams with zero mean velocity

Place two one-kilogram particles at the origin with velocities plus and minus one metre per second. At time zero, position zero and h equal to one metre, each kernel has value fifteen sixteenths per metre. Opposing mass fluxes cancel, but their squared-velocity contributions add. A stationary population can therefore share the same density and mean velocity while differing in momentum flux. This gives an explicit counterexample to identifying the full state from only its first moments.

$$
\rho_h=\frac{15}{8}\ \mathrm{kg/m},\quad j_h=u_h=0,\quad Q_h=S_h=\frac{15}{8}\ \mathrm N.
$$

## A finite display is not the whole line

Integrating over the entire line returns the sum of particle masses. A finite window may contain only part of a kernel, so its measured mass can be smaller. Do not renormalize the visible portion to the global total; doing so would conceal transport or missing support. In the worked pair, the interval from minus one half to plus one half metre contains 203/128 kilograms, while the whole line still contains two kilograms.

$$
\int_{\mathbb R}\rho_h\,dx=\sum_i m_i,\qquad M_{[a,b]}=\sum_i m_i\int_a^bW_h(x-x_i)\,dx.
$$

## Empty support and a zero-length window

Outside every compact support, density and flux are both exactly zero. Their ratio is undefined, not an observed stationary velocity. Likewise a zero-length observation interval has zero mass but no normalized mean density. The teaching implementation explicitly bounds h between 0.02 and 2 metres and accepts positive interval lengths from 10 to the minus eight metres. Unsupported smaller inputs are rejected rather than broadened into a different observation.

A positive density does not guarantee that every velocity moment fits the ordinary display range. Velocities of order 10⁻²⁰⁰ m/s have positive squares that round to zero in an early floating-point multiplication. Here kernel moments and derivatives are accumulated as exact rational expressions of the supplied binary64 values before one final rounding. Dispersion uses an equivalent sum of nonnegative pairwise squared velocity differences, avoiding cancellation between large moments. The arithmetic table distinguishes exact zero, nonzero values outside the linear display range (with sign and logarithm), and undefined means in empty support. Particle coordinates and support membership are still evaluated with ordinary floating-point geometry; this method cannot recover position information already lost there.

## What changes when the smoothing width changes?

Increasing the smoothing width can lower the measured density at a location even when no particle moves. If h changes in time, differentiating the observation introduces an extra bandwidth term. This is not a physical mass source. The momentum observation has an analogous term weighted by particle velocity. Spatially varying widths and kernels renormalized near walls require further terms; they are outside the present uniform-bandwidth construction.

$$
\partial_t\rho_h+\partial_xj_h=\dot h\sum_i m_i\partial_hW_h(x-x_i),\qquad \partial_tj_h+\partial_xQ_h=\dot h\sum_i m_iv_i\partial_hW_h(x-x_i).
$$

## Compute a change caused only by observation

For a stationary one-kilogram particle at the origin, the central density is fifteen sixteenths kilograms per metre at h equal to one metre. Doubling h halves that central value while leaving the spatial integral unchanged. With a width rate of one quarter metre per second at the original width, the density time derivative is minus fifteen sixty-fourths kilograms per metre per second. The physical flux is zero; the observation scale accounts for the change.

$$
\rho_h(0)=\frac{15m}{16h},\qquad \partial_t\rho_h(0)=-\frac{15m\dot h}{16h^2}.
$$

## Dispersion under a change of observation frame

Add the same boost b to every particle velocity and transform the observation position consistently. Relative particle distances stay fixed, so density is unchanged and the mean velocity increases by b. Centred velocity differences, and hence dispersion, are invariant. Mass flux itself changes with the frame. A useful AI test must specify which quantities should transform and which should remain invariant, rather than demanding identical outputs for all channels.

$$
x^{\prime}=x+bt,\quad v_i^{\prime}=v_i+b,\quad j_h^{\prime}=j_h+b\rho_h,\quad u_h^{\prime}=u_h+b,\quad S_h^{\prime}=S_h.
$$

## Experiment sequence and limits of identification

Begin with opposite velocities and compare zero mean motion with nonzero dispersion. Give both particles the same velocity to remove that dispersion. Then reduce h to expose narrow peaks and empty regions, and increase it to merge information from neighbours. Finally vary the width rate to isolate observation-induced terms. Several microscopic configurations can share the same limited averages, so smooth outputs and small conservation residuals do not uniquely recover particle states or establish local equilibrium.

## Exercises with answers

### Exercise 1

Two equal coincident masses move at plus and minus one metre per second. Does zero mean velocity imply zero momentum flux?

**Answer:** No. At h equal to one metre, Q and S are both 15/8 newtons in the worked example. Squared velocities do not cancel.

### Exercise 2

Coincident masses one and three kilograms move at minus and plus one metre per second. Find the observed velocity.

**Answer:** The common kernel cancels, giving (minus one plus three) divided by four, or 0.5 metres per second. The unweighted average is zero.

### Exercise 3

Outside all supports, both density and flux vanish. May the velocity be displayed as zero?

**Answer:** No. The ratio is undefined. An empty observation differs from material observed at rest.

### Exercise 4

For a stationary one-kilogram particle, double h from one to two metres. What happens to central density and total mass?

**Answer:** Central density falls from 15/16 to 15/32 kilograms per metre. Total mass remains one kilogram.

### Exercise 5

The worked pair has less than two kilograms in the interval from minus one half to plus one half metre. Is mass conservation violated?

**Answer:** No. The window contains 203/128 kilograms; the remaining kernel mass lies outside it.

### Exercise 6

Does a zero continuity residual of the smoothed field prove collisions or local equilibrium?

**Answer:** No. Even a single ballistic particle satisfies the same identity. Scale suitability and the physical distribution need separate evidence.

## Sources and next connections

For background on continuum averaging and scale separation, see Natalia Berloff’s [Averaging in Fluid Dynamics](https://www.damtp.cam.ac.uk/user/tong/concepts/fluid2.pdf). For mean free paths and collision times, see David Tong’s [Kinetic Theory §1.2](https://www.damtp.cam.ac.uk/user/tong/kintheory/one.pdf). The compact kernel, finite particle experiments and declared proxy inequalities here are this textbook’s constructions; the sources do not certify the implementation.
