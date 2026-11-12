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

## Research extension: energy moments and an unobserved flux

### Extend the same ballistic observation

Keep the finite positive masses, straight particle trajectories and fixed compact kernel already defined. Set the width rate to zero for this calculation. No force, collision or temperature law is added. A moment is a spatially weighted sum of a power of particle velocity; it is not another particle drawn from a new distribution. M0 and M1 are the existing density and mass flux. On a line the second moment has units energy per length and the third moment has units power. These differ from three-dimensional energy density and heat flux per area.

$$
M_k(x,t)=\sum_i m_i v_i^k W_h(x-X_i-v_it),\quad k=0,1,2,3,\quad \rho=M_0,\quad [M_2]=\mathrm{J/m},\quad[M_3]=\mathrm W.
$$

### Derive the next equation in the hierarchy

Differentiate a kernel along each straight trajectory. Its time derivative contributes minus vi times its spatial derivative, so the kth moment requires the next velocity moment as its flux. For k=2, dividing by two gives exact energy transport. Integrating the normalized kernel over the whole line recovers the sum of the fixed particle kinetic energies. The compact kernel is C1, sufficient for these first derivatives, even at its support edge. A finite observation window instead requires its boundary energy flux; it does not automatically contain all particle energy.

$$
\partial_tM_k+\partial_xM_{k+1}=0\ (k=0,1,2),\quad E=M_2/2,\quad F_E=M_3/2,\quad E_t+\partial_xF_E=0,\quad\int_{\mathbb R}E\,dx=\tfrac12\sum_i m_i v_i^2.
$$

### Separate mean motion from velocity dispersion

Where density is positive, define the local mean velocity and centred second moment. Expanding each squared velocity about that mean gives the split below. K measures energy in mean motion and D energy in the unresolved velocity spread. D is nonnegative, but calling it a thermodynamic internal energy or assigning a temperature requires additional physical assumptions. At a location with no support, raw moments and raw energy are zero, while the observed mean velocity and this ratio-based decomposition are not defined.

$$
\rho>0:\quad u=M_1/\rho,\quad S=M_2-\rho u^2\ge0,\quad K=\tfrac12\rho u^2,\quad D=\tfrac12S,\quad E=K+D.
$$

### An energy flux needs a third centred moment

Expand vi cubed as (u+ci) cubed and use the vanishing weighted first centred moment. The remaining cubic contribution is q. The full transport includes advected energy, dispersion work S u and q. In this one-dimensional ballistic calculation q is a centred kinetic-energy transport, not an assumed Fourier heat-conduction law. Neither S nor q follows from local equilibrium here. A positive second moment places no universal sign on the odd third moment.

$$
q=\tfrac12\sum_i m_i(v_i-u)^3W_h,\qquad M_3=\rho u^3+3uS+2q,\qquad F_E=(E+S)u+q.
$$

### Derive mean-energy transport from the lower moments

Use continuity and momentum, with momentum flux rho u squared plus S. Applying the product and quotient rules to K gives the first identity below. Adding the derivative of S u moves its transport to the left and leaves S times the velocity gradient as a local exchange. This is the one-dimensional dispersion-work term. Its sign depends on the local gradient. Moving a term between a stored rate and a flux changes the reported local budget, so keep the displayed convention when comparing two models.

$$
K_t+\partial_x(Ku)=-u\partial_xS,\qquad K_t+\partial_x[(K+S)u]=S\partial_xu.
$$

### The dispersion budget has the opposite exchange

Subtract the mean-energy equation from total energy. The same exchange appears with a minus sign and cancels when both budgets are added. Thus even collisionless particles can exchange resolved mean and unresolved dispersion energy under the chosen observation. Nonnegative dispersion does not imply nonnegative entropy production, viscous heating or monotone dispersion energy. The process has no collisions and is reversible under reversal of the particle velocities and time. If the smoothing width changes, additional observation terms must be derived before using these fixed-width identities. The bottom figure uses a separate preparation: masses (1,2,3) kg, initial positions (−1/4,0,1/2) m, velocities (−1,1/2,2) m/s, h=3/4 m and t=1/4 s. It plots the two opposite exchange terms, not the complete local time derivative or an irreversible heating rate.

$$
D_t+\partial_x(Du+q)=-S\partial_xu,\qquad [S\partial_xu]=\mathrm{W/m}.
$$

### Construct positive populations with identical lower moments

Place two particles at the same initial position and choose their mass fractions and velocities using a positive asymmetry parameter a and a positive speed reference. The fractions sum to one, the mean is zero and the variance is the same for every a. The third moment changes with a and changes sign under a to one over a. These identities hold for the entire initial spatial field because both particles share the same kernel centre. They do not state that the later fields remain equal: the particles then move to different positions.

$$
a>0:\quad v_-=-a v_{\rm ref},\quad v_+=v_{\rm ref}/a,\quad w_-=\frac1{1+a^2},\quad w_+=\frac{a^2}{1+a^2},\quad\sum w_jv_j=0,\quad\sum w_jv_j^2=v_{\rm ref}^2,\quad\sum w_jv_j^3=(a^{-1}-a)v_{\rm ref}^3.
$$

### Compare two fully specified preparations

Take mass reference 1 kg and speed reference 1 m/s. The symmetric and asymmetric preparations below both have total mass 5 kg, zero mean, the same density, dispersion and total kinetic energy at t=0. Nevertheless their energy fluxes differ wherever the kernel is nonzero. The table evaluates four positive a values with exact fractions before printing mass fractions, common normalized variance and normalized centred flux. Normalizing q by rho times the reference speed cubed is allowed only inside positive support; it is not an observed quotient outside the particles.

$$
m_{\rm total}=5m_{\rm ref}:\quad a=1:\ (m_-,m_+)=(\tfrac52,\tfrac52)m_{\rm ref};\quad a=2:\ (m_-,m_+)=(1,4)m_{\rm ref};\quad\rho=5m_{\rm ref}W_h,\quad S=\rho v_{\rm ref}^2,\quad q_1=0,\quad q_2=-\tfrac34\rho v_{\rm ref}^3\quad(t=0).
$$

### Transform energy as well as velocity

A common Galilean boost transforms the observation position and every particle trajectory together. Kernel weights and centred velocity differences are unchanged, hence S and q are invariant. Total kinetic energy is frame-dependent and changes with mass flux and density as shown. A comparison that boosts vi while retaining the old energy variable is not a symmetry test. The equality is local for corresponding observations, rather than a claim that a fixed laboratory coordinate sees the same particles after the change.

$$
x'=x+bt,\quad v_i'=v_i+b:\quad u'=u+b,\quad S'=S,\quad q'=q,\quad E'=E+bM_1+\tfrac12b^2\rho.
$$

### A closure choice can change the predicted evolution

Retain the actual lower moments and energy, but omit q from the energy flux. The resulting residual is minus its spatial derivative. The symmetric example has q=0 initially; the asymmetric example generally has a nonzero gradient despite identical initial density, mean and dispersion. Thus even those entire initial lower-moment fields do not close the energy evolution. A fitted closure needs declared training states and independent flux observations. The native two-particle lab keeps its original density and second-moment controls; this supplementary calculation does not introduce an energy solver or assert a material constitutive law.

$$
F_{E,\rm omit}=(E+S)u:\qquad E_t+\partial_xF_{E,\rm omit}=-\partial_xq.
$$

| a | negative-velocity mass fraction | positive-velocity mass fraction | variance / v_ref squared | q / (rho v_ref cubed) |
| --- | --- | --- | --- | --- |
| 0.50000000 | 0.80000000 | 0.20000000 | 1.00000000 | 0.75000000 |
| 1.00000000 | 0.50000000 | 0.50000000 | 1.00000000 | 0.00000000 |
| 2.00000000 | 0.20000000 | 0.80000000 | 1.00000000 | -0.75000000 |
| 4.00000000 | 0.05882353 | 0.94117647 | 1.00000000 | -1.87500000 |

```python
from fractions import Fraction as F
# Coincident centres at t=0, positive density; normalize speed by v_ref.
for a in [F(1,2), F(1), F(2), F(4)]:
    wm, wp = 1/(1+a*a), a*a/(1+a*a)
    vm, vp = -a, 1/a
    mean = wm*vm+wp*vp
    variance = wm*(vm-mean)**2+wp*(vp-mean)**2
    flux = (wm*(vm-mean)**3+wp*(vp-mean)**3)/2
    assert mean==0 and variance==1
    values = [a,wm,wp,variance,flux]
    print(" ".join(f"{float(x):.8f}" for x in values))
```

[MIT OCW 16.55, Lectures 13–14](https://ocw.mit.edu/courses/16-55-ionized-gases-fall-2014/resources/mit16_55f14_lecture13-14/) provides general kinetic-moment and energy-flux background. The fixed compact-kernel hierarchy, positive two-stream family and separate mean/dispersion budgets here are derived explicitly for this finite ballistic model.

![Equal initial lower moments, different energy flux and opposite local exchange](/assets/learning/continuum-ballistic-energy-moment-closure.en.svg)

### Research questions with explanations

**Do density, mean velocity and dispersion determine energy flux?** **Answer:** No. The positive two-stream family fixes those initial fields while changing its centred third moment q.

**Is positive dispersion enough to infer irreversible heating?** **Answer:** No. The local exchange has sign S times the velocity gradient and cancels in total energy. This ballistic model contains no collision mechanism.

**Does an empty support measure a mean velocity of zero?** **Answer:** No. Raw energy can be zero there, but velocity requires division by positive density. The central decomposition is not an observed stationary state.

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
