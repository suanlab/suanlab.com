## Why the same deformation can produce different stresses

Stretch a rubber band quickly and then hold it. The force can change while its length barely changes. Repeat the experiment slowly and the force history may differ. A relation that assigns stress using only the current strain cannot explain all these observations. A constitutive law needs to specify which aspects of the deformation history the material remembers.

We compare two elementary arrangements of a spring and a dashpot. They are useful models of mechanisms, not universal descriptions of every polymer or tissue. The spring stores recoverable mechanical energy. The dashpot converts mechanical work into an unmodeled thermal reservoir. A changing stress is therefore neither automatically lost energy nor proof of an active material.

## Assumptions, units and the prescribed experiment

Consider a spatially uniform, small uniaxial engineering strain. We prescribe its history rather than solve the motion of a specimen. Elastic modulus E and viscosity eta are positive constants. Temperature dependence, plasticity, inertia, spatial gradients and large deformations are excluded. Strain and internal elastic strain are dimensionless and limited to small values. Stress is measured in pascals, viscosity in pascal seconds and energy density in joules per cubic meter.

$$
\varepsilon(t)=\varepsilon_0\sin(\omega t),\qquad
\dot\varepsilon=\varepsilon_0\omega\cos(\omega t),\qquad
[E]=\mathrm{Pa},\quad[\eta]=\mathrm{Pa\,s}.
$$

The amplitude can be signed; changing its sign reverses the deformation convention. The angular frequency is nonnegative. The ideal sinusoid begins at zero total strain with a specified initial strain rate. This is a smooth loading history, different from an instantaneous strain step.

## Derive Kelvin–Voigt from parallel elements

Parallel elements undergo the same strain and their stresses add. The spring stress is proportional to strain; dashpot stress is proportional to strain rate. Together they produce an instantaneous relation that depends on both current quantities.

$$
\sigma_{KV}=E\varepsilon+\eta\dot\varepsilon.
$$

At the beginning of our sinusoidal experiment, strain is zero but strain rate need not be zero. Kelvin–Voigt stress can therefore be nonzero immediately. This is not a mysterious memory effect: the dashpot is responding to the specified current rate. An ideal discontinuous strain step would require an impulsive dashpot response, so it must not be drawn as a finite ordinary relaxation curve using the same formula.

## Derive Maxwell from series elements

Series elements carry the same stress, while their strains add. The spring strain is stress divided by E. The dashpot strain rate is stress divided by eta. Differentiating the total strain gives a first-order evolution equation for stress.

$$
\dot\varepsilon=\frac{\dot\sigma}{E}+\frac\sigma\eta,
\qquad \dot\sigma+\frac\sigma\tau=E\dot\varepsilon,
\qquad \tau=\frac\eta E.
$$

The relaxation time tau compares viscous and elastic response. Unlike Kelvin–Voigt, specifying current total strain and rate is insufficient to determine Maxwell stress. We must also provide an internal state, such as initial spring strain. If total initial strain is zero and the spring is initially strained, the dashpot has the opposite initial strain.

## Make initial memory explicit

Let initial elastic strain be e-zero. The homogeneous term carries its memory and the integral carries the response to subsequent loading. The exponential kernel weights recent strain-rate history more strongly than distant history.

$$
\sigma_0=Ee_0,\qquad
\sigma(t)=\sigma_0e^{-t/\tau}
+E\int_0^t e^{-(t-s)/\tau}\dot\varepsilon(s)\,ds.
$$

Two specimens with the same E, eta and current imposed deformation can have different stresses because their initial internal states differ. Their stress difference decays exponentially. Removing the initial memory term changes the actual field; it should be distinguished from omitting the relaxation term in the equation while evaluating an unchanged field.

## Separate startup from the steady periodic response

For the prescribed sinusoid, define dimensionless frequency x. Direct integration gives two periodic coefficients and a decaying transient. These coefficients are stresses, not moduli, because they include strain amplitude.

$$
x=\omega\tau,\qquad a=\frac{E\varepsilon_0x^2}{1+x^2},\qquad
b=\frac{E\varepsilon_0x}{1+x^2},\qquad
\sigma=a\sin(\omega t)+b\cos(\omega t)+(\sigma_0-b)e^{-t/\tau}.
$$

The purely periodic initial stress is b. Choosing zero initial stress usually leaves a startup transient even though the homogeneous contribution from the declared initial stress is zero. Thus “zero initial memory” and “already in the periodic state” are different statements. Plot the actual stress, periodic stress and transient separately before interpreting a loop.

## Stored energy, dissipation and mechanical power

For Kelvin–Voigt, the spring stores energy according to total strain. For Maxwell, only the spring's part of strain stores recoverable energy. The dashpot dissipates at a nonnegative rate in both models.

$$
\psi_{KV}=\frac12E\varepsilon^2,\quad
\mathcal D_{KV}=\eta\dot\varepsilon^2,\qquad
\psi_M=\frac{\sigma^2}{2E},\quad
\mathcal D_M=\frac{\sigma^2}{\eta}.
$$

Differentiate the actual energy and use the constitutive relations to obtain the local power balance. Negative input power can occur when the spring returns stored energy. Passivity requires nonnegative dissipation, not nonnegative input power at every instant.

$$
\sigma\dot\varepsilon=\dot\psi+\mathcal D,
\qquad [\sigma\dot\varepsilon]=[\mathcal D]=\mathrm{W/m^3}.
$$

The dissipated energy leaves the mechanical subsystem. This lesson does not calculate a specimen temperature, heat capacity or thermal boundary exchange. Calling the mechanical dissipation “heat generation” requires that additional physical interpretation; it does not supply a complete heat equation.

## Integrate an actual observation interval

A finite experiment records work over a specified time interval. Integrate actual power and actual dissipation independently, then compare with stored-energy change. Multiply all densities by a physical reference volume only when the assumptions justify that volume conversion.

$$
W_{[t_0,t_1]}=\int_{t_0}^{t_1}\sigma\dot\varepsilon\,dt
=\psi(t_1)-\psi(t_0)+\int_{t_0}^{t_1}\mathcal D\,dt.
$$

The first cycle of a startup experiment generally has different beginning and ending stress. Its path in the stress–strain plane is not a closed loop, and its stored-energy change need not vanish. A small balance residual calculated from shared analytic expressions is a diagnostic. Independent differentiation, convolution and time integration are needed to test the implementation.

## Storage and loss moduli describe the periodic state

Use sine strain as the phase convention. The in-phase coefficient defines storage modulus and the cosine coefficient defines loss modulus. Reversing a complex-exponential convention without adjusting the phase interpretation can reverse an apparent sign.

$$
\sigma_{\rm steady}=\varepsilon_0[E'\sin(\omega t)+E''\cos(\omega t)],\quad
E'_{KV}=E,\ E''_{KV}=\eta\omega,\quad
E'_M=\frac{Ex^2}{1+x^2},\ E''_M=\frac{Ex}{1+x^2}.
$$

For positive frequency, the work of one steady cycle equals its dissipation because stored energy returns to its initial value. It is a per-volume energy, not a power or a dimensionless loop area.

$$
W_{\rm steady\ cycle}=\oint\sigma\,d\varepsilon=\pi\varepsilon_0^2E''\ge0.
$$

At zero frequency our sine history is identically zero. There is no finite cycle. A preloaded Maxwell spring can still relax with zero external power. Very small positive frequency can put a period outside a finite teaching horizon or numerical range; neither case creates a finite zero-frequency period.

## A worked example

Take E=2 Pa, eta=1 Pa s, amplitude 0.002, frequency 1 rad/s and zero initial elastic strain. The numbers are chosen for hand calculation rather than as a fit to a particular material.

$$
\tau=0.5\ \mathrm{s},\quad x=0.5,\quad
a=0.0008\ \mathrm{Pa},\quad b=0.0016\ \mathrm{Pa},\quad
E'_M=0.4\ \mathrm{Pa},\quad E''_M=0.8\ \mathrm{Pa}.
$$

Maxwell stress is 0.0008 sin(t) plus 0.0016 cos(t) minus 0.0016 exp(−2t), in pascals. Its steady-cycle loss is approximately 1.0053 times ten to the minus five J/m³. Its actual first-cycle work is approximately 8.7731 times ten to the minus six J/m³; some of that work remains stored at the endpoint. Kelvin–Voigt has storage modulus 2 Pa and loss modulus 1 Pa in the same experiment.

## Small changes, cancellation and a cubic startup

Directly subtracting initial stress from a nearly equal current stress can lose a real change. A more revealing representation separates the initial stress derivative from the departure of the forcing cosine from one. If initial stress equals eta times initial imposed strain rate, the first derivative vanishes; the second also vanishes at the sine origin.

$$
\sigma_0=\eta\varepsilon_0\omega
\quad\Longrightarrow\quad
\sigma(t)-\sigma_0=-\frac{E\varepsilon_0\omega^3}{6}t^3+O(t^4).
$$

This cubic change is physically nonzero even when the stress display remains unchanged. Stable exponential-minus-one operations, signed logarithms and normalized integration coordinates retain information that rounded endpoint subtraction can discard. Below the linear representation range, report an unavailable nonzero observation with its logarithmic magnitude. Do not force it to zero. Numerical cycle integration can also lose a tiny loss term through cancellation against a much larger elastic contribution; compare it with the analytic steady-cycle expression.

The cubic conclusion requires the initial equality to hold for the actual inputs, not just their displayed decimals. For example, an amplitude entered as 0.001 and an angular frequency entered as 0.7 have binary floating-point representations. Their exact product need not equal a separately entered initial elastic strain of 0.0007. With unit modulus and viscosity, the small initial stress derivative is about −2.27 times ten to the minus twenty Pa/s. Rounding the two driving terms separately would incorrectly erase it and leave only a cubic change. The implementation forms the complete difference and ratio before observing that derivative, including at time zero. Keep an exactly tuned cubic experiment separate from a nearby experiment with a small linear contribution; neither display precision nor a small shared-formula residual proves the equality.

## Experiments and explained exercises

First compare startup and periodic initial conditions under identical loading. Next vary frequency while observing stress phase, storage and loss moduli. Finally inspect actual interval work, stored-energy change and dissipation together. For learned constitutive models, these checks expose missing internal state, incorrect phase conventions and violations hidden by a low pointwise fitting error.

One frequency can be insufficient to distinguish model structure even if both phase and amplitude are measured: with suitable positive parameters, either elementary model can reproduce a positive storage/loss pair at that frequency. Multiple frequencies and startup or relaxation histories supply additional tests. Fitting one curve should therefore be followed by predictions under a different loading history, with initial state, units and stress convention kept explicit.

### Exercise 1 — Parallel versus series

Which quantity is shared by the two elements in each arrangement?

**Answer:** Parallel elements share strain and add stresses. Series elements share stress and add strains. These kinematic and force constraints produce different constitutive equations even when E and eta are identical.

### Exercise 2 — A nonzero initial stress

Can Maxwell stress be nonzero at zero total initial strain?

**Answer:** Yes. The spring can have strain e-zero and the dashpot strain minus e-zero. Their total is zero while spring stress is E times e-zero. State the internal initial condition explicitly.

### Exercise 3 — Negative power

Does negative mechanical power prove the model is active?

**Answer:** No. Stored spring energy can be returned. Check that dissipation is nonnegative and that input work equals stored-energy change plus dissipated energy over the actual interval.

### Exercise 4 — First-cycle loss

May the steady-cycle formula be used as the actual first-cycle work from zero initial stress?

**Answer:** Generally no. The startup transient remains, endpoint stress need not equal initial stress and stored energy changes. Integrate the actual field instead.

### Exercise 5 — Zero frequency

What happens to a preloaded Maxwell model when frequency is zero in this experiment?

**Answer:** Imposed strain and strain rate are zero. The initial stress relaxes exponentially, stored energy decreases and dissipation is positive. There is no finite cycle to label with a cycle loss.

### Exercise 6 — Hidden cubic change

Why can an initially stationary stress still change at small positive time?

**Answer:** Vanishing first derivative does not make the whole function constant. Under the tuned initial condition, the leading change is cubic. Preserve that small change separately from a rounded stress difference.

## Source and limitations

[A. F. Bower, Applied Mechanics of Solids §3.6](https://solidmechanics.org/Text/Chapter3_6/Chapter3_6.htm) discusses small-strain viscoelastic response, loading histories and elementary mechanical models. We use a stated uniaxial convention and derive the smooth-history energy identities above. The two-element examples are not a calibrated description of every material or a general large-deformation viscoelastic law.
