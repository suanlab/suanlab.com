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

## Research extension: identical frequency data and different internal laws

Suppose a harmonic test reports one positive storage modulus and one positive loss modulus. Can a constitutive-learning algorithm infer the arrangement of springs and dashpots uniquely? Use Sobs and Lobs for the observed moduli, keeping them distinct from the earlier a and b stress coefficients that already include strain amplitude. Use a known nonzero small strain amplitude. The observations refer to a **steady periodic state** at a known positive frequency. Startup data are not being silently replaced by a periodic formula.

$$
S_o=E'(\omega_0)>0,\quad L_o=E''(\omega_0)>0,\quad\omega_0>0,\qquad\sigma_{\rm per}(t)=\varepsilon_0[S_o\sin(\omega_0t)+L_o\cos(\omega_0t)],\quad[S_o]=[L_o]=\mathrm{Pa}.
$$

A Kelvin–Voigt model fits these two observations by taking its spring modulus to be Sobs and its viscosity to be Lobs divided by frequency. Both parameters are positive. This explains why a nonzero measured loss does not alone prove the presence of a hidden relaxing spring state: the parallel dashpot already supplies loss.

$$
E_{KV}=S_o,\qquad\eta_{KV}=\frac{L_o}{\omega_0}>0.
$$

A single Maxwell element fits exactly the same pair with different positive parameters. Divide storage by loss to determine its dimensionless frequency, then recover its modulus and relaxation time. For example, unit observed moduli at unit angular frequency give a Kelvin–Voigt spring and viscosity of one, but a Maxwell modulus and viscosity of two in their respective SI units. Both predict the same total periodic stress for the same imposed strain.

$$
E_M=\frac{S_o^2+L_o^2}{S_o},\qquad\tau_M=\frac{S_o}{L_o\omega_0},\qquad\eta_M=E_M\tau_M=\frac{S_o^2+L_o^2}{L_o\omega_0}.
$$

Predict a second frequency without refitting. Write its ratio to the measured frequency as z. The two response functions now follow directly from their constitutive equations. Kelvin–Voigt storage stays constant and its loss grows linearly. Maxwell storage tends to a finite high-frequency modulus while its loss eventually falls. Matching at one point does not make these whole functions equal.

$$
z=\omega/\omega_0:\quad\begin{aligned}E'_{KV}(z)&=S_o,&E''_{KV}(z)&=zL_o,\\E'_M(z)&=\frac{z^2S_o(S_o^2+L_o^2)}{L_o^2+z^2S_o^2},&E''_M(z)&=\frac{zL_o(S_o^2+L_o^2)}{L_o^2+z^2S_o^2}.\end{aligned}
$$

The explicit storage difference proves that any other positive frequency distinguishes **these two ideal noiseless model classes** when both observed moduli are positive. This is an exact-identification statement, not a claim about experimental noise or an arbitrary relaxation spectrum. Close to z = 1 the difference is small, so practical discrimination also depends on measurement accuracy and frequency placement.

$$
E'_M(z)-E'_{KV}(z)=\frac{S_oL_o^2(z^2-1)}{L_o^2+z^2S_o^2}\ne0\qquad(z>0,\ z\ne1).
$$

Now expand the model class to several Maxwell branches in parallel. Each branch carries its own initial internal state and a causal exponential memory; their stresses add. A finite set of positive branch parameters is already more flexible than the single element. It is important to state which model class is being identified before interpreting a fitted parameter list.

$$
\dot\sigma_j+\frac{\sigma_j}{\tau_j}=E_j\dot\varepsilon,\quad E_j,\tau_j>0,\qquad\sigma=\sum_j\sigma_j.
$$

Sum each branch's power balance. The sum stores recoverable energy and dissipates at a nonnegative rate, even though instantaneous input power can be negative when energy is returned. The model is passive relative to its declared initial storage. Positivity and this energy identity constrain physical structure, but they do not imply that a limited experiment identifies the branch parameters uniquely.

$$
\psi=\sum_j\frac{\sigma_j^2}{2E_j},\qquad\mathcal D=\sum_j\frac{\sigma_j^2}{E_j\tau_j}\ge0,\qquad\sigma\dot\varepsilon=\dot\psi+\mathcal D.
$$

At the observed frequency, regard each branch's positive loss contribution as a weight. The storage-to-loss ratio is the weighted average of its dimensionless relaxation times. This turns the inverse problem into a two-moment matching problem. To compare full periodic histories, initialize each branch at stress epsilon-zero times its weight. Choosing zero initial memory instead generally produces a transient and would change the experiment.

$$
x_j=\omega_0\tau_j,\quad w_j=\frac{E_jx_j}{1+x_j^2}>0,\qquad L_o=\sum_jw_j,\quad S_o=\sum_jw_jx_j.
$$

Choose one dimensionless branch time on each side of the observed ratio. The displayed positive weights match both moments exactly, and the recovered moduli and times are positive. There are infinitely many such choices. Thus even adding passivity and a finite causal memory representation does not select a unique two-branch spectrum from a single frequency. Two frequencies distinguish the elementary pair above but need not determine every larger spectrum.

$$
0<x_1<\frac{S_o}{L_o}<x_2:\qquad w_1=L_o\frac{x_2-S_o/L_o}{x_2-x_1},\quad w_2=L_o\frac{S_o/L_o-x_1}{x_2-x_1},\quad E_j=\frac{w_j(1+x_j^2)}{x_j},\quad\tau_j=\frac{x_j}{\omega_0}.
$$

Here is one explicit two-branch fit to the unit target. Its high-frequency storage is the sum of branch moduli, and its low-frequency loss divided by angular frequency tends to the sum of modulus times relaxation time. Both limits differ from the single Maxwell fit, whose corresponding values are two. All three models have the same observed periodic stress and cycle loss at the fitted frequency, yet their internal stored energies can differ at the same phase. A matched output history need not reveal the internal allocation of energy.

$$
S_o=L_o=1\mathrm{\ Pa},\quad\omega_0=1\mathrm{\ s^{-1}}:\qquad(E_1,\tau_1)=(3.4\mathrm{\ Pa},0.25\mathrm{\ s}),\quad(E_2,\tau_2)=(0.85\mathrm{\ Pa},4\mathrm{\ s}),\quad\sum_jE_j=4.25\mathrm{\ Pa},\quad\sum_jE_j\tau_j=4.25\mathrm{\ Pa\,s}.
$$

The table uses Sobs = Lobs = 1 Pa and omega-zero = 1/s. Columns are frequency ratio, Kelvin–Voigt storage/loss and single-Maxwell storage/loss in Pa. The figure adds the positive two-branch example. Its energy panel uses the periodic branch initial states and plots storage divided by nonzero strain amplitude squared, in Pa. Equal total periodic stress does not imply equal stored energy. These examples add no spectrum controls to the current two-element lab.

| frequency ratio | KV storage | KV loss | Maxwell storage | Maxwell loss |
| --- | --- | --- | --- | --- |
| 0.25000000 | 1.00000000 | 0.25000000 | 0.11764706 | 0.47058824 |
| 0.50000000 | 1.00000000 | 0.50000000 | 0.40000000 | 0.80000000 |
| 1.00000000 | 1.00000000 | 1.00000000 | 1.00000000 | 1.00000000 |
| 2.00000000 | 1.00000000 | 2.00000000 | 1.60000000 | 0.80000000 |

```python
# Observed storage=loss=1 Pa at omega0=1/s.
for z in (0.25, 0.5, 1.0, 2.0):
    kv_storage, kv_loss = 1.0, z
    maxwell_storage = 2*z*z/(1+z*z)
    maxwell_loss = 2*z/(1+z*z)
    print(" ".join(f"{v:.8f}" for v in (z, kv_storage, kv_loss, maxwell_storage, maxwell_loss)))
```

The [Bower small-strain viscoelasticity chapter](https://solidmechanics.org/Text/Chapter3_6/Chapter3_6.htm) gives history and harmonic-response background. The inverse constructions and uniaxial sine convention here are explicit separate derivations; they do not inherit the source shear-strain factors automatically. The fitted toy spectra are not a calibration for a named material.

![Positive passive spectra fit one frequency but differ in other frequencies and internal energy](/assets/learning/continuum-viscoelastic-spectrum-identifiability.en.svg)

### Research questions with explanations

**Does positive loss at one frequency identify a unique Maxwell spectrum?** **Answer:** No. The explicit positive two-branch construction matches the same storage and loss for infinitely many time pairs.

**Can zero initial memory be used when comparing complete steady histories?** **Answer:** Generally no. Each Maxwell branch needs its periodic initial stress; zero memory usually leaves a startup transient.

**Does an extra frequency identify every possible model class?** **Answer:** No. It distinguishes the stated Kelvin–Voigt and single-Maxwell fits in the noiseless setting, while larger spectra require further information and an explicit model class.

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
