## Why a temperature field is more than an extra input

A heated solid tends to expand. That familiar observation suggests a simple workflow: solve a heat equation, then use the temperature to compute deformation. Sometimes that approximation is useful. But deformation can also change temperature. A coupled model must specify which directions of influence it retains and how energy crosses between its fields.

We study one spatial mode of a linear thermoelastic solid. Three amplitudes describe displacement, velocity and temperature deviation. This small system lets us inspect reciprocal exchange directly, compare it with a one-way approximation and verify the actual spatial and temporal integrals. It is also a useful model for evaluating an AI surrogate: accurately fitting two curves does not by itself establish that their exchange terms agree.

## Declare the material, reference state and domain

Let u be axial displacement, epsilon its spatial derivative and theta the deviation from a positive reference absolute temperature T-zero. Elastic modulus E, density rho and volumetric heat capacity C are positive. Thermal conductivity kappa is nonnegative. The thermal stress coefficient beta is constant. C is the heat capacity at fixed strain, not at fixed stress; these can differ in a coupled material.

$$
\varepsilon=u_x,\qquad T=T_0+\theta,\qquad
[E]=\mathrm{Pa},\quad[C]=\mathrm{J/(m^3K)},\quad
[\beta]=\mathrm{Pa/K},\quad[\kappa]=\mathrm{W/(mK)}.
$$

The model assumes small strain and small temperature deviations with constant properties. It excludes plasticity, large deformation, changing geometry and a calibrated three-dimensional specimen. We use a periodic cell of length two pi divided by positive wavenumber k, with cross-sectional area A. The symbol k is a wavenumber; kappa is conductivity. Setting k to zero would remove the finite periodic cell and is outside this family.

## Derive both directions from one thermodynamic choice

A quadratic Helmholtz free-energy expansion fixes stress and the entropy perturbation consistently. Here both energy and entropy are per reference volume. Terms constant or linear in the reference state have been omitted from this perturbation description.

$$
\psi=\frac E2\varepsilon^2-\beta\varepsilon\theta
-\frac C{2T_0}\theta^2,\qquad
\sigma=\psi_{\varepsilon}=E\varepsilon-\beta\theta,\qquad
s=-\psi_{\theta}=\beta\varepsilon+\frac C{T_0}\theta.
$$

The negative temperature curvature of free energy is consistent with positive heat capacity. It does not mean that the material has a negative stored thermal inventory. Free energy and internal energy use different independent variables and serve different purposes.

Momentum balance supplies the mechanical equation. Linearizing the entropy relation about T-zero and using Fourier conduction supplies the thermal equation. The same beta appears in the stress response and the reverse thermal coupling.

$$
\rho u_{tt}=E u_{xx}-\beta\theta_x,\qquad
C\theta_t=\kappa\theta_{xx}-T_0\beta\varepsilon_t.
$$

For positive beta, extension can cool the perturbation in an adiabatic process. The sign is not chosen independently to make a plotted energy curve decrease; it follows from the declared thermodynamic convention. A different convention must transform all linked quantities consistently.

## Reduce the PDE to three actual state variables

Choose a sine displacement mode and a cosine temperature mode. Their spatial derivatives then have the phases required to couple to each other. Substituting into the original equations gives a closed linear amplitude system.

$$
u=a(t)\sin(kx),\quad\theta=b(t)\cos(kx),\quad v=\dot a,
\qquad
\dot a=v,\quad
\rho\dot v=-Ek^2a+\beta kb,\quad
C\dot b=-\kappa k^2b-T_0\beta kv.
$$

Initial displacement amplitude, velocity amplitude and temperature amplitude are three distinct pieces of initial data. A snapshot of displacement does not determine velocity or temperature. A negative b is a temperature deviation below the reference, not a negative absolute temperature. The teaching bounds keep deviations and strains small; the reciprocal energy bound provides an additional check over the whole trajectory.

## Choose an energy whose meaning is explicit

For this linearized problem, combine kinetic energy with the quadratic quantity psi plus theta times s. The mixed term cancels. Integrating over the actual periodic cell uses the average of sine squared and cosine squared, each one half. With volume V equal to two pi A divided by k, the result is quadratic availability.

$$
\mathcal A_m=\frac V4(\rho v^2+Ek^2a^2),\qquad
\mathcal A_T=\frac{VC}{4T_0}b^2,\qquad
\mathcal A=\mathcal A_m+\mathcal A_T.
$$

Availability measures disturbance energy relative to the reference state. It is not the full internal energy, an accumulated heat quantity or C times the temperature amplitude. Positive and negative b give positive thermal availability. Heat conduction can reduce availability while smoothing the disturbance; this does not say that physical energy disappears. The linear perturbation model does not track every second-order contribution to mean heating and entropy.

Changing area scales both energies without changing the amplitude dynamics. Changing wavenumber changes the actual cell volume as well as oscillation and diffusion rates. Forgetting the volume factor can therefore produce incorrect comparisons between experiments.

## Watch the exchange cancel and conduction remain

Differentiate both integrated energies using the amplitude equations. Mechanical exchange is signed: it can transfer availability in either direction. The reciprocal thermal exchange has the opposite sign. Conductive availability loss is nonnegative.

$$
\dot{\mathcal A}_m=\frac{V\beta k}{2}vb,\qquad
\dot{\mathcal A}_T=-\frac{V\beta k}{2}vb
-\frac{V\kappa k^2}{2T_0}b^2,\qquad
\dot{\mathcal A}=-\frac{V\kappa k^2}{2T_0}b^2.
$$

With zero conductivity, total availability is conserved although its mechanical and thermal parts can change. With zero beta, the elastic oscillator and the diffusing temperature mode decouple. These are different limits: removing conduction does not remove reciprocal exchange.

For the reciprocal model, positivity also bounds strain amplitude. The bound applies to that model, not automatically to a differently driven or one-way system.

$$
|ka(t)|\le\sqrt{\frac{4\mathcal A(0)}{VE}}.
$$

## Compute the evolution without erasing a small change

Energy-normalized coordinates put the coupling into a particularly transparent matrix. The mechanical and reciprocal thermal off-diagonal entries have opposite signs.

$$
y=\begin{pmatrix}k\sqrt E\,a\\\sqrt\rho\,v\\\sqrt{C/T_0}\,b\end{pmatrix},\qquad
\dot y=\begin{pmatrix}0&\omega&0\\-\omega&0&g\\0&-g&-d\end{pmatrix}y,
\quad\omega=k\sqrt{E/\rho},\quad
g=\beta k\sqrt{T_0/(\rho C)},\quad d=\kappa k^2/C.
$$

The implementation evaluates the matrix exponential numerically using a scaled Taylor expansion and repeated squaring. It propagates the exponential minus the identity directly, so small changes are not formed by subtracting two rounded state values. The full propagator is advanced separately to retain a small decaying tail that would be lost by adding a change close to minus one back to one. Signed logarithms retain nonzero amplitudes and quadratic quantities outside the ordinary floating-point range. These techniques improve the declared computation; they are not a proof of correctly rounded results for every real input.

For finite intervals, independent time quadrature integrates exchange and conduction loss. A balance-derived availability change is reported separately from endpoint subtraction. Agreement between two quantities constructed from the same identity is a diagnostic, not an independent accuracy certificate. The reference calculations instead use physical-variable matrix exponentials, a separate eigensystem and physical-space integrals.

## Work through the initial rates

Take E equals 2 Pa, rho equals 1 kg/m³, C equals 1 J/(m³ K), T-zero equals 300 K, beta equals 0.02 Pa/K, kappa equals 0.1 W/(m K), k equals 1/m and A equals 1 m². Set initial a to 0.001 m, v to 0.001 m/s and b to 0.02 K. These are educational coefficients, not a calibrated material.

The cell volume is two pi m³. Mechanical availability is approximately 4.71239 microjoules; thermal availability is approximately 2.09440 microjoules. At phase zero, stress is 0.0016 Pa. The initial mechanical exchange rate is about positive 1.25664 microwatts, while conductive availability loss is about 0.418879 microwatts. Thermal availability changes at about minus 1.67552 microwatts. Their sum is minus the conductive loss, even though the mechanical part initially grows.

## Compare a one-way approximation honestly

Suppose temperature is evolved by diffusion alone while it continues to drive mechanical stress. The thermal amplitude equation becomes b-dot equals minus d b. The mechanical exchange remains, so the previous closed reciprocal availability identity changes.

$$
\text{one-way:}\quad
\dot b=-db,\qquad
\dot{\mathcal A}=\frac{V\beta k}{2}vb
-\frac{V\kappa k^2}{2T_0}b^2.
$$

This is a different model and gives a different trajectory. It may approximate weak feedback, or represent temperature maintained by an external controller. In the latter interpretation an appropriate heat input must be declared. It should not be presented as the same closed reciprocal system with an unexplained energy gain. Likewise, evaluating an omitted-term residual on the old trajectory is different from solving the changed system.

## Explain the cubic loss near the start

Start with zero displacement and temperature amplitudes but a nonzero velocity amplitude. Temperature is generated linearly at first. Conductive loss rate is therefore quadratic in time, and accumulated conductive loss is cubic.

$$
b(t)\sim-\frac{T_0\beta k v_0}{C}t,\qquad
\int_0^t\!D(s)\,ds
\sim\frac{V\kappa T_0\beta^2k^4v_0^2}{6C^2}t^3.
$$

An endpoint energy can round to its initial value while this positive cubic loss remains mathematically nonzero. The stable integral retains its logarithm even when the ordinary value is unavailable. Exact zero beta, zero conductivity or zero initial state are separate physical cases and must remain exact zeros where appropriate.

## Six exercises with answers

### Exercise 1 — Missing initial data

Does an initial displacement profile uniquely determine this evolution?

**Answer:** No. Specify displacement amplitude, velocity amplitude and temperature amplitude. Each occupies a different state coordinate.

### Exercise 2 — Negative thermal amplitude

Does b below zero imply negative thermal availability or negative absolute temperature?

**Answer:** Neither. Thermal availability is proportional to b squared. Absolute temperature is T-zero plus the small spatial deviation and remains positive in the declared family.

### Exercise 3 — Adiabatic exchange

Set conductivity to zero while retaining beta. Must mechanical availability stay constant?

**Answer:** No. Reciprocal exchange can change both parts. Their sum stays constant; the conductive loss is exactly zero.

### Exercise 4 — Area scaling

Double area without changing material, wavenumber or initial amplitudes. What changes?

**Answer:** The amplitude trajectory is unchanged. Cell volume, integrated availabilities, exchange power and loss power double.

### Exercise 5 — One-way temperature

Can diffusion-only temperature driving a mechanical field use the closed reciprocal availability balance without modification?

**Answer:** No. The reverse exchange was removed. Declare the approximation or the external heat controller and account for the remaining exchange term.

### Exercise 6 — A rounded constant energy

With zero initial temperature and nonzero initial velocity, does an unchanged rounded endpoint energy prove zero early conduction loss?

**Answer:** No. Generated temperature is initially linear in time and accumulated loss is cubic. Inspect its stable integral and logarithm separately from endpoint subtraction.

## Sources and research scope

[P. Kelly, Solid Mechanics Part IV §2.2](https://pkel015.connect.amazon.auckland.ac.nz/SolidMechanicsBooks/Part_IV/Chapter_2_Applications/PDF/02_Thermomech_Applications_02_Thermoelasticity.pdf) develops thermoelastic free-energy relations and the deformation term in heat conduction. [MOOSE's thermal–mechanical tutorial](https://mooseframework.inl.gov/modules/combined/tutorials/introduction/thermomech_step01.html) illustrates connecting thermal and mechanical components. We derive our stated one-dimensional periodic family and availability balance directly. For AI research, test reciprocal exchange, initial data, parameter scales and spatial/time integrals, then investigate the effects of omitted modes and nonlinear physics. A successful single-mode fit is not a general thermoelastic solver.
