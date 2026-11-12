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

## Research extension: dissipation, invariant motions and relaxation

The availability law says that a closed reciprocal mode cannot grow in total norm. It does not yet say whether every disturbance disappears. To ask that stronger question, retain the same energy-normalized variables and examine the actual evolution matrix. Omega is strictly positive because the periodic cell has positive wavenumber; g can have either sign and d is nonnegative. All three matrix coefficients have units of inverse seconds. No external forcing or one-way temperature controller is added.

$$
M=\begin{pmatrix}0&\omega&0\\-\omega&0&g\\0&-g&-d\end{pmatrix},\qquad \dot y=My,\quad\mathcal A=\frac V4\|y\|^2,\qquad\omega>0,\ d\ge0.
$$

Insert a trial exponential state with growth rate lambda. Expanding the determinant gives a cubic rather than three independent scalar decay rates. Its roots describe the asymptotic modes of this particular finite system. A negative real part gives exponential damping of that mode; a purely imaginary root permits persistent oscillation. Because the coefficients are real, any nonreal roots occur as a conjugate pair.

$$
\det(\lambda I-M)=\lambda^3+d\lambda^2+(\omega^2+g^2)\lambda+d\omega^2.
$$

For a monic cubic with positive coefficients, the cubic Hurwitz condition additionally requires the product of the quadratic and linear coefficients to exceed the constant coefficient. Here its margin is d times g squared. Thus positive conduction and nonzero reciprocal coupling place all roots strictly in the left half-plane. The condition is independent of the sign of g. This conclusion concerns the declared single mode, not every continuum boundary condition or nonlinear material.

$$
d>0,\ g\ne0:\qquad d(\omega^2+g^2)-d\omega^2=dg^2>0\quad\Longrightarrow\quad\operatorname{Re}\lambda_j<0.
$$

There is also an intuitive route. A trajectory with no availability loss must keep the thermal coordinate zero. Its differential equation then requires the velocity coordinate to vanish when g is nonzero. The velocity equation finally requires the displacement coordinate to vanish because omega is positive. The only trajectory that can remain forever inside the zero-loss set is the origin. This invariant-set argument explains why indirect thermal dissipation eventually damps mechanical motion, even when the instantaneous loss happens to be zero.

$$
\dot{\mathcal A}=-\frac{Vd}{2}y_3^2,\qquad y_3\equiv0\Rightarrow gy_2\equiv0\Rightarrow\omega y_1\equiv0\quad(d>0,\ g\ne0).
$$

The exact endpoints matter. Removing conduction makes the matrix skew-symmetric: its exponential preserves every norm, and the spectrum includes one zero root. Removing coupling instead leaves an undamped mechanical oscillator alongside an independently diffusing thermal coordinate. Nonincreasing availability is consistent with both of these persistent motions. Replacing an exact zero by an arbitrary small positive coefficient changes the long-time conclusion.

$$
d=0:\ \lambda\in\{0,\,\pm i\Omega\},\quad\Omega=\sqrt{\omega^2+g^2};\qquad g=0:\ \lambda\in\{-d,\,\pm i\omega\}.
$$

The stationary direction in the adiabatic case combines displacement and temperature with zero velocity. Project the initial state onto it. That projection is constant; the orthogonal part rotates forever. This is a decomposition, not convergence to a terminal equilibrium. For example, an initial pure thermal coordinate generally contains both a stationary part and an oscillating part. Y-star has the same units as each energy-normalized coordinate, the square root of pascals. Scale the whole state by a small common amplitude to remain inside the original small-strain and small-temperature regime.

$$
n=\begin{pmatrix}g\\0\\\omega\end{pmatrix},\quad Mn=0\ (d=0),\qquad y_{\parallel}=\frac{nn^T}{\Omega^2}y(0),\quad y(t)=y_{\parallel}+y_{\perp}(t).
$$

For zero conduction the cubic identity reduces the exponential to a sine and cosine expression. This gives an executable example with no numerical ODE library. In the table omega and g both equal one per second and the initial state is Y-star times the third coordinate vector. The columns divide all coordinates by Y-star, so the norm column is one without assigning an unrealistically large physical disturbance. The formula concerns zero conduction only.

$$
M^3=-\Omega^2M,\qquad e^{Mt}=I+\frac{\sin(\Omega t)}{\Omega}M+\frac{1-\cos(\Omega t)}{\Omega^2}M^2\quad(d=0).
$$

Even with both required mechanisms present, increasing conduction is not an unlimited speed improvement. At fixed positive omega and fixed nonzero g, perturb the undamped conjugate pair for small positive d. Its real part starts linearly in d. A separate slow real root starts with a different coefficient. These are expansions in d with the other scales fixed; neither is a universal error bound or a finite-time envelope for every coordinate.

$$
d\to0^+:\qquad\operatorname{Re}\lambda_{\pm}=-\frac{dg^2}{2(\omega^2+g^2)}+o(d),\qquad\lambda_0=-\frac{d\omega^2}{\omega^2+g^2}+o(d).
$$

In the opposite limit temperature relaxes so quickly that its amplitude is small, approximately minus g times velocity divided by d. The mechanical feedback therefore becomes weak again. The conjugate roots approach the original mechanical frequency with damping proportional to inverse d. Thus both very weak and very strong conduction can leave slowly damped mechanical oscillations. The fast thermal root does not describe the slowest surviving disturbance.

$$
d\to\infty:\qquad y_3\simeq-\frac gd y_2,\quad\lambda_{\pm}=\pm i\omega-\frac{g^2}{2d}+o(d^{-1}),\quad\lambda_{\rm fast}=-d+O(d^{-1}).
$$

The figure scans a dimensionless conduction ratio while holding the coupling ratio at one. Its positive spectral relaxation rate tends to zero at either extreme. That is an asymptotic indicator; the matrix can be nonnormal when conduction is present, so it does not equal every transient component slope. For scientific AI, assess the availability inequality, zero-case spectrum and observed trajectories separately. Fitting a decaying temperature trace alone cannot establish that mechanical energy ultimately decays. The existing interactive controls and propagator remain unchanged.

$$
\delta=d/\omega,\quad\gamma=g/\omega,\qquad r_{\rm spec}=-\max_j\operatorname{Re}(\lambda_j/\omega)\quad(\delta>0,\ \gamma\ne0).
$$

Columns are time in seconds, y1/Y-star, y2/Y-star, y3/Y-star and squared norm divided by Y-star squared. This is a zero-conduction trajectory, not a positively damped one.

| t | y1/Ystar | y2/Ystar | y3/Ystar | norm2/Ystar2 |
| --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 0.00000000 | 1.00000000 | 1.00000000 |
| 0.50000000 | 0.11987770 | 0.45936268 | 0.88012230 | 1.00000000 |
| 1.00000000 | 0.42202815 | 0.69845600 | 0.57797185 | 1.00000000 |
| 2.00000000 | 0.97568156 | 0.21783962 | 0.02431844 | 1.00000000 |

```python
from math import sqrt, sin, cos
# omega=g=1 /s, d=0; y(0)=Y-star*(0,0,1).
omega = sqrt(2.0)
for t in (0.0, 0.5, 1.0, 2.0):
    y1 = (1-cos(omega*t))/2
    y2 = sin(omega*t)/omega
    y3 = (1+cos(omega*t))/2
    norm2 = y1*y1+y2*y2+y3*y3
    print(" ".join(f"{v:.8f}" for v in (t, y1, y2, y3, norm2)))
```

[Kelly’s thermoelasticity chapter](https://pkel015.connect.amazon.auckland.ac.nz/SolidMechanicsBooks/Part_IV/Chapter_2_Applications/PDF/02_Thermomech_Applications_02_Thermoelasticity.pdf) supplies the thermodynamic background used in the original model. The characteristic cubic, invariant directions and fixed-parameter asymptotics are derived here from its stated three-coordinate reduction.

![Adiabatic persistent motion, a conduction-dependent relaxation spectrum and finite-time availability](/assets/learning/continuum-thermoelastic-relaxation-spectrum.en.svg)

### Research questions with explanations

**Does a nonincreasing availability prove that all coordinates decay?** **Answer:** No. Zero conduction preserves the whole norm, and zero coupling retains an undamped mechanical oscillator. Inspect the invariant trajectories or the characteristic roots.

**Must more conduction always damp mechanics faster?** **Answer:** No. At fixed coupling, very large d suppresses the thermal response and makes the mechanical damping scale as inverse d. The fast thermal root is not the slowest rate.

**Does an adiabatic stationary projection mean the trajectory converges to it?** **Answer:** No. The orthogonal component continues to oscillate. Projection and asymptotic convergence are different statements.

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
