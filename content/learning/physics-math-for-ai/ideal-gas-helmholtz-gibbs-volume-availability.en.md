## Which environment decides the direction?

An ideal gas can lower its Helmholtz free energy by expanding. Does that mean it must expand without limit inside a room with positive external pressure? The missing ingredient is the environment: moving a boundary also changes the mechanical energy exchanged with the pressure reservoir.

We will compare uniform gas states at the same temperature and amount. A curve of these states helps identify an equilibrium and compare endpoint budgets. It does not specify a moving piston's position as a function of time. The central question is which function includes the constraints that are actually held fixed.

## The controlled model

Let n be the amount in mol, T the absolute temperature in K, V the gas volume in m³, and P the fixed external pressure in Pa. The gas has the classical ideal-gas equation of state. Define the energy scale A:

$$
p_{\rm gas}(V)=\frac{nRT}{V},\qquad A=nRT>0.
$$

Heat into the gas and work by the gas are positive. Pressure times volume is an energy because Pa·m³=J. The numerical model uses n between 0.25 and 2 mol, T between 250 and 800 K, P between 25000 and 200000 Pa, and positive query/reference volumes between 0.001 and 1 m³. These are illustrative model bounds, not a claim that every real gas is ideal throughout this range.

We supply a reference volume Vr and a total reference Helmholtz value Fr between −10000 and 10000 J. Fr belongs to the selected n and T. Particle exchange, reactions, phase changes, piston inertia and a friction law are absent.

## Integrate the isothermal pressure

For a simple compressible equilibrium system at fixed amount, the Helmholtz differential gives the volume derivative. Integrating the ideal-gas pressure along this isotherm determines the volume dependence:

$$
\left(\frac{\partial F}{\partial V}\right)_{T,n}=-p_{\rm gas},\qquad
F(V)=F_r-A\ln\frac{V}{V_r}.
$$

Since pressure is positive, this F decreases as V increases. Its volume derivative never vanishes at a finite positive V. Looking for a minimum of F over unconstrained V therefore solves a different problem from mechanical equilibrium at a specified external pressure.

For the ideal gas, internal energy is independent of volume at fixed n,T. The entropy difference on the isotherm follows from ΔF=−TΔS:

$$
\Delta U_{V_r\to V}=0,\qquad
\Delta S_{V_r\to V}=nR\ln\frac{V}{V_r}.
$$

These are differences between equilibrium states. The entropy integral is evaluated along a reversible comparison path even if a different actual process is irreversible.

## What the reference does not determine

The input Fr specifies a value at the chosen n,T,Vr. It does not supply the full function Fr(T,n). Consequently this model does not report absolute internal energy, absolute entropy, or a heat capacity inferred by differentiating an invented temperature dependence.

Changing T in a new run changes the isothermal problem. Comparing absolute free energies across temperatures would require a consistent reference function as additional information. Changing n likewise requires a consistent amount dependence of that reference. Within one run, volume derivatives and differences are determined without that information. This is a useful distinction between integrating an equation of state and specifying a complete fundamental thermodynamic relation.

## Two functions with pressure terms

For a gas state at volume V, its internal pressure is pgas(V). Its Gibbs value at that pressure is F(V)+pgas(V)V. The fixed external pressure instead defines a constrained potential Φ:

$$
G_{\rm gas}(V)=F(V)+p_{\rm gas}(V)V=F(V)+A,\qquad
\Phi(V)=F(V)+PV.
$$

The first expression traces gas states whose internal pressures vary with V. The second keeps the environment's pressure P fixed while comparing possible volumes. They are not interchangeable away from mechanical equilibrium:

$$
\Phi(V)-G_{\rm gas}(V)=PV-A.
$$

At equilibrium pgas=P they agree. Calling every value of F(V)+PV an equilibrium Gibbs state at pressure P would silently assume the very pressure balance we are trying to determine.

## Equilibrium from force and curvature

The slope of Φ is the external-minus-internal pressure. It changes sign at a unique volume, while its curvature is strictly positive:

$$
\frac{d\Phi}{dV}=P-\frac{A}{V},\qquad
V_* = \frac{A}{P},\qquad
\frac{d^2\Phi}{dV^2}=\frac{A}{V^2}>0.
$$

Below V*, gas pressure exceeds external pressure and increasing V lowers Φ. Above V*, decreasing V lowers Φ. The sign describes an energetic tendency. It is not a velocity: a motion law would additionally need a piston area, mass, dissipation and initial data.

The equilibrium volume is generally not exactly representable as a binary floating-point number. Evaluating the model at its rounded displayed coordinate need not satisfy the pressure balance exactly.

## A nonnegative gap

Introduce the dimensionless ratio x=V/V*. Subtracting the analytic equilibrium potential produces a reference-independent excess:

$$
\Phi(V)-\Phi(V_*)=A\bigl(x-1-\ln x\bigr)\ge0,\qquad x>0.
$$

The inequality follows from ln x≤x−1, with equality only at x=1. Thus the excess is positive on both sides of the minimum. It measures how far this state lies above the equilibrium potential, not the sign of F itself.

Near the minimum the first-order terms cancel. With δ=x−1, the local shape is quadratic:

$$
x-1-\ln x=\frac{\delta^2}{2}-\frac{\delta^3}{3}+\frac{\delta^4}{4}-\cdots.
$$

A state can have a tiny nonzero pressure mismatch even when the two displayed potentials agree. The excess must be evaluated directly rather than obtained by subtracting rounded displayed energies.

## Natural variables of the equilibrium Gibbs function

Minimizing over volume gives the equilibrium Gibbs function at the selected external pressure:

$$
G_{\rm eq}(T,P)=F_r-A\ln\frac{A}{PV_r}+A,\qquad
\left(\frac{\partial G_{\rm eq}}{\partial P}\right)_{T,n}=\frac{A}{P}=V_*,\qquad
\left(\frac{\partial^2G_{\rm eq}}{\partial P^2}\right)_{T,n}=-\frac{A}{P^2}<0.
$$

There is no contradiction between positive curvature of Φ with respect to V and negative curvature of Geq with respect to P. In the first test V is an internal variable being optimized while P is fixed. In the second, P labels different imposed environments. “Minimize Gibbs energy at fixed pressure” does not mean “minimize over the imposed pressure.”

## A conditional entropy budget

Suppose the gas reaches V* through heat exchange with a reservoir at T and work against the constant external pressure P. Assume no other work and no net initial/final mechanical kinetic-energy store. These conditions determine endpoint budgets without specifying the transient motion:

$$
W_{\rm ext}=Q_{\rm gas}=P(V_*-V),\qquad
\Delta S_{\rm gas}=nR\ln\frac{V_*}{V},\qquad
\Delta S_{\rm bath}=-\frac{Q_{\rm gas}}{T}.
$$

Adding the two entropy changes yields a nonnegative production budget:

$$
\Delta S_{\rm universe}=\frac{\Phi(V)-\Phi(V_*)}{T}\ge0.
$$

The gas entropy alone may decrease during compression. The combined entropy still increases because the thermal reservoir receives heat. The quoted production is in J/K, not W/K, and gives no equilibration time. A real apparatus must account for every mechanical store and dissipation channel before using this endpoint budget.

## Reversible work as a comparison

A reversible isothermal path between the same gas endpoints uses a varying quasistatic load. It is a different protocol from sudden equilibration against a constant external pressure. Its work by the gas is −ΔF. The deficit of the constant-pressure protocol is exactly the positive potential excess:

$$
W_{\rm rev}=-A\ln x,\qquad
W_{\rm rev}-W_{\rm ext}=\Phi(V)-\Phi(V_*)=T\Delta S_{\rm universe}.
$$

For compression both work values are negative. A larger work-by-gas value then means a smaller required work input. A positive deficit does not require the device to produce positive work.

## A numerical example

Choose n=1 mol, T=256 K, Vr=V*=1/32 m³ and Fr=1000 J. Set P=nRT/V*, approximately 68112.078 Pa. Then A≈2128.502430 J. The pressure is selected consistently with the same numerical R used in the calculation.

| V/V* | F (J) | Φ (J) | Ggas (J) | Φ−Φ* (J) |
| --- | --- | --- | --- | --- |
| 1/2 | 2475.365 | 3539.617 | 4603.868 | 411.114 |
| 1 | 1000.000 | 3128.502 | 3128.502 | 0 |
| 2 | −475.365 | 3781.639 | 1653.137 | 653.137 |

The largest-volume state has the smallest F in this table but is not the fixed-pressure equilibrium. Compressing it to V* raises F by about 1475.365 J while lowering Φ by about 653.137 J. Identifying the controlled environment resolves the apparent contradiction.

## Read nearby states and references carefully

A plot of F slopes downward; a plot of Φ has its minimum at V*; the force plot crosses zero at the same exact balance. Pressure, slope, energy and entropy have different units and should have separate axes or clearly stated normalizations. Moving a marker between states is not automatically a physical relaxation animation.

Adding a constant C to Fr adds C to F, Ggas, Φ and Geq. It changes none of their volume derivatives, the equilibrium volume, the excess or the conditional entropy-production budget. An absolute negative F is therefore not a standalone instability criterion.

### Reproduce the table in the experiment

Set n=1, T=256, P=68112.07776791134, Vr=0.03125 and Fr=1000 in the model settings. This pressure uses the same encoded R as the calculation and makes V*=0.03125 exactly in this specified model. Try V=0.015625, 0.03125 and 0.0625 in that order. Compare F, Φ and Ggas, then inspect Φ−G* and Sgen. At the middle volume the pressure balance and excess are structurally zero; the two neighboring preparations have a positive excess. The default “Expanded state” preset instead uses T=300 K and P=100000 Pa, so it is a different isotherm from the numerical table.

The horizontal coordinate is log₁₀(V/m³). A shift by one means ten times the volume, not one additional cubic metre. The energy ordinate is rescaled by the factor printed beside the figure: multiply by that factor to recover joules. The 33-state scan changes only V on the selected isotherm. Compare compression and expansion manually before playing it; neither the scan index nor the speed determines a piston velocity. Reset or reload a preset to restore the initial preparation.

The numerical calculation also distinguishes a structural zero from a cancellation involving rounded logarithms. For example, the binary64 number `Math.E` is not the exact mathematical e, although `Math.log(Math.E)` rounds to 1. A constructed anchored free energy can consequently evaluate to 0 numerically without being a proven physical zero. That cancellation is reported as unresolved. The positive equilibrium excess is evaluated separately from absolute anchored energies.

The tiny examples test the encoded mathematical model, including its supplied binary inputs and approximate R. They do not establish experimental accuracy beyond the precision of physical parameters or constants. Numerical resolution and physical uncertainty are separate questions.

## Research extension: an inertial piston can pass through the potential minimum

Place a piston exactly at the pressure-balanced volume, but give it an outward velocity. The potential slope is zero, yet the piston keeps moving. Some kinetic energy becomes potential excess. A potential minimum specifies a resting equilibrium, not every state that happens to pass that coordinate.

We explicitly add a deterministic isothermal model. Keep A=nRT>0, fixed external P>0 and positive volume. A uniform ideal gas remains at an ideal thermostat temperature T; a piston has mass m, constant area $A_p$ and linear viscous coefficient b. All viscous heat goes to that same thermostat. The external pressure load is an ideal work store. Noise, finite heat-transfer response, dry friction and other mechanical stores are absent. This is a macroscopic dynamical idealization, not the canonical fluctuation law of a microscopic piston.

Choose the piston coordinate z=V/$A_p$. Dividing Newton's equation by the piston area gives the effective volume inertia $m_V$ in kg/m⁴ and damping ζ in Pa·s/m³. The force per area is internal minus external pressure.

$$
m_V\ddot V+\zeta\dot V=A/V-P,\qquad m_V=m/A_p^2>0,\quad\zeta=b/A_p^2\ge0.
$$

Add the piston kinetic energy to the original constrained-potential excess. Both terms have joule units. Knowing V alone no longer specifies this available mechanical energy.

$$
K=\tfrac12m_V\dot V^2,\qquad\mathcal L=K+\Phi(V)-\Phi(V_*),\qquad V_*=A/P.
$$

Differentiate along the added mechanical law. Total available energy decreases through viscous dissipation. The potential part alone need not decrease: it can receive kinetic energy during overshoot.

$$
\frac{d\mathcal L}{dt}=\dot V[m_V\ddot V+P-A/V]=-\zeta\dot V^2\le0.
$$

Scale volume by $V_*$ and time by the inertial time $t_*$. The dimensionless damping d compares viscous and inertial effects. Primes below denote derivatives in τ, while dots in the preceding dimensional equations denote seconds.

$$
t_*=\sqrt{m_V V_*^2/A},\quad x=V/V_*,\quad\tau=t/t_*,\quad d=\zeta t_*/m_V.
$$

The dimensionless state includes x and its velocity. The potential h retains the same minimum as the original landscape, while the kinetic term supplies a second state coordinate.

$$
x^{\prime\prime}+d x^\prime=1/x-1,\qquad h(x)=x-1-\log x,\qquad e=\mathcal L/A=\tfrac12(x^\prime)^2+h(x).
$$

Only total e has the nonpositive derivative. The derivative of h can have either sign. In the example x(0)=1 and x′(0)=0.8, h starts at zero and subsequently grows while e decreases.

$$
e^\prime=-d(x^\prime)^2,\qquad h^\prime(\tau)=(1-1/x)x^\prime.
$$

Linearization near x=1 gives damped-oscillator roots. For 0<d<2 the local motion is underdamped; this approximation requires small displacement. In the full model h diverges as x approaches zero or infinity, so a finite-energy continuous solution stays in a positive bounded volume range. For d>0 its only invariant zero-dissipation state is rest at x=1, giving asymptotic relaxation. For d=0 nontrivial trajectories conserve energy rather than relax.

$$
x=1+y:\quad y^{\prime\prime}+d y^\prime+y=0\quad\text{to first order},\qquad r_\pm=(-d\pm\sqrt{d^2-4})/2.
$$

The thermal boundary must match the added store. Let $Q_{\rm sys}$ be net heat from the thermostat into gas plus piston. It includes heat entering the isothermal gas and viscous heat returned to the same bath. Internal gas energy is constant, so the change in piston kinetic energy remains in the first law. This net heat differs from heat into the gas alone.

$$
Q_{\rm sys}=P(V_f-V_i)+(K_f-K_i).
$$

Gas entropy changes by nR log($V_f$/$V_i$); the ideal mechanical piston and work store add no entropy. Combining that change with −$Q_{\rm sys}$/T gives the production integral below. It includes the kinetic-energy difference that was absent from the original no-mechanical-store endpoint example.

$$
\Sigma=nR\log(V_f/V_i)-Q_{\rm sys}/T=\frac{\mathcal L_i-\mathcal L_f}{T}=\int\frac{\zeta\dot V^2}{T}\,dt\ge0.
$$

After complete relaxation with positive damping, the bath has received the initial available energy as dissipation measured by TΣ. Starting at $V_*$ with nonzero velocity therefore still produces entropy. The old potential-only expression is recovered when initial and final kinetic stores vanish. None of this gives a finite equilibration time or validates arbitrary numerical time steps.

$$
\Sigma_\infty=\frac{K_i+\Phi(V_i)-\Phi(V_*)}{T}\qquad(\zeta>0,\ V_f\to V_*,\ \dot V_f\to0).
$$

### Read overshoot and the decreasing total

The upper panel plots x versus inertial time τ for d=0.4 and initial (x,x′)=(1,0.8). The lower panel separates h, kinetic energy divided by A, and their total e. Rising h is compatible with decreasing e. The original reader still compares static volumes; the extra clock belongs only to this declared model.

![An initially moving piston crosses the pressure-balanced volume; potential excess may rise while kinetic plus potential energy decreases through damping.](/assets/learning/free-energy-inertial-piston-budget.en.svg)

### Reproduce four numerical states

The supplement uses fixed-step RK4 with Δτ=0.001, not the existing reader calculator. x and x′ are dimensionless, h and e are normalized by A, and the last column is TΣ/A. These are numerical approximations checked against an independent local Taylor expansion and step refinement. Monitor positive volume and the energy budget; a general positivity or error certificate is not supplied.

| τ | x | x′ | h(x) | Total e | TΣ/A |
| --- | --- | --- | --- | --- | --- |
| 0.00000000 | 1.00000000 | 0.80000000 | 0.00000000 | 0.32000000 | 0.00000000 |
| 0.50000000 | 1.34988601 | 0.58551421 | 0.04986586 | 0.22127930 | 0.09872070 |
| 1.00000000 | 1.57967682 | 0.33320315 | 0.12245654 | 0.17796871 | 0.14203129 |
| 3.00000000 | 1.42541791 | -0.36864137 | 0.07095287 | 0.13890110 | 0.18109890 |

```python
from math import log
d, x, v, dt = 0.4, 1.0, 0.8, 0.001
initial = v*v/2+x-1-log(x)
def rhs(x, v):
    return v, 1/x-1-d*v
for step in range(3001):
    if step in [0, 500, 1000, 3000]:
        h = x-1-log(x)
        energy = v*v/2+h
        generated = initial-energy  # T*Sigma/A
        print(f"{step*dt:.8f} {x:.8f} {v:.8f} {h:.8f} "
              f"{energy:.8f} {generated:.8f}")
    if step == 3000:
        break
    a, b = rhs(x, v)
    c, f = rhs(x+dt*a/2, v+dt*b/2)
    g, j = rhs(x+dt*c/2, v+dt*f/2)
    k, l = rhs(x+dt*g, v+dt*j)
    x += dt*(a+2*c+2*g+k)/6
    v += dt*(b+2*f+2*j+l)/6
```

### Three research questions with explanations

#### 1. Why can h increase after starting at its minimum?

**Explanation.** The piston starts with kinetic energy and positive velocity. Pressure balance cancels the pressure force at that coordinate, not the velocity. Kinetic energy can move into h while viscosity reduces their total. Position-only data would miss the energy store driving this overshoot.

#### 2. Does the same equilibrium imply the same trajectory or convergence?

**Explanation.** No. Inertia, damping and initial velocity are extra information. Positive damping gives relaxation in the stated deterministic model; zero damping generally preserves oscillation energy. The landscape alone chooses none of these dynamics, and adding thermal fluctuations would require a different stochastic model.

#### 3. When is the old potential-only entropy budget insufficient?

**Explanation.** When a mechanical store has a nonzero initial or final change. Here total production depends on the decrease of K plus constrained potential, and net thermostat heat equals pressure-load work plus ΔK. If both kinetic endpoints vanish, the earlier conditional endpoint expression is recovered.

### Source and research boundary

[David Tong, Classical Thermodynamics, section 4.4](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) supplies the fixed-environment potential convention. Newtonian force balance, the viscous law and thermostat routing are additional assumptions derived into the displayed energy and entropy balances here. This supplement does not establish finite-rate heat-transfer feasibility or identify an arbitrary learned gradient flow with physical piston motion.

## Research interpretation

The same discipline applies when building an optimization objective: specify what is fixed, which variables are free, and what each added term means. A pressure-volume term has units, a reservoir interpretation and a thermodynamic derivation. An arbitrary machine-learning regularizer does not inherit those properties just because it changes an objective's minimum.

The potential landscape determines equilibria and local stiffness, while a trajectory requires additional dynamics. Gradient descent on Φ can be an algorithm for finding the minimizer; without a specified mobility or mechanical equation it is not a solved piston motion. Nonideal equations of state, mixtures, phase coexistence and chemical potentials require further models.

## Exercises with worked answers

### Exercise 1: Recover the isotherm

Given pgas=A/V at fixed n,T and F(Vr)=Fr, derive F(V) and the gas entropy difference from Vr.

**Answer:** Integrate dF=−A dV/V to obtain Fr−A ln(V/Vr). Since ΔU=0 on this ideal-gas isotherm, ΔF=−TΔS gives ΔS=nR ln(V/Vr). The calculation determines a difference; it does not determine absolute entropy from the supplied anchor.

### Exercise 2: Test both sides of the minimum

Find the slope signs and normalized excesses at V=V*/2 and V=2V*.

**Answer:** At half volume, pgas=2P and dΦ/dV=−P; at double volume, pgas=P/2 and dΦ/dV=P/2. The excesses divided by A are ln2−1/2 and 1−ln2, respectively. Both are positive, consistent with the strictly positive volume curvature. F itself remains decreasing in V.

### Exercise 3: Account for compression entropy

For initial V=2V*, evaluate the constant-pressure endpoint budget. Explain why an increasing F does not violate the second law here.

**Answer:** Work and heat into the gas are both −A. The gas entropy change is −nR ln2, the bath change is nR, and total production is nR(1−ln2)>0. In the numerical example it is about 2.551316 J/K. The gas F rises by A ln2, but the fixed-pressure constrained potential falls by A(1−ln2). The relevant combined-system budget includes pressure work.

### Exercise 4: Shift the energy reference

Replace Fr by Fr+C at the same n,T,Vr. Which physical comparisons change?

**Answer:** All four anchored energies shift by C. Their slopes, the pressure mismatch, V*, Φ−Φ*, work differences and entropy differences remain unchanged. An additive reference must not change an equilibrium classification. This statement concerns a constant shift within the isothermal comparison, not an unspecified temperature-dependent reference function.

### Exercise 5: Interpret matching displays

A nearby state has δ≈10⁻¹⁶, yet both displayed potentials agree. Is the potential excess or entropy production necessarily zero? What if a separately constructed anchored F computes to 0 through `Math.log(Math.E)`?

**Answer:** The excess is approximately Aδ²/2, about 10⁻²⁹ J for the example, and remains positive. Production is that excess divided by T. Matching rounded energies cannot establish equilibrium. An anchored numerical cancellation is a different issue: the logarithm is approximate, so the zero is unresolved unless the defining structural conditions establish it. Neither case justifies snapping the state to equilibrium.

### Exercise 6: Explain Gibbs concavity

Differentiate Geq with respect to the imposed pressure at fixed n,T. Does the negative second derivative contradict volume stability?

**Answer:** Differentiating the logarithm gives dGeq/dP=A/P=V* and d²Geq/dP²=−A/P². Volume stability instead tests d²Φ/dV²=A/V²>0 at a fixed P. The variables varied and the environments held fixed are different, so the two curvature signs are consistent.

## Sources

[David Tong, Classical Thermodynamics, section 4.4](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) discusses thermodynamic potentials and their natural variables. The pressure/force integrals, finite-precision examples and conditional budgets above are worked derivations for this specified model.

[NIST CODATA constant tables](https://physics.nist.gov/cuu/pdf/all.pdf) give the SI molar gas constant. Calculations here use its binary64 approximation R=8.31446261815324 J/(mol K).
