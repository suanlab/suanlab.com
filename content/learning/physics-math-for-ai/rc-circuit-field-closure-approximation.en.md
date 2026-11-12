## Does a correct circuit guarantee correct fields?

A neural model might predict the voltages of a resistor-capacitor circuit accurately. Does that establish accurate E and B throughout the device? A circuit compresses complicated internal fields into terminal quantities. Conservation at that reduced level matters, but it does not validate arbitrary spatial fields reconstructed afterwards. This unit keeps the exact solution of an ideal circuit separate from a particular approximate field reconstruction. We will find an example where the circuit balances work while a field equation retains a nonzero residual.

## Assumptions and prerequisites

Review charge conservation, capacitance, electrical power, cylindrical Ampere law, Faraday law, and second time derivatives. Assume positive R and C, constant supply voltage, and a specified initial capacitor voltage. The gap reconstruction uses circular parallel plates with constant positive epsilon and mu. Fringing, wires, distributed inductance, propagation, radiation, and complete device boundary data remain outside this ideal lumped model. These omissions are part of its definition. They should be assessed when applying the model rather than hidden inside a numerical tolerance.

## Plate charge and capacitance

Let the plate radius be a and the gap be d. Use a uniform axial electric field V/d between the plates. Multiplying displacement by area gives the charge, so capacitance follows from geometry and constitutive data. Choose positive current to increase the positive plate charge. This sign convention will also determine the energy transfer signs. The formula is the familiar parallel-plate approximation, not an exact statement about every finite plate geometry. The ratio of gap to radius will help us discuss one limitation of that approximation.

$$
A=\pi a^2,\qquad E_z=V/d,\qquad q=\epsilon A E_z=CV,\qquad C=\epsilon A/d.
$$

## From charge conservation to the RC equation

The series current equals the capacitor charge rate, while the resistor voltage drop is RI. Combining those terminal relations gives the RC differential equation. This derivation already assumes that a lumped terminal description is appropriate. In a general time-dependent electromagnetic configuration, electric circulation can be nonzero and arbitrary path integrals cannot all be replaced by one scalar voltage difference. The circuit equation is therefore a reduced closure with a stated scope, not a replacement for checking the spatial equations when those fields are also being claimed.

$$
I=\frac{dq}{dt}=C\dot V,\qquad V_s=V+RI,\qquad RC\dot V+V=V_s.
$$

## Separate initial data and the time scale

Define the time constant T=RC and normalized time theta=t/T. Theta equal to one means one RC time constant, not one second. Initial voltage and equilibrium voltage supply different information. If they are equal, there is no transient current, although a static electric field can remain in the gap. A zero supply also does not imply zero initial charge: a previously charged capacitor can discharge through the resistor. These distinctions are useful checks against controls that accidentally replace one physical experiment with another.

$$
T=RC,\quad \theta=t/T,\quad V(t)=V_s+(V_i-V_s)e^{-\theta},\quad I(t)=\frac{V_s-V_i}{R}e^{-\theta}.
$$

## Retain a small voltage change

For very small theta, directly evaluating one minus exp(-theta) can subtract two numbers that have rounded to equality. Expm1 preserves the small change when it is representable. Recording voltage change separately also helps when the final voltage rounds back to the initial value. This does not recover information already lost when the input voltages were rounded to the same number. An unrepresentable nonzero change should remain unavailable rather than be reported as a physical zero. Normalized time, physical time, voltage, and voltage change can have different representation limits.

$$
\Delta V=(V_s-V_i)[-\operatorname{expm1}(-\theta)],\qquad V=V_i+\Delta V.
$$

## Read the direction of power transfer

Differentiating the capacitor electric energy gives VI. Source power is Vs times I and can be negative: the ideal voltage source may absorb energy. Resistive heating is RI squared and is nonnegative. Check these signs in discharging cases and when the initial voltage exceeds the supply. Removing a negative source power as an error would erase a legitimate direction of energy transfer. The balance below is the circuit model balance; it should not silently be expanded into a claim that all magnetic and radiated energy of an actual device has been included.

$$
U_E=\tfrac12CV^2,\qquad \dot U_E=VI,\qquad P_s=V_sI,\qquad P_R=RI^2,\qquad P_s=P_R+\dot U_E.
$$

## Compute accumulated work and heat

Integrate the power curves to compare supplied work, heat, and storage change. The storage change can be evaluated through delta V times the sum of the initial and final voltages, avoiding subtraction of large nearby stored energies. This is a stable rearrangement within the same analytic family, not independent certification. Solving the initial-value problem numerically and integrating its powers provides another computational route. That route still has numerical error. Do not confuse a small observed residual with a rigorous error bound for all parameter choices or a complete real-device energy budget.

$$
W_s=V_sC\Delta V,\quad Q_R=\tfrac12C(V_s-V_i)^2(1-e^{-2\theta}),\quad \Delta U_E=\tfrac12C\Delta V(2V_i+\Delta V),\quad W_s=Q_R+\Delta U_E.
$$

## Worked example: the half-charged point

For an ideal circuit take R=2 ohms, C=2 farads, supply one volt, and initial voltage zero. At theta=ln(2), physical time is 4ln(2) seconds, voltage is half a volt, and current is a quarter ampere. Source power is 0.25 watt; half becomes resistor heating and half increases electric storage. Accumulated source work is one joule, heat is 0.75 joule, and stored-energy change is 0.25 joule. These circuit numbers do not by themselves establish the geometric validity of a particular parallel-plate device.

$$
R=2,\ C=2,\ V_s=1,\ V_i=0,\ \theta=\ln2:\quad V=\tfrac12,\ I=\tfrac14,\ W_s=1,\ Q_R=\tfrac34,\ \Delta U_E=\tfrac14.
$$

## Reconstruct B from displacement current

Inside the gap there is no bulk conduction current in this reconstruction, but a changing electric field produces displacement current. Apply cylindrical symmetry to an Ampere loop of radius r within the plate radius. The resulting B-phi is proportional to r. Displacement current across the full plate area agrees with the circuit current. This construction uses Ampere law and the assumed uniform E. It has not yet checked Faraday law. That missing check is essential before describing the reconstructed pair as an exact Maxwell solution.

$$
I_D(r)=\epsilon\pi r^2\dot V/d,\qquad 2\pi r B_\phi=\mu I_D(r),\qquad B_\phi=\frac{\mu\epsilon r}{2d}\dot V,\qquad I_D(a)=C\dot V=I.
$$

## The remaining Faraday residual

The spatially uniform axial E has zero curl. The reconstructed B depends on the voltage rate, so it changes when voltage acceleration is nonzero. Consequently curl E plus the B time derivative remains nonzero. The symmetry axis gives a pointwise zero because r is zero there; it does not establish compatibility over the gap. If initial and supply voltages match, the family is static and this residual vanishes everywhere. The lesson is to inspect the actual retained and omitted equations, rather than treating every discrepancy as a numerical integration failure.

$$
(\nabla\times\mathbf E+\partial_t\mathbf B)_\phi=\frac{\mu\epsilon r}{2d}\ddot V,\qquad \ddot V=-\frac{V_s-V_i}{(RC)^2}e^{-\theta}.
$$

## Where does the reconstructed magnetic energy belong?

Integrating the reconstructed magnetic energy density over the gap gives a separate local magnetic-energy estimate. It is an integral of this approximate field, not the total magnetic energy of the device. Simply adding it to the capacitor storage budget does not automatically preserve the original circuit balance. An extended model needs consistent inductance and boundary power as well. The magnetic-to-electric energy ratio is undefined when electric storage is zero. For example, an ideal charging start can have zero capacitor voltage and a nonzero reconstructed B, highlighting the limits of the instantaneous lumped initialization.

$$
U_{B,\rm gap}=\int_{\rm gap}\frac{B^2}{2\mu}\,d^3x=\frac{\mu\epsilon^2\pi a^4}{16d}\dot V^2.
$$

## Two length ratios for assessing the approximation

The ratio a/(cT) compares plate size with the distance a disturbance propagates in one RC time. Gap divided by radius addresses a different geometric approximation. A small value of one does not settle the other or control propagation along long external wires. These parameters guide an order-of-magnitude assessment; they are not universal rigorous error bounds for arbitrary devices. Varying R can change the transient time and the field reconstruction defect while preserving the same normalized circuit solution. That comparison makes the difference between reduced-model accuracy and field accuracy concrete.

$$
c=1/\sqrt{\mu\epsilon},\qquad \delta=\frac{a}{cRC},\qquad \gamma=\frac{d}{a}.
$$

## The ideal step and zero limits

An ideal voltage step sets the circuit current immediately in the lumped description. Finite propagation and compatible initial field data of an actual device are not part of that claim. Distinguish the positive-RC circuit solution from a physical account of the switching event, and examine the smooth transient away from that event. Setting R or C exactly to zero changes the mathematical problem and can invalidate the original initial condition. Do not implement that limit by silently inserting a small positive constant. Zero drive, static equilibrium, an axial query, and arithmetic unavailability are also different cases.

$$
V(0)=V_i,\qquad I(0)=(V_s-V_i)/R,\qquad \lim_{t\to\infty}V(t)=V_s.
$$

## Independent calculations for verification

Solve the voltage ODE numerically and integrate source power minus heating to compare with storage change. Then differentiate the reconstructed fields in space and time to check Ampere and Faraday separately. Circuit and field residuals have different units and should be displayed with their own scales. A zero obtained by writing the same analytic relation twice is not independent verification. The selected numerical cases supply evidence within the declared family; they do not certify every material, geometry, or complete device. This is a useful pattern for evaluating physics-informed models that combine circuit outputs and spatial predictions.

## Specify what success means for an AI model

If the target is terminal-voltage prediction, check the circuit equation, initial condition, and power budget. If the model also claims E and B, check the spatial equations, sources, boundaries, propagation, and approximation conditions. Combining these different targets into one loss can hide an important failure. The background on lumped elements and low-frequency assumptions is in [Feynman II §22–1](https://www.feynmanlectures.caltech.edu/II_22.html); the distinction between quasistatic equation sets is discussed in [Haus and Melcher §3.2](https://web.mit.edu/6.013_book/www/chapter3/3.2.html). The explicit RC reconstruction and counterexample here follow from the declared model.

## Research extension: passive polarization memory and initial layers

### A capacitor can carry an internal memory

The ideal capacitor uses charge C V as an instantaneous relation. Add one relaxing polarization charge p in coulombs, distinct from a polarization density vector. The instantaneous capacitance is Cinf; DeltaC is the extra low-frequency capacitance. This is a supplementary lumped constitutive model with a constant ideal supply and positive parameters, not a new field solver or an additional control in the existing RC lab. The terminal current charges both the instantaneous part and the internal branch.

$$
Q=C_\infty V+p,\qquad I=\dot Q=(V_s-V)/R,\qquad C_\infty,\Delta C,R,\tau_p>0.
$$

### Build the branch from passive circuit elements

Connect a capacitor DeltaC in series with a resistor tau_p/DeltaC, and put that series branch in parallel with Cinf. The branch capacitor voltage is p/DeltaC and its current is p-dot. Its resistor drop supplies the remaining voltage, yielding the first equation below. Terminal charge conservation then gives the second. Two initial data, Vi and pi, are now required. Specifying the initial terminal voltage alone does not determine the internal state. All voltages and currents keep the original passive sign convention.

$$
\tau_p\dot p+p=\Delta C\,V,\qquad \dot V=\frac{(V_s-V)/R-(\Delta C\,V-p)/\tau_p}{C_\infty}.
$$

### Write its causal memory explicitly

Solving the first-order polarization equation gives a decaying preparation term plus an integral over past voltage. Only values up to the observation time appear. This makes causality explicit for this model rather than inferring it from a fit at one frequency. If V changes rapidly compared with tau_p, p cannot follow its equilibrium value DeltaC V immediately. Conversely, replacing p by DeltaC V is an approximation requiring a time scale and compatible initial preparation. The model omits propagation, fringing and the microscopic mechanisms behind its chosen relaxation time.

$$
p(t)=p_i e^{-t/\tau_p}+\frac{\Delta C}{\tau_p}\int_0^t e^{-(t-u)/\tau_p}V(u)\,du.
$$

### Storage and dissipation are separate

Multiply the terminal current by V and use the branch equation. The instantaneous and branch capacitors store the two terms of U; the internal resistor dissipates Pd. Adding the external resistor gives the full lumped balance. Pd is nonnegative even when the ideal source absorbs rather than supplies power. Using only one-half of the low-frequency capacitance times V squared would miss an independent internal state away from equilibrium. This consistent enlarged circuit budget still does not make the original approximate gap E/B reconstruction an exact whole-device Maxwell solution.

$$
U=\tfrac12C_\infty V^2+\frac{p^2}{2\Delta C},\quad P_d=\frac{\tau_p}{\Delta C}\dot p^2\ge0,\qquad V_sI=RI^2+\dot U+P_d.
$$

### Check the same passivity in frequency space

For the admittance take zero initial state and a Laplace variable s. For peak phasors with exp(−i omega t), substitute s=−i omega. Its real part is nonnegative, so the average internal input power is one-half Re(Y) times voltage magnitude squared. Low-frequency capacitance tends to Cinf+DeltaC, while the coefficient of the leading high-frequency imaginary term tends to Cinf. The internal loss remains distinct from the external R loss. A fitted single branch need not describe every frequency or every physical dielectric.

$$
Y(s)=sC_\infty+\frac{s\Delta C}{1+s\tau_p},\qquad \operatorname{Re}Y(-i\omega)=\frac{\Delta C\,\tau_p\omega^2}{1+\omega^2\tau_p^2}\ge0.
$$

### Two stable rates replace one RC rate

Subtract the equilibrium V=Vs, p=DeltaC Vs and form the two-by-two linear system. Its decay-rate polynomial is lambda squared minus S lambda plus eta rho. The displayed positive discriminant and product make both rates real, distinct and positive. Compute the slow rate through the product formula to avoid subtracting nearly equal positive numbers. A sum of two decaying modes can show changing slope on a semilog plot; treating that trace as one exponential can erase the polarization state. This passive RC network has no oscillatory natural modes.

$$
\eta=(RC_\infty)^{-1},\quad\rho=\tau_p^{-1},\quad\beta=\Delta C/C_\infty,\quad S=\eta+(1+\beta)\rho,\quad d=(\eta-\rho)^2+2\beta\rho(\eta+\rho)+\beta^2\rho^2>0,\quad\lambda_f=\frac{S+\sqrt d}{2},\quad\lambda_s=\frac{2\eta\rho}{S+\sqrt d},\quad\lambda_s\lambda_f=\eta\rho.
$$

### Follow a fully specified discharge

The table uses the stated SI parameters and an initially unpolarized branch. The source voltage is zero, but the instantaneous capacitor initially stores half a joule. Polarization charge first grows and later decays while total storage decreases. The code computes the two modes, both state variables, storage and instantaneous internal loss; current is minus V for R=1 ohm. Independent ODE integration and power-history integration provide another route to the state and energy balance. They retain numerical error and do not constitute a general device error bound.

$$
C_\infty=\Delta C=1\,\mathrm F,\ R=1\,\Omega,\ \tau_p=1\,\mathrm s,\ V_s=0,\ V_i=1\,\mathrm V,\ p_i=0:\quad\theta=t/(1\,\mathrm s),\quad r_{s,f}=\frac{3\mp\sqrt5}{2},\quad\lambda_{s,f}=r_{s,f}/(1\,\mathrm s),\quad\frac{V}{1\,\mathrm V}=\frac{r_f-2}{\sqrt5}e^{-r_s\theta}+\frac{2-r_s}{\sqrt5}e^{-r_f\theta},\quad\frac{p}{1\,\mathrm C}=\frac{e^{-r_s\theta}-e^{-r_f\theta}}{\sqrt5}.
$$

### Fast relaxation can change the effective initial voltage

Take tau_p toward zero while keeping R, Cinf, DeltaC and a bounded supply fixed. During the short internal layer, the external current transfers vanishing net charge to leading order, so Q is approximately conserved. Internal redistribution selects V_after, which need not equal Vi. The layer time follows from the isolated branch imbalance V−p/DeltaC. The formulas for the selected state are exact for the isolated fixed-charge redistribution; their use as initial matching data for the full finite-R circuit is a singular-limit statement, not exact equality at a finite tau_p.

$$
Q_i=C_\infty V_i+p_i,\qquad V_{\rm after}=\frac{Q_i}{C_\infty+\Delta C},\qquad p_{\rm after}=\Delta C\,V_{\rm after},\qquad t_{\rm layer}\sim\frac{\tau_p C_\infty}{C_\infty+\Delta C}.
$$

### The layer can dissipate finite energy

Subtract the storage at the matched state from the prepared storage. The nonnegative square below is exactly the heat of complete isolated redistribution. Thus an arbitrarily short layer can dissipate finite energy when preparation is incompatible. For Cinf=DeltaC=1, Vi=1 and pi=0, the voltage after the layer tends to one-half and the lost storage is one-quarter joule. If pi=DeltaC Vi, this preparation loss is zero. A calculation that simply keeps Vi while changing the total capacitance silently changes the initial experiment.

$$
U_i-U_{\rm after}=\frac{C_\infty\Delta C}{2(C_\infty+\Delta C)}\left(V_i-\frac{p_i}{\Delta C}\right)^2\ge0.
$$

### Match the outer reduced circuit, then assess fields separately

At fixed positive time outside the shrinking layer, the reduced capacitance is Cinf+DeltaC and the initial matching value is V_after. The limits t toward zero and tau_p toward zero need not commute. Setting tau_p exactly to zero removes a differential equation; it is not an ordinary evaluation of the two-state formulas. Removing DeltaC is a separate branch-removal model requiring no remaining branch charge. The native ideal RC lab is that no-branch comparison. Its field compatibility, propagation scale and missing Faraday equation still need their original checks.

$$
R(C_\infty+\Delta C)\dot V_{\rm outer}+V_{\rm outer}=V_s,\qquad V_{\rm outer}(t)=V_s+(V_{\rm after}-V_s)e^{-t/[R(C_\infty+\Delta C)]}\quad(t>0\text{ fixed as }\tau_p\to0).
$$

| t (s) | V (V) | p (C) | U (J) | Pd (W) |
| --- | --- | --- | --- | --- |
| 0.00000000 | 1.00000000 | 0.00000000 | 0.50000000 | 1.00000000 |
| 0.25000000 | 0.62727759 | 0.17406835 | 0.21188848 | 0.20539862 |
| 1.00000000 | 0.24142772 | 0.27260894 | 0.06630149 | 0.00097227 |
| 4.00000000 | 0.05999745 | 0.09703209 | 0.00650746 | 0.00137157 |

```python
from math import sqrt, exp
# Cinf=DeltaC=1 F, R=1 ohm, tau_p=1 s, Vs=0, Vi=1 V, pi=0 C.
root = sqrt(5)
slow, fast = (3-root)/2, (3+root)/2
for t in [0, 0.25, 1, 4]:
    es, ef = exp(-slow*t), exp(-fast*t)
    voltage = (fast-2)/root*es + (2-slow)/root*ef
    polarization_charge = (es-ef)/root
    energy = (voltage**2 + polarization_charge**2)/2
    dielectric_power = (voltage-polarization_charge)**2
    values = [t, voltage, polarization_charge, energy, dielectric_power]
    print(" ".join(f"{x:.8f}" for x in values))
```

[Feynman II §22–1](https://www.feynmanlectures.caltech.edu/II_22.html) and [Haus–Melcher §3.2](https://web.mit.edu/6.013_book/www/chapter3/3.2.html) provide lumped-circuit and quasistatic context. The passive relaxing branch, its energy, rates and initial-layer matching are derived here from the displayed circuit equations.

![Relaxing capacitor states, two dissipation channels and initial-layer matching](/assets/learning/electromagnetism-relaxing-capacitor-energy.en.svg)

### Research questions with explanations

**Is initial voltage enough to specify the relaxing capacitor?** **Answer:** No. The polarization charge is an independent state; identical Vi with different pi gives different subsequent voltage and internal loss.

**Does a vanishing layer duration imply vanishing heat?** **Answer:** No. An incompatible fixed-charge preparation can lose the finite excess storage shown by the nonnegative square.

**Does the enlarged circuit balance validate the gap Maxwell fields?** **Answer:** No. It validates this lumped storage-and-loss model. The separate field reconstruction still needs sources, boundaries and both curl equations.

## Exercises with explanations

### Exercise 1

If initial and supply voltage both equal 2 V, must both current and gap E vanish?

**Answer:** Current is zero, but E=2/d remains. Absence of a transient does not imply absence of a field.

### Exercise 2

For R=2ohms and C=2farads, what physical time is theta=1?

**Answer:** T=RC=4seconds, so theta=1 means4seconds. Normalized time itself does not carry seconds.

### Exercise 3

At the half-charged point of the worked example, find storage power.

**Answer:** VI=0.5 times0.25=0.125W, equal to source0.25W minus heating0.125W.

### Exercise 4

Does zero circuit energy residual certify the reconstructed E/B?

**Answer:** No. The Faraday residual generally remains proportional to voltage acceleration. Circuit conservation and field compatibility require separate checks.

### Exercise 5

If electric storage is zero but reconstructed magnetic storage is nonzero at initialization, what is UB/UE?

**Answer:** It is undefined because the denominator is zero. Do not report a measured ratio of zero; also examine the initialization approximation.

### Exercise 6

How should a zero direct voltage change and nonzero expm1 change be interpreted?

**Answer:** Distinguish loss against a rounded background from a representable small change. Already-identical input values cannot be recovered, and unrepresentable changes are not exact physical zeros.
