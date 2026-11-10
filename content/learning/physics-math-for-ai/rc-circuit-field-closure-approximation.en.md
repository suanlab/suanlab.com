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
