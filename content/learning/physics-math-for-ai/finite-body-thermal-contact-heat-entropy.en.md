## Does a hotter object necessarily transfer more energy?

Put a small hot spoon into a large amount of warm water. The spoon's temperature can change rapidly while the water's changes very little. Temperature does not measure total energy. The energy required for a temperature change depends on the material and its amount. Heat capacity C, measured in J/K, describes the change in internal energy per kelvin under the conditions specified here.

We retain just two bodies. **Heat transfer** names the process by which energy crosses their boundary because of a temperature difference. Internal energy belongs to the state of a body; heat describes a mode of transfer. We do not introduce a separate stored quantity called heat content.

## Specify the system before solving it

Each body has a spatially uniform temperature and a constant positive heat capacity C₁ or C₂. The combined system is insulated from its surroundings. There is no volume change, matter transfer, mechanical work or phase change. Temperatures are positive absolute temperatures in kelvin. The connection has thermal conductance K in W/K; the symbol K for this quantity is distinct from the temperature unit kelvin.

This lumped approximation is useful when internal temperature equilibration is much faster than the changes caused by the connection. It does not resolve temperature fields inside a large body or diffusion near a contact surface.

## From energy balance to a time-dependent solution

Take heat transfer from body 1 to body 2 as positive:

$$
\Phi=K(T_1-T_2),\qquad C_1\dot T_1=-\Phi,\qquad C_2\dot T_2=\Phi.
$$

The transfer rate Φ is a power in W=J/s. Adding the two balances shows that total internal energy does not change. Subtracting the temperature equations gives an equation for the temperature difference d(t)=T₁−T₂:

$$
\dot d=-K(1/C_1+1/C_2)d=-\lambda d,
\qquad C_e=\frac{C_1C_2}{C_1+C_2},\qquad\lambda=K/C_e.
$$

Thus d(t)=d₀ exp(−λt). Energy conservation fixes the equilibrium temperature as a heat-capacity-weighted average:

$$
T_e=\frac{C_1T_{10}+C_2T_{20}}{C_1+C_2},\quad
T_1=T_e+\frac{C_2}{C_1+C_2}d_0e^{-\lambda t},\quad
T_2=T_e-\frac{C_1}{C_1+C_2}d_0e^{-\lambda t}.
$$

For K>0, the time constant τ=1/λ=Cₑ/K is the time needed to reduce the difference to 1/e of its initial value. It is not the time of exact equilibrium. When K=0 the connection is absent, so there is no finite relaxation time.

## Distinguish power from accumulated transfer

The energy that has crossed the boundary by time t is the integral of power:

$$
Q_{1\to2}(t)=\int_0^t\Phi(s)\,ds
=C_ed_0(1-e^{-\lambda t}),\qquad
\Delta U_1=-Q_{1\to2},\quad\Delta U_2=Q_{1\to2}.
$$

Q is measured in J, whereas Φ is measured in W. The rate can decrease while the accumulated transferred energy continues to grow. Reversing the initial temperature ordering reverses the sign of Q. Since the reference value of internal energy can be chosen, we compare its **changes** here.

## Energy is conserved. What changes?

For changes between internally equilibrated states, dS=C dT/T. This does not assert that the actual thermal contact is reversible. A state-function difference can be calculated using a reversible path between the same endpoints.

$$
\Delta S=C_1\ln(T_1/T_{10})+C_2\ln(T_2/T_{20}),\qquad
\dot S=\frac{K(T_1-T_2)^2}{T_1T_2}\ge0.
$$

The hot body loses entropy and the cold body gains entropy. Receiving the same small energy at a lower temperature produces a larger contribution Q/T, so the total grows. Entropy is generated **inside** the combined system even though no heat crosses its outer boundary. Energy conservation and entropy increase are distinct statements.

## Follow the numbers

Use C₁=2 J/K, C₂=3 J/K, T₁₀=330 K, T₂₀=270 K and K=0.4 W/K. Then Cₑ=1.2 J/K, Tₑ=294 K and τ=3 s. At 3 seconds the temperature difference is 60/e≈22.073 K: T₁≈307.244 K and T₂≈285.171 K. The transferred energy is Q≈45.513 J, with ΔU₁≈−45.513 J and ΔU₂≈45.513 J. Total entropy has increased by approximately 0.021097 J/K.

Compare the hot body's energy loss, 2×(330−307.244), with the cold body's gain, 3×(285.171−270). They agree up to rounding. This check uses actual changes in energy rather than treating the average temperature as conserved.

## Read the visualization

The experiment plots both temperatures, transferred energy, individual and total entropy changes, the instantaneous heat-transfer rate, entropy production and the remaining temperature difference. The horizontal axis is physical time in seconds. Read each vertical unit separately: K, J, J/K, W or W/K. Selected points on the temperature curves mark the current time; the equilibrium line is the heat-capacity-weighted temperature. Compare the signed transfer with the changes of the two bodies in the table. These curves describe the lumped model, rather than particles carrying a stored substance called heat.

Changing heat capacity changes both equilibrium temperature and relaxation time. Doubling K alone moves along the same thermal trajectory twice as quickly, without changing the final temperature or infinite-time total transfer. Comparing these experiments separates faster transfer from a larger final transferred amount.

## Six questions with worked answers

### Exercise 1: Why is (330+270)/2=300 K the wrong equilibrium temperature?

**Answer:** The heat capacities differ. At 300 K the energy changes would be −60 J and +90 J, requiring 30 J from outside. The weighted average 294 K gives changes −72 J and +72 J.
### Exercise 2: What is the initial transfer rate?

**Answer:** Φ(0)=0.4×60=24 W. Multiplying this by 3 seconds gives 72 J, an overestimate because the rate decreases. The actual integral is approximately 45.513 J.
### Exercise 3: What if both heat capacities double?

**Answer:** Tₑ stays fixed, while Cₑ and τ double. Less relaxation occurs by the same time, but the infinite-time transferred energy doubles.
### Exercise 4: Are equal initial temperatures and K=0 the same case?

**Answer:** Both give Q=0 and ΔS=0. In the first case there is no temperature difference; in the second there can be a difference but no transfer path. Disconnected bodies at unequal temperatures have not reached thermal equilibrium with each other.
### Exercise 5: Does the hot body's decreasing entropy violate the second law?

**Answer:** That body exports heat across its boundary. Apply the nonnegative entropy-change condition to the combined insulated system, rather than to one body in isolation from its transfer terms.
### Exercise 6: Should the computed rate be forced to zero when displayed temperatures become equal?

**Answer:** Display rounding can hide a small remaining difference. Retain the model's d₀ exp(−λt) and corresponding rate. Exact zero differs from a value below display resolution.

## Whole-history changes and interval changes

To find transfer between 3 and 8 seconds, calculate the change during that interval. Subtracting Q(3) from Q(8) works at ordinary scales, but subtracting two large cumulative values can lose significant digits for a very short interval. Let dₐ=d₀ exp(−λa) be the temperature difference at the start and Δt=b−a the duration:

$$
Q_{a\to b}=C_e d_a(1-e^{-\lambda\Delta t}).
$$

For a positive duration this avoids subtracting accumulated transfers. The interval entropy change can similarly be calculated by integrating its production rate over that interval. A zero-duration interval has exactly zero change. An end before the start is outside the forward-time interval used here.

## Interpreting very small changes

Energy can flow even when individual temperature changes are below display resolution. At smaller scales, the computer's finite number range may be unable to represent the transferred amount itself. Replacing it with physical zero would confuse weak transfer with a disconnected system. Retaining the sign and logarithm of its magnitude preserves information about its direction and scale.

Computing total entropy only as a sum of two nearly opposite contributions is also vulnerable to cancellation. This unit instead integrates a nonnegative production rate, using numerical quadrature within declared material and temperature bounds. It does not provide an exact method for arbitrary temperatures, nonlinear connections or spatial temperature fields.

## Connection to AI research

These balances form a small dynamical system with a conservation constraint. A neural model may fit both temperature curves closely while violating total energy conservation. Separate checks for positive temperature, energy balance and the sign of entropy production reveal different failures. A decreasing training loss is not automatically a physical entropy statement: the physical states, units and boundaries must first be defined.

## Sources and the next connection

The discussion of temperature, the first law and entropy in [David Tong, Classical Thermodynamics](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) supports the distinctions between systems, boundaries and transferred quantities. The constant-capacity bodies, linear connection and time-dependent solution are the specialization declared in this unit. Next, compare ideal-gas paths with identical endpoints but different heat and work.
