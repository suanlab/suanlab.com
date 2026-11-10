# Does an energy-balanced engine have to be possible?

> Manuscript draft. The product model, figures and registration are not implemented. This port model does not solve an irreversible piston trajectory.

## Is conservation enough?

Imagine an engine that absorbs 1000 J from a hot reservoir and delivers all 1000 J as work. Its internal-energy change is zero after a cycle, and incoming and outgoing energies agree. The first law alone finds no inconsistency. Why can this not be an ordinary cyclic heat engine with no other effects?

The second law constrains direction and entropy budgets in addition to energy quantities. We will examine boundary ports before drawing a detailed internal trajectory.

## Declare the system and signs

Use two ideal reservoirs at constant temperatures Th>Tc>0 and a closed device that returns to its initial state after each cycle. Qh>0 enters from the hot reservoir; Qc is heat into the device from the cold reservoir. Thus rejection to the cold reservoir has Qc<0. W is work performed by the device.

Heat and work are energies per cycle in J, not powers. Entropy production per cycle is in J/K. No operating duration, internal temperature field or friction law is specified, so these budgets must not be presented as a solved piston motion.

## The laws perform different checks

Because ΔU=0 after a cycle, the first law requires

$$
W=Q_h+Q_c
$$

Device entropy also returns to its initial value. The combined reservoir entropy change must satisfy

$$
S_{\rm gen}=-\frac{Q_h}{T_h}-\frac{Q_c}{T_c}\ge0
$$

A reservoir loses entropy when it loses heat and gains entropy when it receives heat. This condition is separate from energy conservation. Combining both conditions produces a bound on work.

## Construct ports using an entropy-generation coefficient

Introduce a relative generation coefficient σ:

$$
Q_c=-\frac{T_c}{T_h}Q_h(1+\sigma),\qquad
S_{\rm gen}=\frac{Q_h}{T_h}\sigma
$$

The second law requires σ≥0. A negative σ is a counterfactual input for comparison, not a physically feasible engine. For σ≥−1 the cold port rejects heat or has zero heat transfer.

Net work is

$$
W=Q_h\left[1-\frac{T_c}{T_h}(1+\sigma)\right]
$$

Greater generation reduces the work available from the same hot input. Interpret W/Qh as heat-engine efficiency only when the device operates as an engine with W>0.

## Compare four budgets

Hold Th=600 K, Tc=300 K and Qh=1000 J fixed.

| σ | Qc (J) | W (J) | Sgen (J/K) | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 0 | −500 | 500 | 0 | Reversible limiting ports |
| 0.5 | −750 | 250 | About 0.8333 | Positive generation and lost work |
| 2 | −1500 | −500 | About 3.3333 | Additional external work input |
| −0.5 | −250 | 750 | About −0.8333 | Energy-balanced but second-law violating |

In the third row the device absorbs heat from the hot reservoir and rejects a larger amount to the cold one. Supplied work also ends up at the cold port. Calling it a refrigerator that moves heat from cold to hot would reverse the actual heat directions.

## Lost work and entropy

Let Wrev denote reversible limiting work with the same Qh and the same two reservoirs. Then

$$
W_{\rm rev}-W=T_c S_{\rm gen}
$$

This relates generation to lost work under these boundary conditions. Do not apply the same expression automatically to every open system or arbitrary environmental temperature. Matter exchange and other work ports require their own availability budget.

## Read the proposed diagram

First inspect the signs and magnitudes of hot, cold and work arrows. Then inspect energy balance and reservoir entropy balance separately. Negative-σ examples must be explicitly counterfactual diagrams, rather than animations of feasible operating devices. A production amount in J/K cannot become a rate in W/K without time information.

## Six questions with worked answers

### Exercise 1: Does converting 1000 J of heat into 1000 J of work violate the first law?

**Answer:** Not on energy balance alone. Taking only heat from one positive-temperature reservoir and converting it entirely into cyclic work violates the second law through negative total entropy production.

### Exercise 2: Does σ=0 mean internal temperature is constant throughout?

**Answer:** No. Zero generation specifies reversible limiting ports. Internal states can change, as along Carnot adiabats. This port model does not determine the internal trajectory.

### Exercise 3: Is the table's σ=2 device a refrigerator?

**Answer:** No. It rejects heat to the cold reservoir rather than extracting heat from it. Negative W alone does not determine refrigeration operation.

### Exercise 4: Does a smooth negative-σ calculation describe a possible device?

**Answer:** Mathematical smoothness does not establish physical admissibility. Negative combined reservoir entropy is a counterfactual comparison here.

### Exercise 5: How much work is lost when generation increases by 0.1 J/K at fixed Qh?

**Answer:** Under these two-reservoir conditions the loss is Tc×0.1 J. At Tc=300 K it is 30 J. Use absolute temperature in kelvin.

### Exercise 6: Does the universe return to its initial entropy when the device returns to its initial state?

**Answer:** No. Reservoir entropy changes remain even though the device state returns. Check positive production with the external budget included.

## Connection to AI research

A model trained only on conservation losses can still predict negative entropy-production ports. Validate input conditions, energy balances, admissibility inequalities and operating classifications separately. Label impossible examples explicitly when using them in training. Next, examine why a classical constant heat capacity cannot simply be extrapolated to zero temperature.
