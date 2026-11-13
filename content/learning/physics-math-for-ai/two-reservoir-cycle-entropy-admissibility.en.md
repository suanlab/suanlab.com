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

## Read the port diagram

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

## Small generation versus rounded displays

For very small generation, the displayed reservoir entropy changes may sum to zero. Computing production by subtracting those rounded observations would erase weak irreversibility. Calculate production and the work deficit directly from the input budget instead.

Exactly zero hot input gives zero transfer: a relative coefficient's sign alone does not establish a second-law violation. Conversely, a nonzero negative generation below the number range must not become a physical zero and an admissible classification. Read the value, sign, logarithmic magnitude and budget meaning together.

## Research extension: apparent excess work from an unrestored device

A run can exceed the cyclic Carnot work for its recorded hot input without violating the second law if part of the device ends in a different state. It may have consumed a stored resource. Calling that run a cycle hides the resource and misclassifies its entropy budget.

Keep two ideal reservoirs at T_h>T_c>0 and no matter exchange, but now allow device changes ΔU and ΔS. Include every auxiliary in this device boundary. Work is positive out, heat positive in; all transfers are per run, not rates. The existing lab still models a restored cyclic device. This supplement examines what changes when that assumption fails.

The first law now retains the device's energy change. A closed material boundary does not by itself imply a cyclic state.

$$
\Delta U=Q_h+Q_c-W.
$$

The reservoirs' entropy changes must be added to the device's. Nonnegative combined generation constrains this complete budget, rather than the reservoir terms alone.

$$
S_{\rm gen}=\Delta S-\frac{Q_h}{T_h}-\frac{Q_c}{T_c}\ge0.
$$

Solving the entropy balance gives the cold heat port including both stored device entropy and generation. Its sign still determines whether heat is absorbed from or rejected to the cold reservoir.

$$
Q_c=T_c\Delta S-\frac{T_c}{T_h}Q_h-T_cS_{\rm gen}.
$$

At the fixed cold reference temperature define the endpoint resource change ΔF_c. Eliminating Q_c gives the exact work identity below. This endpoint combination does not assume that the device was in equilibrium with that reference reservoir throughout the run.

$$
\Delta F_c=\Delta U-T_c\Delta S,\qquad W=Q_h(1-T_c/T_h)-\Delta F_c-T_cS_{\rm gen}.
$$

Negative ΔF_c can supply work beyond the cyclic reversible value. W/Q_h is then an apparent ratio with an additional consumed resource, not the efficiency of a device restored using only two heat ports.

$$
W-W_{\rm cyc,rev}=-\Delta F_c-T_cS_{\rm gen},\qquad W_{\rm cyc,rev}=Q_h(1-T_c/T_h).
$$

If a data pipeline silently imposes ΔS=0, it reports the apparent generation below. A negative result can reflect missing device-state information rather than a negative true production. Conversely, merely inventing an unmeasured entropy change is not evidence of feasibility.

$$
S_{\rm apparent}=-Q_h/T_h-Q_c/T_c=S_{\rm gen}-\Delta S.
$$

Reset the device exactly, using only the cold reservoir and a work port, with no other stored resource change. The reset's entropy and energy balances give a signed work-input bound. For the examples with ΔF_c≤0 this is a positive restoration cost. Unaccounted reset reservoirs or auxiliaries would change this formula.

$$
W_{\rm reset,on}=-\Delta F_c+T_c S_{\rm gen,reset}\ge-\Delta F_c.
$$

Adding run and reset cancels the endpoint resource change. The total hot input is still Q_h because reset uses only the cold reservoir. The completed operation obeys the original cyclic bound, with both generation contributions retained.

$$
W_{\rm net}=W-W_{\rm reset,on}=W_{\rm cyc,rev}-T_c(S_{\rm gen}+S_{\rm gen,reset})\le W_{\rm cyc,rev}.
$$

At 600 K and 300 K, hot input 1000 J has a cyclic reversible work of 500 J. A device gaining 1 J/K of entropy with no energy change can instead output 800 J reversibly during the unrestored run. Its apparent reservoir-only production is −1 J/K, while the complete production is zero.

$$
T_h=600\ {\rm K},\ T_c=300\ {\rm K},\ Q_h=1000\ {\rm J},\ \Delta U=0,\ \Delta S=1\ {\rm J/K},\ S_{\rm gen}=0:\quad W=800\ {\rm J},\ Q_c=-200\ {\rm J}.
$$

A concrete resource is an auxiliary ideal gas expanded reversibly at T_c. Its temperature and internal energy stay fixed but its volume and entropy change. Combine that expansion with a reversible engine. Choosing nR log(V_b/V_a)=1 J/K supplies the extra 300 J. Reversible compression to the starting volume costs that same 300 J. There is no repeated free conversion of cold heat to work with all states restored.

$$
\Delta S_{\rm aux}=nR\log(V_b/V_a),\quad W_{\rm aux}=T_c\Delta S_{\rm aux},\quad W_{\rm reset,on}^{\rm rev}=T_c\Delta S_{\rm aux}.
$$

### Read work before and after restoration

The upper panel shows work from four endpoint budgets; some exceed the 500 J cyclic reference. The lower panel subtracts reversible reset work. Entropy generation then lowers net output below that reference. These bars specify energy/entropy accounting, not a solved mechanical time history.

![Apparent work above the cyclic reference consumes a device free-energy resource; resetting that resource restores the cyclic bound.](/assets/learning/laws-unrestored-device-reset.en.svg)

### Reproduce the endpoint budgets

All rows use ΔU=0, T_h=600 K, T_c=300 K and Q_h=1000 J. The final column assumes a reversible reset using only the cold reservoir; an irreversible reset gives less net work. Budget consistency is a necessary check, not by itself a construction of every device.

| ΔS (J/K) | S_gen (J/K) | W (J) | ΔF_c (J) | Reset input (J) | Net W (J) |
| --- | --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 500.00000000 | 0.00000000 | 0.00000000 | 500.00000000 |
| 0.50000000 | 0.00000000 | 650.00000000 | -150.00000000 | 150.00000000 | 500.00000000 |
| 1.00000000 | 0.00000000 | 800.00000000 | -300.00000000 | 300.00000000 | 500.00000000 |
| 1.00000000 | 0.20000000 | 740.00000000 | -300.00000000 | 300.00000000 | 440.00000000 |

```python
Th, Tc, Qh = 600.0, 300.0, 1000.0
for dS, generated in [(0,0), (0.5,0), (1,0), (1,0.2)]:
    dF = -Tc*dS if dS else 0.0  # all examples have dU = 0
    W = Qh*(1-Tc/Th)-dF-Tc*generated
    reset = -dF if dF else 0.0  # reversible reset using only the cold reservoir
    print(f"{dS:.8f} {generated:.8f} {W:.8f} "
          f"{dF:.8f} {reset:.8f} {W-reset:.8f}")
```

### Three research questions with explanations

#### 1. Does the 800 J example disprove the Carnot bound?

**Explanation.** No. The device is not restored: its auxiliary gas has expanded and gained entropy. It consumed 300 J of resource measured by the decrease in U−T_c S. Restoring the auxiliary reversibly costs 300 J, leaving 500 J net for the complete operation.

#### 2. Should a negative reservoir-only entropy sum always be relabeled as physical?

**Explanation.** No. It is necessary to measure or justify the omitted device state and every auxiliary. With a genuinely restored device it remains a violation. The example explains a specific missing-boundary mechanism, not permission to excuse arbitrary model outputs.

#### 3. Why is the reset temperature and boundary specified?

**Explanation.** The bound follows from a reset exchanging heat only with T_c and work only with the recorded work port. Another reservoir or consumed resource can change the apparent reset cost; those changes must be included in the completed operation before comparing limits.

### Source and research boundary

[David Tong, Classical Thermodynamics, sections 4.2–4.4](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) supplies state/transfer and entropy/free-energy conventions. The noncyclic budget and explicit expansion/reset comparison are derived here. For AI-generated device plans, verifying numerical conservation is insufficient unless state closure and resource boundaries are also supplied.

## Connection to AI research

A model trained only on conservation losses can still predict negative entropy-production ports. Validate input conditions, energy balances, admissibility inequalities and operating classifications separately. Label impossible examples explicitly when using them in training. Next, examine why a classical constant heat capacity cannot simply be extrapolated to zero temperature.

Source: [David Tong, Classical Thermodynamics](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf).
