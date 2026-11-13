## Where does energy come from in a cycle?

An expanding gas can push a piston and perform work. Returning the gas to its original state, however, requires compression. To determine net engine work we must include both the outward and return journeys. The Carnot cycle organizes this question into four reversible processes.

The system contains n mol of a monatomic ideal gas. Two ideal reservoirs maintain positive absolute temperatures Th and Tc, with Th>Tc. There is no matter transfer. Unlike the previous finite-body contact experiment, reservoir temperatures do not change when they exchange heat.

## The zeroth law: compare temperatures with a thermometer

If A is in thermal equilibrium with thermometer C, and B is in thermal equilibrium with C in that same thermometer state, A and B are also in mutual thermal equilibrium. This transitive relation supports comparing different materials using a shared temperature label. Thermal equilibrium concerns their behavior when thermal contact is allowed.

Merely disconnecting bodies and observing no change is insufficient. In the earlier contact experiment, K=0 prevented flow even at unequal temperatures. Equal temperatures also do not imply equal heat capacities or equal total energies.

The zeroth law permits a common temperature label without uniquely fixing its numerical scale. Here we use absolute temperature in kelvin, connected to Carnot heat ratios. Substituting an arbitrary empirical temperature label into those ratios would require another justification.

## Signs, units and reversibility

Heat Q is positive into the gas; work W is positive when performed by the gas. Thus ΔU=Q−W. Q and W are energies per cycle in J. Without time information, work cannot be interpreted as power in watts. Temperature is in K, volume in m³, pressure in Pa, and molar heat capacity Cᵥ=3R/2 in J/(mol K).

Volume control is frictionless and quasistatic. Each isothermal stage exchanges heat reversibly with its reservoir in the limit of an infinitesimal temperature difference. Adiabatic stages exchange no heat. Moving slowly alone does not eliminate friction or a finite temperature difference.

## Construct the four corners

Choose initial volume Va and isothermal expansion ratio r>1. Let γ=5/3 and α=(Th/Tc)^(3/2).

| State | Temperature | Volume |
| --- | --- | --- |
| A | Th | Va |
| B | Th | rVa |
| C | Tc | αrVa |
| D | Tc | αVa |

A→B is hot isothermal expansion, B→C adiabatic expansion, C→D cold isothermal compression, and D→A adiabatic compression. These volumes are not arbitrary rectangle coordinates. Conservation of TV^(2/3) along the adiabats connects the two temperatures and closes the cycle.

## Integrate the isotherms and adiabats separately

An isothermal ideal-gas stage has ΔU=0 and Q=W. Integrating p=nRT/V over physical volume gives

$$
Q_h=W_{AB}=nRT_h\ln r,\qquad
Q_c=W_{CD}=-nRT_c\ln r.
$$

Qc is negative because the gas releases heat to the cold reservoir. Always check whether a source instead uses Qc for the positive magnitude of rejected heat.

An adiabatic stage has Q=0. Integrating dU=nCᵥdT=−p dV gives TV^(2/3)=constant and

$$
W_{BC}=nC_v(T_h-T_c),\qquad
W_{DA}=-nC_v(T_h-T_c).
$$

The two adiabatic works cancel over the complete cycle, but neither is individually zero. Temperature and internal energy change during these stages.

## Net work and efficiency

The four internal-energy changes sum to zero. Therefore

$$
W_{\rm cycle}=Q_h+Q_c=nR(T_h-T_c)\ln r,\qquad
\eta=\frac{W_{\rm cycle}}{Q_h}=1-\frac{T_c}{T_h}.
$$

Net work is the signed area enclosed in the p–V plane. Expanding while hot and compressing while cold produces more outward work than is required on the return journey. Arbitrarily deleting cold heat rejection would violate the conditions of this reversible cycle.

## Read the entropy picture

Gas entropy increases by Σ=nR ln r during the hot isotherm and decreases by the same amount during the cold isotherm. It remains constant along reversible adiabats:

$$
\frac{Q_h}{T_h}+\frac{Q_c}{T_c}=0.
$$

The T–S diagram is a rectangle of width Σ and height Th−Tc. Its area also equals net heat and net work. Each reservoir entropy change has the opposite sign to the corresponding transfer into the gas. Total entropy production vanishes in the reversible limit. Returning the working gas to its initial state does not make entropy production vanish for an arbitrary irreversible engine.

## Check the numbers

Use n=1 mol, Th=600 K, Tc=300 K, Va=0.010 m³ and r=2. Then α≈2.828, Vb=0.020, Vc≈0.05657 and Vd≈0.02828 m³. The gas absorbs approximately 3457.888 J from the hot reservoir and releases 1728.944 J to the cold one. Net work is approximately 1728.944 J and efficiency is 0.5. The adiabatic works are approximately +3741.508 J and −3741.508 J.

The large positive adiabatic work is not the engine's net output. Account for all four stages of the closed cycle.

## Reverse it to operate a refrigerator

Following the same reversible cycle backward reverses all complete heat and work transfers. External work moves heat from the cold reservoir to the hot one. The refrigeration and heating coefficients of performance are

$$
{\rm COP}_{c}=\frac{T_c}{T_h-T_c},\qquad
{\rm COP}_{h}=\frac{T_h}{T_h-T_c}.
$$

Reversing the example consumes approximately 1728.944 J of work, removes the same amount of heat from the cold reservoir, and delivers 3457.888 J to the hot one. Refrigeration COP is 1 and heating COP is 2. A heating COP above one is possible because delivered heat includes heat taken from the cold reservoir as well as the supplied work.

## Partial-path work differs from whole-cycle operation

The reversed cycle begins with adiabatic expansion A→D. The gas can perform positive work during this stage by reducing its internal energy. Completing all three remaining stages and returning to the initial state nevertheless gives negative net work. The complete refrigerator cycle requires external work input.

Do not classify an engine or refrigerator from the sign of work accumulated only so far. Check a complete cycle's ports and state closure together. Likewise, subtracting a tiny reverse progress from the number one can erase the change through rounding; the reverse path is evaluated directly.

## Degenerate cases and progress

When Th=Tc, net work vanishes. When r=1, isothermal heat transfers vanish. If a denominator is zero, the actual W/Qh or COP must remain undefined. Distinguish a temperature-based Carnot bound from an undefined ratio for a degenerate cycle.

Progress in a figure denotes position along four stages. It does not prescribe equal durations or constant power. Finite-rate reservoir heat transfer, piston response and friction require another dynamical model and an irreversible entropy budget.

## Six questions with worked answers

### Exercise 1: Can the adiabats be omitted because their works cancel?

**Answer:** No. They connect the isotherms while changing temperature and restoring the state. Their individual internal-energy changes are nonzero.

### Exercise 2: Does ΔU=0 over a cycle imply zero heat?

**Answer:** It implies Qnet=Wnet through ΔU=Qnet−Wnet=0. Neither quantity must individually vanish.

### Exercise 3: Why is efficiency 50% between 600 K and 300 K?

**Answer:** η=1−300/600=0.5. Half of the absorbed heat is rejected to the cold reservoir in this reversible cycle.

### Exercise 4: Can Celsius temperatures be used in the ratio?

**Answer:** No. Use absolute temperatures in kelvin. The Celsius offset changes temperature ratios.

### Exercise 5: Does reversed heating COP=2 violate energy conservation?

**Answer:** No. Heat delivered to the hot reservoir equals supplied work plus heat extracted from the cold reservoir. Work alone is not the only energy source.

### Exercise 6: Do equal reservoir temperatures produce an infinitely capable refrigerator?

**Answer:** The degenerate process has zero work input, so the usual refrigeration COP ratio is undefined. It cannot be interpreted as performance for moving heat against a finite temperature difference.

## Research extension: maximum work from finite hot and cold bodies

The original engine uses reservoirs whose temperatures never change. A hot drink and a cool metal block are different: as work is extracted, the source cools and the sink warms. Multiplying all extracted heat by the initial Carnot efficiency overestimates the work. We now change the source model explicitly, while retaining the reversible engine as an infinitesimal intermediary.

Take two internally equilibrated bodies with constant positive heat capacities Cₕ,C꜀ in J/K and initial temperatures Tₕ₀>T꜀₀>0. The outer boundary admits only extracted work, with no matter or heat exchange. An ideal engine completes infinitesimal reversible cycles between the instantaneous temperatures and retains no net energy or entropy. There is no time, conductance or friction law. Aₕ and A꜀ below are positive amounts removed from the hot body and deposited in the cold body; the engine's signed cold heat is −A꜀.

Reversible exchange conserves the entropy of both bodies together. A small amount received at the lower temperature compensates the hot body's entropy loss, not its energy loss.

$$
C_h\frac{dT_h}{T_h}+C_c\frac{dT_c}{T_c}=0,\qquad dA_h=-C_h\,dT_h>0,\quad dA_c=C_c\,dT_c>0.
$$

Integrating the entropy balance gives the entire relation between the two temperatures. This path closes at a common temperature, rather than keeping the initial cold temperature fixed.

$$
T_c=T_{c0}\left(\frac{T_{h0}}{T_h}\right)^{C_h/C_c},\qquad T_h\downarrow T_f,\quad T_c\uparrow T_f.
$$

The final temperature is the capacity-weighted geometric mean. The ratio form below keeps the logarithms dimensionless. It lies between the initial temperatures.

$$
T_f=T_{h0}\left(\frac{T_{c0}}{T_{h0}}\right)^{C_c/(C_h+C_c)}.
$$

The difference between the two cumulative heat magnitudes is extracted work. Both bodies' internal-energy changes must be included; the working engine has no accumulated state change over its completed cycles.

$$
A_h=C_h(T_{h0}-T_h),\quad A_c=C_c(T_c-T_{c0}),\qquad W=A_h-A_c.
$$

Evaluate that balance when the two temperatures meet. For equal capacities of 1 J/K and initial temperatures 600 K and 300 K, T_f≈424.264069 K and maximum work is about 51.471863 J, not the work obtained by applying 50% efficiency to all hot-body heat.

$$
W_{\max}=C_hT_{h0}+C_cT_{c0}-(C_h+C_c)T_f.
$$

Why is this an upper bound? Consider any process with the same insulated outer boundary, an intermediary restored to its initial state, and a final common positive body temperature T_end. Nonnegative total entropy implies the inequality below. Energy balance then bounds work by the reversible value. Other remaining resources, unequal final temperatures or an unrestored intermediary require their own end-state budget.

$$
S_{\rm gen}=(C_h+C_c)\log(T_{\rm end}/T_f)\ge0\quad\Longrightarrow\quad T_{\rm end}\ge T_f,\quad W\le W_{\max}.
$$

Directly contacting the bodies without extracting work instead gives the weighted arithmetic mean T_a. Concavity of log implies the arithmetic mean is at least the geometric mean, explaining the nonnegative maximum work. The two operations have different final temperatures despite identical initial conditions.

$$
T_a=\frac{C_hT_{h0}+C_cT_{c0}}{C_h+C_c},\qquad W_{\max}=(C_h+C_c)(T_a-T_f)\ge0.
$$

Direct contact generates the positive entropy difference below. It is not the same thermodynamic process run faster: useful work has been discarded and the final state differs.

$$
\Delta S_{\rm contact}=C_h\log(T_a/T_{h0})+C_c\log(T_a/T_{c0})=(C_h+C_c)\log(T_a/T_f).
$$

Divide total work by total hot-body heat only after doing both integrals. For equal capacities in the example, this gives approximately 0.292893, below the initial instantaneous value 0.5.

$$
\eta_{\rm total}=\frac{W_{\max}}{C_h(T_{h0}-T_f)},\qquad 0<\eta_{\rm total}<1-\frac{T_{c0}}{T_{h0}}\quad(T_{h0}>T_{c0}).
$$

The overall efficiency is a heat-weighted average of instantaneous efficiencies. The latter decrease strictly toward zero as the bodies approach equality, giving the strict bound for unequal starting temperatures. Equal starting temperatures give zero available work and zero hot input, so the efficiency ratio is undefined. Approaching the reversible endpoint says nothing about a finite operating duration.

$$
\eta_{\rm total}=\frac{\int(1-T_c/T_h)\,dA_h}{\int dA_h},\qquad \eta_{\rm instantaneous}\longrightarrow0\quad(T_h,T_c\to T_f).
$$

### Read changing temperatures and cumulative transfers

The upper panel follows the two temperatures along a chosen path coordinate. The lower panel separates heat removed, heat deposited and work; their difference balances exactly. Path position is not seconds or power. These curves describe the new finite-body idealization, not a change to the existing fixed-reservoir lab.

![Finite hot and cold bodies approach a common geometric-mean temperature while extracted work equals removed heat minus deposited heat.](/assets/learning/laws-finite-source-work.en.svg)

### Compare capacities and a degenerate preparation

Capacities are in J/K. The first three rows start at 600 K and 300 K; the last starts with both bodies at 300 K. Heat capacities are not the molar heat capacity of the working gas. The last efficiency is intentionally undefined.

| Cₕ (J/K) | C꜀ (J/K) | T_f (K) | Maximum W (J) | Overall efficiency |
| --- | --- | --- | --- | --- |
| 1.00000000 | 1.00000000 | 424.26406871 | 51.47186258 | 0.29289322 |
| 2.00000000 | 1.00000000 | 476.22031559 | 71.33905323 | 0.28816947 |
| 1.00000000 | 2.00000000 | 377.97631497 | 66.07105509 | 0.29758562 |
| 1.00000000 | 1.00000000 | 300.00000000 | 0.00000000 | undefined |

```python
from math import exp, log
for Ch, Cc, Th, Tc in [(1,1,600,300), (2,1,600,300),
                        (1,2,600,300), (1,1,300,300)]:
    Tf = Th*exp(Cc/(Ch+Cc)*log(Tc/Th))
    W = Ch*Th+Cc*Tc-(Ch+Cc)*Tf
    Ah = Ch*(Th-Tf)
    eta = f"{W/Ah:.8f}" if Ah > 0 else "undefined"
    print(f"{Ch:.8f} {Cc:.8f} {Tf:.8f} {W:.8f} {eta}")
```

### Three research questions with explanations

#### 1. Why can the final temperature be lower than the no-work contact temperature?

**Explanation.** The engine exports work through the outer boundary, leaving less total internal energy in the bodies. Reversible extraction preserves their total entropy and selects the geometric mean; direct contact exports no work, preserves body energy and selects the arithmetic mean.

#### 2. Can the initial Carnot efficiency be applied to the total removed heat?

**Explanation.** No. Both source temperatures change, and the instantaneous efficiency decreases as they approach each other. Integrating the work or taking the heat-weighted average gives the stated total. The fixed-reservoir formula remains valid only at each infinitesimal step under this idealization.

#### 3. Does maximum reversible work specify achievable power or completion time?

**Explanation.** No. The derivation supplies an end-state bound and a reversible limiting path. It contains no transport rate, engine speed or dissipation model. Those additions may lower extracted work, and the ideal endpoint must not be advertised as finite-time operation.

### Source and research boundary

[David Tong, Classical Thermodynamics, sections 4.2–4.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) supplies the energy, entropy and Carnot conventions. The finite-capacity path and optimal-work comparison are derived here under the declared boundary conditions. A data-driven engine model must retain the source states and capacities when comparing these finite bodies with ideal reservoirs.

## Connection to AI research

Check cycle closure, energy balance, reservoir entropy and operating direction together. Small pressure-curve errors can coexist with incorrect net-work signs or heat ports. Distinguish state coordinates, path progress and physical time in training data. Next, compare energy-conserving ports that nevertheless violate the second law.

Source: [David Tong, Classical Thermodynamics](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf), the Carnot-cycle and temperature-scale discussion. Signs have been converted to this manuscript's work-by convention.
