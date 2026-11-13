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

## Research extension: which thermal parameters do temperatures identify?

Imagine two experiments with precisely the same pair of temperature curves. In one, every heat capacity and the conductance are doubled. The bodies can carry twice the energy while following the same temperatures at every time. Better temperature fits alone cannot select one of these physical systems. This is a missing absolute scale, distinct from noisy estimation.

Retain constant positive heat capacities, a linear conductance, spatially uniform positive temperatures and calibrated thermometers and clock. First assume K>0, unequal initial temperatures and an ideal continuous, error-free record of both temperatures. These assumptions describe structural identifiability; numerical differentiation of noisy finite records is a separate problem. We later add an explicitly calibrated boundary heat input in a separate experiment.

Define two positive rates a and b, both in s⁻¹. They, rather than three independent material parameters, determine the unforced temperature dynamics.

$$
d=T_1-T_2,\quad a=K/C_1,\quad b=K/C_2,\qquad \dot T_1=-ad,\quad\dot T_2=bd.
$$

The original exponential solution can be written entirely in terms of those rates and the initial temperatures. Its equilibrium is a capacity-weighted temperature, not a fitted source of new absolute energy information.

$$
\lambda=a+b,\quad T_e=\frac{bT_{10}+aT_{20}}{a+b},\quad d=d_0e^{-\lambda t},\quad T_1=T_e+\frac a\lambda d,\quad T_2=T_e-\frac b\lambda d.
$$

At an instant with nonzero d, slopes identify a and b. The fitted equilibrium and decay give the equivalent capacity ratio and sum of rates. The finite-time logarithm requires t₂>t₁ and a resolved nonzero difference of the same sign at both times. It is not a recipe for differentiating rounded or noisy temperatures without an uncertainty model.

$$
a=-\frac{\dot T_1}{d},\quad b=\frac{\dot T_2}{d},\quad \frac{C_1}{C_2}=\frac ba=\frac{T_e-T_{20}}{T_{10}-T_e},\quad \lambda=-\frac{\log[d(t_2)/d(t_1)]}{t_2-t_1}.
$$

A common positive scale s preserves both rates and therefore the entire temperature record. It changes every energy transfer, internal-energy change and entropy change by s. The sign of entropy production remains correct for every member, so adding only conservation and entropy-sign penalties to a temperature loss does not choose an absolute scale.

$$
(C_1,C_2,K)\mapsto(sC_1,sC_2,sK),\quad s>0:\quad (T_1,T_2)\mapsto(T_1,T_2),\quad (Q,\Delta S,\Phi)\mapsto s(Q,\Delta S,\Phi).
$$

Make the ambiguity local as well as global. Choose fixed positive reference capacities C*, conductance K* and rates a*, b*. The Jacobian of the two log rates has rank two and a one-dimensional null direction. This is a sensitivity Jacobian, not a Fisher matrix: no detector likelihood has been specified.

$$
\theta=(\log(C_1/C_*),\log(C_2/C_*),\log(K/K_*)),\qquad J=\frac{\partial(\log(a/a_*),\log(b/b_*))}{\partial\theta}=\begin{pmatrix}-1&0&1\\0&-1&1\end{pmatrix},\quad J(1,1,1)^{\mathsf T}=0.
$$

The nondegenerate assumption matters. Equal initial temperatures leave a constant unforced record for every K and capacity pair. With an unequal initial difference and positive finite capacities, an exactly constant error-free record implies K=0 in this model, but neither capacity is determined. Log-rate formulas and the three-parameter reconstruction below do not apply to that disconnected case.

$$
Kd_0=0\quad\Longrightarrow\quad \dot T_1=\dot T_2=0\quad\text{in the unforced experiment}.
$$

To supply an absolute scale, add known heat power P>0 through body 1's outer boundary. This changes the combined system boundary: the two-body system is now supplied with energy. P is a calibrated thermal power in W, not a newly inferred function of the same temperature record.

$$
C_1\dot T_1=-K(T_1-T_2)+P,\qquad C_2\dot T_2=K(T_1-T_2).
$$

Switch P from zero to its known value with temperatures continuous and material parameters unchanged. Superscripts − and + denote the immediate left and right derivatives at the same switching state. The slope jump isolates C₁ because the contact term is identical on both sides. Recover K and C₂ using the a,b already identified in a nondegenerate unforced record. A comparison at different states would not cancel the contact term. Real finite-response thermometers and heat delivery require a different observation/actuation model.

$$
\dot T_1^{+}-\dot T_1^{-}=P/C_1,\quad \dot T_2^{+}-\dot T_2^{-}=0,\qquad C_1=\frac{P}{\dot T_1^{+}-\dot T_1^{-}},\quad K=aC_1,\quad C_2=K/b.
$$

For a controlled boundary heat input, the two bodies' energy grows at P. Their entropy change includes P/T₁ as well as internal contact production. Calling the whole expression the old isolated-system production would silently change the boundary. This supplement specifies heat input; an electrical work source and heater would require their own entropy accounting.

$$
\dot U_{\rm bodies}=P,\qquad \dot S_{\rm bodies}=\frac{Kd^2}{T_1T_2}+\frac P{T_1}.
$$

If an ideal thermal source at positive Tₛ supplies that power, its entropy decreases at P/Tₛ. The combined expression is nonnegative when Tₛ≥T₁. This condition must hold during the modeled interval. An imposed P is a calibration idealization, not a passive conductance law or a guarantee that an arbitrary device can maintain it.

$$
\dot S_{\rm bodies+source}=\frac{Kd^2}{T_1T_2}+P\left(\frac1{T_1}-\frac1{T_s}\right)\ge0\qquad(P>0,\ T_s\ge T_1>0).
$$

### Read the ambiguity diagram

The upper panel shows a single pair of unforced temperature curves shared by three parameter scales. The lower panel shows their different transferred energies. Both panels use seconds on the horizontal axis; the lower vertical axis is J rather than K. The separate calibration step in the table is not applied to these unforced curves.

![Three common scales of heat capacities and conductance produce identical temperatures but different transferred energies.](/assets/learning/heat-work-contact-scale.en.svg)

### Reproduce the common-scale family

Use C₁=2s J/K, C₂=3s J/K, K=0.4s W/K and initial temperatures 330 K and 270 K. Columns T₁ and Q refer to the unforced state at t=3 s. The last column instead predicts the immediate slope jump for a separate calibrated P=2 W heat-input step. Its change with s supplies information absent from the unforced temperatures.

| s | C₁ (J/K) | K (W/K) | T₁ at 3 s (K) | Q at 3 s (J) | Slope jump (K/s) |
| --- | --- | --- | --- | --- | --- |
| 0.50000000 | 1.00000000 | 0.20000000 | 307.24365988 | 22.75634012 | 2.00000000 |
| 1.00000000 | 2.00000000 | 0.40000000 | 307.24365988 | 45.51268024 | 1.00000000 |
| 2.00000000 | 4.00000000 | 0.80000000 | 307.24365988 | 91.02536047 | 0.50000000 |
| 5.00000000 | 10.00000000 | 2.00000000 | 307.24365988 | 227.56340118 | 0.20000000 |

```python
from math import exp
for s in [0.5, 1.0, 2.0, 5.0]:
    C1, C2, K = 2*s, 3*s, 0.4*s
    Ce = C1*C2/(C1+C2)
    lam = K/Ce
    Te = (C1*330+C2*270)/(C1+C2)
    T1 = Te + C2/(C1+C2)*60*exp(-lam*3)
    Q = Ce*60*(1-exp(-lam*3))
    jump = 2.0/C1  # separate calibrated 2 W heat-input step
    print(f"{s:.8f} {C1:.8f} {K:.8f} {T1:.8f} {Q:.8f} {jump:.8f}")
```

### Three research questions with explanations

#### 1. Would recording the same unforced temperature curves for longer remove the common scale?

**Explanation.** No. The scaling identity preserves the entire ideal trajectory, including all finite intervals and the limiting temperature. Longer recording may help noisy rate estimation, but it cannot remove an exact structural ambiguity. A measured energy transfer in J, an independently calibrated capacity, or the specified known heat input changes the information supplied.

#### 2. Do energy conservation and nonnegative entropy production identify C₁, C₂ and K?

**Explanation.** No. Every common positive scale satisfies those properties for the same temperature record, with scaled energy and entropy magnitudes. Their signs and dimensionally consistent balances do not fix magnitudes. The rank-two rate Jacobian makes the missing direction explicit.

#### 3. Can one use P divided by the observed temperature slope to obtain C₁?

**Explanation.** Not generally. While contact transfer is present, C₁ times that slope equals P−Kd. The difference of immediate slopes at the same continuous state cancels Kd, giving P/C₁. Reconstruction of both capacities and K also requires the separately identified positive a,b; equal-temperature unforced data or a disconnected contact do not supply them.

### Source and model boundary

[David Tong, Classical Thermodynamics, sections 4.1–4.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) provides the system-boundary, heat/work and entropy conventions. The constant-capacity inverse problem, common-scale Jacobian and calibrated slope jump are derived here for the stated lumped model. They do not identify a spatial conductivity field or certify a real calorimeter. The Python supplement and diagram add research reasoning to the existing reader; they do not introduce a new interactive lab control.

## Sources and the next connection

The discussion of temperature, the first law and entropy in [David Tong, Classical Thermodynamics](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) supports the distinctions between systems, boundaries and transferred quantities. The constant-capacity bodies, linear connection and time-dependent solution are the specialization declared in this unit. Next, compare ideal-gas paths with identical endpoints but different heat and work.
