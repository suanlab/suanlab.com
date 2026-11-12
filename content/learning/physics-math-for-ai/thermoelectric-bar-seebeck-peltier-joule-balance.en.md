## Why electrical resistance is not the whole thermal story

An electric current through a resistor produces Joule heating. A thermoelectric material also couples electrical transport to temperature gradients and carries reversible heat with the current. Treating all heat transport as resistive heating misses both the induced voltage and the boundary heat that can support cooling or generation.

We solve a single steady one-dimensional bar with constant properties. The aim is to follow temperature, electric potential, local work and actual boundary heat through one consistent calculation. This is useful when evaluating an AI model with separate thermal and electrical outputs: each field can look plausible while their combined power balance fails.

## Declare the bar and the directions of its ports

Length L, cross-sectional area A, electrical resistivity r and thermal conductivity K are positive. The Seebeck coefficient S is constant and signed. End temperatures are positive absolute temperatures TL and TR. Current density j is positive in the positive x direction. We prescribe j and calculate the dependent terminal voltage; we do not impose inconsistent current and voltage data simultaneously.

$$
0\le x\le L,\qquad T(0)=T_L,\quad T(L)=T_R,\qquad
[r]=\Omega\,\mathrm m,\quad[K]=\mathrm{W/(mK)},\quad
[S]=\mathrm{V/K},\quad[j]=\mathrm{A/m^2}.
$$

The bar is immobile and steady, with no volumetric heat loss, radiation, contact resistance or temperature-dependent properties. The coefficient and terminal reference are declared for this single leg. The model does not include every junction, return conductor or heat exchanger of a complete device. Very large computed temperatures require reassessing the constant-property idealization rather than interpreting the example as calibrated hardware.

## Couple heat and current with consistent signs

Electric field E is minus the spatial derivative of potential V. The current law contains the Seebeck contribution. Heat flux combines Fourier conduction with the Peltier contribution, whose coefficient is S times absolute temperature.

$$
E=-V_x,\qquad j=\frac1r(E-ST_x),\qquad
\Pi=ST,\qquad q=\Pi j-KT_x,\qquad j_x=0.
$$

Current continuity makes j spatially constant in this uniform bar. It does not imply that heat flux is constant: electrical work is being transferred into the thermal transport balance. Using only temperature deviation in the Peltier coefficient would lose the reference-temperature contribution. For instance, 300 K multiplied by S is not replaced by zero merely because the local deviation from 300 K is zero.

S is constant here, so the temperature derivative of S and the corresponding Thomson term vanish. That is an explicit material assumption, not a general statement that the Thomson effect never matters.

## Derive the temperature profile

In steady state, the derivative of heat flux equals local electrical work jE. Substituting both constitutive laws makes the reversible terms cancel from the differential equation for temperature. The remaining curvature is set by Joule heating.

$$
q_x=jE,\qquad -KT_{xx}=rj^2,\qquad
T=T_L+(T_R-T_L)\xi+
\frac{rj^2L^2}{2K}\xi(1-\xi),\qquad\xi=x/L.
$$

The last term is a nonnegative Joule temperature rise above the straight boundary interpolation. It is zero at both endpoints and largest at the midpoint. The full temperature maximum need not be at the midpoint when the end temperatures differ. A plot must distinguish the maximum of the added bulge from the maximum of the whole temperature field.

Positive boundary temperatures and a nonnegative bulge keep temperature positive in this family. Current reversal leaves the Joule bulge unchanged because it depends on j squared. It can still reverse Peltier transport, change terminal electrical work and switch the signs of the boundary heat ports.

## Recover electric potential and terminal voltage

Integrating E equals rj plus S times the temperature gradient gives potential at every point. A reference potential only chooses the zero of the graph. The terminal drop is defined from left to right.

$$
V(x)=V(0)-rjx-S[T(x)-T_L],\qquad
\Delta V=V(0)-V(L)=rjL+S(T_R-T_L).
$$

Adding the same constant to all potentials changes no electric field, current, heat flux or power. A potential value is therefore different from a potential drop. The stable potential change from the left endpoint is useful when a tiny drop is lost by adding it to a large reference value.

The same terminal relation can be rearranged to solve j from a prescribed voltage. That is an alternative parameterization. It is not an extra independent condition to add while j is already fixed. In particular, exact open-circuit current can be specified as zero without inferring it from two almost equal rounded voltages.

## Inspect local heat and work before integrating

The local electrical power density is jE. Its Joule part is nonnegative, but its reversible thermoelectric part can be signed. Differentiating the actual heat flux supplies the same sum.

$$
jE=rj^2+SjT_x,\qquad
q_x=rj^2+SjT_x.
$$

Each term has units W/m³. The heat flux itself has units W/m². Multiplying a flux density by area gives watts; integrating a volumetric work density over length and area also gives watts. These operations are different and must not lose the actual dimensions of the bar.

A local electrical work density can differ substantially from Joule heating. This does not make Joule heating negative. It reflects the reversible coupling contribution and the chosen direction of electrical work.

## Close the actual boundary-port balance

Let QL denote heat entering at the left face in the positive x direction and QR heat leaving at the right face in the same coordinate direction. Their values can be negative, indicating transport opposite to those labels. Outward normals differ at the two faces, so the net heat leaving the bar is QR minus QL.

$$
Q_L=Aq(0),\qquad Q_R=Aq(L),\qquad
P_{\rm in}=Aj\Delta V=Q_R-Q_L.
$$

The input electrical power separates into total Joule heating and reversible electrical work. The reversible part is not an additional irreversible loss.

$$
P_{\rm Joule}=Arj^2L,\qquad
P_{\rm rev}=ASj(T_R-T_L),\qquad
P_{\rm in}=P_{\rm Joule}+P_{\rm rev}.
$$

For a partial interval, use the actual potential drop to that position and the heat difference between that position and the left boundary. A steady bar stores a temperature distribution, but this calculation does not integrate a transient heat capacity or claim that steady state means zero thermal inventory.

## Work through a dimensional example

Choose L equals 1 m, A equals 1 m², r equals 1 ohm meter, K equals 1 W/(m K), S equals 0.01 V/K, TL equals 300 K, TR equals 310 K and j equals 0.9 A/m². The terminal drop is 1 V and current is 0.9 A. At the midpoint, temperature is 305.10125 K.

Left heat input is minus 7.705 W and right heat output is minus 6.805 W. Thus heat enters from the right reservoir and leaves toward the left, while electrical input adds 0.9 W to the net boundary transport. Joule heating is 0.81 W and reversible electrical work is 0.09 W. The difference of boundary heat equals their sum. Using Joule heating alone would miss one tenth of the total electrical input in this example.

These coefficients define an educational idealization. The calculation demonstrates units and signs; it is not a performance specification for a particular thermoelectric product.

## Classify cooling and generation by the actual ports

Positive electrical input does not guarantee cooling of the left reservoir. Cooling at that port requires positive QL as well. Define a cooling coefficient of performance only when both quantities are positive.

$$
\mathrm{COP}_L=\frac{Q_L}{P_{\rm in}}
\quad\text{only if }Q_L>0\text{ and }P_{\rm in}>0.
$$

As a second example, retain the same geometry and end temperatures but choose r equals 0.01 ohm meter, K equals 0.1 W/(m K), S equals 0.05 V/K and j equals 10 A/m². The dependent terminal drop is 0.6 V. QL is 148.5 W, QR is 154.5 W and electrical input is 6 W, giving COP 24.75. This remains an ideal single-leg example with explicitly chosen coefficients.

Returning to the first material with j equal to minus 0.05 A/m² gives a terminal drop of positive 0.05 V and electrical input of minus 0.0025 W. The bar delivers electrical power under the declared terminal conditions. Label this as generation; do not attach the same positive-input cooling ratio to it. A missing or zero denominator is not a zero COP.

## Preserve zero-current limits and tiny nonzero effects

At exact zero current, Joule heating and Peltier heat transport vanish. Fourier heat conduction can remain, and a nonzero terminal voltage can remain. Open circuit is therefore not synonymous with zero temperature gradient or zero voltage.

$$
j=0:\qquad
\Delta V=S(T_R-T_L),\quad
q=-K\frac{T_R-T_L}{L},\quad P_{\rm in}=0.
$$

If the end temperatures are equal and current is tiny but nonzero, the temperature bulge and Joule power are quadratic in current. The leading Peltier boundary transport is linear in current.

$$
T-T_{\rm linear}=\frac{rj^2L^2}{2K}\xi(1-\xi),\qquad
T_R=T_L:\quad P_{\rm in}=Arj^2L.
$$

Ordinary boundary heat values may then round to the same number even though their difference is positive. We retain the stable power difference, small temperature rise, sign and logarithm separately. Exact zero is displayed differently from a nonzero observation outside the ordinary numeric range. A very large finite cooling ratio can also be available only through a logarithm; it is not mathematical infinity.

## Separate reversible transport from entropy production

For this constant-property family, differentiating the entropy flux q/T gives a nonnegative sum. The reversible cross terms cancel, whereas electrical resistance and Fourier conduction produce entropy.

$$
\frac{d}{dx}\left(\frac qT\right)
=\frac{rj^2}{T}+\frac{K(T_x)^2}{T^2}\ge0.
$$

This identity helps explain why Peltier transport should not be counted as a second Joule loss. It does not make the whole device reversible: resistance and conduction still matter. When comparing with heat reservoirs, use their absolute temperatures and the actual signs of heat entering and leaving them.

## Use the family to test scientific AI assumptions

A thermal surrogate and an electrical surrogate should share geometry, material conventions and terminal conditions. Compare the independent derivative of potential with E, the derivative of heat flux with jE and the integrated electrical work with the boundary heat difference. Also test potential-reference shifts and exact zero-current cases.

A small algebraic residual assembled from the same predicted quantities is not independent evidence. Use separate field derivatives, physical-coordinate integrals and held-out parameter cases. The present family excludes variable Seebeck coefficients, contact jumps, nonlinear properties and time-dependent storage. Adding these mechanisms requires extending the equations and evidence rather than merely extrapolating a curve beyond its assumptions.

## Check a nearly cancelled terminal voltage

A displayed decimal is stored as a binary input. With r=L=1, S=0.01 V/K, TR−TL=10 K and j=−0.1 A/m², the two voltage terms look equal and opposite. For those actual stored inputs, their sum is about −3.46945×10⁻¹⁸ V. With A=1 m², electrical input is about +3.46945×10⁻¹⁹ W. Calling it generation or exact zero would give the wrong terminal regime. By contrast, S=0.015625 V/K, TR−TL=8 K and j=−0.125 A/m² are exactly cancelling binary inputs.

The constant-property bar uses exact rational operations on its supplied binary inputs before projecting each observable. Voltage, local electrical work, heat-flux derivative and physical interval integrals therefore retain the same small signed contribution. Ordinary boundary heat subtraction can still round to zero; compare the separately retained difference. This exact algebra applies to this rational bar family, not to every transcendental calculation elsewhere in the book.

## Research extension: load matching and the objective of a generator

The earlier bar permits an imposed signed current. A generator connected to a passive resistor has a different experimental control: the load determines that current together with the terminal law. Keep the original right end hot and the left end cold, positive constant Seebeck coefficient and fixed reservoir temperatures. Define I positive in the generating direction, opposite to the original positive-x current density. R is an electrical resistance in ohms; H is a thermal conductance in watts per kelvin, not the thermal conductivity K.

$$
T_R=T_h>T_c=T_L>0,\quad S>0,\quad\Delta T=T_h-T_c,\quad I=-Aj>0,\qquad R=\frac{rL}{A},\quad H=\frac{KA}{L},\quad\mathcal E=S\Delta T.
$$

A positive load resistance receives the terminal voltage and current. Substitute the generating current sign into the old terminal relation and solve the resulting circuit, rather than adding an arbitrary voltage condition to an already fixed current. The ratio m compares external and internal electrical resistances. This ideal circuit uses the declared single-leg terminal reference and an ideal return; contact jumps, lead thermopower and heat exchangers are still excluded.

$$
m=R_{\rm load}/R>0,\qquad\Delta V=\mathcal E-RI=R_{\rm load}I,\qquad I=\frac{\mathcal E}{R(1+m)}.
$$

Useful electrical power is minus the original electrical input. It can be computed from either the load or the internal emf and loss. Differentiate its load-ratio factor: the unique maximum for positive m occurs at resistance matching. A temperature profile that looks favorable does not itself identify this electrical operating point. The half-open limits m approaching zero and infinity both yield zero useful power even though their heat and current states differ.

$$
P_{\rm out}=I\Delta V=\mathcal E I-RI^2=\frac{\mathcal E^2}{R}\frac m{(1+m)^2},\qquad\frac{d}{dm}\frac m{(1+m)^2}=\frac{1-m}{(1+m)^3},\quad m_{\rm power}=1.
$$

Track the actual hot and cold heat ports before defining efficiency. Heat drawn from the hot right reservoir is minus the old right-going QR. Heat rejected into the cold left reservoir is minus QL. Evaluate the original temperature gradient at each endpoint to obtain the two expressions. Half of the distributed Joule heat appears at each end for these constant properties and fixed temperatures; that split is not a general contact or nonlinear-material law.

$$
Q_h=-Q_R=ST_hI+H\Delta T-\frac{RI^2}{2},\qquad Q_c=-Q_L=ST_cI+H\Delta T+\frac{RI^2}{2},\qquad Q_h-Q_c=P_{\rm out}.
$$

Both heat quantities are positive for the positive-load generating family. The identity closes the first law but does not determine the efficiency optimum. Define the thermal-to-electrical merit parameter Z and its dimensionless product with mean absolute temperature. Geometry cancels from Z for this uniform leg. Increasing area scales both conductances and power, but it does not increase Z for unchanged material properties.

$$
Z=\frac{S^2}{RH}=\frac{S^2}{rK},\quad[Z]=\mathrm{K}^{-1},\qquad\overline T=\frac{T_h+T_c}{2},\quad Z\overline T>0,\qquad\eta=\frac{P_{\rm out}}{Q_h}.
$$

Substitution of the load current gives a compact efficiency curve. This expression still includes the heat leak through the material and the Joule contribution to the hot port. Its denominator is positive in the declared family. A large Seebeck voltage alone is insufficient to predict efficiency because electrical resistance and thermal conduction also enter. Use absolute reservoir temperatures here, not deviations from an arbitrary reference.

$$
\eta(m)=\frac{\Delta T\,m}{T_h(1+m)-\Delta T/2+(1+m)^2/Z}.
$$

Differentiate this ratio while keeping temperatures and material properties fixed. Its derivative has the sign of one plus Z times mean temperature minus m squared. Thus the unique efficiency optimum has a larger load resistance than the maximum-power point. The distinction is an optimization question, not a failure of either calculation: the objective in the denominator changed from nothing to consumed hot-side heat.

$$
\operatorname{sgn}\eta^{\prime}(m)=\operatorname{sgn}(1+Z\overline T-m^2),\qquad m_{\rm efficiency}=\sqrt{1+Z\overline T}>1.
$$

At that load the familiar closed expression follows directly. For finite positive Z it is below the Carnot ratio. Its formal approach to Carnot as Z grows is a limit of this ideal constant-property model; it is not a claim about finite-power manufactured hardware or coefficients that remain unchanged over arbitrarily large temperature ranges. Maximum efficiency and maximum power generally occur at distinct currents.

$$
\eta_{\max}=\frac{\Delta T}{T_h}\frac{\sqrt{1+Z\overline T}-1}{\sqrt{1+Z\overline T}+T_c/T_h}<\frac{\Delta T}{T_h}.
$$

The second law supplies a separate check using both reservoirs. The positive entropy increase is a Fourier heat-leak term plus an electrical-resistance term. The reversible Seebeck/Peltier transport cancels from this expression. To test a coupled surrogate, independently integrate the local entropy-production density over the actual bar and compare with these reservoir ports; rearranging the port identity alone supplies no independent field evidence.

$$
\dot S_{\rm res}=\frac{Q_c}{T_c}-\frac{Q_h}{T_h}=\frac{H(\Delta T)^2}{T_cT_h}+\frac{RI^2(T_h+T_c)}{2T_cT_h}=A\int_0^L\left[\frac{rj^2}{T}+\frac{K T_x^2}{T^2}\right]dx>0.
$$

Keep the limiting observations distinct. An open circuit still conducts heat, so its efficiency tends to zero at positive temperature difference. A short circuit also gives zero useful load power while carrying finite current and dissipating heat. Exact equal reservoir temperatures remove the thermal drive: passive-load current, output and hot heat are zero, so the efficiency is undefined rather than an assigned zero ratio. The existing lab still accepts prescribed current; this resistor-coupled family is an additional worked model.

$$
m\to\infty:\ I,P_{\rm out},\eta\to0,\ Q_h\to H\Delta T>0;\qquad m\to0^+:\ P_{\rm out},\eta\to0;\qquad\Delta T=0:\ \eta=0/0\ \text{undefined}.
$$

The teaching table fixes R=0.1 ohm, H=0.2 W/K, S=0.03 V/K and reservoirs at 300 and 400 K. Columns are load ratio, current in amperes, output in watts, hot heat in watts and dimensionless efficiency. These declared coefficients are not a calibrated product.

| m | I | Pout | Qhot | efficiency |
| --- | --- | --- | --- | --- |
| 0.50000000 | 20.00000000 | 20.00000000 | 240.00000000 | 0.08333333 |
| 1.00000000 | 15.00000000 | 22.50000000 | 188.75000000 | 0.11920530 |
| 4.09267639 | 5.89081216 | 14.20226970 | 88.95466257 | 0.15965739 |
| 10.00000000 | 2.72727273 | 7.43801653 | 52.35537190 | 0.14206788 |

```python
from math import sqrt
# Ideal leg: R=0.1 ohm, H=0.2 W/K, S=0.03 V/K.
r, h, s, tc, th = 0.1, 0.2, 0.03, 300.0, 400.0
z = s*s/(r*h)
opt = sqrt(1+z*(th+tc)/2)
for m in (0.5, 1.0, opt, 10.0):
    current = s*(th-tc)/(r*(1+m))
    power = current*current*m*r
    hot = s*th*current+h*(th-tc)-r*current*current/2
    efficiency = power/hot
    print(" ".join(f"{v:.8f}" for v in (m, current, power, hot, efficiency)))
```

[COMSOL’s thermoelectric theory](https://doc.comsol.com/6.3/doc/com.comsol.help.heat/heat_ug_theory.07.083.html) documents the linked transport laws. The load circuit, optimization and reservoir-entropy comparison here are derived from our original bar and its explicit port signs.

![Distinct maximum-power and maximum-efficiency loads and their reservoir entropy](/assets/learning/continuum-thermoelectric-load-power-efficiency.en.svg)

### Research questions with explanations

**Does resistance matching maximize both output and efficiency?** **Answer:** No. Maximum power uses m=1, whereas maximum efficiency uses sqrt(1+Z times mean temperature) under the stated fixed-temperature, constant-property assumptions.

**Does an open circuit eliminate heat drawn from the hot reservoir?** **Answer:** No. Fourier conduction remains at positive temperature difference. Electrical output and efficiency tend to zero while hot heat stays positive.

**May equal reservoir temperatures be assigned a measured zero efficiency?** **Answer:** No. With this passive load, both output and hot heat vanish. The ratio is undefined, and a missing thermal drive is a separate control.

## Six exercises with answers

### Exercise 1 — Independent terminal data

Can arbitrary current and arbitrary terminal voltage both be imposed on this same steady bar?

**Answer:** They must satisfy the terminal relation. Choose one as the drive and solve for the other; otherwise the data can be incompatible.

### Exercise 2 — Reference potential

Add 3 V to the reference potential. What happens to electrical input and heat flux?

**Answer:** Neither changes. All potentials shift together, while electric field and potential differences remain invariant.

### Exercise 3 — Open circuit

With positive S and TR greater than TL, must open circuit have zero terminal voltage and zero heat flow?

**Answer:** No. Current and electrical power are zero, but the Seebeck voltage is S times the temperature difference and Fourier heat still flows from hot to cold.

### Exercise 4 — Boundary signs

In the first example, why is electrical input positive when both listed heat ports are negative?

**Answer:** Net outward heat is QR minus QL: minus 6.805 minus minus 7.705 equals positive 0.9 W. The signs describe coordinate directions, not a sum of two outward fluxes.

### Exercise 5 — Cooling ratio

May a positive electrical input automatically be assigned a positive cooling COP?

**Answer:** No. Verify positive heat drawn from the specified cold port as well. Generation, zero input and heat rejection at that port require different labels.

### Exercise 6 — Tiny current

When equal-temperature boundary heat readings round to the same value, is their physical difference necessarily zero?

**Answer:** No. The Peltier terms can dominate the readings while the smaller positive Joule power is lost by subtraction. Preserve the quadratic power and its logarithm separately.

## Source and limitations

[COMSOL's Thermoelectric Effect theory](https://doc.comsol.com/6.3/doc/com.comsol.help.heat/heat_ug_theory.07.083.html) describes the linked current and heat-flux laws and thermoelectric relations. We specialize those conventions to a stated constant-property one-dimensional bar and derive its profiles and port balances directly. The model does not establish the cooling or generating performance of a complete manufactured device with unmodeled contacts and return paths.
