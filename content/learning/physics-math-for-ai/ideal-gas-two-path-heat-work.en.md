## Can endpoints determine energy expenditure?

Two routes between the same places can require different amounts of work. Likewise, initial and final thermodynamic states do not separately determine heat and work. We will calculate two processes with **the same internal-energy and entropy changes but different heat and work**.

Consider a closed system containing n mol of a monatomic ideal gas. T is measured in K, V in m³ and p in Pa. Use R=8.31446261815324 J/(mol K) and the constant molar heat capacity Cᵥ=3R/2. This is a specified ideal-gas, constant-heat-capacity model, rather than a description of real gases at every pressure and temperature.

## Fix the boundary and signs

Heat Q is positive into the gas. Work W is positive when the gas does work on its surroundings. With no matter transfer or other work modes,

$$
\Delta U=Q-W,\qquad pV=nRT,\qquad dU=nC_v\,dT.
$$

Expansion gives W>0 and compression gives W<0. Pressure is calculated from T and V; it is not an additional independent control. The work formula W=∫p dV applies here to frictionless, quasistatic volume control in the limit where boundary pressure matches the gas pressure. It does not describe an arbitrary sudden expansion.

## Construct two two-stage paths

The initial state is (T₀,V₀), and the final state is (T₁,V₁).

| Path | First stage | Second stage |
| --- | --- | --- |
| A | Hold V₀; change T₀→T₁ | Hold T₁; change V₀→V₁ |
| B | Hold T₀; change V₀→V₁ | Hold V₁; change T₀→T₁ |

A constant-volume stage is vertical on a p–V plot. Pressure changes, but dV=0 and the volume work vanishes. An isothermal stage follows p=nRT/V. If T₁>T₀, A's expansion curve lies above B's and encloses a larger area under the curve.

$$
W_A=nRT_1\ln(V_1/V_0),\qquad W_B=nRT_0\ln(V_1/V_0).
$$

These follow directly from ∫(nRT/V)dV=nRT ln(V₁/V₀). The logarithm takes a dimensionless volume ratio. For compression that ratio is below one, giving negative work.

## Calculate heat stage by stage

At fixed volume, W=0 and Q=nCᵥ(T₁−T₀). At fixed temperature, an ideal gas has ΔU=0 and therefore Q=W. Combining the stages gives

$$
\Delta U_A=\Delta U_B=nC_v(T_1-T_0),\qquad
Q_A=\Delta U+W_A,\quad Q_B=\Delta U+W_B.
$$

If A performs more work while achieving the same final internal energy, it must receive correspondingly more heat. The difference Q−W determines the state change; Q and W individually remain path quantities.

## Does entropy depend on the path?

Specify reversible heat exchange for the realization of both paths. Reversible constant-volume heating requires a succession of reservoirs that differ infinitesimally in temperature from the gas at each stage. Contact with a single reservoir at the final temperature is not that reversible process. Slow control alone does not remove friction or irreversibility due to a finite temperature difference.

$$
\Delta S=\int\frac{\delta Q_{\rm rev}}{T}
=nC_v\ln(T_1/T_0)+nR\ln(V_1/V_0).
$$

Temperature changes during heating, so dividing total Q by final T is generally incorrect. Both paths give the same gas entropy change. Total entropy production for the gas and reservoirs is zero in the specified reversible realization. **The gas's entropy change** is distinct from **entropy production in the combined system**.

## Numerical example and figures

Set n=1 mol, T₀=300 K, T₁=330 K, V₀=0.010 m³ and V₁=0.020 m³. Then ΔU≈374.151 J, W_A≈1901.838 J and W_B≈1728.944 J. Consequently Q_A≈2275.989 J and Q_B≈2103.095 J. Both entropy changes are approximately 6.951826 J/K. A receives an additional 172.894 J of heat and performs the same additional amount of work.

The experiment shows the current position on each p–V path, together with plots of Q, W, ΔU and ΔS accumulated **along that path**. Equal progress fractions need not correspond to equal intermediate states. Progress denotes path position, not physical time, speed or power. Pressure, curve area and accumulated quantities must all respond to the same state equation.

## Six questions with worked answers

### Exercise 1: Do different heat inputs with equal ΔU violate conservation?

**Answer:** No. Q_A−Q_B=W_A−W_B, so Q−W gives the same ΔU in both cases.
### Exercise 2: What if V₀=V₁?

**Answer:** Both volume works vanish and both paths reduce to the same constant-volume temperature change, with Q=ΔU.
### Exercise 3: What if T₀=T₁?

**Answer:** Both paths reduce to the same isothermal process. ΔU=0 and Q=W=nRT₀ ln(V₁/V₀).
### Exercise 4: What happens when the whole process is reversed?

**Answer:** Reversing the old A produces the new B as defined from the swapped endpoints; reversing the old B produces the new A. Thus new W_A=−old W_B. Negating the old A while retaining the same path label would compare different paths.
### Exercise 5: Is a slowly expanding gas always reversible?

**Answer:** No. Friction and finite-temperature-difference heat transfer can still generate entropy. Quasistatic mechanical control is a different condition from reversibility of the complete process.
### Exercise 6: What happens if we follow A forward and B backward?

**Answer:** The resulting cycle has ΔU=ΔS=0. Its net work is nR(T₁−T₀)ln(V₁/V₀), and net heat equals net work. This measures the difference between the paths; it does not imply that all heat is supplied by a single reservoir.

## Progress is different from time

Assign half of the progress range to each stage. In A's first half, temperature varies linearly from T₀ to T₁. In its second half, volume varies linearly from V₀ to V₁. B reverses this order. Pressure is recalculated from the current temperature and volume. At progress 0.5, the intermediate states of A and B are supposed to differ.

This is a parametrization for tracing the paths. It does not prescribe equal physical durations, speeds or powers. A physical time model would need heat-transfer laws for the reservoirs and control and response equations for the piston. This unit does not solve those dynamics.

## Do equal-entropy endpoints imply an adiabatic path?

No. The endpoints of this ideal gas have equal entropy when

$$
\left(\frac{T_1}{T_0}\right)^3\left(\frac{V_1}{V_0}\right)^2=1.
$$

For example, T₀=252 K, T₁=343 K, V₀=343/32768 m³ and V₁=216/32768 m³ satisfy this condition exactly. At n=1 mol, total heat is approximately −183.925 J along A and 165.974 J along B. Both give ΔS=0, but neither has zero net heat. Equal values of a state function at the endpoints differ from an adiabatic process with no heat transfer at any stage.

## Reading nearly cancelling heat contributions

When internal-energy change and work have opposite signs, heat can be very small. Consider path A with T₀=251 K, T₁=271 K, V₀=0.015625 m³, V₁=0.013987598122283916 m³ and n=1 mol. Adding the rounded ΔU and W gives zero, whereas evaluating the supplied inputs gives a positive Q of approximately 3.47×10⁻¹⁴ J.

This does not claim that an experiment could measure such a small heat. It distinguishes mathematical zero from numerical cancellation in the declared model. The calculation narrows upper and lower logarithm bounds to determine the sign and magnitude in this case. A result that cannot be resolved to the specified accuracy remains unresolved rather than being replaced with physical zero.

## Connection to AI models

A regression model given only final and initial (T,V) can represent U and S differences, but cannot determine process-dependent Q and W without path information. This is missing input information, rather than merely too little training data. Even when a model receives the path, check conservation, units and the validity range of the state equation separately.

## Research extension: can total heat and work reveal the path?

The two original paths establish that endpoints do not determine heat and work. The reverse question is equally useful: if endpoints and the total heat and work are known, does that identify the intermediate path? We construct different smooth positive-temperature paths whose totals agree exactly, then choose an extra observation that separates them.

Keep a closed monatomic ideal gas, known n>0, Cᵥ=3R/2, positive endpoint temperatures and volumes, and work positive out of the gas. Assume frictionless quasistatic volume control and the reversible heat-exchange realization specified earlier. These are equilibrium paths, not solutions for a finite-speed piston or a heat-transfer controller. The variable u is dimensionless path position, not time.

Use logarithmic volume position and two temperature-shape coefficients b,c, measured in K. Both shape functions vanish at the endpoints, so every member has the same initial and final state. This smooth family supplements the original two-stage paths; it does not change how their existing progress control is defined.

$$
u\in[0,1],\quad L=\log(V_1/V_0),\quad V=V_0e^{Lu},\quad T_{b,c}(u)=T_0+(T_1-T_0)u+b f(u)+c h(u),\quad f=u(1-u),\quad h=u(1-u)(1-2u).
$$

A convenient sufficient bound keeps every temperature positive. It is intentionally conservative: failing this bound does not prove that a particular path becomes negative. All plotted and tabulated members satisfy it.

$$
|f|\le\tfrac14,\quad |h|\le\tfrac14,\qquad T_{b,c}\ge\min(T_0,T_1)-\frac{|b|+|c|}{4}>0\quad\text{if }|b|+|c|<4\min(T_0,T_1).
$$

Substitute the actual pressure and volume derivative into the work integral. The f contribution integrates to 1/6; the antisymmetric h contribution integrates to zero. Work therefore changes with b but is completely blind to c.

$$
W_{b,c}=\int_0^1\frac{nRT_{b,c}}{V}\,\frac{dV}{du}\,du=nRL\left[\frac{T_0+T_1}{2}+\frac b6\right].
$$

Internal-energy change is fixed by the endpoint temperatures. The first law then fixes total heat once work is known. Thus measuring Q as well as W provides an energy-balance check, but no independent path-shape information in this specified model.

$$
\Delta U=nC_v(T_1-T_0),\quad Q_{b,c}=\Delta U+W_{b,c},\quad W_{b_2,c_2}-W_{b_1,c_1}=Q_{b_2,c_2}-Q_{b_1,c_1}=\frac{nRL}{6}(b_2-b_1).
$$

In the reversible realization, divide the differential heat by the instantaneous temperature before integrating. The temperature derivative contributes only the logarithm of the endpoint ratio. All family members have the same entropy change, even when heat changes with b. No quotient of total Q by an arbitrarily selected temperature is used.

$$
\frac{\delta Q_{\rm rev}}{T}=nC_v\frac{T_{b,c}^\prime}{T_{b,c}}\,du+nRL\,du,\qquad \Delta S=nC_v\log(T_1/T_0)+nRL.
$$

At fixed b, changing c produces a genuine history difference invisible to W,Q and endpoint entropy. The zero integral is an observation null direction, not numerical cancellation. We have exhibited nonuniqueness even inside a two-parameter family; it is not a claim that these two coefficients describe all possible processes.

$$
h(0)=h(1)=0,\qquad\int_0^1h(u)\,du=0,\qquad \partial_c W=\partial_c Q=\partial_c\Delta S=0.
$$

An extra sensor at the logarithmic-volume midpoint still misses c: the perturbation h also vanishes there. Repeating that same midpoint measurement more accurately cannot distinguish the two c signs. The corresponding physical volume is √(V₀V₁), rather than the arithmetic midpoint of the two volumes.

$$
T_{b,c}(1/2)=\frac{T_0+T_1}{2}+\frac b4,\qquad \partial_c T_{b,c}(1/2)=0.
$$

At u=1/4, h is nonzero. A temperature reading there, together with an already determined b, recovers c within this exact family. The physical probe location is V₀ exp(L/4). Calibration errors, an uncertain state equation or shapes outside the family would require a larger inverse problem.

$$
T_{b,c}(1/4)=T_0+\frac{T_1-T_0}{4}+\frac{3b}{16}+\frac{3c}{32},\qquad c=\frac{32}{3}\left[T(1/4)-T_0-\frac{T_1-T_0}{4}-\frac{3b}{16}\right].
$$

If a differentiable increasing schedule u(t) is additionally prescribed, rates acquire a factor u̇. Changing the schedule changes power while leaving these ideal path integrals fixed. These chain-rule expressions supply no heat-transfer law or achievable device speed. Exact reversible exchange remains an ideal limiting construction; the existing experiment still displays path progress rather than time.

$$
\dot W=nRLT_{b,c}(u)\dot u,\qquad \dot Q=\left[nC_vT_{b,c}^\prime(u)+nRLT_{b,c}(u)\right]\dot u.
$$

The observation Jacobian for total work and quarter-position temperature has a nonzero determinant when L≠0. Total work fixes b and the quarter probe fixes c. At equal endpoint volumes, L=0, work is zero and this particular observation pair loses rank. Do not apply the division by L in that case; other temperature observations or a different experimental design would be needed.

$$
\frac{\partial(W,T(1/4))}{\partial(b,c)}=\begin{pmatrix}nRL/6&0\\3/16&3/32\end{pmatrix},\qquad \det=\frac{nRL}{64}.
$$

### Read the indistinguishable-path diagram

The upper panel plots temperatures against logarithmic volume position for b=40 K and c=0,±80 K. They meet at both endpoints and at u=1/2, while the quarter readings differ. The lower panel plots work against c for two b values. A horizontal work line is an exact null sensitivity, not a flat trajectory in time. Units distinguish temperature in K from total work in J.

![Different positive temperature histories share endpoints, midpoint and total work; a quarter-position temperature observation distinguishes them.](/assets/learning/heat-work-path-null-direction.en.svg)

### Reproduce equal totals and different histories

Use n=1 mol, T₀=300 K, T₁=330 K, V₀=0.010 m³ and V₁=0.020 m³. Every row has ΔU≈374.151 J and ΔS≈6.951826 J/K. The last three rows have identical work, heat and midpoint temperature; the quarter readings are different. These are declared path examples, not the original A and B presets.

| b (K) | c (K) | W (J) | Q (J) | T(1/4) (K) | T(1/2) (K) |
| --- | --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 1815.39109132 | 2189.54190913 | 307.50000000 | 315.00000000 |
| 40.00000000 | 0.00000000 | 1853.81206680 | 2227.96288461 | 315.00000000 | 325.00000000 |
| 40.00000000 | 80.00000000 | 1853.81206680 | 2227.96288461 | 322.50000000 | 325.00000000 |
| 40.00000000 | -80.00000000 | 1853.81206680 | 2227.96288461 | 307.50000000 | 325.00000000 |

```python
from math import log
R, n, T0, T1 = 8.31446261815324, 1.0, 300.0, 330.0
L = log(2.0)
for b, c in [(0, 0), (40, 0), (40, 80), (40, -80)]:
    W = n*R*L*((T0+T1)/2+b/6)
    Q = n*(1.5*R)*(T1-T0)+W
    quarter = T0+(T1-T0)/4+3*b/16+3*c/32
    middle = (T0+T1)/2+b/4
    print(f"{b:.8f} {c:.8f} {W:.8f} {Q:.8f} {quarter:.8f} {middle:.8f}")
```

### Three research questions with explanations

#### 1. If both total heat and total work agree, must the thermodynamic paths agree?

**Explanation.** No. At fixed b, every permitted c shares the same W and Q, while its temperature history changes by c h(u). Endpoint energy and entropy are also unchanged. Conservation relates Q and W, so counting them as two independent shape observations would overstate what was measured.

#### 2. Why does a midpoint thermometer fail although a quarter-position thermometer succeeds?

**Explanation.** The chosen null perturbation h is zero at u=1/2 and equals 3/32 at u=1/4. The midpoint has zero sensitivity to c; the quarter reading has nonzero sensitivity. This is specific to the declared shape family. A sensor that works for these two coefficients does not reconstruct an arbitrary unknown function.

#### 3. Does the same work imply the same power or a realizable finite-time reversible device?

**Explanation.** No. Work integrates the path, whereas power also depends on u̇ and requires an actual time schedule. The displayed chain rule does not determine piston dynamics, reservoir conductances or finite losses. A real finite-speed process needs those laws, and its actual heat and work may differ from the ideal quasistatic values.

### Source and research boundary

[David Tong, Classical Thermodynamics, sections 4.2–4.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) supports the distinction between state changes and path-dependent transfer and the reversible entropy construction. Work here has the opposite sign to work on the gas in that source. The smooth path family, observation null direction and probe Jacobian are direct derivations in this supplement. They do not establish identifiability of arbitrary thermodynamic processes or convert a training progress coordinate into physical time.

## Sources and the next connection

See the first-law, reversibility and entropy discussion in [David Tong, Classical Thermodynamics](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf). Unlike the source's work-on-system convention, work here is positive when performed by the gas. The SI definition R=N_A k_B follows the [NIST constants table](https://physics.nist.gov/cuu/Constants/Table/allascii.txt); calculations use its binary approximation. Next are thermodynamic laws and boundary conditions, including engines and refrigerators.
