## Changing identity while a cloud moves

Imagine a dissolved tracer whose molecules switch between two observable states, A and B. A moving cloud can become less blue and more red even while its total material stays constant. A detector sensitive only to A would report a decreasing signal. That observation alone cannot distinguish chemical conversion from material leaving the detector's field of view. We need a balance for each state and another balance for their sum.

This lesson constructs an exactly solvable family rather than a general reaction solver. The total cloud follows the Gaussian advection-diffusion field from the preceding lesson. A spatially uniform composition changes through reversible first-order kinetics. Factoring those two processes lets us derive every concentration, derivative and boundary flux, then test the result against independent calculations. The central question is practical: how much of a measured change comes from transport, and how much from conversion?

## Assumptions and units

The tube is an infinite straight line with constant cross-sectional area A. The two labels have the same mass basis and share a constant velocity U and diffusivity D. Forward and reverse rates are constant, nonnegative and independent of concentration. The initial fraction of label A is the same at every position. We prescribe transport; pressure, temperature, momentum and feedback on the flow are outside this model. Equal labels are useful for learning, but they do not describe arbitrary molecular masses or reaction stoichiometry.

| Quantity | Unit | Interpretation |
| --- | --- | --- |
| c_A, c_B | kg/m³ | mass concentration of each label |
| j_A, j_B | kg/(m² s) | signed advective plus diffusive flux |
| k_f, k_b | 1/s | forward and reverse rate constants |
| S_A, S_B | kg/(m³ s) | net conversion sources |
| M | kg | finite total mass on the entire line |

$$
j_i=Uc_i-D\partial_xc_i,\qquad \partial_tc_i+\partial_xj_i=S_i,\qquad i\in\{A,B\}.
$$

Positive flux points toward increasing x. A positive source increases the named species locally; it need not increase the total material. Keeping these meanings separate prevents a common sign error when interpreting a rising or falling concentration.

## Derive the coupled balance

Each forward event removes A and adds the same mass of B. Each reverse event does the opposite. The net A source is reverse turnover minus forward turnover. Adding the two species equations cancels these sources exactly. This cancellation is a statement about the assumed stoichiometry, not a numerical achievement of a particular discretization.

$$
S_A=-k_fc_A+k_bc_B,\qquad S_B=-S_A,\qquad c=c_A+c_B,\qquad c_t+Uc_x=Dc_{xx}.
$$

The sum obeys a source-free transport equation even when each species changes rapidly. With different diffusivities the sum still has a conservation law, but its flux generally cannot be written using one common D times the total concentration gradient. That is one reason the following separation needs its assumptions.

## Separate composition from the Gaussian carrier

Use a smooth initial cloud with positive initial width. Its mass is M, its initial center is mu-zero and its initial standard deviation is sigma-zero. Write the species as fractions of that common carrier. This is consistent because the initial fractions are spatially uniform and both labels experience the same transport operator.

$$
v(t)=\sigma_0^2+2Dt,\quad \mu(t)=\mu_0+Ut,\quad c=\frac{M}{A\sqrt{2\pi v}}e^{-(x-\mu)^2/(2v)},\quad c_A=f_A(t)c,\quad c_B=f_B(t)c.
$$

Area converts the mass-per-length Gaussian into volume concentration. It is a physical cross-section, not a probability normalization. Substituting the product into the A equation leaves the carrier equation multiplied by f_A plus the additional term f_A' c. The carrier terms cancel, reducing the reaction problem to a scalar ordinary differential equation.

$$
f_A'=-k_ff_A+k_b(1-f_A),\qquad f_B=1-f_A,\qquad f_A+f_B=1.
$$

These are identities of the analytic model. In floating-point calculations, storing the two positive fractions separately can retain information that subtracting a rounded fraction from one would lose.

## Relaxation and the zero-rate exception

When the sum of rates is positive, define its reciprocal as the relaxation timescale. The equilibrium composition balances forward and reverse turnover. The approach to equilibrium is exponential; it does not depend on the cloud's location, width or speed under the assumptions above.

$$
\lambda=k_f+k_b>0,\qquad f_A^*=\frac{k_b}{\lambda},\qquad f_A(t)=f_A^*+(f_{A0}-f_A^*)e^{-\lambda t},\qquad \tau=\lambda^{-1}.
$$

If both rates vanish, every initial composition remains stationary. There is no uniquely preferred equilibrium fraction and no meaningful relaxation time from this formula. Assigning equilibrium zero in that case would invent a preference absent from the equation. If M is zero, the declared kinetic fraction can still be evolved mathematically, but the measured fraction obtained by dividing species mass by total mass is undefined.

## Gross turnover is different from net conversion

An equilibrium mixture can exchange labels continually. Equal forward and reverse turnover gives zero net source without requiring either turnover to vanish. This distinction matters whenever an instrument measures event counts or energy expenditure rather than just concentration change.

$$
R_f=k_fc_A,\qquad R_b=k_bc_B,\qquad S_A=R_b-R_f,\qquad S_A+S_B=0.
$$

For equal positive rates and equal fractions, the source is zero while both turnover rates are positive wherever the carrier concentration is positive. Conversely, a source-free observation at one point may simply occur because no material is present there. A useful visualization should therefore show carrier concentration, both turnovers and the signed net source together rather than presenting a zero net source as proof that nothing happens.

## Derivatives reveal the source term

Differentiate the actual product field. Its time derivative includes the changing carrier and changing composition. Its spatial derivative has no fraction-gradient contribution because the fraction is spatially uniform. The source term supplies precisely the additional time dependence.

$$
\partial_tc_A=f_Ac_t+f_A'c,\qquad \partial_xj_A=f_A\partial_xj,\qquad \partial_tc_A+\partial_xj_A=f_A'c=S_A.
$$

Omitting reaction from the equation while evaluating this same field leaves a residual equal to the source. This is a useful negative control: a source-free equation should fail on a genuinely reacting field. Replacing the field with a different nonreacting solution would hide that failure and would answer a different question.

## A worked composition calculation

Choose rates 1/s and 0.5/s, initial A fraction 0.8 and total mass 2 kg. After one second, the mixture has moved substantially toward its equilibrium A fraction of one third. Round only for presentation; use the exact expression when computing the subsequent balance.

$$
f_A(1)=\frac13+\left(0.8-\frac13\right)e^{-1.5}\approx0.43746,\qquad M_A(1)\approx0.87492\ \mathrm{kg},\qquad M_B(1)\approx1.12508\ \mathrm{kg}.
$$

The A mass has fallen from 1.6 kg, but the total remains 2 kg. At this time f_A' is approximately minus 0.15619 per second. Consequently the whole-line A source is approximately minus 0.31238 kg/s and the B source has the opposite sign. Changing U moves both clouds together; changing D spreads both clouds. Neither change alters this composition curve in the separated family.

## Finite windows need boundary flux and reaction

An instrument sees a finite region, not the whole infinite tube. Integrate the local equation over a fixed window and multiply by area. Right-boundary flux minus left-boundary flux is net outflow. Stored mass can decrease through positive net outflow, a negative integrated source, or both.

$$
M_i^{[a,b]}=A\int_a^bc_i\,dx,\qquad \frac{dM_i^{[a,b]}}{dt}+A[j_i(b)-j_i(a)]=A\int_a^bS_i\,dx.
$$

Adding the A and B window budgets cancels the integrated reaction source, but boundary transport remains. Therefore total mass in the window need not be constant. By contrast, on the whole line the Gaussian boundary flux vanishes at infinity, so total mass is constant. A plot should label whether a displayed mass belongs to the observation window or the entire line.

## Stable small changes and tiny tails

Near an initial pure-A state, the newly formed B fraction can be positive even when the displayed A fraction rounds to one. Compute both fractions with positive expressions and use an exponential-minus-one operation for small relaxation increments. Let d-zero denote the initial A-fraction derivative.

$$
d_0=k_b(1-f_{A0})-k_ff_{A0},\qquad f_A'(t)=d_0e^{-\lambda t},\qquad \Delta f_A=\frac{d_0}{\lambda}(1-e^{-\lambda t}).
$$

The final expression applies only for positive lambda. With a forward rate of 10 to the minus 20 per second, no reverse rate and one second of evolution, a pure-A state produces a B fraction close to 10 to the minus 20. Ordinary subtraction of the rounded A fraction from one returns zero. At the other extreme, a rapidly decaying transient can be smaller than the smallest representable positive number. Its linear observation should be unavailable while its sign and logarithmic magnitude remain usable; it should not be relabeled an exact zero. Likewise, a Gaussian tail outside the visible cloud remains part of the model.

Very small rates also need care before the exponential is evaluated. A derivative initially halfway between zero and the smallest positive number becomes slightly smaller in magnitude after any positive relaxation time. Rounding the decay factor to one first can place that observation on the wrong side of a rounding boundary. For a small dimensionless exponent, the implementation retains polynomial corrections and complete input products and ratios before rounding. A mass-weighted source can remain representable even when the fraction derivative is unavailable, so the two observations are evaluated separately. This treatment does not turn the Gaussian quadrature or every transcendental calculation into an exact arithmetic solver.

Exact stationarity is a separate condition. With rates two and one per second the mathematical equilibrium is one third. A binary floating-point input written as one third is a nearby rational number; its small initial derivative need not be exactly zero even if both displayed fractions look identical. Equal rates with an initial fraction of one half provide an exactly stationary contrast. Compare the declared inputs, signed derivative and stable change before calling a rounded equality an equilibrium measurement.

## A visualization experiment and a research checklist

First hold transport fixed and vary the forward rate. Compare the two colored species, their sum and the composition curve. Next freeze the rates and vary velocity; the combined cloud moves without changing the whole-line species fractions. Finally place a narrow observation window near one edge. Follow storage, boundary outflow and integrated reaction independently. The window experiment explains why a local concentration record alone generally cannot identify a reaction rate without transport information.

For a physics-informed model, a small training residual is only one diagnostic. Check concentrations and derivatives against an independently constructed field, integrate actual sources over physical windows and test the source-omission control. The accompanying references use high-precision differentiation and integration, separate from the product implementation. Reusing the same formula to define and validate every term can make a sign error invisible. Extensions with spatially varying initial fractions, unequal diffusivities or temperature-dependent rates require new derivations and tests.

## Exercises with worked answers

### Exercise 1 — Find the equilibrium

For forward rate 2/s and reverse rate 1/s, what A fraction is stationary?

**Answer:** One third. At that fraction, forward turnover is twice one third times c and reverse turnover is one times two thirds times c. They agree, although each is nonzero for positive c.

### Exercise 2 — Explain a disappearing A signal

An A-sensitive detector reports decreasing concentration in a fixed window. Has total material necessarily disappeared?

**Answer:** No. A can convert into B, and either species can cross the window boundary. Measure or infer both species and boundary fluxes, then compare the summed window budget. Whole-line conservation does not imply constant window mass.

### Exercise 3 — Handle two zero rates

Both rates are zero and the initial A fraction is 0.7. What fraction and equilibrium should be reported?

**Answer:** The fraction stays 0.7. Every fraction is stationary, so report a nonunique equilibrium family rather than a selected equilibrium value or a finite relaxation time.

### Exercise 4 — Detect an omitted source

For the reacting field, what does the A residual become if reaction is omitted from the equation?

**Answer:** It equals f_A' c, the actual A source. Its sign indicates net production or loss. The same-field control tests whether the equation needs reaction without changing the underlying transport field.

### Exercise 5 — Interpret a rounded fraction

A rounds to fraction one while a stable calculation reports a small positive B fraction. Must the implementation force B to zero?

**Answer:** No. A lost the small difference through rounding. Preserve the independently calculated B value and distinguish a rounded sum from an exact analytic normalization. Clamping or renormalizing would discard the information being studied.

### Exercise 6 — Break the separation

Would the product construction remain valid if the initial A fraction varied along the tube?

**Answer:** Generally no. Spatial derivatives of the fraction introduce additional gradient terms. The two species then require a spatial reaction-transport solution; the single time-dependent fraction no longer describes the full composition field.

## Sources and scope

[Eymard, Hilhorst and Olech, Numerical approximation of a reaction-diffusion system with fast reversible reaction](https://arxiv.org/html/0808.0251v1) provides a primary example of reversible coupled reaction-diffusion equations and discusses bounded-domain conditions. Our infinite-line, common-transport Gaussian family is a separate teaching specialization derived above. The paper's numerical convergence results do not certify this implementation or transfer its boundary conditions to our model.
