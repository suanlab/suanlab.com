## Conserving the total does not select the right wave

Imagine a line of traffic in which the speed depends on the local state. A congested region can develop a sharp front; a spreading region can open into a fan. A numerical model can preserve the total amount on the road and still move that front incorrectly, or keep an expanding jump that should spread out.

This chapter makes that distinction visible with inviscid Burgers transport. We advance actual cell averages with two shared-flux methods: Godunov and uncorrected Roe. Both have a conservative update. For a rarefaction crossing zero characteristic speed, their interface fluxes differ, and the Roe method can preserve the wrong stationary expansion jump.

We examine three levels separately: the continuum admissibility of a shock, the entropy defect at a numerical interface, and the entropy budget after a finite time step. A successful check at one level does not replace the other two. In particular, global entropy cancellation can hide a bad interface.

## Define a nondimensional nonlinear conservation law

The state q, coordinate x, and time t are nondimensional in this example. We do not assign the quadratic quantity below units of thermodynamic entropy or physical energy. Burgers is a scalar model for studying nonlinear transport, not a complete compressible fluid simulation.

$$
q_t+\partial_x f(q)=0,\qquad f(q)=\frac{q^2}{2},\qquad f'(q)=q.
$$

For a smooth solution the transport speed equals the state itself. Larger positive values move right faster; more negative values move left faster. Characteristics can converge and form a shock, or diverge and form a rarefaction. Once a discontinuity exists, the pointwise product q times its spatial derivative is not a substitute for the integral conservation law.

Our initial data have one jump, from a constant left state to a constant right state. The jump is aligned with a cell boundary at one quarter, one half, or three quarters of the computational interval. This alignment keeps initialization unambiguous: no cell begins with an accidentally sampled mixture.

## Select the admissible continuum wave

A compressive jump has a larger left state than right state. Its Rankine–Hugoniot speed follows from conservation across the moving front. Introduce the convex mathematical entropy eta and its physical entropy flux psi.

$$
s=\frac{q_L+q_R}{2},\qquad
\eta(q)=\frac{q^2}{2},\qquad\psi(q)=\frac{q^3}{3},\qquad
[\psi]-s[\eta]=-\frac{(q_L-q_R)^3}{12}.
$$

Brackets mean right value minus left value. For a compressive shock the last expression is nonpositive, as required by this entropy inequality. Reversing the jump gives a positive defect: the resulting expansion shock may satisfy conservation but fails admissibility.

If the left state is smaller, the admissible solution is a continuous fan. At positive time, values inside the fan equal the similarity coordinate measured from the initial jump. The two fan edges move with the left and right states. Equal states give a constant solution, while equal and opposite compressive states produce a stationary shock.

## Integrate the reference over each cell

The exact reference lives on the whole real line, even though we calculate on a finite interval. The rarefaction is defined by the similarity variable xi; a shock uses its moving discontinuity instead.

$$
\xi=\frac{x-x_0}{t},\qquad
q_{\mathrm{rare}}(x,t)=
\begin{cases}q_L,&\xi<q_L,\\ \xi,&q_L\le\xi\le q_R,\\ q_R,&\xi>q_R,\end{cases}
\qquad
q_{\mathrm{shock}}(x,t)=\begin{cases}q_L,&x<x_0+st,\\q_R,&x>x_0+st.\end{cases}
$$

A discontinuity's value at the single front point does not change a cell integral. The reference for a numerical cell is its average over the whole cell. When a front cuts through a cell, that average mixes the two shock states by their occupied lengths. For a fan, integrate its linear profile over the part of the cell inside the fan.

$$
\bar q_i(t)=\frac1{\Delta x}\int_{x_i}^{x_{i+1}}q(x,t)\,dx,\qquad
\mathcal E_1=\sum_i\Delta x\,|q_i-\bar q_i(t)|.
$$

We also integrate the continuum entropy density itself. Averaging q-squared divided by two is different from squaring the average q. A partially filled shock cell or a fan cell generally has a positive averaging gap. Comparing those two quantities without acknowledging that gap would mix a continuum entropy budget with a different discrete quadratic diagnostic.

We separately integrate the absolute error of the piecewise-constant numerical reconstruction against the continuous exact profile. A coarse cell can have an almost correct average while hiding a sharp unresolved shock inside it. Consequently, cell-average error need not decrease monotonically with refinement. Reconstruction error and measured shock position provide additional resolution checks at the same physical time.

## Choose the Godunov interface flux

Godunov uses the exact local Riemann solution along the stationary interface. For a rarefaction, the interface sees the left state if the entire fan moves right, the right state if it moves left, and zero if the fan straddles zero. For a shock, the sign of the shock speed selects the trace.

| Local states | Interface selection | Numerical flux |
|---|---|---|
| Increasing, both nonnegative | Left state | Left state squared / 2 |
| Increasing, both nonpositive | Right state | Right state squared / 2 |
| Increasing across zero | Sonic value zero | Zero |
| Decreasing, positive shock speed | Left trace | Left state squared / 2 |
| Decreasing, negative shock speed | Right trace | Right state squared / 2 |

At a zero-speed compressive shock, the two physical fluxes agree. The implementation declares the left trace for the flux selection. That convention does not mean the discontinuous solution has one uniquely defined point value at the front.

For this convex scalar flux, an equivalent independent construction takes the minimum physical flux over an increasing interval and the maximum of the endpoint physical fluxes over a decreasing interval. This gives a useful check of the trace-selection implementation.

## Expose the uncorrected Roe counterexample

Roe replaces the local nonlinear problem by transport at the arithmetic mean of its two states. The resulting central flux with viscosity is conservative, but it needs additional treatment for a transonic rarefaction.

$$
\widehat q=\frac{a+b}{2},\qquad
F_{\mathrm{Roe}}(a,b)=\frac{f(a)+f(b)}2-\frac{|a+b|}{4}(b-a),\qquad
F_{\mathrm{Roe}}(-1,1)=\frac12,\quad F_{\mathrm{Godunov}}(-1,1)=0.
$$

The mean speed in the example is zero. Roe keeps a fictitious stationary discontinuity, whereas the exact entropy solution opens into a fan extending from minus t to plus t. Both fluxes can be shared perfectly between neighboring cells. Conservation therefore cannot tell us which wave was selected.

The comparison deliberately uses uncorrected Roe. It should not be read as a claim that every Roe-based solver has this failure: practical methods can apply entropy fixes. The purpose is to isolate the specific missing condition and verify that our diagnostics detect it.

## Advance actual states and declare the boundaries

At each old time, compute every interface flux from the old cell states, then update all cells together. Each scheme evolves its own trajectory. Godunov is not merely a reference label attached to a Roe run.

$$
r_i^n=\frac{F_{i-1/2}^n-F_{i+1/2}^n}{\Delta x},\qquad
q_i^{n+1}=q_i^n+\Delta t\,r_i^n.
$$

The two external ghost cells receive exact full-line Riemann cell averages at that old time. These are prescribed exterior data. They are not a claim of a perfect nonreflecting numerical boundary. Interior truncation errors can still affect the interface fluxes.

This boundary definition remains explicit when a wave leaves the computational interval. The exact full-line solution continues to provide both exterior averages and the interior reference. We do not freeze the original far-field states after a fan or shock has crossed a boundary. The time step is executed as entered, including excessive steps; the model does not silently replace an unstable experiment with a smaller step.

## Close the numerical and continuum boundary ledgers separately

Multiply the actual update by cell width and sum. Every interior face cancels with its neighbor, leaving the two numerical boundary fluxes.

$$
\mathcal M_h^n=\sum_i\Delta x\,q_i^n,\qquad
\mathcal M_h^{n+1}-\mathcal M_h^n
=\Delta t(F_{\mathrm{left}}^n-F_{\mathrm{right}}^n).
$$

The numerical ledger accumulates those actual fluxes. It does not insert the exact continuum boundary supply to force agreement. We also sum the computed cell changes independently, so cancellation or accumulation errors remain observable.

The continuum reference has its own time-integrated boundary flux. Before a wave reaches either end, the traces remain the far-field states. After crossing, a boundary may lie inside the fan or behind a shock. The independent reference integration splits time at those crossings. Its mass integral and its integrated boundary supply must agree over the entire interval, rather than only in the early-time regime.

## Distinguish entropy flux from entropy potential

For the quadratic entropy, the entropy variable is q and the potential is q times the physical conservation flux minus psi. It is one half of the physical entropy flux. Confusing the two changes the numerical budget.

$$
\Psi(q)=qf(q)-\psi(q)=\frac{q^3}{6},\qquad
\widehat\psi(a,b;F)=\frac{a+b}{2}F-\frac{a^3+b^3}{12},\qquad
D(a,b;F)=\frac{b^3-a^3}{6}-(b-a)F.
$$

Here F is the conservation flux of the selected method. The numerical entropy flux is a declared two-state quantity; it need not equal the physical entropy flux of the chosen trace. For the stationary shock from one to minus one, its value is zero, while the physical entropy fluxes on the two sides are one third and minus one third.

Godunov gives nonnegative D at every interface for this entropy. Uncorrected Roe at minus one to one gives D equal to minus two thirds. The reverse compressive interface gives plus two thirds. A periodic two-cell pattern contains both interfaces, so the bad and good contributions cancel globally. They even cancel in the adjacent cell budgets. Inspecting D face by face is essential.

## Retain the finite-time entropy correction

Collecting the two face contributions in a cell gives an exact semidiscrete identity. Expanding the actual Euler step adds a positive time-discretization term.

$$
\Delta x\,\Delta\eta_i+\Delta t(\widehat\psi_{i+1/2}-\widehat\psi_{i-1/2})
=-\frac{\Delta t}{2}(D_{i-1/2}+D_{i+1/2})
+\frac{\Delta t^2\Delta x}{2}r_i^2.
$$

A sufficient local condition for the right side to be nonpositive is the following. It concerns this declared numerical entropy flux and this quadratic entropy, rather than every entropy or an optimal universal CFL constant.

$$
\Delta t\,\Delta x\,r_i^2\le D_{i-1/2}+D_{i+1/2}.
$$

We separately report the actual characteristic Courant indicator, based on the largest absolute current cell or ghost state. A value above one signals that the usual one-cell propagation condition is exceeded. A value below one does not repair Roe's transonic interface choice. At an exact threshold, floating-point rounding can affect a strict inequality, so inspect the numerical value as well as its Boolean flag.

The total quadratic entropy in a finite interval need not decrease: boundary entropy supply can outweigh internal dissipation. A periodic sum removes that supply, but can still hide compensating positive and negative interface defects.

## Work one periodic step by hand

For a separate periodic teaching stencil, take four unit-width cells with states minus one, minus one, one, one, and time step 0.1. The increasing interface crosses zero; the closing interface is a stationary compressive shock. This is a periodic one-step example, not the finite-interval full-line reference experiment.

Godunov gives face fluxes one half, zero, one half, one half. The rates are zero, one half, minus one half, zero. The new states are minus one, minus 0.95, 0.95, one. Total mass stays zero. The quadratic entropy falls from two to 1.9025: minus 0.1 from the semidiscrete term plus 0.0025 from the positive Euler correction.

Roe gives one half at every face, so no cell changes. Its global entropy change is zero despite the inadmissible increasing interface. The calculation can be reproduced using the same actual update functions used by the model.

```typescript
import { burgersPeriodicStep } from '@/lib/learning-continuum-godunov';

const q = [-1, -1, 1, 1];
const good = burgersPeriodicStep(q, 1, 0.1, 'godunov');
const bad = burgersPeriodicStep(q, 1, 0.1, 'roe');
console.log(good.values.map(v => v.value)); // [-1, -0.95, 0.95, 1]
console.log(bad.values.map(v => v.value));  // [-1, -1, 1, 1]
```

## Compare a nonconservative update and preserve tiny errors

For positive periodic states, a one-sided discretization of q times its derivative is a useful negative control. It is not the same algorithm as a shared discretization of the derivative of q-squared divided by two.

$$
q_i^{\mathrm{adv}}=q_i-\frac{\Delta t}{\Delta x}q_i(q_i-q_{i-1}),\qquad
\Delta\mathcal M_{\mathrm{adv}}=-\frac{\Delta t}{2}\sum_i(q_i-q_{i-1})^2.
$$

Thus the advective update generally loses mass even when its output looks smooth. With states one, one half, zero, one quarter and step 0.1, the mass change is minus 0.05625. The shared Godunov periodic update has zero boundary supply and zero algebraic mass change.

Small errors need separate treatment. Before the exact wave has traveled beyond the adjacent cells, its cell-average changes are exactly linear in time. We store the initial linear numerical response and its accumulated nonlinear remainder separately. This retains a genuine numerical error even when it is hundreds of orders smaller than the full cell value or its leading change.

Signed logarithms preserve nonzero products beyond ordinary numeric range. Large boundary and dissipation terms can also cancel while a tiny quadratic increment remains; the model evaluates their combined shared-face work algebraically before summation and retains the separately rounded component sum as a diagnostic. It does not clip tiny values to zero. The arithmetic still has finite precision, and direct cell sums or near-threshold signs can be ill conditioned.

## Use the counterexamples when developing AI models

A learned conservative flux model should be evaluated on compressive shocks, transonic rarefactions, constant states, negative transport, and waves crossing boundaries. Inspect wave position, cell-average error, numerical boundary balance, and face entropy defects. A low global residual can conceal a locally wrong wave selection.

For a compressive shock, compare the exact front location with threshold crossings of the numerical cell averages. A smeared front has a transition width, and an unstable profile can have several crossings. Report that ambiguity rather than pretending every run contains one accurately located front. Once the front leaves the interval, an interior position estimate is no longer available.

Training only on smooth data may fail to expose the distinction between conservative and advective forms. A face-sharing architecture helps enforce internal cancellation, but it does not choose the correct entropy solution, boundary condition, or stable time step. These require additional mathematical structure and targeted validation.

## Exercises with worked answers and primary sources

1. **Find the wave for left state two and right state zero.** It is a compressive shock with speed one. At time 0.2 it has moved 0.2 length units. Its entropy defect is minus eight divided by twelve, or minus two thirds. The speed follows from the flux jump divided by the state jump.
2. **Why is minus one to one different?** It is an increasing transonic pair. The exact fan has a zero state at its stationary interface, so Godunov's flux is zero. Roe uses mean speed zero and returns one half, preserving an inadmissible expansion jump.
3. **Verify the four-cell entropy change.** Only two cells change, each from magnitude one to 0.95. Their combined quadratic change is 0.95 squared minus one, or minus 0.0975. The mass changes cancel. Roe's unchanged result does not pass the face entropy test.
4. **Does zero periodic entropy change certify every face?** No. In the two-cell Roe pattern, the increasing face has defect minus two thirds and the decreasing face plus two thirds. Both the total and adjacent sums cancel. Inspect the individual defects, not just the sum or a cell flag.
5. **Check the advective mass defect.** The squared neighboring jumps for states (1, 1/2, 0, 1/4) sum to 9/8. Multiplying by minus 0.1/2 gives minus 9/160, or minus 0.05625. Shared conservative fluxes cannot create that interior loss on the same periodic stencil.
6. **What changes when a wave exits the finite interval?** The exact full-line reference remains valid, but boundary traces may stop equaling the initial far-field values. Supply exact exterior cell averages at each old time, integrate the continuum boundary flux across crossing times independently, and compare the numerical mass ledger with actual numerical boundary fluxes.

The primary [Clawpack Burgers chapter](https://www.clawpack.org/riemann_book/html/Burgers.html) develops the shock and rarefaction solutions. Its [Approximate Burgers solver chapter](https://www.clawpack.org/riemann_book/html/Burgers_approximate.html) explains the Roe transonic failure and entropy-fix motivation. The entropy potential, numerical entropy flux, and distinction between semidiscrete and fully discrete stability follow the framework in [Tadmor, Acta Numerica 2003](https://math.umd.edu/~tadmor/pubs/files/Tadmor_Acta2003.pdf), specialized here by direct algebra. These references establish the theory; the actual implementation is checked separately against independent high-precision calculations and integrated boundary budgets.
