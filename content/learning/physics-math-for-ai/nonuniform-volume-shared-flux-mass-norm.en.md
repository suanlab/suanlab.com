## Start with containers of different sizes

Imagine three connected containers. One holds twice the volume of either of the others. A pipe transfers one gram from the first container into the second. Both containers must record the same transferred mass with opposite signs. Their concentrations, however, change by different amounts: the larger container dilutes the incoming mass more.

This distinction is the core of a conservative discretization. Sharing a pipe flux is necessary, but the storage volume used to convert mass into concentration must also be correct. A graph network can share edge messages and still violate physical mass if it updates concentrations without the appropriate node weights.

Our experiment uses a periodic tube with unequal cells. It actually advances their average concentrations, compares a deliberately incorrect storage rule, and measures both mass and a volume weighted quadratic norm. These are separate diagnostics. A mass conserving calculation can become negative or unstable.

## Declare the physical model and its units

Use a constant cross-sectional area A, a constant signed transport speed a, and a nonnegative diffusivity D. Concentration c is passive: changing it does not change the prescribed speed. The domain closes into a ring, so the rightmost interface connects back to the first cell. There are no external boundary sources.

$$
\partial_t c+a\partial_x c=D\partial_{xx}c,\qquad
c(x+L,t)=c(x,t),\qquad A>0,\quad D\ge0.
$$

Length and position have units m, area has units m², velocity has units m/s, diffusivity has units m²/s, concentration has units kg/m³, and time has units s. An integrated face flux therefore has units kg/s. Multiplying a concentration by its cell volume produces mass in kg.

This is a one-dimensional constant-coefficient transport problem. It has no moving mesh, reaction, variable area, wall layer, or coupling to momentum. Those extensions require their own balance laws. In particular, a changing cell volume would introduce a storage derivative that the present experiment does not contain.

## Build the cells before assigning their values

Choose N equal parameter intervals and warp their physical boundaries with a smooth periodic mapping. The warp magnitude is at most 0.75, keeping the mapping increasing. The endpoints are fixed at zero and L. Each cell center is its geometric midpoint, and the last center-to-center distance wraps around the ring.

$$
x_i=L\left[\frac{i}{N}+\frac{g}{2\pi}\sin\left(\frac{2\pi i}{N}\right)\right],\quad
h_i=x_{i+1}-x_i,\quad V_i=Ah_i,\quad
\gamma_i=\frac{DA}{d_i}.
$$

Here d-i is the positive distance between neighboring centers, including the periodic distance for the closing face. The conductance gamma has units m³/s. Multiplying it by a concentration difference gives an integrated diffusive mass flux.

The implementation declares ordinary floating-point coordinates as its mesh geometry. A warp smaller than coordinate resolution may leave the stored mesh unchanged. Signed logarithms preserve small physical products later in the calculation; they do not make these coordinates symbolic or exact.

## A cell average is an integral

Start from a smooth cosine profile with mean concentration c-zero, relative contrast epsilon between minus one and one, integer mode m, and phase phi in cycles. Its cell value is an integral over the entire cell, rather than the value at the center.

$$
k=\frac{2\pi m}{L},\qquad c(x,0)=c_0[1+\epsilon\cos(kx+2\pi\phi)],\qquad
q_i^0=c_0+c_0\epsilon\operatorname{sinc}(kh_i/2)\cos(kx_{c,i}+2\pi\phi),
\quad\operatorname{sinc}(z)=\frac{\sin z}{z}.
$$

A wide cell smooths more of the oscillation. The sinc factor expresses that averaging. Replacing the integral by a center sample changes the initial numerical problem, especially for a high mode or a coarse mesh. When comparing resolutions, initialize each mesh by its own physical cell integrals.

For example, if k times the cell width is pi, the oscillatory part of the average is multiplied by two divided by pi, about 0.637. A center sample would retain the entire local amplitude. This discrepancy exists before the first time step and should not be blamed on the evolution algorithm.

## Compute every interface once

Orient face i from cell i to its next neighbor j. Upwind advection uses the left cell for nonnegative speed and the right cell for negative speed. Diffusion sends material down the discrete concentration gradient. The same face value leaves one cell and enters the other.

$$
F_i=aAq_{\mathrm{up},i}+\gamma_i(q_i-q_j),\qquad
V_i r_i=F_{i-1}-F_i,\qquad
q_i^{n+1}=q_i^n+\Delta t\,r_i^n.
$$

The periodic index on the left flux matters. There is no special unpaired face at the ends. Positive F means transport in the chosen orientation; negative F means transport against it. Negative velocity changes the advective donor, not the meaning of conservation.

A useful implementation order is: compute all shared faces from the current state, assemble all rates, then update every cell together. Updating a cell in place before computing its neighbor's face would create a different algorithm. Our calculation is explicit Euler with a prescribed time step and a prescribed integer number of steps.

## Derive conservation with the mass matrix

Let B have plus one at each face's source cell and minus one at its destination. Let M be the diagonal matrix of physical cell volumes. Each column of B sums to zero, including the closing face.

$$
M\dot q=-BF,\qquad \boldsymbol1^TB=0,\qquad
\mathcal M=\boldsymbol1^TMq,\qquad \dot{\mathcal M}=0.
$$

Multiplying the actual Euler update by M and summing gives the same zero mass change in exact arithmetic. The cancellation is algebraic and does not depend on the initial cosine profile. It would also hold for a learned face flux that is shared consistently.

The sum of concentrations has no such physical meaning on an unequal mesh. Its terms count one small cell and one large cell equally. The correct total uses volume weights. In a plot, concentration corresponds to bar height; mass corresponds to height times physical cell width times cross-sectional area. Counting bar heights alone discards the widths.

## Run the wrong storage rule as a control

The negative control keeps the same shared flux formula but divides every cell's net inflow by a single reference volume. It evolves its own state at subsequent steps. It is not a rescaled label attached to the correct trajectory.

$$
V_{\mathrm{ref}}=\frac{1}{N}\sum_iV_i,\qquad
\dot q_i^{\mathrm{wrong}}=\frac{F_{i-1}-F_i}{V_{\mathrm{ref}}},\qquad
\dot{\mathcal M}_{\mathrm{wrong}}
=-\sum_{i\to j}\frac{V_i-V_j}{V_{\mathrm{ref}}}F_i.
$$

This rule preserves the unweighted concentration sum while generally changing physical mass. It is a useful failure because its output may remain smooth and its edge messages still cancel. The mistake is in storage, not in whether a face value was shared.

On a uniform mesh every volume equals the reference volume, so the two algorithms coincide. Always include that control: a discrepancy there would indicate an additional implementation error. Conversely, agreement on a uniform mesh cannot certify correct handling of irregular cells.

## Check one small example by hand

Use three cell volumes 1, 2, and 1 m³, face conductances all equal to 1 m³/s, zero advection, and initial concentrations 1, 0, and 0 kg/m³. The oriented fluxes are 1, 0, and minus 1 kg/s. The negative last flux transfers mass from the first cell into the last cell across the closing face.

| Quantity | Physical storage | Reference-volume storage |
|---|---|---|
| Concentration rates | −2, 1/2, 1 | −3/2, 3/4, 3/4 |
| State after 0.1 s | 0.8, 0.05, 0.1 | 0.85, 0.075, 0.075 |
| Physical mass after 0.1 s | 1 kg | 1.075 kg |
| Unweighted concentration sum | 0.95 | 1 |

The correct concentration sum decreases while mass stays exactly one kilogram in the mathematical update. That decrease is not leakage. The larger middle cell receives less concentration per unit incoming mass. The wrong rule makes the concentration sum look reassuring while creating 0.075 kg of material.

The quadratic norm starts at 0.5 kg²/m³ and becomes 0.3275 kg²/m³ under physical storage. These hand values are checked independently of the periodic mesh constructor, so a correct mesh initializer cannot conceal a wrong update rule.

```python
import numpy as np
V = np.array([1., 2., 1.])
q = np.array([1., 0., 0.])
gamma = np.ones(3)
dt = 0.1
F = gamma * (q - np.roll(q, -1))  # zero advection
net = np.roll(F, 1) - F
correct = q + dt * net / V
wrong = q + dt * net / V.mean()
print(V @ correct, V @ wrong)  # 1.0, 1.075
print(0.5 * V @ (correct**2 - q**2))  # -0.1725
```

## Derive the quadratic norm budget

Define a volume weighted quadratic norm. It measures concentration variation and magnitude, but its units are not joules. Calling it physical kinetic energy would be incorrect for this passive concentration model.

$$
E=\frac12\sum_iV_iq_i^2,\qquad
\dot E=-\sum_{i\to j}\gamma_i(q_i-q_j)^2
-\frac{|a|A}{2}\sum_{i\to j}(q_i-q_j)^2\le0.
$$

To obtain the diffusion term, multiply each cell balance by its concentration and collect the two contributions at a face. They give minus conductance times the squared jump. For constant-speed periodic upwinding, telescoping the difference of neighboring squares leaves the additional nonnegative loss shown above.

The first loss comes from the specified physical diffusion. The second is numerical dissipation from first-order upwind transport. When D is zero, a smooth continuum profile translates without losing this quadratic norm, while a finite upwind mesh usually damps it. Numerical smoothing and physical diffusion should not be interpreted as the same mechanism.

## Time discretization changes the conclusion

A nonpositive continuous-time norm derivative does not by itself guarantee a finite Euler step decreases the norm. Expanding the square gives an additional positive term that must be retained.

$$
E(q+\Delta t r)-E(q)
=\Delta t\,q^TMr+\frac{\Delta t^2}{2}r^TMr.
$$

For the hand example, the first term is minus 0.2 and the positive correction is 0.0275, leaving minus 0.1725. Increasing the time step strengthens the positive term quadratically. A conservative method can therefore amplify the norm while its face mass ledger still closes.

A sufficient condition for nonnegative update coefficients is cell dependent. Incoming neighbors have nonnegative coefficients, while the central coefficient must remain nonnegative.

$$
C_i=\frac{\Delta t}{V_i}\bigl(|a|A+\gamma_{i-1}+\gamma_i\bigr)\le1
\quad\text{for every cell }i.
$$

Under this condition each new value is a convex combination of old neighboring values. Constant preservation and the invariant volume weights then give quadratic norm contraction by Jensen's inequality. The smallest cells can set the restriction. Replacing all volumes by an average in this check can miss the dangerous cell.

## Compare with an exact continuum reference

For the chosen initial profile, the continuum solution is a translating and diffusively decaying Fourier mode. Its exact cell average uses the same cell integral as initialization.

$$
\bar c_i(t)=c_0+c_0\epsilon e^{-Dk^2t}
\operatorname{sinc}(kh_i/2)\cos(k(x_{c,i}-at)+2\pi\phi),\qquad
\mathcal E_1=\sum_iV_i|q_i-\bar c_i(t)|.
$$

The reference is valid for all times in this periodic constant-coefficient problem. It is not a pointwise center value, and it is not a second run of the same discrete algorithm. The reported error has units kg because it integrates absolute concentration error over physical volume. The maximum cell-average error has concentration units.

Refine the mesh while reducing the time step enough to satisfy every local restriction. Compare at the same physical final time. Merely adding cells at fixed step size can turn a stable coarse run into an unstable fine run, especially because the explicit diffusive restriction scales quadratically with cell width.

## Preserve small changes and expose rounding diagnostics

The calculation stores the initial departure from the mean and the accumulated update separately. It never needs to recover a tiny update by subtracting two nearly equal full concentrations. For a linear operator R, the cumulative update z follows an equivalent recurrence.

$$
z^0=0,\qquad
z^{n+1}=z^n+\Delta t\,[R(q^0-c_0\boldsymbol1)+Rz^n],\qquad
q^n=c_0\boldsymbol1+(q^0-c_0\boldsymbol1)+z^n.
$$

Signed logarithms distinguish exact zero from a nonzero quantity too small or too large for an ordinary floating-point value. An unavailable ordinary value is not silently changed into zero. Small products such as diffusivity times area and velocity times area are formed in that representation before division or summation.

There are still finite precision limits. Cell geometry is rounded, logarithms are floating point, and violent instability can produce huge positive and negative masses whose direct sum is ill conditioned. The shared-face mass ledger expresses the algebraic conservation identity; the assembled cell mass and directly subtracted rounded masses expose numerical summation error. Agreement should be assessed relative to the absolute masses being cancelled, rather than treating a large cancelling sum as accurately resolved.

## Connect the structure to AI models

A learned model can predict one integrated flux per oriented face and apply minus B times that vector. Using the physical mass matrix then enforces the global internal cancellation. Predicting one independent concentration increment per node lacks that guarantee unless additional constraints are imposed.

Volume weights also affect training. A loss that counts every cell equally gives a region with many tiny cells more influence than an equally sized region with fewer large cells. A volume weighted error approximates a physical integral. Which loss is appropriate depends on the scientific objective; the weights must be declared.

Conservation is only one requirement. Learned fluxes can be inconsistent on constant fields, violate positivity, destabilize time integration, or generalize poorly to a new mesh. Flux identification also has a null space: adding the same circulation to every edge of this ring leaves B times the flux unchanged. Cell trajectories alone cannot identify that circulation without additional information.

## Research extension: physical volumes, stochastic averaging and convex entropy

The previous chapter shows that a sufficiently small Euler step is a convex average and contracts a quadratic norm. Here we ask a stronger question: which structure preserves every convex measure of the concentration distribution? Restrict this supplement to zero advection, fixed positive cell volumes and symmetric nonnegative diffusive conductances. Write c for the same concentration state previously denoted q. Count each undirected edge once in an edge sum. A periodic ring is the original geometry; the same derivation works on a finite connected network.

$$
\dot c=Qc,\qquad Q_{ij}=\frac{g_{ij}}{V_i}\ (i\ne j),\quad Q_{ii}=-\sum_{j\ne i}\frac{g_{ij}}{V_i},\qquad g_{ij}=g_{ji}\ge0,\quad V_i>0.
$$

The matrix has nonnegative off-diagonal entries and zero row sums. Its invariant weights are physical volume fractions, rather than equal weights on cells of unequal size. Symmetry of the conductance gives detailed balance. These are finite reversible-generator properties. Concentration itself is not a probability of occupying a cell: different volumes store different masses at the same concentration. Keep that distinction when borrowing Markov-chain notation.

$$
Q\boldsymbol1=0,\quad\pi_i=\frac{V_i}{V_{\rm tot}},\quad V_{\rm tot}=\sum_iV_i,\qquad\pi^TQ=0,\quad\pi_iQ_{ij}=\frac{g_{ij}}{V_{\rm tot}}=\pi_jQ_{ji}.
$$

Choose a rate at least as large as every total departure rate; taking the maximum is convenient. Adding a scaled generator to the identity then gives a row-stochastic matrix. Each row computes a convex combination and the same volume weights remain invariant. The rate has units inverse seconds. If every conductance is zero, the generator is zero and the solution is the initial state exactly; division by a zero rate is unnecessary and invalid.

$$
\lambda=\max_i(-Q_{ii})>0,\qquad P=I+Q/\lambda,\quad P_{ij}\ge0,\quad P\boldsymbol1=\boldsymbol1,\quad\pi^TP=\pi^T;\qquad Q=0:\ c(t)=c(0).
$$

Expand the exponential using Q equal to lambda times P minus I. The coefficients are Poisson probabilities, so the exact evolution is a convex mixture of stochastic powers. This construction is called uniformization. It proves positivity and mass preservation without choosing an Euler step. The infinite identity is exact; a finite numerical sum still has truncation and rounding errors. The current interactive lab continues to execute explicit Euler and does not silently switch to this propagator.

$$
S(t)=e^{tQ}=e^{-\lambda t}\sum_{n=0}^{\infty}\frac{(\lambda t)^n}{n!}P^n,\qquad S(t)\boldsymbol1=\boldsymbol1,\quad\pi^TS(t)=\pi^T.
$$

Jensen applies row by row to any convex function defined on the concentration range. Multiplying by the invariant volumes and summing gives a whole-distribution contraction. The old quadratic norm is one member of this family. Absolute value and relative-entropy functions answer other questions. A stochastic row also keeps every concentration within the initial minimum and maximum. A correct mass ledger alone does not provide any of these additional statements.

$$
\Phi\ \text{convex}:\qquad\sum_iV_i\Phi(c_i(t))\le\sum_iV_i\Phi(c_i(0)),\qquad\min_jc_j(0)\le c_i(t)\le\max_jc_j(0).
$$

For nonnegative states with positive total mass, normalize actual cell masses into probabilities. Their stationary distribution is the volume fraction. The resulting relative entropy of concentrations equals total mass times a dimensionless Kullback–Leibler divergence. Its units are kilograms, not joules or thermodynamic entropy. The logarithm uses a dimensionless concentration ratio. The linear terms cancel because the physical mass and mean are fixed.

$$
M_0=\sum_iV_ic_i>0,\quad\bar c=M_0/V_{\rm tot},\quad p_i=V_ic_i/M_0,\qquad\mathcal H=\sum_iV_i\left[c_i\log(c_i/\bar c)-c_i+\bar c\right]=M_0\sum_ip_i\log(p_i/\pi_i)\ge0.
$$

For strictly positive concentrations, differentiate this functional using the physical face exchanges. Each edge pairs a concentration difference with the same-sign logarithmic difference. This makes its dissipation nonnegative. That argument is independent of a particular initial cosine. It is also a structural constraint for a learned diffusive conductance: sharing a flux and using nonnegative symmetric conductance provide different pieces of the proof.

$$
\frac{d\mathcal H}{dt}=-\sum_{\{i,j\}}g_{ij}(c_i-c_j)\log(c_i/c_j)\le0\qquad(c_i>0).
$$

A zero cell concentration is allowed in the entropy itself by the continuous zero-log-zero convention when the mean is positive. The logarithmic derivative at that initial state can be singular; do not report a fabricated finite zero. A connected positive-edge graph spreads any nonzero nonnegative inventory to every cell at positive time. Disconnected components instead retain separate masses and approach their own volume-weighted means. The all-zero state is a separate zero-inventory case, not a ratio with a manufactured mean.

$$
0\log0:=0\quad(\bar c>0),\qquad\bar c_{\mathcal C}=\frac{\sum_{i\in\mathcal C}V_ic_i(0)}{\sum_{i\in\mathcal C}V_i},\qquad c(0)=0\Rightarrow c(t)=0.
$$

The Euler matrix is stochastic only when its central coefficients stay nonnegative. Beyond that restriction, one quadratic norm can still decrease while the minimum principle fails. For three unit-volume cells on a ring with unit conductance, the nonconstant generator eigenvalues are both minus three. Quadratic contraction permits steps up to two thirds, while positivity for every nonnegative input requires a step no larger than one half. These are different intervals, even for this tiny exact system.

$$
c^{n+1}=(I+\Delta tQ)c^n,\qquad\Delta t\lambda\le1\Rightarrow I+\Delta tQ\ge0;\qquad Q=\begin{pmatrix}-2&1&1\\1&-2&1\\1&1&-2\end{pmatrix}:\quad\Delta t_{\rm pos}\le\frac12,\quad\Delta t_{L^2}\le\frac23.
$$

The explicit counterexample retains mass and decreases the quadratic norm while creating a negative concentration. Its absolute-value convex sum increases, demonstrating that one successful convex diagnostic does not establish all of them. Relative entropy is undefined for that negative state; clipping it would change both the numerical experiment and its mass. The table below compares several separate one-step experiments from the same initial state, not successive steps along one trajectory.

$$
c^0=(0,1,0),\quad\Delta t=\frac35:\quad c^1=\left(\frac35,-\frac15,\frac35\right),\quad\sum_ic_i^1=1,\quad\frac12\sum_i(c_i^1)^2=\frac{19}{50}<\frac12,\quad\sum_i|c_i^1|=\frac75>1.
$$

Table columns are step in seconds, minimum concentration in kg/m³, mass in kg, quadratic norm in kg²/m³ and volume-weighted absolute concentration in kg. Each row starts from the same unit-volume ring. The last two rows preserve mass and do not increase the quadratic norm, while violating positivity.

| dt | minimum | mass | quadratic | absolute |
| --- | --- | --- | --- | --- |
| 0.10000000 | 0.10000000 | 1.00000000 | 0.33000000 | 1.00000000 |
| 0.50000000 | 0.00000000 | 1.00000000 | 0.25000000 | 1.00000000 |
| 0.60000000 | -0.20000000 | 1.00000000 | 0.38000000 | 1.40000000 |
| 0.66666667 | -0.33333333 | 1.00000000 | 0.50000000 | 1.66666667 |

```python
from fractions import Fraction as F
# Unit volumes (m^3), unit conductances (m^3/s), c0=(0,1,0) kg/m^3.
for dt in (F(1,10), F(1,2), F(3,5), F(2,3)):
    c = (dt, 1-2*dt, dt)
    mass = sum(c)
    quadratic = sum(v*v for v in c)/2
    absolute = sum(abs(v) for v in c)
    print(" ".join(f"{float(v):.8f}" for v in (dt, min(c), mass, quadratic, absolute)))
```

[Ward Whitt’s CTMC chapter, §3.4](https://www.columbia.edu/~ww2040/3106F11/CTMCchapter121906.pdf) develops uniformization. We apply the row-generator convention to the stated physical volumes and derive the entropy and Euler counterexample directly.

![Stochastic diffusion versus Euler positivity and convex-diagnostic failures](/assets/learning/continuum-volume-markov-convex-entropy.en.svg)

### Research questions with explanations

**Does a smaller quadratic norm certify a nonnegative concentration?** **Answer:** No. The exact three-cell Euler example has mass one and quadratic norm 19/50, yet its middle concentration is minus one fifth. Its relative entropy is undefined.

**Why do physical volume fractions appear as invariant probabilities?** **Answer:** Detailed balance uses conductance divided by the receiving row volume. Equal concentration means cell mass is proportional to volume; concentration itself is not cell probability.

**What if conductances disconnect the graph?** **Answer:** Each connected component preserves its own inventory and has its own weighted mean. A global zero-entropy equilibrium need not be reached.

## Exercises with worked answers and further reading

1. **Why can equal transferred masses produce unequal concentration changes?** Divide the incoming mass by each receiving volume. A 0.1 kg transfer changes a 1 m³ cell by 0.1 kg/m³ and a 2 m³ cell by 0.05 kg/m³. Conservation concerns mass, not equal concentration increments.
2. **Reconstruct the three-cell rates.** The net inflows from fluxes (1, 0, −1) are (−2, 1, 1) kg/s. Dividing by volumes (1, 2, 1) gives (−2, 1/2, 1). Their weighted sum is −2 + 1 + 1 = 0. Their unweighted sum is −1/2.
3. **Find the wrong rule's mass production.** The reference volume is 4/3. Dividing the same inflows by it gives (−3/2, 3/4, 3/4). The physical weighted sum is 3/4 kg/s, so a 0.1 s step produces 3/40 kg. A later step must recompute fluxes from that rule's own state.
4. **Separate the two norm contributions.** In the same example, the squared jump sum is two, so the continuous norm rate is −2. The weighted squared rate is 11/2. At step 0.1, the positive correction is 0.01 times 11/4 = 0.0275. Adding it to −0.2 gives −0.1725.
5. **Does a closed mass ledger certify a stable calculation?** No. Increase diffusivity and use an excessive prescribed step on a warped mesh. The shared faces still cancel, but a central coefficient can become negative, concentrations can turn negative, and the positive Euler correction can dominate. Check the actual local coefficients and the resulting state.
6. **Can a model infer every face flux from cell changes?** No. On the ring, adding one common constant flux changes no net cell inflow. The data determine flux differences but leave a circulating component unobserved. Boundary measurements, a constitutive law, or an additional convention can supply missing information.

The finite-volume update and telescoping argument follow the framework explained in the primary [Riemann Problems and Jupyter Solutions: Approximate Solvers](https://www.clawpack.org/riemann_book/html/Approximate_solvers.html). The physical inflow/outflow interpretation and constant-speed translating solution are discussed in its [Advection chapter](https://www.clawpack.org/riemann_book/html/Advection.html). The nonuniform storage counterexample, quadratic budgets, and signed-log implementation here are our explicit specializations; those references do not certify this implementation.
