## Replace a curve by probability accounts

The experiments below compare independently checked finite models. Exact rational time schemes, semigroup enclosures and display approximations are distinguished.

Imagine a row of connected containers. The content of each container is probability, and a shared pipe transfers exactly the amount one container loses to its neighbor. At an exterior wall, either close the pipe or collect the escaped probability in a separate account. This picture explains a finite-volume method: integrate the conservation law over cells before approximating the fluxes between them.

The previous chapter solved a Gaussian law on the entire real line. Here we deliberately choose a finite interval, a piecewise constant initialization and boundary rules. This is a different model. Our exact reference will be the continuous-time law of the selected finite grid, which still approximates a continuum equation. Keeping those two levels distinct prevents a well-solved grid model from being mistaken for a verified continuum solution.

## Specify cells, masses, densities and units

On the interval from −L to L, consider linear drift b(x)=c−gamma x and constant diffusion D. Divide the interval into K equal cells of width dx. Let mi be the probability in cell i, and let rhoi=mi/dx be its cell-average density. Initially assign nonnegative integer weights and divide by their sum. The resulting piecewise constant density is the stated initial condition, rather than an unspecified Gaussian sampled at cell centers.

$$
\partial_t p=-\partial_xJ,\qquad
J=bp-D\partial_xp,\qquad
m_i=\int_{x_{i-1/2}}^{x_{i+1/2}}p(x,t)\,dx,
\qquad \dot m_i=J_{i-1/2}-J_{i+1/2}.
$$

Mass is dimensionless, density has units m⁻¹, c has units m/s, gamma has units s⁻¹, and D has units m²/s. Face flux J has units s⁻¹. A cell containing probability 1/4 has density 1/4m⁻¹ if its width is one meter, but density 1/2m⁻¹ if its width is half a meter. Taller density bars after refinement do not automatically mean more total probability.

The [TU Delft finite-volume introduction](https://mude.citg.tudelft.nl/book/2024/fvm/fvm.html) provides the control-volume viewpoint. The derivations below specify this chapter's one-dimensional numerical flux and boundary accounts.

## Give each interior face one shared flux

Define bplus=max(b,0) and bminus=min(b,0), evaluated at the face. Upwind drift takes the density from the side the flow comes from, while a centered difference approximates the density gradient:

$$
J_{i+1/2}=b_{i+1/2}^{+}\rho_i+
b_{i+1/2}^{-}\rho_{i+1}
-D\frac{\rho_{i+1}-\rho_i}{\Delta x}.
$$

A positive face velocity uses the left cell; a negative one uses the right cell. The [Clawpack advection chapter](https://www.clawpack.org/riemann_book/html/Advection.html) explains transport by a specified velocity. Our flux also includes diffusion and spatially varying linear drift.

Use exactly the same face value when subtracting from the left cell and adding to the right cell. Summing the cell equations then cancels every internal face. Computing two inconsistent fluxes for the same face would create or destroy probability inside the domain.

Writing the flux in terms of cell masses reveals nonnegative transfer rates:

$$
r_{i\to i+1}=\frac{b_{i+1/2}^{+}}{\Delta x}
                +\frac{D}{\Delta x^2},\qquad
r_{i+1\to i}=\frac{-b_{i+1/2}^{-}}{\Delta x}
                +\frac{D}{\Delta x^2}.
$$

Both have units s⁻¹. The resulting finite generator has nonnegative off-diagonal entries and a diagonal equal to minus its total departure rate. Its cell transitions are a representation of a discretized density equation. They are not measured microscopic jump mechanisms of the original diffusion.

## Set reflecting and absorbing faces explicitly

A reflecting face has total flux exactly zero. With drift present this is a condition on bp−Dpx, not generally px=0. Closing the boundary flux directly implements the selected discrete no-flux rule.

An absorbing face has no incoming exterior probability. For diffusion we approximate the zero boundary density over the half-cell distance from the cell center to the boundary. For drift we use zero exterior density on inflow and the interior upwind value on outflow. The resulting outward escape rates are

$$
\ell=\frac{\max(-b(-L),0)}{\Delta x}
          +\frac{2D}{\Delta x^2},\qquad
r=\frac{\max(b(L),0)}{\Delta x}
          +\frac{2D}{\Delta x^2}.
$$

The factor two in the diffusive term comes from the half-cell boundary distance. Replacing it with D/dx² silently uses a different boundary approximation. A reflecting face sets its entire escape rate to zero. Either side can be chosen independently.

For zero diffusion, the boundary prescription reduces to zero external inflow and upwind outflow. A reflecting transport limit can have singular boundary behavior; this small grid calculation is not a proof of a smooth continuum density or of boundary-limit convergence.

## Preserve global probability while tracking loss inside

Append two bookkeeping states, escaped-left and escaped-right. Probability transferred to either stays there; there is no return. If q is the augmented row law and Q is its generator,

$$
\dot q=qQ,\qquad Q\mathbf1=0,\qquad
\frac d{dt}\sum_{i=1}^{K}m_i=J_{1/2}-J_{K+1/2},
\qquad
\sum_{i=1}^{K}m_i+a_L+a_R=1.
$$

At a left absorbing face the positive escaped-left rate is minus the oriented left flux. At a right absorbing face the escaped-right rate equals the right flux. Mixing outward and rightward sign conventions is a common source of a wrong mass budget.

Loss of interior probability under absorption is intended behavior. Dividing surviving cell masses by their sum creates a conditional distribution given survival, which answers a different question. The prepared model reports unconditional interior masses and both escape accounts, without that normalization. The bookkeeping states do not represent exterior density bins with a finite spatial width.

## Apply explicit Euler and inspect its whole kernel

For a physical step h, explicit Euler uses the previous law:

$$
q^{n+1}=q^n(I+hQ),\qquad
h\max_i(-Q_{ii})\leq1.
$$

The displayed inequality is necessary and sufficient for this Euler matrix to be a stochastic kernel: it keeps every diagonal nonnegative, while all off-diagonal entries already are nonnegative. Every row sums to one regardless of this condition. Thus conservation alone cannot certify a probability approximation.

Above the bound, keep signed values and label the failure. Clipping negative cells and renormalizing would replace the method, change its flux budget and hide the diagnostic. A favorable stationary initialization may also hide an invalid kernel: multiplying a particular stationary row can preserve it even when the kernel fails for other initial laws.

For diffusion-dominated interior cells the largest departure rate scales like D/dx². Refining space therefore requires a step of order dx² for explicit positivity. Drift adds a scale of order |b|/dx. The exact bound comes from all departures of each actual cell, including absorbing boundary rates, rather than from one interior rule quoted without checking the walls.

## Compare implicit Euler without calling it exact

Implicit Euler evaluates the generator action at the new law:

$$
q^{n+1}(I-hQ)=q^n,\qquad
q^{n+1}=q^n(I-hQ)^{-1}.
$$

For this finite conservative generator and h>=0, the inverse is a stochastic matrix. One way to see its positivity and normalization is the resolvent representation

$$
(I-hQ)^{-1}=\int_0^\infty e^{-s}e^{hsQ}\,ds.
$$

Each semigroup kernel in the integral is stochastic, and the positive weights integrate to one. Differentiating e^(−s)e^(hsQ) verifies the inverse identity; decay at infinity follows because the finite stochastic kernel is bounded. At h=0 the identity is recovered directly.

This formula also explains why unconditional positivity does not remove time error. One implicit step averages exact grid evolution over a random duration hS with S exponentially distributed with mean one. It does not equal evolution at the fixed time h. With N steps summing to T, those auxiliary durations have mean T and variance T²/N. This is an interpretation of the method's kernel, not a claim that the physical experiment has a random clock.

## Enclose the exact grid law with a separate time calculation

Choose nu equal to the maximum departure rate. If nu>0, set P=I+Q/nu. Then

$$
e^{TQ}=e^{-\nu T}\sum_{k=0}^\infty
\frac{(\nu T)^k}{k!}P^k.
$$

This is the same potential-event construction developed for master equations, now on all cells and escape accounts. At Q=0 use the identity. A finite cutoff retains a subprobability law. A positive suffix bound encloses the omitted probability and gives componentwise intervals for the complete grid law. A small cutoff can yield broad intervals; it does not justify using their midpoint as a certified accurate answer.

The model uses exact rational conditional laws and Euler kernels with directed256bit exponential bounds. It compares explicit and implicit final rows with the same grid law at the same physical T. It does not normalize the retained Poisson series. Changing the cutoff improves this time reference; it does not refine the spatial mesh or move the walls.

## Separate space error, time error and boundary choice

Increasing N at fixed cells, interval, boundaries and initialization tests the time discretization. Increasing the Poisson cutoff tightens the independent grid-time reference. Increasing K changes the spatial generator and may require a smaller explicit step. Enlarging L changes the modeled domain. Replacing reflection with absorption changes the physical boundary question.

The four-cell and eight-cell preparations represent the same initial uniform density on the interval from −1m to 0m within the domain from −2m to 2m. The coarse cell's probability is split evenly between its two fine children. Copying the coarse mass into each child would double probability; placing all mass in one child would change the initial spatial law. One pair of grids is an illustration, not a proof of convergence or an error estimate for all resolutions.

At reflecting boundaries with D>0, the grid has a unique interior stationary law. Adjacent masses satisfy detailed flux balance:

$$
\pi_{i+1}=\pi_i\frac{r_{i\to i+1}}{r_{i+1\to i}},
\qquad\sum_i\pi_i=1.
$$

This upwind stationary law generally differs from cell integrals of the continuum stationary density at finite dx. The prepared stationary readout is supplied only for the stated closed diffusive case. At D=0, zero rates may split the grid into different closed classes; no unique law is fabricated by dividing by a zero backward rate.

## Diagrams and controlled comparisons

| Diagram | Readout | Condition to preserve |
| --- | --- | --- |
| Cell row with face arrows | One shared signed flux per face | Same sign convention for both neighbors |
| Cell mass and density bars | Probability and probability per meter | Label width and units; escape accounts have no density |
| Interior plus two escaped accounts | Global total and boundary loss | No conditional-survival normalization |
| Explicit and implicit time records | Positivity and exact budget residuals | Keep failed signed explicit values |
| Poisson cutoff history | Partial mass, positive tail, full-law intervals | Broad enclosures remain visibly unresolved |
| Matched-time comparison | Both signed terminal time errors | Same grid, boundaries, initialization and T |

The preparation has2..8 cells, L from1/2 to4m, gamma up to2/s, wind between−2 and2m/s, D up to1m²/s, T up to10s,1..128 time steps and cutoff0..256. Maximum departure rate times T is limited to64 for the independent time reference; reduce T or adjust the grid when this joint computation bound fails. These limits bound interaction cost and do not limit the curriculum. The reader uses native workers and separate view-only record selectors; changing preparation cancels the previous calculation.

## Research extension: where and when does probability escape?

### Turn escaped mass into a first-passage question

The two escape accounts tell how much probability has left by a selected time. A researcher may instead ask which wall is eventually reached and how long that takes. For the same finite grid, these questions have exact linear equations. They require no Euler time stepping or arbitrary terminal cutoff.

Retain the chapter's row-generator convention. Let A be the interior transient block and bL,bR its columns of escape rates. Assume every interior state reaches an absorbing account with probability one and finite mean time. This holds for this finite connected grid with D>0 and at least one absorbing wall. Define the Green matrix G:

$$
Q=\begin{pmatrix}A&b_L&b_R\\0&0&0\\0&0&0\end{pmatrix},\qquad A\mathbf1+b_L+b_R=0,\qquad G=\int_0^\infty e^{tA}\,dt=(-A)^{-1}.
$$

Entry Gij is expected time spent in cell j before absorption, starting from cell i; its unit is seconds. This is a matrix of occupation times, not a stochastic transition kernel. To prove the inverse identity, integrate the derivative of exp(tA); its limit is zero because the finite interior class is transient. A reflecting grid has no such decay and makes this inverse unavailable. At D=0 verify accessibility before solving.

### Integrate survival and the escape flux

For a specified starting cell i, survival is the sum of its interior transition probabilities. The identity E[T]=integral of P(T>t) gives the mean time vector tau. Integrating the right escape flux gives eventual right-exit probability uR:

$$
S_i(t)=(e^{tA}\mathbf1)_i,\qquad \tau=G\mathbf1,\qquad u_R=Gb_R,\quad u_L=Gb_L,\qquad u_L+u_R=\mathbf1.
$$

Multiplying these identities by −A produces backward equations:

$$
-A\tau=\mathbf1,\qquad -Au_R=b_R,\qquad -Au_L=b_L.
$$

One can also derive them by conditioning on the first holding time and next jump. This connects to the [Cambridge Markov-chain notes, Section3](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf), which develop first-step equations in discrete time. Here holding times convert that argument into the stated finite continuous-time equations; the formulas are derived for this chapter's grid. With an initial interior probability row alpha, use alpha tau for mean time and alpha uR for right-exit probability. No survival renormalization appears.

### Conditioning on an exit direction changes the time question

Define wR as expected T times the indicator of eventual right exit. Its integrand includes t because the right escape flux is the density of exit time with that label. Integration by parts gives

$$
w_R=\int_0^\infty t e^{tA}b_R\,dt=G^2b_R=Gu_R,\qquad \mathbb E_i[T\mid R]=\frac{(w_R)_i}{(u_R)_i}\quad((u_R)_i>0).
$$

The conditional mean is undefined when the corresponding exit probability is zero. It is generally not the unconditional mean tau. For a mixed initial row alpha, the ratio is (alpha wR)/(alpha uR); averaging cellwise conditional means using the original alpha weights is incorrect because right-exit conditioning changes the starting-cell weights.

Take the earlier two one-meter cells, no drift, D=1m²/s, and both walls absorbing. The half-cell boundary rate is2/s, so

$$
A=\begin{pmatrix}-3&1\\1&-3\end{pmatrix}\mathrm{s}^{-1},\quad b_R=\binom02\mathrm{s}^{-1},\quad G=\frac18\begin{pmatrix}3&1\\1&3\end{pmatrix}\mathrm{s}.
$$

Direct multiplication yields

$$
\tau=\binom{1/2}{1/2}\mathrm{s},\quad u_R=\binom{1/4}{3/4},\quad w_R=\binom{3/16}{5/16}\mathrm{s},\quad \mathbb E_1[T\mid R]=\frac34\mathrm{s}.
$$

Starting in the left cell, right exit is less likely and its conditional mean3/4s exceeds the unconditional mean1/2s. Left exit has probability3/4 and conditional mean5/12s. Weighting the two means by their exit probabilities gives (1/4)(3/4)+(3/4)(5/12)=1/2s. The same generator supplies all four quantities; none is a physical trajectory animation.

### An exact grid solution still has spatial error

For a continuum Brownian particle with diffusion D>0 on [0,ell] and absorbing endpoints, the backward ODE and its endpoint conditions yield

$$
D\tau_c^{\prime\prime}(x)=-1,\quad \tau_c(0)=\tau_c(\ell)=0,\qquad \tau_c(x)=\frac{x(\ell-x)}{2D},\qquad u_{R,c}(x)=\frac x\ell.
$$

Translate the chapter's interval by L; here ell=2L is total length, distinct from its half-width L. Divide it into K equal cells and retain the original half-cell escape rule. For this zero-drift uniform grid, substitution into every interior and endpoint row gives the exact discrete results

$$
x_i=(i+\tfrac12)\Delta x,\quad \tau_i=\frac{x_i(\ell-x_i)}{2D}+\frac{\Delta x^2}{8D},\qquad (u_R)_i=\frac{x_i}{\ell},\quad 0\leq i<K.
$$

Inside, the second difference of a quadratic equals its second derivative. At the boundary, adding the constant dx²/(8D) supplies the correction required by the half-cell rate2D/dx². Thus the splitting probability agrees with the continuum value at cell centers, while the mean time is too large by the same positive amount at every center. Agreement of one observable cannot certify all observables. This center-start comparison treats a discrete cell state as a point at its center; averaging continuum times over a uniformly occupied initial cell is a different comparison.

![Two-cell exit-direction conditional times differ from the unconditional time, and pure-diffusion mean-time grid bias decreases quadratically.](/assets/learning/fokker-research-grid-first-passage.en.svg)

The upper panel compares exact two-cell conditional exit times with the unconditional time. The lower panel shows center-start grid mean-time error on [0,2]m at D=1m²/s. The slope refers to spatial refinement of this particular pure-diffusion model. No general order claim for drift, irregular meshes or singular limits follows.

```python
from fractions import Fraction as F
for cells in [2, 4, 8, 16]:
    length, diffusion = F(2), F(1)
    dx = length / cells
    x = dx / 2
    tau = x * (length - x) / (2 * diffusion) + dx**2 / (8 * diffusion)
    right = x / length
    bias = dx**2 / (8 * diffusion)
    print(cells, dx, tau, right, bias)
```

All values below are exact fractions. The table lists cell count, width, mean time from the first cell, right-exit probability there, and the largest center-start mean-time error across the grid:

| K | dx [m] | tau_0 [s] | u_R,0 | max bias [s] |
| --- | --- | --- | --- | --- |
| 2 | 1 | 1/2 | 1/4 | 1/8 |
| 4 | 1/2 | 1/4 | 1/8 | 1/32 |
| 8 | 1/4 | 1/8 | 1/16 | 1/128 |
| 16 | 1/8 | 1/16 | 1/32 | 1/512 |

The16-cell example extends beyond the interactive grid's8-cell bound and runs only in this standalone code. A linear solve of the transient matrix independently verifies these formulas. Solving it more accurately or increasing a Poisson cutoff removes no spatial bias. Changing walls instead changes the first-passage question itself.

### Research self-checks

#### R1. Which time does conditioning report?

Start the two-cell example in its left cell. What are the right-exit probability, unconditional mean and right-conditioned mean?

**Research explanation.** They are1/4,1/2s and3/4s. The joint first moment wR is3/16s, so divide it by1/4 to condition. Dividing the unconditional time1/2s by1/4 would give2s and answer no valid conditional expectation here.

#### R2. Can a correct splitting probability hide a wrong time?

For K=2, ell=2m and D=1m²/s, compare the first-cell grid and continuum center-start results.

**Research explanation.** The center is x=1/2m. Both right-exit probabilities are1/4. The continuum mean is3/8s, whereas the grid mean is1/2s, a bias1/8s. The half-cell boundary rate preserves the linear splitting solution but requires a constant correction to the quadratic time solution.

#### R3. When must the inverse be refused?

Make both boundaries reflecting, or set all transfer and escape rates to zero. Can G=(−A)⁻¹ still produce a finite mean exit time?

**Research explanation.** No. A now has a closed interior class and a zero eigenvalue. Exit never occurs in the zero-rate example, so mean time is infinite and conditioning on right exit is undefined. A pseudoinverse or a forced normalization would conceal the failed transience assumption rather than solve the specified absorption problem.

## Worked exercises

### 1. Convert mass to density under refinement

A one-meter cell has probability1/4. Split it into two equal children while preserving its uniform density. What masses and densities should the children have?

**Explanation.** Each child receives probability1/8 and has width1/2m. Each density is therefore1/4m⁻¹, matching the parent. Assigning probability1/4 to each child doubles mass. Assigning all probability to one child changes the initial density even though global normalization survives.

### 2. Find a conservative signed failure

Take two cells of width1m, no drift, D=1m²/s, reflecting boundaries and initial row(1,0). Use one step h=3/2s. Compare explicit and implicit Euler.

**Explanation.** The interior generator is[[-1,1],[1,-1]]/s. Explicit Euler gives(−1/2,3/2), whose sum is one but which is not a probability law. Implicit Euler gives(5/8,3/8), a valid law. The exact grid row at this time is((1+exp(−3))/2,(1−exp(−3))/2). Implicit positivity does not make its row equal to that reference.

### 3. Include the half-cell absorbing distance

For the same two cells and diffusion, make both boundaries absorbing. What is each boundary escape rate, and how does total interior survival evolve?

**Explanation.** Each boundary escape rate is2/s. The transient interior matrix is[[-3,1],[1,-3]]/s: one interior transfer plus two units of escape. Summing its two equations gives survival derivative−2 times survival, hence survival exp(−2t) from an initially interior law. The missing mass is in the two escape accounts, whose sum is1−exp(−2t).

### 4. Track the orientation of the left flux

At an absorbing left face with no drift, D=1m²/s, dx=1m and current first-cell mass1/4, calculate the oriented flux and escaped-left rate.

**Explanation.** The density is1/4m⁻¹, the half-cell gradient points rightward, and the right-oriented boundary flux is−1/2s⁻¹. The escaped-left account increases at its negative,1/2s⁻¹. Adding the signed flux itself to that account would incorrectly remove escaped probability and break global conservation.

### 5. Recognize a different stationary spatial law

Use four one-meter cells on[−2,2]m, b=−x/s and D=1/2m²/s with reflecting boundaries. Determine the stationary cell masses.

**Explanation.** The face velocities at−1,0,1m are1,0,−1m/s. Adjacent rate ratios are3,1,1/3, giving weights1:3:3:1 and masses(1,3,3,1)/8. This is the exact stationary law of the chosen grid generator. It is not an assertion that the continuum Gaussian has these exact cell integrals.

### 6. Diagnose three different refinements

A plot changes when N increases, the cutoff increases, or K increases. Which changes address which error? Can all three be called a smaller physical time step?

**Explanation.** Increasing N at fixed T decreases the Euler step and addresses time integration. Increasing cutoff tightens the semigroup enclosure at unchanged time and grid. Increasing K changes space and usually tightens explicit positivity restrictions. None changes the specified physical observation time T. Check initialization and boundaries before interpreting a changed picture as improved accuracy.
