## 1. Energy can change without a state change

Imagine a system moving between two floors. Energy can change because it moves to another floor, or because an external controller raises the floor it occupies. These are different mechanisms. We will call them a state transition and a change of the energy levels, and enter them into separate heat and work ledgers.

An individual observation can have negative work or negative total entropy production. Statements about averages additionally depend on the initial preparation. Define signs and initialization before applying an inequality.

## 2. Specify energies and the thermal reservoir

There are two states,0 and1. Set E₀=0 and the gap ε=kBT ln r, with r>0. For r<1, state1 lies lower; it should not always be called the excited state. One thermal reservoir maintains a specified, constant positive temperature T.

We first measure energy in kBT and entropy in nats, meaning entropy divided by kB. Multiply energies by kBT to obtain joules, using kB=1.380649×10⁻²³J/K. Doubling T while holding r fixed doubles the physical gaps in this model. That comparison differs from changing temperature at fixed energies in joules.

## 3. Put a physical condition on the rates

Choose a positive relaxation clock ν in each holding interval and use these rates.

$$
q_{01}=\frac{\nu}{1+r},\qquad
q_{10}=\frac{\nu r}{1+r},\qquad
\ln\frac{q_{01}}{q_{10}}=-\beta\varepsilon=-\ln r,\quad
\beta=(k_BT)^{-1}.
$$

These rates impose local detailed balance as a physical modeling assumption. An energy difference fixes their ratio, but does not uniquely determine the kinetic clock ν. The canonical occupancy is π₁=1/(1+r). We include no internal state entropy, chemical reaction or particle exchange here.

A Markov model learned from arbitrary observations does not automatically inherit this heat interpretation. First identify its energies, reservoirs and transition mechanisms.

## 4. Two quenches and two periods of waiting

Prepare a distribution at the initial gap ratio r₀. Then execute the following ordered protocol.

$$
r_0\xrightarrow{\mathrm{quench}}r_1
\xrightarrow{\tau_1,\nu_1}r_1
\xrightarrow{\mathrm{quench}}r_2
\xrightarrow{\tau_2,\nu_2}r_2.
$$

A quench changes the energy levels without changing the state. A holding interval keeps r fixed while allowing any number of state transitions. Setting τ=0 removes that hold, rather than removing the quenches around it. The initial ensemble gives both states strictly positive probability.

An illustration should distinguish the initial state, first quench, first hold endpoint, second quench and final hold endpoint. An instantaneous energy jump is not a smooth relaxation curve.

## 5. The first law for one observation

Let x be the state at the first quench, y the state at the second, and z the final state. Each is0 or1. W is work done on the system; Q is heat entering the system.

$$
\beta W=x\ln\frac{r_1}{r_0}+y\ln\frac{r_2}{r_1},\qquad
\beta Q=(y-x)\ln r_1+(z-y)\ln r_2,\qquad
\Delta U=W+Q.
$$

For example, an upward0→1 jump during the first hold absorbs ε₁ when the gap is positive. The reservoir loses that heat. A subsequent1→0 jump at the same gap cancels it. Any sequence of backtracking jumps during a constant gap telescopes to the endpoint energy change.

Consequently, eight joint endpoint events x,y,z determine all heat, work and endpoint-energy values for this protocol. They are not eight sampled trajectories or a cutoff of an infinite path sum. Each conditional transition probability already integrates every intervening jump count.

## 6. Probability relaxes exponentially

At fixed r, let p denote the probability of state1. Its exact evolution is

$$
p(t)=\pi_1+\bigl(p(0)-\pi_1\bigr)e^{-\nu t},
\qquad \pi_1=\frac{1}{1+r}.
$$

Changing r changes the target equilibrium occupancy. Changing ν changes the speed of approach to the same target. Reading more time records is different from changing this physical clock.

An unresolvably small finite-time deviation is not proof of exact equilibration. Inspect its enclosure and representation state. Zero time and an exactly canonical initial law admit separate exact control cases.

## 7. Add system and reservoir entropy

The ensemble Shannon entropy is H(p)=−p ln p−(1−p)ln(1−p). The reservoir entropy change is−βQ. For each hold and for the complete protocol, mean total production obeys

$$
\Sigma=H(p_f)-H(p_i)-\beta\langle Q\rangle\ge0.
$$

The system entropy change itself can be negative. If the system absorbs heat, the reservoir entropy decreases. Making each term nonnegative or clipping its sign would alter the physical ledger.

For one observation the system entropy change is ln pᵢ(x)−ln p_f(z). Here p_f is the actual final distribution. Substituting the final canonical distribution changes the boundary condition of the comparison.

## 8. Use the free energy of the actual distribution

A nonequilibrium distribution at a specified energy landscape has a free-energy functional.

$$
\beta F[p,E]=\beta\langle E\rangle-H(p)
=-\ln Z+D(p\Vert\pi).
$$

Combining the mean first law with the entropy balance gives ⟨W⟩−ΔF[p,E]=kBTΣ. If we instead use the difference of canonical free energies, the remaining correction is

$$
\beta(\langle W\rangle-\Delta F_{\mathrm{eq}})
=\Sigma+D(p_f\Vert\pi_f)-D(p_i\Vert\pi_i).
$$

Even a canonical start need not end in equilibrium. Dissipated work relative to canonical free energies can therefore differ from total entropy production. A nonequilibrium initial distribution can also supply a resource for extracting work. Negative mean work alone does not violate nonnegative mean entropy production.

In an AI energy-based model, D(p∥π) measures the mismatch of an approximating law p from its target π. When physical energies and T are specified as in this chapter, the same relative entropy also gives a free-energy gap. If p is estimated from data, uncertainty in H, D and the inferred thermodynamic balance needs examination as well.

## 9. Four atoms describe the entire work distribution

Write P₁(x,y) for the first hold's conditional transition probability. No further work is done after the second quench, so these four terms determine the complete work law.

$$
\Pr(x,y)=p_i(x)P_1(x,y),\qquad
e^{-\beta W_{xy}}
=\left(\frac{r_0}{r_1}\right)^x
 \left(\frac{r_1}{r_2}\right)^y.
$$

When different x,y produce the same work, their masses can be grouped. At τ₁=0 the off-diagonal events have exactly zero probability. Do not present the work or entropy of an unreachable event as an observed outcome.

Zero mean work, zero variance and zero work for every observation are distinct conditions. A cyclic protocol initialized in the first hold's canonical distribution can have zero mean work while individual work values fluctuate positively and negatively.

## 10. The initialization condition in Jarzynski's relation

If the initial distribution is canonical for r₀, then

$$
\left\langle e^{-\beta W}\right\rangle
=e^{-\beta\Delta F_{\mathrm{eq}}}
=\frac{Z_2}{Z_0},\qquad Z_j=1+\frac1{r_j}.
$$

The left side weights smaller work values more strongly. It cannot be replaced by exp(−β⟨W⟩). The relation does not itself require the final holding interval to reach equilibrium.

If changing the initial distribution changes the left side away from this canonical value, check initialization before claiming a violation. Conversely, observing the trivial equality1=1 when no energy level changes does not prove that the initial distribution was canonical.

## 11. Reverse the protocol for a Crooks comparison

The reverse experiment starts in the canonical law at r₂. It first performs the final hold in reverse order, then quenches r₂→r₁, performs the first hold and quenches r₁→r₀. Its first hold leaves the already canonical distribution unchanged.

$$
\frac{\Pr_F(x,y)}{\Pr_R(y,x)}
=e^{\beta(W_{xy}-\Delta F_{\mathrm{eq}})}
\frac{p_i(x)}{\pi_0(x)}.
$$

For a canonical forward start, the final correction factor is1. Relabeling arrows while leaving the temporal order unchanged does not define the reversed protocol. When zero holding time makes both event probabilities zero, their ratio is not defined.

For the integral relation of total entropy, the reverse initial distribution is the actual p_f. This is a different experiment from the canonical reverse initialization in a work comparison. Confusing them loses the final relative-entropy correction.

## 12. Complete a numerical ledger

Choose r₀=1,r₁=3,r₂=2,ν₁=1/s,ν₂=2/s,τ₁=τ₂=1s and initial p₁=1/2. Occupancy after the first hold is approximately0.341970; final occupancy is approximately0.334502.

| Observation | Value in kBT or nats |
| --- | ---: |
| Mean work ⟨W⟩/(kBT) | 0.410649 |
| Mean heat entering the system ⟨Q⟩/(kBT) | −0.178790 |
| Mean energy increase ΔU/(kBT) | 0.231859 |
| Total entropy production Σ | 0.122964 |
| Canonical free-energy change ΔF_eq/(kBT) | 0.287682 |

Work plus heat gives the energy increase. Canonical dissipated work is about0.122967. Its difference from Σ, approximately0.000003072, is the relative entropy of the actual final law with respect to the final canonical law. The exponential work average is exactly3/4.

Change only the initial occupancy to4/5 and the exponential work average becomes approximately0.552591. The same energy protocol with another initial preparation need not obey the uncorrected canonical-start relation.

## Research extension: changing the preparation of an exponential-work estimate

### Correct an initial distribution before testing an identity

A work identity has both a dynamical protocol and an initial sampling law. Keep this chapter's two quenches, one heat bath and fixed energy scales. Let pi0 be the canonical initial law and let p be another initial distribution with both probabilities positive. Define the normalized exponential work:

$$
Y=e^{-\beta(W-\Delta F_{\rm eq})},\qquad\mathbb E_{\pi_0}Y=1,\qquad\beta=(k_BT)^{-1}.
$$

The canonical-start condition is the one in [Jarzynski's original equality, equation(2)](https://arxiv.org/pdf/cond-mat/9610209). Changing p generally changes the unweighted mean of Y. Reweighting only the initial state gives a different estimator with the same canonical target:

$$
Z=\frac{\pi_0(X_0)}{p(X_0)}Y,\qquad\mathbb E_pZ=\sum_x\pi_0(x)\mathbb E[Y\mid X_0=x]=1\quad(p(x)>0).
$$

The conditional dynamics given X0 must remain the same protocol. This is an exact finite-support change of measure, not a statement that the uncorrected physical work relation holds for every preparation. If an initial state is omitted while pi0 assigns it positive mass, the missing contribution cannot be recovered by a finite weight.

### Minimize variance over initial-state proposals

For each possible initial state, compute a conditional second moment. The second moment of the corrected weight separates into two initial contributions:

$$
b_x=\mathbb E[Y^2\mid X_0=x],\quad c_x=\pi_0(x)^2b_x,\qquad\operatorname{Var}_pZ=\sum_x\frac{c_x}{p(x)}-1.
$$

Cauchy–Schwarz, or a Lagrange multiplier with sum p=1, gives

$$
p_*(x)=\frac{\sqrt{c_x}}{\sum_y\sqrt{c_y}},\qquad\min_p\operatorname{Var}_pZ=\left(\sum_x\sqrt{c_x}\right)^2-1.
$$

The optimal proposal is proportional to the canonical probability times the conditional root-mean-square exponential work. It generally differs from the canonical initial law. Only the initial distribution is optimized; the hold dynamics and quenches are unchanged. Estimating these moments from pilot data adds a separate uncertainty problem.

The normalization Y uses the known analytic free-energy difference of this solvable example. If that difference is unknown, the same optimization can use exp(−beta W) instead: multiplying all weights by one constant does not change p_star. The identity with mean one is a theoretical check, not an instruction to assume an unknown answer.

### Calculate the four atoms before choosing a proposal

The first hold's exact transition kernel is

$$
P_1(x,y)=\lambda\mathbf1_{x=y}+(1-\lambda)\pi_{r_1}(y),\qquad\lambda=e^{-\nu_1\tau_1}.
$$

For the existing example r0=1,r1=3,r2=2, the normalized work values are

$$
r_0=1,\ r_1=3,\ r_2=2:\qquad\pi_0=(1/2,1/2),\quad Y_{xy}=\begin{pmatrix}4/3&2\\4/9&2/3\end{pmatrix}_{xy}.
$$

They include the factor Z0/Z2=4/3. The final hold does no work and therefore does not enter this table or the optimal initial proposal. Sum the two conditional endpoint probabilities in each row to get

$$
\mathbb E[Y\mid0]=\frac{9-\lambda}{6},\quad\mathbb E[Y\mid1]=\frac{3+\lambda}{6},\qquad b_0=\frac{21-5\lambda}{9},\quad b_1=\frac{7+5\lambda}{27}.
$$

These formulas remain valid at zero first-hold time, lambda=1. Off-diagonal endpoint events then have zero probability, rather than constituting missing sampled paths. At lambda=0 the formula describes the fully relaxed limit, not exact equilibration after a finite positive time.

![Changing initial state probability changes the corrected work variance and the uncorrected exponential mean, with a distinct optimum proposal.](/assets/learning/entropy-thermal-research-initial-work-reweighting.en.svg)

The upper panel shows the variance of the corrected normalized work estimator against initial state 1 probability, with the analytic optimum marked. The lower panel shows its raw, uncorrected exponential mean. The first hold is 1s at clock 1/s; no sampled trajectories or confidence bands are drawn.

```python
from math import exp, log, sqrt

attenuation = exp(-1.0)  # clock = 1/s, first hold = 1s
pi_hold = [0.75, 0.25]
transition = [[attenuation * (x == y)
               + (1 - attenuation) * pi_hold[y]
               for y in range(2)] for x in range(2)]
y_weight = [[4/3, 2.0], [4/9, 2/3]]
c = [0.25 * sum(transition[x][y] * y_weight[x][y]**2
                for y in range(2)) for x in range(2)]
optimal_one = sqrt(c[1]) / (sqrt(c[0]) + sqrt(c[1]))
for initial_one in [0.2, 0.5, 0.8, optimal_one]:
    initial = [1 - initial_one, initial_one]
    raw = corrected = second = 0.0
    for x in range(2):
        for y in range(2):
            mass = initial[x] * transition[x][y]
            weight = 0.5 / initial[x] * y_weight[x][y]
            raw += mass * y_weight[x][y]
            corrected += mass * weight
            second += mass * weight**2
    preparation_kl = sum(p * log(2*p) for p in initial)
    print(f"{initial_one:.6f} {raw:.6f} {corrected:.6f} "
          f"{second - 1:.6f} {preparation_kl:.6f}")
```

Columns list initial state 1 probability, raw E[Y], corrected E[Z], Var[Z] and the preparation relative entropy D(p||pi0). All rows use the same quenches and first-hold dynamics. Values involving exponentials and square roots are floating calculations, not directed enclosures.

| p(state 1) | E[Y] | E[Z] | Var[Z] | preparation KL |
| --- | --- | --- | --- | --- |
| 0.200000 | 1.263212 | 1.000000 | 0.074530 | 0.192745 |
| 0.500000 | 1.000000 | 1.000000 | 0.228170 | 0.000000 |
| 0.800000 | 0.736788 | 1.000000 | 1.763503 | 0.192745 |
| 0.281684 | 1.191545 | 1.000000 | 0.031514 | 0.098611 |

Changing the initial law is also a physical preparation change. For the specified energies its nonequilibrium free-energy excess is kBT D(p||pi0). Reducing estimator variance does not make that preparation resource free or prove that physical entropy production is minimized. If the physical experiment requires a canonical start, p cannot simply be replaced without changing the experiment.

### An unbiased exponential mean does not give an unbiased free energy

For M independent complete repetitions with a fixed full-support proposal,

$$
\overline Z_M=\frac1M\sum_{m=1}^M Z_m,\qquad\mathbb E\overline Z_M=1,\quad\operatorname{Var}(\overline Z_M)=\frac{\operatorname{Var}Z}{M}\quad\text{for independent runs}.
$$

Taking the logarithm changes the statistical question. Positivity and Jensen's inequality imply

$$
\widehat{\Delta F}=\Delta F_{\rm eq}-\beta^{-1}\ln\overline Z_M,\qquad\mathbb E\widehat{\Delta F}\geq\Delta F_{\rm eq},\quad\mathbb E\widehat{\Delta F}-\Delta F_{\rm eq}=\frac{\operatorname{Var}Z}{2\beta M}+o(M^{-1})\quad(M\to\infty).
$$

The last expression is a large-M delta-method expansion, not an exact finite-sample correction. In this fixed finite four-atom model all weights are positive and bounded, so its moment conditions hold. Optimizing the initial proposal reduces this asymptotic bias and variance, but does not remove uncertainty from a small data set. Correlated repetitions or a proposal fitted on the same observations require further analysis.

### Research self-checks

#### R1. Why is support important?

Prepare only state 0 although the original canonical law assigns probability1/2 to state 1. Can initial weighting recover the full canonical exponential mean?

**Research explanation.** No. The second state's canonical contribution is absent and its formal weight would divide by zero. Full support is needed for this exact initial-state change of measure. Assigning an observed mean of one after renormalizing the remaining events would change the target.

#### R2. Does the best proposal minimize physical dissipation?

Interpret the variance-minimizing row in the table relative to the canonical row.

**Research explanation.** It minimizes the corrected estimator variance over initial probabilities with fixed dynamics. It changes the physical initial resource, measured here by D(p||pi0), and generally leaves raw E[Y] different from one. No thermodynamic optimization follows from this sampling criterion alone.

#### R3. Can an initial proposal give zero variance in a limit?

Set the first hold to zero, keeping r0=1 and r2=2. Use p(state 1)=1/3.

**Research explanation.** The two quenches become one net quench. Only x=y remains; Y is 4/3 for state 0 and 2/3 for state 1. Multiplying by pi0/p gives Z=1 in both cases. The corrected variance is zero, but raw E[Y]=10/9. This exact importance-weight identity does not restore the uncorrected canonical-start relation for the changed preparation.

## 13. Exercises with explanations

**1. Is raising a level without changing the state heat or work?** In this protocol it is work supplied by the external controller. A state transition at a fixed gap exchanges heat instead. Calling both an energy change does not identify the mechanism.

**2. Does τ₁=0 erase both quenches?** No. The two quenches occur consecutively, with x=y and work x kBT ln(r₂/r₀). Every observation has zero work only when this consecutive protocol is cyclic, r₂=r₀.

**3. Does zero mean work imply zero variance?** No. Set r₀=r₂=1,r₁=1/9 and p₁=9/10. The first hold starts in its own canonical law, so its distribution does not change and mean work is zero. Observations with different states at the two quenches still have positive or negative work, giving positive variance.

**4. Does extending the final hold change the work distribution?** Not for this protocol: no work is performed after the last quench. Heat, final occupancy, system entropy and total production can nevertheless change. Work and heat are different random variables.

**5. Is extracting mean work from a nonequilibrium preparation contradictory?** No. Initial relative entropy can supply a resource. For r₀=r₂=1,r₁=1/9,p₁=19/20, a first hold of1s and no final hold, mean work is negative while total production is positive. Test the sign of ⟨W⟩−ΔF[p,E].

**6. If doubling temperature leaves occupancy curves unchanged, is heat irrelevant?** No. Holding r andν fixed also doubles the gaps in joules. Dimensionless curves can remain unchanged while physical work and heat double. Changing temperature at fixed physical energies requires recomputing r.

## 14. Scope and further study

We assumed a single reservoir, even states and weak-coupling Markov thermodynamics. Four work atoms and eight endpoint atoms close this problem because the protocol has two instantaneous quenches and piecewise constant energies. The same endpoint description does not automatically determine work under general continuous driving or reservoir-resolved heat.

Next, distinguish the channels when two reservoirs produce the same observed state transition. The aggregate stationary generator can have zero current while physical heat conduction still produces entropy. Chemical potentials, particle exchange, internal state entropy, odd variables, strong coupling, feedback and finite-sample inference remain in the full curriculum.

For the underlying concepts, continue with [Seifert's review](https://arxiv.org/pdf/1205.4176), [Jarzynski's original paper](https://arxiv.org/pdf/cond-mat/9610209) and [Crooks's original paper](https://arxiv.org/pdf/cond-mat/9901352). The two-state calculations and worked examples here are derived directly under the stated assumptions.
