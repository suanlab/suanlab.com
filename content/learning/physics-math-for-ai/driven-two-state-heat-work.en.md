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
