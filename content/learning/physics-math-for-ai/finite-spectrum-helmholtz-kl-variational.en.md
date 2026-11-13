## Why the lowest energy is not the whole answer

Imagine many independently prepared copies of a system with two possible energies. Putting every copy in the lower level minimizes mean energy. Nevertheless, a system in contact with a warm reservoir has some probability of occupying the upper level. To understand that probability, we must include the entropy associated with distributing probability among states.

The question here is precise: for a fixed finite energy spectrum and a fixed positive reservoir temperature, which probability distribution minimizes the Helmholtz functional? We will derive an answer, measure its advantage over other distributions, and distinguish a comparison of states from a physical trajectory.

## State space, constraints and units

There are m explicitly enumerated microstates with fixed energies Eᵢ in joules. Volume, particle number and all parameters of the Hamiltonian are held fixed. A trial distribution p has nonnegative entries summing to one. It may describe repeated preparations; we do not assume every trial distribution is already a thermal equilibrium.

The reservoir has temperature T>0 in kelvin, and kB has units J/K. Natural logarithms give dimensionless information entropy H in nats. Mean energy and Helmholtz functional have units J per copy, while physical entropy has units J/K per copy:

$$
\sum_{i=1}^{m}p_i=1,\qquad U[p]=\sum_i p_iE_i,\qquad H[p]=-\sum_i p_i\ln p_i,\qquad S[p]=k_BH[p],\qquad F_T[p]=U[p]-TS[p].
$$

We extend 0 ln0 continuously to0. For N independent identically distributed copies with additive energies, the joint entropy and mean energy are additive, so total F is N times the per-copy value. Correlated copies require their joint distribution; multiplying a marginal entropy by N can miss mutual information.

## Derive the canonical target

Minimize F over the probability simplex. At an interior stationary point, a Lagrange multiplier λ imposes normalization. Differentiation with respect to pᵢ gives a logarithmic equation; solving it and normalizing determines π:

$$
E_i+k_BT(\ln p_i+1)+\lambda=0,\qquad
\beta=\frac{1}{k_BT},\qquad Z=\sum_i e^{-\beta E_i},\qquad \pi_i=\frac{e^{-\beta E_i}}{Z}.
$$

Every πᵢ is strictly positive for finite energies and finite T>0. This calculation finds the candidate. The global argument below proves it beats every boundary distribution too. Canonical probabilities, information entropy and equilibrium free energy use the standard finite-state framework developed in [Tong, sections1.3.1–1.3.4](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf).

## Derive the free-energy gap

Take the logarithm of π, substitute it into the definition of relative entropy, and use normalization. No approximation or large-system limit is needed:

$$
\ln\pi_i=-\beta E_i-\ln Z,\qquad
D(p\Vert\pi)=\sum_{i:p_i>0}p_i\ln\frac{p_i}{\pi_i}
=-H[p]+\beta U[p]+\ln Z.
$$

Evaluating F at π gives the equilibrium value. Multiplying the preceding identity by kBT then yields the variational certificate:

$$
F_{\rm eq}=-k_BT\ln Z,\qquad
F_T[p]-F_{\rm eq}=k_BT D(p\Vert\pi)\ge0.
$$

This is an energy difference, not a probability and not an elapsed time. For independent copies its total is multiplied by N. The direction is D(p∥π): swapping the arguments changes the quantity and generally destroys this particular identity.

## See positivity one microstate at a time

The usual sum p ln(p/π) contains positive and negative terms. Normalization lets us add terms that sum to zero. Each resulting contribution is nonnegative:

$$
D(p\Vert\pi)=\sum_i b(p_i,\pi_i),\qquad
b(a,q)=a\ln(a/q)-a+q\ge0,\qquad b(0,q)=q.
$$

For a>0, the derivative with respect to a is ln(a/q), and the second derivative is1/a. Thus each contribution has its unique minimum0 at a=q. At a=0 it is positive because q>0. Consequently the total vanishes exactly when every microstate probability equals its target. This proves uniqueness of the global minimum, including the boundary.

An intuitive picture is a bowl above the tangent to x ln x. Individual signed log terms can cancel; the distances above their tangents cannot. The size of that bowl measures how much the trial allocation departs from the thermal allocation.

## A two-state graph you can reconstruct

Choose a reference temperature T₀ and fixed energies0 and Δ=kBT₀ ln3. At T=T₀ the canonical distribution is (3/4,1/4). Let x be the excited-state probability. The horizontal axis is x, and the vertical axis is F/(kBT₀):

$$
f(x)=x\ln3+(1-x)\ln(1-x)+x\ln x,\qquad
f'(x)=\ln\frac{3x}{1-x},\qquad
f''(x)=\frac1x+\frac1{1-x}>0.
$$

Draw a straight line x ln3 for energy, a curved nonpositive entropy term, and their sum. The energy line favors x=0. The entropy term favors x=1/2. Their sum has its unique minimum at x=1/4. The infinite endpoint slopes are compatible with finite endpoint values; evaluate the values using the continuous extension.

| Trial allocation | U/(kBT₀) | H | F/(kBT₀) | Gap/(kBT₀) |
| --- | ---: | ---: | ---: | ---: |
| Ground state only | 0 | 0 | 0 | 0.287682 |
| Canonical | 0.274653 | 0.562335 | −0.287682 | 0 |
| Uniform | 0.549306 | 0.693147 | −0.143841 | 0.143841 |

Uniformity maximizes H here but does not minimize F. A larger entropy alone is not the equilibrium criterion when the competing distributions have different mean energies.

## Count degeneracy exactly once

Suppose energy group a contains gₐ distinct microstates. Write its total probability as Pₐ and its conditional distribution inside the group as rₐⱼ. Terms from empty groups are defined by continuity. Expanding the logarithm of pₐⱼ=Pₐrₐⱼ gives:

$$
H[p]=H[P]+\sum_a P_a H[r_a],\qquad
\Pi_a=\frac{g_a e^{-\beta E_a}}{Z},\qquad
Z=\sum_a g_a e^{-\beta E_a}.
$$

At equilibrium, equal-energy microstates are equally likely. A trial group probability alone does not establish that property. The exact decomposition of the gap is:

$$
D(p\Vert\pi)=D(P\Vert\Pi)+\sum_aP_aD(r_a\Vert u_a),\qquad (u_a)_j=\frac1{g_a}.
$$

If we assume uniform conditionals, H gains Pₐ ln gₐ. If we enumerate all microstates, that contribution is already present. Including both an expanded microstate list and another degeneracy factor would count states twice. A plot of only group bars must state which internal conditional distributions it uses.

## Shift the zero without changing the problem

Add the same constant C to every energy. This changes the absolute energy reference but leaves energy differences, probabilities and entropy unchanged:

$$
E_i'=E_i+C,\qquad Z'=e^{-\beta C}Z,\qquad
U'[p]=U[p]+C,\qquad F_T'[p]=F_T[p]+C,\qquad F_{\rm eq}'=F_{\rm eq}+C.
$$

The free-energy gap therefore remains invariant. Its nonnegativity has physical content even when an arbitrary choice of zero makes both plotted free energies negative. A dimensionless loss cannot acquire joule units merely by being renamed energy; the scale and modeling assumptions must be supplied.

## Change temperature while keeping the Hamiltonian fixed

In the two-state example, Δ is fixed using T₀ once. At another temperature T, the excited-to-ground probability ratio is3^(−T₀/T), not always1/3. Raising T to2T₀ gives approximately (0.633975,0.366025). Recomputing Δ=kBT ln3 at each slider position would change the Hamiltonian together with the bath.

The same distinction applies to positive weights v supplied at one reference temperature:

$$
E_i=C-k_BT_0\ln v_i\quad\Longrightarrow\quad
\pi_i(T)=\frac{v_i^{T_0/T}}{\sum_jv_j^{T_0/T}}.
$$

Normalizing v directly is correct at T₀. It does not implement the fixed-spectrum temperature map for arbitrary T. The finite-weight KL kernel provides a comparison primitive. Separate modules now evaluate the fixed-Hamiltonian temperature map and its positive Helmholtz gap, using directed intervals and a logarithmic cold-tail decomposition rather than rounded target weights.

## Resolve small probabilities and small gaps

For numerical evaluation, subtract the minimum energy before exponentiating. This prevents a large common energy offset from overflowing every weight. Keep log probabilities as well as probabilities:

$$
a_i=\frac{E_i-E_{\min}}{k_BT},\qquad
L=\ln\sum_j e^{-a_j},\qquad
\ln\pi_i=-a_i-L,\qquad F_{\rm eq}=E_{\min}-k_BTL.
$$

This algebra alone does not solve every binary64 limit. At extremely small positive T, the division defining aᵢ may itself exceed the number range. A vanishing displayed exponential is then an underflow observation, not a proof of a forbidden microstate. Raw scaled products, logarithmic evidence and explicit unresolved observations are needed.

Near p=π, subtracting two rounded free energies can lose a positive gap. For the two-state example with x=1/4+ε, the leading gap divided by kBT₀ is(8/3)ε². The linear change vanishes at the minimum. Comparing a stable positive remainder against an independent high-precision calculation is more informative than clipping a negative subtraction to zero.

$$
\frac{F_{T_0}[p]-F_{\rm eq}}{k_BT_0}=\frac83\varepsilon^2+O(\varepsilon^3),\qquad p=(3/4-\varepsilon,1/4+\varepsilon).
$$

## Positive temperature and the zero-temperature limit

As T approaches zero from above, a finite fixed spectrum concentrates on its ground-state manifold. If there are g₀ exactly degenerate ground microstates, the canonical limit is uniform on those states and H approaches ln g₀. At high temperature the finite spectrum approaches a uniform distribution over all m microstates.

These are limits of the canonical family. They do not authorize substituting T=0 into β or treating0 times an infinite KL as an ordinary product. Nor does residual entropy of this finite model by itself settle the thermodynamic third-law behavior of a macroscopic material. The order of system-size and temperature limits requires its own analysis.

## A variational landscape does not supply a clock

The identity orders candidate distributions at a fixed T and Hamiltonian. It supplies no transition rates, friction coefficient, integration step, or time variable. A line joining p to π is a comparison path unless a dynamical law has been specified.

For a reservoir relaxation experiment with no work input and a fixed Hamiltonian, an endpoint balance can relate a free-energy decrease to total entropy production under the appropriate system–bath assumptions. Arbitrary optimization updates do not automatically meet those assumptions. The distinction between fixed-temperature and fixed-pressure potentials is developed in [Tong, section4.4](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf).

## Research extension: a quench, a new target and thermal relaxation

A distribution can be exactly canonical, then suddenly have a positive free-energy gap because the energy levels changed while its probabilities did not. That jump is work stored in a nonequilibrium preparation; it is not yet entropy produced by relaxation. We specify a transition law to see where the stored gap subsequently goes.

Use a separate two-microstate example with ground energy zero and excited energy $E_j$, not the original three-state lab. The bath has fixed T>0, and both gaps are finite and positive. Initially the system is equilibrated with gap $E_a$. At time zero an ideal externally controlled classical level shift changes it to $E_b$ without changing occupations or exchanging heat. No coherence, measurement feedback or bath-temperature change is included. All energies below are per copy.

Define dimensionless gaps and the two canonical targets. The target changes from $\pi_a$ to $\pi_b$ at the quench. Relative entropy must always name which of these targets is being used.

$$
e_j=E_j/(k_BT),\quad Z_j=1+e^{-e_j},\quad p_j=\frac1{1+e^{e_j}},\quad\pi_j=(1-p_j,p_j)\qquad(j=a,b).
$$

Because the probabilities and their entropy are unchanged at the instantaneous shift, the change of mean energy is work on the system. This work-on convention is explicit; there is no boundary volume work in this example.

$$
W_{\rm on}=(E_b-E_a)p_a,\qquad Q_{\rm quench}=0,\qquad\Delta S_{\rm quench}=0.
$$

Subtract the two equilibrium free energies. The excess work above that equilibrium difference is exactly the new nonequilibrium gap. Calling this positive excess immediate entropy generation would be wrong: the ideal quench itself exchanged no heat and changed no entropy.

$$
\Delta F_{\rm eq}=-k_BT\log(Z_b/Z_a),\qquad W_{\rm ex}=W_{\rm on}-\Delta F_{\rm eq}=k_BT D(\pi_a\Vert\pi_b).
$$

Now hold $E_b$ and T fixed and supply a particular Markov transition law. The forward and reverse stationary fluxes agree. The rate scale γ has units s⁻¹ and is an added modeling choice; it cannot be inferred from the free-energy landscape alone.

$$
k_{01}=\gamma p_b,\quad k_{10}=\gamma(1-p_b),\quad(1-p_b)k_{01}=p_b k_{10},\qquad\gamma>0.
$$

The master equation gives exponential relaxation. Distinct positive γ values reach the same equilibrium with different clocks. This explicitly added dynamics does not turn the existing lab's comparison scan into a physical trajectory.

$$
\dot p=k_{01}(1-p)-k_{10}p=-\gamma(p-p_b),\qquad p(t)=p_b+(p_a-p_b)e^{-\gamma t}.
$$

During relaxation there is no work input. The change of mean energy is heat into the system, which can be negative. System entropy can change with either sign, so it is essential to include the bath contribution.

$$
Q(t)=E_b[p(t)-p_a],\qquad\Delta S(t)=k_B\{H[p(t)]-H[p_a]\},\quad H[p]=-p\log p-(1-p)\log(1-p).
$$

The complete system–bath production equals the decrease of the fixed-target KL. This follows by inserting the canonical log probability into the endpoint entropy balance, with no further large-system limit.

$$
\Sigma(t)=\Delta S(t)-Q(t)/T=k_B\{D(\pi_a\Vert\pi_b)-D((1-p(t),p(t))\Vert\pi_b)\}\ge0.
$$

Differentiate the KL along the specified master equation. The two factors have the same sign because log odds increase with p. Thus production is nonnegative for the interior probabilities used here. This is a property of this declared thermal dynamics, not a general rule for arbitrary optimizer updates.

$$
\dot\Sigma=k_B\gamma(p-p_b)\log\frac{p(1-p_b)}{p_b(1-p)}\ge0,\qquad 0<p,p_b<1.
$$

Account for both stages together. The initial energy uses $E_a$, while the final energy uses $E_b$. Omitting the quench work would make the subsequent heat look like a failure of energy conservation.

$$
E_b p(t)-E_a p_a=W_{\rm on}+Q(t).
$$

At a finite observation time, excess work is split between a remaining free-energy gap and produced entropy multiplied by T. Only after complete relaxation does the first term vanish. The target-relative gap can jump upward at the externally driven quench even though it decreases during the subsequent fixed-target relaxation.

$$
W_{\rm ex}=k_BT D((1-p(t),p(t))\Vert\pi_b)+T\Sigma(t).
$$

### Read the target jump and remaining resource

The upper panel shows continuous excited occupancy and the discontinuous canonical target at t=0. The lower panel separates the current-target KL from accumulated Σ/kB. Their sum after the quench is the initial excess divided by kBT. Before the quench the old-target gap is zero. The time axis is seconds only because γ was specified.

![A sudden target change creates a free-energy gap with unchanged occupancy; fixed-target thermal relaxation converts that gap into system–bath entropy production.](/assets/learning/free-energy-quench-relaxation.en.svg)

### Reproduce one finite relaxation

Take $E_a$/(kBT)=log3, $E_b$/(kBT)=log9 and γ=1 s⁻¹. Initial excited occupancy is 1/4 and the new equilibrium occupancy is 1/10. The table evaluates the exact finite-model solution at ordinary scales. Heat is normalized by kBT, whereas production is divided by kB.

| t (s) | Excited p(t) | Remaining D | Σ/kB | Q/(kBT) |
| --- | --- | --- | --- | --- |
| 0.00000000 | 0.25000000 | 0.09233152 | 0.00000000 | 0.00000000 |
| 0.50000000 | 0.19097960 | 0.03734530 | 0.05498621 | -0.12968108 |
| 1.00000000 | 0.15518192 | 0.01473671 | 0.07759480 | -0.20833662 |
| 3.00000000 | 0.10746806 | 0.00030324 | 0.09202828 | -0.31317468 |

```python
from math import exp, log
ea, eb, gamma = log(3.0), log(9.0), 1.0
pa, pb = 1/(1+exp(ea)), 1/(1+exp(eb))
def kl(p):
    return p*log(p/pb)+(1-p)*log((1-p)/(1-pb))
initial_gap = kl(pa)
for t in [0.0, 0.5, 1.0, 3.0]:
    p = pb+(pa-pb)*exp(-gamma*t)
    gap = kl(p)
    generated = initial_gap-gap
    heat = eb*(p-pa)  # Q/(kB T)
    print(f"{t:.8f} {p:.8f} {gap:.8f} {generated:.8f} {heat:.8f}")
```

### Three research questions with explanations

#### 1. Does an upward KL jump at the quench violate nonnegative production?

**Explanation.** No. The reference distribution changed through external work. The ideal shift produces no entropy immediately; it stores an excess relative to the new equilibrium. The monotonicity claim applies only after the Hamiltonian and bath are held fixed with the specified transition law.

#### 2. Can the variational minimum determine γ?

**Explanation.** No. Any positive γ in these rates gives the same detailed-balance target and a different relaxation time. A microscopic coupling model or measured kinetics is needed to select that scale. A scan index or training iteration is not automatically seconds.

#### 3. Where is the excess work before relaxation is complete?

**Explanation.** It is divided between kBT times the remaining KL and T times the accumulated system–bath production. The latter is not the system entropy change alone. The energy balance separately includes work from the level shift and heat from the fixed-gap relaxation.

### Source and research boundary

[David Tong, Statistical Physics, sections 1.3.1–1.3.4](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf) supplies the fixed-spectrum canonical/free-energy conventions; [Classical Thermodynamics, sections 4.2–4.4](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) supplies work and entropy bookkeeping. The quench protocol and Markov rates are explicitly added here and checked independently. They are not a newly implemented control in the original three-state comparison lab.

## Connection to learning and inference

For a normalized finite target proportional to exp(−energy/scale), an objective containing expected energy and negative entropy can be rewritten as a scaled KL plus a constant. This is an algebraic bridge to variational inference. The target, scale, support and direction of KL must all be specified.

A restricted approximation family may not contain π. Minimizing within that family gives the best permitted approximation under this objective, rather than proving the physical system cannot reach equilibrium. Correlated variables, continuous measures and quantum coherence require additional definitions; this finite diagonal probability model does not resolve them.

## Worked exercises

### Exercise 1: Uniform trial at the reference temperature

For energies0 and kBT₀ ln3, evaluate U, H and F for p=(1/2,1/2) at T₀. How far is it above equilibrium?

**Answer:** U/(kBT₀)=ln3/2=0.549306 and H=ln2=0.693147. Therefore F/(kBT₀)=−0.143841. Since Z=4/3, Feq/(kBT₀)=−ln(4/3)=−0.287682, so the gap is0.143841 kBT₀. Direct KL gives one half of ln(2/3) plus one half of ln2, the same dimensionless gap.

### Exercise 2: A boundary distribution

Put all probability in the ground state of the same spectrum. Is its entropy undefined, and is it the warm equilibrium?

**Answer:** H=0 by the continuous extension. U=F=0. The gap is kBT₀ ln(4/3)>0, and D((1,0)∥(3/4,1/4))=ln(4/3). In the nonnegative decomposition, the empty excited state contributes1/4; it must not be silently dropped from that decomposition even though its p ln(p/π) term is0.

### Exercise 3: Hidden imbalance inside a degenerate group

There are two ground microstates and one excited microstate with energy kBT₀ ln2. Compare p=(0.8,0,0.2) with equilibrium at T₀.

**Answer:** Z=2+1/2=2.5 and π=(0.4,0.4,0.2). Both have group probabilities (0.8,0.2) and the same mean energy. The trial has H=0.500402; equilibrium has H=1.054920. The entropy deficit and dimensionless gap are0.8 ln2=0.554518. Group-only KL is0, while the ground-group conditional term recovers the full positive gap.

### Exercise 4: Move the energy reference

Add5kBT₀ to each energy in Exercise1, keeping the bath at T₀. What changes?

**Answer:** U, F and Feq each increase by5kBT₀. Z is multiplied by exp(−5). Its normalization cancels that common factor, so π, H and D are unchanged. The gap stays0.143841 kBT₀. Comparing the shifted trial F to the old unshifted Feq would mix two energy references.

### Exercise 5: Double the bath temperature

Keep the original two-state energy gap fixed and set T=2T₀. Find the new equilibrium and identify the control mistake that would keep it at (3/4,1/4).

**Answer:** The weight ratio is exp(−ln3/2)=1/√3. Normalization gives ground probability √3/(1+√3)=0.633975 and excited probability0.366025. Replacing the fixed gap by kBT ln3 at every temperature would preserve the old ratio by changing the spectrum itself.

### Exercise 6: Read an almost-equilibrium result

Let x=1/4+ε with small nonzero ε. A screen shows identical rounded F and Feq. May it label the state exactly at equilibrium?

**Answer:** No. Strict convexity gives a positive gap for every admissible nonzero ε. Taylor expansion gives gap/(kBT₀)=(8/3)ε²+O(ε³). Exact normalization and a stable positive remainder must determine the equality condition. A value outside the linear display range should retain its sign and logarithmic evidence rather than become a physical zero.

## Explore the experiment

This reader’s experiment uses **three microstates**, with two equal-energy ground states and one excited state. The earlier two-state graph explains the variational argument; its equilibrium excited probability is not the target for this three-state experiment. With an energy gap Δ and a common reference C, the implemented spectrum and comparison family are:

$$
(E_0,E_1,E_2)=(C,C,C+\Delta),\qquad
p(x)=\left(\frac{1-x}{2},\frac{1-x}{2},x\right),\qquad
\pi_2=\frac{e^{-\Delta/(k_BT)}}{2+e^{-\Delta/(k_BT)}}.
$$

Each blue bar is a trial microstate probability; each orange bar is its canonical target. Input weights are normalized to obtain p. The energy-gap and energy-reference fields are in kelvin because they specify Δ/kB and C/kB; the energy table itself uses joules. Hold these fields fixed when changing only the bath temperature.

| Example to select | What to compare | Interpretation |
| --- | --- | --- |
| Hidden conditional imbalance | Weights (4,0,1) versus (2,2,1) at the same T and gap | Group probabilities stay (0.8,0.2); redistributing the ground group reduces F by 0.8 kBT ln2. |
| Nearly canonical | Blue and orange bars, then the gap status | The binary inputs approximate the ideal gap kBT ln2; close bars do not prove exact equality. |
| Exact degenerate equilibrium | Zero energy gap and three equal weights | All energies match and p is structurally canonical; the gap is exactly zero. |
| Gap recovered in the total | F−Feq versus N(F−Feq), including sign and log columns | A missing per-copy linear value can become representable after multiplying the original quantity by N. |

The F and gap curves use p(x) with equal ground conditionals. A selected trial with unequal ground weights can lie above those curves at the same p₂. Read the point together with all three bars. The temperature plot shows 1–1200 K; the observation table still evaluates an allowed T outside that plotting range. A comparison scan replaces the trial weights with p(x); it does not preserve a hidden ground imbalance or simulate thermal relaxation. Reset or reselect the example to return to its preparation.

The figures, guided experiments and observation table are available in this reader. A finite precision ceiling can still leave a calculation explicitly unresolved; that status never establishes equilibrium. Subtracting displayed F and Feq does not certify their positive difference. Continue to the ensemble and partition-function units to connect this finite variational picture to reservoir models and fluctuations.
