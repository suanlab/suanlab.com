## A proposal is not yet a target-preserving transition

Suppose a sampler proposes moving clockwise around three states more often than counterclockwise. A destination with a larger target weight may look attractive, but attraction alone does not account for how often that destination proposes the reverse move. Ignoring this second asymmetry can change the distribution the sampler actually preserves.

The previous chapter separated a stationary distribution from detailed balance and convergence. Here we build a transition that preserves a specified nonuniform target. We will keep three objects separate: the proposal probability, acceptance conditional on that proposal, and the final transition including rejection. A high acceptance number is not a proof that these objects have been combined correctly.

For the general proposal-ratio construction, see [Hastings (1970), section 2.2](https://hedibert.org/wp-content/uploads/2013/12/1970Hastings.pdf), particularly its test ratio and the separate requirement to check communication. [Levin–Peres, section 3.2.2](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf) gives a textbook treatment. Below we derive the chosen rule directly from symmetric pair flows and calculate our own finite examples. You need finite probabilities, row-vector transitions and the preceding chapter's flow ledger.

## Define the target and the proposal independently

There are three named states, 0, 1 and 2. Give them positive integer target weights w0,w1,w2. Separately choose nonnegative proposal weights f,b,s for clockwise, counterclockwise and self proposals. Each weight is at most 1,000,000, the target weights are at least one, and the proposal-weight sum must be positive.

$$
W=w_0+w_1+w_2,\qquad \pi_i=\frac{w_i}{W},\qquad D=f+b+s,\qquad Q=\frac1D\begin{pmatrix}s&f&b\\b&s&f\\f&b&s\end{pmatrix}.
$$

Qij means that the sampler proposes j when currently at i. It does not yet mean that the next recorded state will be j. The target assigns probability to states; Q assigns conditional proposal probability to possible destinations. Normalizing one does not normalize or correct the other.

| Object | Question it answers | Constraint here |
| --- | --- | --- |
| πi | How much target probability belongs to state i? | Positive, total one |
| Qij | How often is j proposed from i? | Nonnegative, each departure row totals one |
| αij | Given that proposal, how often is it accepted? | Between zero and one; absent proposals are not assigned a conditional value |
| Pij | Where is the next chain state after accept/reject? | Nonnegative, each row totals one |

The full-support target restriction matters: no state has zero target probability. We are not silently specifying how a continuous density, a zero-target region or an adaptive neural proposal should be handled. This experiment analyzes a fixed finite kernel; it does not generate a sampled trajectory.

## Start with a symmetric pair-flow budget

Detailed balance asks for equal target-weighted final traffic in both directions. For distinct states, a simple construction accepts the smaller of the two available target-weighted proposal flows:

$$
\pi_iP_{ij}=\min\!\left(\pi_iQ_{ij},\pi_jQ_{ji}\right),\qquad i\ne j.
$$

This choice cannot exceed the traffic proposed in either direction. It therefore leaves room for a valid acceptance probability. The minimum is unchanged when i and j are exchanged, so the accepted off-diagonal flow is symmetric by construction.

This is one valid choice of a symmetric flow budget, not a claim that every target-preserving transition must use it. Nonreversible kernels can also preserve a target, as the previous chapter showed. Here the symmetric construction is valuable because it gives a direct, checkable proof of target invariance.

## Derive conditional acceptance and treat absent proposals explicitly

For an event that can actually be proposed, divide its accepted flow by its proposed flow:

$$
\alpha_{ij}=\min\!\left(1,\frac{\pi_jQ_{ji}}{\pi_iQ_{ij}}\right)\qquad\text{when }Q_{ij}>0.
$$

The target denominator is positive by the input contract. The proposal denominator is positive only in this branch. The resulting accepted flow is

$$
\pi_iQ_{ij}\alpha_{ij}=\min\!\left(\pi_iQ_{ij},\pi_jQ_{ji}\right)\qquad(i\ne j).
$$

If Qij is zero, that proposal never occurs. Its conditional acceptance is not provided; its unconditional contribution is exactly zero. If Qij is positive but Qji is zero, the proposal occurs and is rejected with probability one. These situations have different meanings even though both contribute zero off-diagonal transition probability.

| Proposal situation | Conditional acceptance | Final off-diagonal contribution |
| --- | --- | --- |
| Qij=0 | Not provided for an absent event | 0 |
| Qij>0,Qji=0 | Exactly 0 | 0 |
| Qii>0 | Self proposal accepted by convention | Remains at i |

For a present self proposal, the ratio is one. Accepting it does not move the state. Adding a tiny artificial reverse probability would change the proposal model; it is not a way to resolve an absent conditional event.

## Put rejected mass back on the departure diagonal

For distinct states, Pij=Qijαij. A rejected proposal leaves the sampler at its current state. The departure diagonal must include both self proposals and all rejected proposals:

$$
P_{ij}=\begin{cases}Q_{ij}\alpha_{ij},&i\ne j,\\1-\sum_{k\ne i}Q_{ik}\alpha_{ik},&i=j.\end{cases}
$$

It helps to keep an elementary ledger of accepted and rejected joint events:

$$
A_{ij}=Q_{ij}\alpha_{ij},\qquad B_{ij}=Q_{ij}-A_{ij},\qquad h_i=\sum_jB_{ij},\qquad P_{ii}=Q_{ii}+h_i.
$$

For an absent proposal, A and B are both zero even though α is not provided. A present self proposal has Bii=0. Consequently every final row sums to one. The diagonal probability can be large even if the original proposal had no self loop.

Do not remove rejected mass and normalize the remaining off-diagonal row as though it were the same chain. That produces a different observation clock. We will calculate the corresponding state-change chain later in this chapter.

## Work through a nonuniform three-state example

Use target weights (1,2,4) and proposal weights (2,1,1). The target and proposal matrix are

$$
\pi=\left(\frac17,\frac27,\frac47\right),\qquad Q=\begin{pmatrix}1/4&1/2&1/4\\1/4&1/4&1/2\\1/2&1/4&1/4\end{pmatrix}.
$$

From state 0 to state 1, the target-weighted proposal flows are equal: (1/7)(1/2)=(2/7)(1/4). That proposal is accepted with probability one. Going from state 2 to state 0 needs a different correction:

$$
\alpha_{20}=\min\!\left(1,\frac{(1/7)(1/4)}{(4/7)(1/2)}\right)=\frac18,\qquad P_{20}=\frac12\frac18=\frac1{16}.
$$

The target-only ratio would have given 1/4, missing the reverse/forward proposal ratio of 1/2. From state 2 to state 1 the corrected acceptance is one, not the target-only value 1/2. Different directions can need opposite changes.

The final corrected transition matrix is

$$
P=\begin{pmatrix}1/4&1/2&1/4\\1/4&1/4&1/2\\1/16&1/4&11/16\end{pmatrix}.
$$

Its large final diagonal entry at state 2 is not an arbitrary adjustment. The 2→0 proposal contributes rejection (1/2)(7/8)=7/16. Adding the self-proposal probability 1/4 gives 11/16.

| Departure state | Self proposal | Total rejection | Final holding | Actual movement |
| --- | --- | --- | --- | --- |
| 0 | 1/4 | 0 | 1/4 | 3/4 |
| 1 | 1/4 | 0 | 1/4 | 3/4 |
| 2 | 1/4 | 7/16 | 11/16 | 5/16 |

For example, target-weighted traffic on 0↔2 is 1/28 in both directions. The same check works for every pair. Summing those pair equalities proves πP=π.

## Calculate what the target-only comparison gets wrong

Consider the explicitly uncorrected comparison that omits the proposal ratio:

$$
\widetilde\alpha_{ij}=\min(1,\pi_j/\pi_i),\qquad \widetilde P=\begin{pmatrix}1/4&1/2&1/4\\1/8&3/8&1/2\\1/8&1/8&3/4\end{pmatrix}.
$$

Its rows are valid probability distributions. Valid rows alone do not establish the intended target. Exact multiplication and a separate stationary solve give

$$
\pi\widetilde P-\pi=\left(0,-\frac1{28},\frac1{28}\right),\qquad \rho=\left(\frac17,\frac5{21},\frac{13}{21}\right),\qquad \rho\widetilde P=\rho.
$$

The target loses probability at state 1 and gains it at state 2 in one step. The comparison's actual stationary distribution ρ differs from π by total variation 1/21. A sampler can therefore settle into a stable, normalized distribution while sampling the wrong target. Calling the discrepancy “burn-in” would not repair its invariant law.

## Label target-prepared and stationary ledgers differently

When π is not invariant for a comparison kernel, traffic weighted by π is a one-step target-prepared ledger, not stationary traffic. The actual stationary ledger uses the separately solved ρ:

$$
F^{\pi}_{ij}=\pi_i\widetilde P_{ij},\qquad F^{\rho}_{ij}=\rho_i\widetilde P_{ij},\qquad \sum_j(F^{\pi}_{ij}-F^{\pi}_{ji})=\pi_i-(\pi\widetilde P)_i.
$$

In the worked comparison, target-prepared clockwise net currents are 1/28,1/14,1/28 along 0→1,1→2,2→0. Their unequal values produce the nonzero divergence just calculated. At the actual stationary law, all three clockwise currents are 1/24 and their vertex divergences vanish.

Both ledgers are informative. Their preparation labels are essential. A chart showing target-weighted arrows cannot be called a stationary-current chart merely because the algorithm has some other stationary law. For the corrected irreducible kernel, ρ=π, so the two preparations coincide and detailed-balance currents vanish.

## Understand the cases where omission happens to preserve the target

A symmetric proposal makes its reverse/forward ratio one, so both acceptance rules coincide. A uniform target under our cyclic proposal also remains invariant for the uncorrected rule: it accepts every proposal and retains a doubly stochastic Q. However, asymmetric clockwise and counterclockwise traffic still violates detailed balance.

There is a less obvious nonuniform example. Set target weights (1,1,4), keeping proposal weights (2,1,1). The uncorrected rule preserves the target but has clockwise current 1/24 on every edge. It is not a general justification for dropping the proposal ratio.

The exact uncorrected target-prepared current on a clockwise edge explains the coincidence:

$$
\widetilde J^{\pi}_{i,i+1}=\frac{f-b}{DW}\min(w_i,w_{i+1}).
$$

For three states, stationarity requires the three clockwise currents to be equal. When f≠b, this occurs exactly when the three pair minima are equal, which is equivalent to the minimum target weight occurring at least twice. With distinct weights (1,2,4) the condition fails. Testing only symmetric proposals, uniform targets or repeated-minimum targets can therefore conceal an incorrect general implementation.

| Preparation | Uncorrected target invariant? | Uncorrected detailed balance? |
| --- | --- | --- |
| Symmetric proposal, any admitted target | Yes | Yes |
| Asymmetric proposal, uniform target | Yes | No |
| Asymmetric proposal, target (1,1,4) | Yes | No |
| Proposal (2,1,1), target (1,2,4) | No | No |

These statements are about the admitted cyclic three-state family. They are not a characterization of every proposal on every state space.

## Check communication after applying the correction

A proposal graph can be strongly connected even when its corrected graph is not. If the proposal moves only clockwise, every proposed off-diagonal move lacks a reverse proposal. The correct acceptance is zero on each such move. The final kernel is the identity, regardless of whether the proposal also has a self loop.

The target is preserved, but so is every other distribution. There is no unique stationary distribution or exploration. Conditional acceptance for an absent reverse event is not invented, and a selected target is not returned as though it were a uniquely computed equilibrium. A correct construction can expose an unsuitable proposal design.

If both directional proposal weights are positive, every pair of distinct states has positive corrected transition probability because all target weights are positive. In this triangle, returns of lengths two and three are then possible, giving period one even without self proposals. This particular connected family converges to its unique target from every initialization. The triangle argument does not make aperiodicity a universal consequence of Hastings correction. The next two-state chapter will show a periodic reversible chain.

## Distinguish acceptance, movement and the preparation used for averaging

Acceptance conditional on a proposed destination differs from average acceptance. At a target-prepared step, average acceptance includes self proposals while movement excludes them:

$$
a_{\pi}=\sum_i\pi_i\sum_jA_{ij},\qquad h_{\pi}=\sum_i\pi_ih_i,\qquad m_{\pi}=\sum_i\pi_i\sum_{j\ne i}P_{ij},\qquad a_{\pi}+h_{\pi}=1.
$$

For the corrected worked example, aπ=3/4, hπ=1/4 and mπ=1/2. The difference between acceptance and movement is the self-proposal probability 1/4. For a proposal that only proposes the current state, average acceptance is one and movement is zero. A nearly self-only proposal can have acceptance near one while still doing essentially nothing.

For the uncorrected example, target-prepared movement is 3/7. Its actual stationary movement, averaged using ρ instead, is 23/56. These are different preparations. Neither an acceptance statistic nor a one-step movement statistic alone certifies Monte Carlo efficiency. Compare invariant-law error, communication, dependence of the intended observable and the actual computational cost.

## Do not silently change the clock by deleting all holding

Suppose we record only state changes, deleting both rejected repeats and accepted self proposals. Write ci=1−Pii for the probability of leaving state i. When every ci is positive, the state-change transition matrix is

$$
T_{ij}=\frac{P_{ij}}{c_i}\quad(i\ne j),\qquad T_{ii}=0,\qquad c_i=1-P_{ii}.
$$

If ρ is the stationary distribution of P, the stationary distribution of this jump chain is

$$
\nu_i=\frac{\rho_i c_i}{\sum_k\rho_kc_k},\qquad (\nu T)_j=\frac{\sum_{i\ne j}\rho_iP_{ij}}{\sum_k\rho_kc_k}=\frac{\rho_jc_j}{\sum_k\rho_kc_k}=\nu_j.
$$

States with a large leaving probability are overrepresented in the jump record relative to the original step clock. For the corrected worked example, the intended target is stationary under P, but the jump record has

$$
\nu=\left(\frac3{14},\frac37,\frac5{14}\right),\qquad \|\nu-\pi\|_{\mathrm{TV}}=\frac3{14}.
$$

Thus keeping only a visually tidy list of changed states changes the sampling law. Ordinary step-based MCMC estimates retain repeated states after rejection. If using a jump record, account for the omitted holding times with an appropriate estimator; its unweighted state frequencies need not represent π. Deleting only rejected proposals while keeping accepted self proposals is another clock and is not the same jump chain defined here.

For an identity kernel there is no leaving event. The jump transition and its stationary law are not provided. Replacing those missing objects by zero or the chosen target would manufacture an observation process that does not occur.

## Keep exact ratios, zeros and displayed residuals distinct

Target normalization cancels from the ratio, but proposal asymmetry remains:

$$
\frac{\pi_jQ_{ji}}{\pi_iQ_{ij}}=\frac{w_jQ_{ji}}{w_iQ_{ij}}\qquad(Q_{ij}>0).
$$

Common positive scales of target weights or proposal weights therefore do not change the kernel within the admitted ranges. This cancellation does not make a nonnormalizable target into a valid probability law. Here the three positive finite target weights guarantee a normalizable target before the ratio is used.

All calculations in the finite model use exact rational numbers. Conditional acceptance is computed only when its conditioning proposal has positive probability. Missing proposals are not repaired by epsilon denominators. Rejection is retained exactly, and classifications use exact signs and residuals, not the appearance of rounded bars.

For example, take target weights (L,L−1,L) and proposal weights (L,L−1,L), with L=1,000,000 and D=3L−1. The uncorrected one-step drift is

$$
\pi\widetilde P-\pi=\frac1{D^2}(1,0,-1).
$$

A six-decimal marginal display hides this roughly 10^−13 discrepancy, but it is not exact invariance. Every reported rational quantity retains its exact endpoints and sign. The Number and magnitude-log fields are display projections; they are not additional proofs of equality. The corrected residual is exactly zero for algebraic reasons, not because a tolerance accepts a small error.

## Connect the construction to AI with the target and proposal still explicit

A learned proposal can be useful if its actual conditional probability and the corresponding reverse probability can be evaluated and its corrected support is suitable. The ratio is not obtained by substituting a score that merely ranks destinations. A finite-state proof also does not establish that an approximate neural likelihood, a truncated decoding distribution or a changing proposal has the required probability law.

When a proposal changes with training history, the fixed-kernel argument needs additional analysis. Continuous state spaces require densities with respect to an appropriate reference measure and careful support conditions. Zero-target states need an explicit initialization and transition convention outside this chapter's full-support contract. These are further mathematical problems, not details solved by copying the same fraction into code.

Our probabilities refer to discrete proposal-and-decision steps. They do not encode an energy reservoir, heat or physical entropy production. Detailed balance as an algorithmic construction and local detailed balance in a thermodynamic model require distinct assumptions.

## Connect proposals, acceptance and movement to the experiment

The default uses target weights 1,2,4 and forward, backward and self-proposal weights 2,1,1. Find the probability of proposing state 1 from state 0, the conditional probability of accepting that proposal and the resulting transition probability in their respective matrices. Separating how often an application is submitted from how often it passes review explains why the two probabilities multiply.

| Action | Observation to inspect | Question to explain |
| --- | --- | --- |
| Switch between ‘Hastings corrected’ and ‘Target-ratio-only comparison’ | Transition matrix, target residual πP−π and actual stationary law ρ | Why can the actual stationary law change while the specified target stays fixed? |
| Choose ‘Missing reverse proposal, corrected identity’ | The corrected transition is identity and stationary uniqueness is false | Can target invariance and pairwise balance alone guarantee movement or mixing? |
| Choose ‘High acceptance, low movement’ | Self proposal, rejection holding, total holding, mean acceptance and mean movement | Why can high acceptance coexist with infrequent state changes? |

Automatic scanning compares distinct fixed-kernel examples; it does not evolve time or an initial distribution. Matrix buttons reveal different ledgers of the same preparation, and switching correction does not overwrite the specified target. When comparing target, actual stationary and jump-stationary bars, explain how removing holding changes the clock and the jump law. An unavailable law is neither a uniform distribution nor a zero vector.

## Predict, calculate and explain

### 1. Separate three probabilities

For target (1,2,4) and proposal (2,1,1), compare the 0→1 and 2→0 proposal probabilities, conditional acceptances and final off-diagonal transitions.

**Explanation.** Q01=1/2, α01=1 and P01=1/2. Q20=1/2, α20=1/8 and P20=1/16. Equal proposal probabilities need not have equal acceptance or equal final transition probabilities. The reverse-proposal probabilities enter the test ratios.

### 2. Reconstruct the departure diagonal

In the same example, how much rejection comes from the 2→0 proposal, and what is P22? What are average acceptance and movement at the target?

**Explanation.** Rejection is (1/2)(7/8)=7/16. Add the self-proposal probability 1/4 to obtain P22=11/16. Target-averaged rejection is 1/4, acceptance is 3/4 and movement is 1/2. Acceptance includes accepted self proposals, which do not contribute to movement.

### 3. Inspect a missing reverse proposal

Let the proposal move only clockwise. What are α01, α02 and the final corrected kernel?

**Explanation.** The 0→1 proposal occurs but has no reverse, so α01=0. The 0→2 proposal never occurs, so α02 is not provided. Every off-diagonal move is rejected and P is the identity. The target is invariant, but stationary distributions are nonunique and the chain does not explore.

### 4. Find the wrong target rather than waiting longer

For the uncorrected target-only comparison in the worked example, give its target drift, actual stationary distribution and distance from the target.

**Explanation.** The drift is (0,−1/28,1/28). Solving for its actual stationary law gives ρ=(1/7,5/21,13/21), at total variation 1/21 from π. Waiting longer does not change this invariant-law error. Its stationary clockwise current is 1/24 on each edge, while its target-prepared currents have nonzero divergence.

### 5. Explain an accidental success

Change the target to weights (1,1,4) while keeping proposal weights (2,1,1). Does uncorrected target preservation prove the proposal-ratio factor unnecessary?

**Explanation.** No. The three pair minima all equal one, making the uncorrected clockwise target currents equal to 1/24. Their divergences vanish, so this particular target is preserved. The currents are nonzero, detailed balance fails, and the distinct-weight target (1,2,4) already provides a counterexample to the general omission.

### 6. Diagnose a state-change-only record

For the corrected worked kernel, what stationary distribution appears if every holding step is deleted?

**Explanation.** Leaving probabilities are (3/4,3/4,5/16). Weight π by these probabilities and normalize by the mean movement 1/2. The result is ν=(3/14,3/7,5/14), at total variation 3/14 from π. This record counts changes rather than ordinary steps; its unweighted frequencies do not follow the original target law.

## Continue to periodicity, support and broader theory

We have derived a correct finite transition from pair-flow budgets, tracked its rejection mass, and compared intended and actual stationary laws. We also found two traps: an uncorrected rule can pass a special target check, and deleting holding can change the sampling law even when the full kernel is correct.

The next unit studies a reversible two-state chain to separate stationary balance, periodicity and convergence rates. Further work includes zero-target regions, general support graphs, continuous densities, adaptive proposals, dependence-aware error estimates and high-dimensional algorithms. The three-state construction supplies a complete introductory calculation; it is not a completion claim for those generalizations or research questions.
