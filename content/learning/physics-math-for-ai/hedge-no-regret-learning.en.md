## The question is relative performance, not perfect prediction

Imagine choosing between two learning pipelines before seeing today's evaluation. Tomorrow's conditions can differ from today's, so selecting the historically best pipeline with complete confidence can be fragile. No-regret learning asks for a weaker, precise target: compare cumulative performance with each single action held fixed over the same observed sequence. It does not demand knowing the best action every day.

Use rewards in[0,1]. A round begins with a mixture p_t over two actions. The environment produces a vector g_t containing both rewards. The learner's fractional reward is the dot product p_t·g_t. After choosing the mixture, the learner receives the entire vector, including the reward of the action it did not favor. This is full-information feedback. In a bandit problem, only the chosen action's feedback would be observed and this update could not simply read the other coordinate.

This lab evaluates mixtures directly. It does not draw a random action or display one realized random trajectory. The distinction becomes essential when interpreting an adversary and a theorem.

## Keep the external comparator fixed

Let G_n(i) be the sum of coordinate i in the first n reward vectors and S_n the sum of fractional rewards. Signed external regret is

$$
R_n=\max_i G_n(i)-S_n,\qquad
G_n(i)=\sum_{t=0}^{n-1}g_t(i),\qquad
S_n=\sum_{t=0}^{n-1}p_t\cdot g_t.
$$

The action i is fixed for the entire prefix. The maximum is selected afterward. This differs from the stronger oracle that changes its action after seeing each current vector. It also differs from rerunning a reactive environment under a different learner policy.

A changing mixture can outperform every fixed action. Signed regret can therefore be negative. A convenient no-regret convention is that its positive part divided by the horizon tends to zero; equivalently, the signed average has nonpositive limsup. A universal upper guarantee does not imply that the absolute signed average tends to zero on every sequence.

## From following a leader to softening a leader

Follow the Leader, or FTL, picks an action maximizing past cumulative rewards. Our tie rule selects action0. FTL's certainty makes its response discontinuous at a tie: a tiny cumulative difference can flip the entire choice.

Hedge instead assigns exponential scores to past performance. Begin with a strictly positive prior pi_i summing to1. For a learning rate eta, define

$$
p_t(i)=\frac{\pi_i\exp(\eta G_t(i))}
{\sum_j\pi_j\exp(\eta G_t(j))}.
$$

Here G_t contains rounds0 through t−1, so the current vector has not entered the decision. After feedback, add g_t to G_t and compute p_(t+1). A large rate emphasizes accumulated reward differences strongly; a small rate changes preferences slowly. At eta0, probabilities stay at the prior forever. FTL ignores both rate and prior controls.

[Arora, Hazan and Kale, §2.1](https://theoryofcomputing.org/articles/v008a006/v008a006.pdf) describe the exponential update within the broader multiplicative-weights family. Their displayed theorem uses its own cost range and constants; the bound below is derived for this lab's normalized rewards and full-support prior.

## A two-round calculation before using the chart

Take a uniform prior, eta1 and fixed feedback(1,0). Before any observation, p_0=(1/2,1/2), so the first reward is1/2. After one observation the score ratio is e:1, giving

$$
p_1(0)=\frac e{1+e}\approx0.731059,
\qquad p_1(1)=\frac1{1+e}.
$$

The second fractional reward is0.731059. After two rounds, S_2≈1.231059 and the best fixed action earns2, so R_2≈0.768941. Using p_1 for the first reward would leak current feedback into the decision.

After eight rounds, S_8≈7.036367 and R_8≈0.963633. The upper bound at that prefix is log2+1≈1.693147. Observed regret and an upper bound are different quantities; the bound need not tightly match this favorable environment.

## A bound with visible assumptions

Assume full feedback, coordinate rewards in[0,1], a positive fixed eta and a full-support prior. Let

$$
C=\log\frac1{\min_i\pi_i}.
$$

Then for every prefix of this fractional-reward run,

$$
R_n\le B_n(\eta)=\frac C\eta+\frac{\eta n}{8}.
$$

A uniform two-action prior gives C=log2. A more concentrated prior can start closer to the useful action but increases this worst-comparator complexity. The displayed C is conservative: a comparator-specific proof can use log(1/pi_i) for each particular i.

The coefficient1/8 uses reward range1. Multiplying all rewards by a physical scale changes the regret units and requires changing the bound consistently. This is a deterministic inequality for the fractional dot products, not a confidence interval from repeated sampling.

## Why the exponential potential gives this inequality

Define Z_n as the sum over i of pi_i exp(eta G_n(i)); Z_0=1. One update satisfies

$$
\log Z_{t+1}-\log Z_t
=\log\sum_i p_t(i)e^{\eta g_t(i)}
\le\eta p_t\cdot g_t+\frac{\eta^2}{8}.
$$

The inequality is Hoeffding's lemma for a variable taking values in[0,1]. It applies to the finite distribution p_t at that round. No independence between successive reward vectors is required for this algebraic step. Summing gives log Z_n≤eta S_n+n eta²/8. For every comparator i, the single term pi_i exp(eta G_n(i)) also gives log Z_n≥log pi_i+eta G_n(i). Combine the two bounds, divide by positive eta and maximize over i to obtain B_n.

[Bartlett's exponential-weights lecture](https://www.stat.berkeley.edu/~bartlett/talks/Lecture1.pdf) presents the Hoeffding-potential proof for bounded losses. Replacing loss by1−reward and retaining the prior term produces the reward convention used here. This paragraph states which potential and comparator are being compared rather than treating a green plot as proof.

For this particular fixed-rate Hedge run, Jensen also gives log Z_n ≥ eta S_n, while Z_n ≤ exp(eta max_i G_n(i)). Thus its mathematical regret is nonnegative. The earlier statement about possible negative signed regret concerns general changing policies, not a claim that this particular algorithm produces negative values. Display arithmetic may leave tiny floating-point residuals.

## A horizon guarantee needs a rate schedule

For one fixed eta,

$$
\frac{R_T}{T}\le\frac{C}{\eta T}+\frac\eta8.
$$

The second term remains as T grows. This inequality alone does not prove no regret for a single indefinitely continued positive-rate run. If a horizon T is known, choose the constant rate for that run as

$$
\eta_T=\sqrt{\frac{8C}{T}},\qquad
B_T(\eta_T)=\sqrt{\frac{CT}{2}}.
$$

Now the positive average upper bound decreases as1/√T. These are different horizon-tuned runs. The interface reports the continuous optimum as a reference; the integer rate control is eta times100, may round that value and has a finite maximum. The displayed certificate always uses the actual entered rate.

For an unknown horizon, a doubling construction restarts a horizon-tuned learner at planned block lengths1,2,4,... . Each block has its own fixed-comparator guarantee; summing the square-root block bounds is still order√T, including an incomplete final block. Alternatively, a time-varying update needs its own analysis. Quietly replacing eta by a new value in an old proof is insufficient. The lab itself implements fixed-rate prefixes, not a hidden doubling algorithm.

## An explicit FTL counterexample

The mixture-reactive environment gives reward0 to the action with highest current probability and1 to the other; equal probabilities penalize action0. It can inspect the public mixture but not a future random draw.

FTL chooses action0 at the initial tie, receives0, then chooses action1 because it earned1 in hindsight. That action is now penalized. Cumulative rewards tie again, and the process repeats. In16 rounds, FTL earns0 while each fixed action earns8, so R_16=8. At even horizons this family has average regret1/2. This is a counterexample to universal no regret for the specified deterministic FTL rule.

Hedge is evaluated against the same environment rule, but its different mixtures can produce a different reward sequence. This is not a controlled comparison on one identical recorded sequence. Each regret compares its learner with fixed coordinates on its own generated record.

## Adaptive timing and the sampled-action distinction

The potential inequality holds for every resulting vector sequence, including one generated by inspecting the current mixture. With fractional rewards the proof is algebraic, and no sample needs to be hidden. In a randomized implementation, however, the identity between a dot product and conditional expected sampled reward requires that the reward vector not depend on the learner's current unseen draw.

A realized sampled regret would contain additional randomness. Expected or high-probability statements, and comparison with an adaptive hindsight best action, require stating the adversary timing and the appropriate expectation or martingale argument. The deterministic fractional certificate should not be relabeled as an automatic sample-path guarantee. It also does not become policy regret: a reactive environment rerun under a different policy may change future feedback.

## Two learners connect regret to minimax averages

The final mode uses original zero-sum matrix A=(1,−1;−1,1). Its value is0. To reuse[0,1] feedback, Player1 receives G=(A+1)/2, the identity matrix; Player2 receives1−G. Player1's vector against column mixture q is(q0,q1), and Player2's vector against row mixture p is(p1,p0). Both mixtures are selected from their own previous cumulative vectors, then both receive current full feedback simultaneously.

Let p-bar and q-bar be arithmetic means of the mixtures played in the displayed prefix. The original A security bounds are

$$
L_A=2\min(\bar p_0,1-\bar p_0)-1,
\qquad U_A=2\max(\bar q_0,1-\bar q_0)-1.
$$

Thus L_A≤0≤U_A. Summing both normalized regrets cancels the actual rewards because the two normalized rewards sum to1 each round. Bilinearity then gives the exact mathematical identity

$$
U_A-L_A=\frac{2(R_n^{(1)}+R_n^{(2)})}{n}.
$$

The interface recomputes the security gap directly from average mixtures and also from the two regrets, exposing their floating-point residual. It does not use one expression as an independent proof of the other. Opposite priors have the same C, so the two Hedge bounds yield gap≤4B_n/n when eta is positive.

This guarantee concerns averaged mixtures. Last iterates can oscillate. It also uses the zero-sum structure: small external regret in a general-sum game does not automatically imply that independent last mixtures converge to a Nash profile. Empirical joint play, coarse correlated equilibrium and stronger swap-regret notions require distinguishing their comparators. They are research extensions, not extra certificates claimed by this mode.

## Read prefixes without inventing evidence

Every table, reward sum, bound and average uses only the displayed prefix. The midpoint environment switches at floor(planned horizon/2), so changing that horizon changes its schedule. Other single-player modes preserve their existing prefixes when the horizon is extended.

A query round clamps to the last displayed round. With an empty prefix, no choice, feedback, average regret or averaged game gap is invented. The first unplayed next mixture is still mathematically defined from the prior. At eta0, Hedge remains that fixed mixture and no positive-eta certificate is displayed. No Hedge certificate is displayed for FTL either. Curves show signed regret and an available upper bound; they do not show one sampled reward trajectory.

## Research extension: regret controls a joint distribution, not its independent marginals

For repeated pure joint actions, let $\mu_T$ assign mass $1/T$ to each played profile. Rewriting player i's external regret gives

$$\max_b\mathbb E_{\mu_T}[u_i(b,a_{-i})-u_i(a)] = R_T^i/T.
$$

Thus upper bounds $R_T^i/T\le\epsilon$ establish an $\epsilon$ coarse correlated equilibrium (CCE). A deviation is fixed before seeing a recommendation. Correlated equilibrium instead permits deviations conditioned on the recommended action; its unnormalized constraints contain $1\{a_i=c\}$ for every recommendation c and alternative b. Swap-regret reasoning addresses that stronger comparator.

For fractional simultaneous play, the corresponding joint law is the average of the per-round product laws, not generally the product of their average marginals. Alternate the coordination equilibria00 and11. Each player's average payoff is3/2 and its best fixed replay payoff is1, so signed regret is negative. The empirical law is even a correlated equilibrium. Its independent fair-marginal product instead gives payoff3/4 and unilateral gain1/4.

| Even horizon | Each signed regret | Independent-average Nash gain |
|---|---:|---:|
| 2 | -1 | 1/4 |
| 4 | -2 | 1/4 |
| 8 | -4 | 1/4 |

![Empirical diagonal joint play and the product of its identical fair marginals assign different cell masses and utilities.](/assets/learning/game-joint-product-en.svg)

```python
from fractions import Fraction as F
from itertools import product
A=[2,0,0,1]; mu=[F(1,2),F(0),F(0),F(1,2)]
for player,recommended,alternative in product(range(2),range(2),range(2)):
    gain=F(0)
    for a,b in product(range(2),repeat=2):
        own=a if player==0 else b
        if own==recommended:
            changed=2*alternative+b if player==0 else 2*a+alternative
            gain+=mu[2*a+b]*(A[changed]-A[2*a+b])
    assert gain<=0
for T in [2,4,8]:
    actual=F(3*T,2); fixed=max(T,F(T,2)); regret=fixed-actual
    product_payoff=F(sum(A),4); best_response=F(1)
    assert regret==F(-T,2) and product_payoff==F(3,4)
    assert best_response-product_payoff==F(1,4)
    print(T,regret,best_response-product_payoff)
```

This constructed path is not claimed to be generated by native Hedge. It explains why the native zero-sum average-mixture theorem cannot be transferred unchanged to a general-sum game.

## Exercises and explained solutions

Exercise1: compute the first two probabilities and rewards for eta1, uniform prior and fixed vector(1,0). What goes wrong if the first probability is e/(1+e)?

Solution: probabilities before rounds0 and1 are1/2 and e/(1+e). Rewards sum to1.231059 and regret is0.768941. The alternative first probability already uses the first feedback and violates the information order.

Exercise2: for uniform prior and T64, calculate eta_T and the optimized average upper bound. Does leaving eta1 unchanged prove the same asymptotic statement?

Solution: eta_T=√(8log2/64)≈0.294353; the optimized average bound is√(log2/128)≈0.073588. Eta1 instead leaves an upper bound log2/T+1/8. It may work well on a sequence, but that expression does not prove vanishing average regret.

Exercise3: why does the game gap use a factor2, and why is it not a last-iterate certificate?

Solution: normalized feedback uses G=(A+1)/2, so differences in A units are twice differences in G units. Cumulative comparators against the opponent's played sequence depend on its average mixture through bilinearity. They do not bound the regret of only the final pair of mixtures.
