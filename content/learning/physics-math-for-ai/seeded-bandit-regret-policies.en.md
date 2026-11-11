## Compare decisions, not just a lucky reward total

The previous chapter averaged over a Bayesian prior and solved a short planning problem. Here the true Bernoulli means mu0,mu1 are fixed by the experiment. Policies do not receive them; they observe only their chosen rewards. This lets us study exploration rules and distinguish noisy observed reward from the loss caused by choosing a lower-mean arm.

Every policy first pulls arm0 and arm1 once. The later rules use their own observed histories. This common initialization is explicit; the Thompson variant here includes it too. All score ties choose arm0. A shared experiment does not mean identical histories: different actions reveal different rewards and create different information.

## Share potential rewards without revealing the future

For each arm, generate a seeded sequence of Bernoulli potential rewards indexed by that arm's pull count. The nth pull of arm i receives the nth entry in its sequence, regardless of the wall-clock round. Policies share these per-arm sequences, while exploration uses separate pseudorandom streams. A policy's action is selected before the next reward entry is read.

The design preserves shorter-horizon prefixes and avoids changing environmental noise merely because one algorithm consumes more random numbers. It does not make all policy comparisons deterministic mathematical expectations. The stochastic model assumes stationary IID rewards; the finite seeded implementation is a reproducible numerical experiment under that model.

## Constant epsilon-greedy

After initialization, with probability epsilon choose a uniformly random arm; otherwise choose the higher empirical reward mean. The random branch can choose the same arm as exploitation, so an exploration event is not necessarily a suboptimal action. Epsilon=0 is purely empirical greedy after the initial pulls and can become stuck after misleading data.

This lab uses constant epsilon, rather than a decreasing schedule. With two unequal means and gap Delta, the random branch alone contributes expected regret epsilon*Delta/2 per later round. Including the initial suboptimal pull gives

$$
\mathbb E[R_T]\geq \Delta+\frac{\epsilon\Delta}{2}(T-2),\qquad T\geq2.
$$

Thus a fixed positive exploration rate can cause linear asymptotic regret even if exploitation eventually identifies the better arm. A logarithmic guarantee for a carefully decreasing schedule should not be attached to this constant-epsilon implementation.

## UCB1: value an uncertain estimate optimistically

At executed round t, let m=t-1 be the total completed pulls. For each arm use its pre-action count n_i and empirical mean:

$$
U_i(t)=\widehat\mu_i+\sqrt{\frac{2\ln(t-1)}{n_i}}.
$$

Choose the larger index. The logarithm is natural. Every n_i is positive after initialization. An index can exceed one: it is an optimistic score, not a Bernoulli probability. Using t-1 matches the completed-play clock in the original UCB1 sketch; a different clock convention must be stated rather than silently mixed into a hand calculation.

For bounded IID rewards, Hoeffding's inequality at a fixed sample count n controls the chance that an empirical mean misses its true mean by a given amount. Substituting a bonus sqrt(2*ln(m)/n) makes the corresponding one-sided probability at most m^-4. Since the number of samples is chosen adaptively, a full argument must also account for possible counts and times. A gap-based analysis limits typical suboptimal pulls on the order of ln(T)/Delta² and expected regret on the order of ln(T)/Delta, with additional terms. These are expected guarantees under assumptions, not a requirement that UCB win every short seeded experiment.

## Thompson Sampling: sample a plausible environment

For each arm use the uniform Beta(1,1) prior. With S observed successes and F failures, sample a candidate mean from Beta(S+1,F+1), then choose the arm with the larger sample. Sampling an arm as plausibly best differs from choosing the largest posterior mean. It also differs from the finite Bayesian Bellman optimum: posterior sampling does not explicitly optimize this remaining budget.

The sampler here exploits integer parameters. If a+b-1 independent uniform numbers are drawn, their a-th smallest value has Beta(a,b) distribution. Indeed,

$$
\Pr(U_{(a)}\leq x)=\sum_{j=a}^{a+b-1}\binom{a+b-1}{j}x^j(1-x)^{a+b-1-j}.
$$

The event says at least a draws fall below x. This gives the Beta CDF and provides a simple transparent implementation for the small experiment. Tests check deterministic order-statistic examples, moments and CDF frequencies; they do not infer a regret theorem from sampler tests.

## Two quantities that should not be clipped together

Let mu_star=max(mu0,mu1). Cumulative pseudo-regret on the selected action path is

$$
R_T=\sum_{t=1}^T(\mu_{star}-\mu_{A_t}).
$$

It is nonnegative. It is also random because the actions depend on earlier observations. The realized mean-benchmark shortfall is

$$
S_T=T\mu_{star}-\sum_{t=1}^T Y_t.
$$

This can be negative when observed rewards exceed the expected benchmark. It is not a comparison with a separately sampled clairvoyant reward path. Under the stated nonanticipating IID model, E[S_T]=E[R_T], but the two values on one run need not match. Clipping negative shortfall would change the statistic and distort its average.

The native bars compare final pseudo-regret for all three policies. The blue outline identifies the selected one. The lower blue staircase is its cumulative pseudo-regret; orange is UCB1. Values are held between numbered rounds for display, with decisions and jumps at the integer rounds. If UCB1 is selected, both paths overlap. The table exposes pre-action scores, actions, rewards, counts and both performance quantities. One seed does not establish an expected ranking.

## Research extension: separate learning regret from reward noise

Let $S_T=T\mu_* -\sum_tY_t$ be realized shortfall and $R_T=\sum_t(\mu_*-\mu_{A_t})$ pseudo-regret. Their difference is $S_T-R_T=\sum_t(\mu_{A_t}-Y_t)$, a martingale with zero conditional increments when actions are chosen before each independent Bernoulli reward. Conditional increments have range length one; Hoeffding-Azuma gives

$$P\left(|S_T-R_T|\ge\sqrt{(T/2)\log(2/\delta)}\right)\le\delta.$$

This bounds reward noise around random pseudo-regret, not pseudo-regret itself. For all prefixes $t\le T_{max}$ and three policies, replace the threshold by $\sqrt{(t/2)\log(6T_{max}/\delta)}$. The union bound needs no independence between policies sharing reward streams.

| $T$, $\delta=0.05$ | Fixed-time threshold |
|---|---|
| 2 | $\sqrt{\log40}$ |
| 8 | $2\sqrt{\log40}$ |
| 32 | $4\sqrt{\log40}$ |

```python
from fractions import Fraction as F
from itertools import product
from math import sqrt,log
mu=[F(1,4),F(3,4)]; delta=.05
for T in [2,4,8]:
    expected=F(0); tail=F(0); total=F(0)
    threshold=sqrt(T/2*log(2/delta))
    for rewards in product([0,1],repeat=T):
        arm=0; mass=F(1); gap=F(0)
        for y in rewards:
            mass*=mu[arm] if y else 1-mu[arm]
            gap+=mu[arm]-y
            if not y:arm=1-arm
        total+=mass; expected+=mass*gap
        if abs(gap)>=threshold:tail+=mass
    assert total==1 and expected==0 and float(tail)<=delta
    print(T,expected,float(tail))
```

The exact enumeration checks a separate adaptive stay-on-success policy. A finite set of deterministic seeds is not the ideal IID probability space, and one visually superior seeded path does not establish expected policy superiority.

## Worked exercise: deterministic UCB decisions

Set mu0=1,mu1=0,T=4. Initial rewards are 1 from arm0 and 0 from arm1. Before round three, m=2 and both counts are one. UCB scores are 1+sqrt(2*ln(2)) and sqrt(2*ln(2)), so choose arm0. Before round four, counts are 2 and 1; scores are 1+sqrt(ln(3)) and sqrt(2*ln(3)), again favoring arm0.

**Solution.** The actions are 0,1,0,0. Total observed reward is three, pseudo-regret is one, and realized benchmark shortfall is also one. This deterministic case separates the score convention from random outcomes and checks that counts used in a score precede the chosen action.

For a second exercise use mu0=mu1=0.5,T=2,seed=1. The two initial observed rewards are both one. Every action gap is zero, so pseudo-regret is zero. Yet shortfall is 2*0.5-2=-1. The third guided example preserves that negative result rather than changing it to zero.

## Limits and research connections

Changing a seed can reverse a short-run ranking. Changing the means, horizon or stationarity assumption can change the problem itself. The current experiment has no delayed rewards, contextual features, changing means or adversarial feedback. Such settings need different models and guarantees.

Bandit evaluation in AI should state what information a policy receives, how reward randomness is shared, whether exploration is constant or adaptive, and which expectation defines regret. Multiple seeds and uncertainty estimates are useful for empirical claims; this interactive reference focuses on inspectable individual decisions and exact accounting. It complements the prior-averaged finite planner without conflating their objectives.

## Primary references

Auer, Cesa-Bianchi and Fischer, [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf), supplies the original UCB1 sketch and bounded-reward expected-regret setting. Agrawal and Goyal, [Analysis of Thompson Sampling](https://proceedings.mlr.press/v23/agrawal12.html), provides the Bernoulli posterior-sampling framework. The shared-stream experiment, constant-epsilon calculation, examples and original visualizations here are teaching constructions.
