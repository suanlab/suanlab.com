## Overlapping information has diminishing returns

A sensor can be valuable when little is known and redundant after other sensors have already covered its targets. The value of a collection is therefore not usually the sum of separate sensor values. Similar effects arise when selecting representative examples or monitoring locations. The relevant property must be established for the chosen objective; calling information “diverse” does not by itself prove submodularity.

This unit uses four sensors and four targets. Each sensor's target mask sums bits 1,2,4,8. A different mask selects sensors: bit i means sensor i. Do not confuse a sensor-set mask with the target-union mask it produces. Query sets A and B are definition checks, independent of the optimization budget k.

## Define the set function and marginal gain

If sensor i covers targets T_i, define

$$
f(S)=\left|\bigcup_{i\in S}T_i\right|,\qquad
\Delta(e\mid S)=f(S\cup\{e\})-f(S).
$$

The empty set has value zero, so the function is normalized. Adding sensors never removes covered targets, so it is monotone. A marginal gain counts targets covered by e that were not already in the union for S. It can be zero even when e is a nonempty sensor.

The displayed values are exact small integers. There is no sampling noise, learned score or hidden uncertainty model in this example. A learned surrogate objective would require a new analysis of its properties and estimation errors.

## Submodularity is a conditional comparison

The diminishing-return definition requires

$$
A\subseteq B,\quad e\notin B
\quad\Longrightarrow\quad
\Delta(e\mid A)\ge\Delta(e\mid B).
$$

For coverage, targets still uncovered under B are a subset of those still uncovered under A. Intersect both sets of uncovered targets with T_e and count: the inequality follows target by target. The larger collection cannot make an already covered target new again.

An equivalent set-lattice definition is

$$
f(A)+f(B)\ge f(A\cup B)+f(A\cap B).
$$

For coverage this also follows per target. A target covered by both A and B contributes two on the left; on the right the union contributes one and the intersection contributes at most one. A target covered by exactly one side contributes one on each side. The equivalence of these definitions holds for finite set functions; normalization and monotonicity are additional properties, not part of submodularity itself.

The controls allow an invalid premise deliberately. Raw marginal gains are still calculated, but their certified comparison is unavailable unless A is a subset of B and e is outside B. A failed premise is not a discovered counterexample to the theorem.

## Modular values and complementarity

An additive or modular function sums fixed item values and satisfies the lattice inequality with equality. Overlapping coverage is generally not additive. Conversely, complementarity can make an item more valuable after another is selected. For example, let h(S)=1 when both sensors zero and one are selected and zero otherwise. This function is normalized and monotone, but the gain from adding sensor one is zero at the empty set and one after sensor zero. It is not submodular.

This complementarity function is a conceptual counterexample, not an implemented alternate objective in the interactive model. It explains why monotonicity alone cannot justify the greedy guarantee discussed below.

## Greedy selects by current marginal gain

Under a cardinality budget |S|<=k, greedy starts empty and repeatedly adds an unselected sensor with the largest current marginal gain. Exact ties choose the smaller sensor index. The code computes these choices without consulting the exact optimum. It fills the requested budget even when remaining gains are zero; stopping at zero would give the same value in this monotone model.

A separate enumeration evaluates all sixteen sensor subsets of size at most k. It reports the optimal value and all optimal feasible sets. The representative optimum uses the smallest cardinality, then smallest mask; a displayed representative does not prove uniqueness. This explicit optimum is available because the teaching instance is tiny, not because greedy solves general submodular optimization exactly.

## A worked example where greedy is not exact

Use target masks (3,5,10,1), meaning T0={0,1}, T1={0,2}, T2={1,3}, T3={0}. At budget two, sensors zero, one and two initially tie with gain two. Greedy selects sensor zero by index. Sensors one and two then each add one target, so greedy selects sensor one. Its sensor mask is three, target union mask seven, and value three.

Sensors one and two instead cover all four targets, giving optimal sensor mask six and value four. The observed ratio is 3/4 and the additive loss is one. At query A empty, B containing sensor zero, and e=1, the marginal gains are two and one. At A={0}, B={0,1}, e=3, both gains are zero because target zero is already covered.

Set A={1}, B={0}, e=1. Raw gains are zero and one, but A is not a subset of B. The apparent increase is not a submodularity violation; the certified comparison stays unavailable.

## Derive the guarantee and keep its assumptions

For a normalized monotone submodular function and a cardinality budget k>=1, let O be an optimal set and G_i the greedy set after i steps. Monotonicity and diminishing returns imply

$$
f(O)-f(G_i)\le\sum_{e\in O\setminus G_i}\Delta(e\mid G_i)
\le k\max_{e\notin G_i}\Delta(e\mid G_i).
$$

Greedy takes that largest gain. Thus the next remaining gap is at most (1-1/k) times the previous gap. Starting from f(empty)=0 and taking k steps gives

$$
f(G_k)\ge\left[1-\left(1-\frac1k\right)^k\right]f(O)
\ge(1-1/\mathrm e)f(O).
$$

Here mathematical e is the exponential base, while the query sensor e is an element index. The finite-k factor at k=2 is 3/4, attained by the worked example. The guarantee is a lower bound on performance, not a promise of equality or exactness. Different constraints, nonmonotone objectives or complementarity require different analysis.

At k=0 both feasible optimum and greedy value are zero; the factor formula is not evaluated. If optimum is zero, the observed ratio is 0/0 and remains unavailable. A positive-k guaranteed value can still equal zero. The display distinguishes these cases instead of assigning an artificial perfect ratio.

## Read the two diagrams and budget curves

The matrix shows which sensor covers each target. Left A/B markers indicate sensor membership, and a blue row outline identifies queried e. Below it, filled target disks show existing coverage; dashed rings show newly covered targets. The second diagram shows greedy's sensor order, current gain and accumulated coverage, with the current budget outlined.

The lower curves vary sensor budget, not training time. Blue is greedy value and orange is the independent optimum. The instance ratio, generic guarantee factor and guaranteed value have separate readouts. Their equality in one example is not evidence that every instance attains the worst-case bound.

## Research extension: quantify an inaccurate marginal oracle

A learned selection score may approximate marginal gain. Assume every queried estimate has absolute error at most $\eta$, simultaneously for all adaptive queries. If greedy chooses the largest estimated gain, its true gain is at least the best true gain minus $2\eta$: one error can lower the best score and another can raise the selected score.

The usual gap recurrence becomes $D_{i+1}\le(1-1/k)D_i+2\eta$, where $D_i=f(O)-f(G_i)$. Unrolling it yields

$$f(G_k)\ge\alpha_k f(O)-2k\alpha_k\eta\ge\alpha_k f(O)-2k\eta,\qquad\alpha_k=1-(1-1/k)^k.$$

The first bound is sharper; the second is easier to report. With the native four-target example, $k=2$ and $f(O)=4$, the latter gives the following conservative guarantees. Negative bounds can be replaced by zero only because this coverage objective is nonnegative.

| Uniform error $\eta$ | Conservative guaranteed value |
|---|---:|
| 0 | 3 |
| 1/4 | 2 |
| 1/2 | 1 |

```python
from fractions import Fraction as F
from itertools import permutations
masks=[3,5,10,1]
def value(S):
    union=0
    for i in S:union|=masks[i]
    return union.bit_count()
for eta in [F(0),F(1,4),F(1,2)]:
    lower=F(3)-4*eta; tested=0
    for order in permutations(range(4),2):
        S=[]; valid=True
        for chosen in order:
            best=max(value(S+[i])-value(S) for i in range(4) if i not in S)
            if value(S+[chosen])-value(S)<best-2*eta:valid=False
            S.append(chosen)
        if valid:
            assert value(S)>=lower; tested+=1
    assert tested>0
    print(eta,lower,tested)
```

The exact enumeration checks every two-step choice sequence compatible with the error condition. This is an auxiliary imperfect-oracle analysis; native gains remain exact. Mean squared prediction error on held-out data does not establish the required uniform bound over adaptive queries. A probabilistic oracle must supply a joint event covering those queries before this deterministic argument applies.

## Exercises and solutions

Exercise: all sensor masks are zero and k=2. What are the optimum, ratio and guarantee? Solution: both values are zero, the ratio is unavailable, the finite-k factor is 3/4 and its guaranteed value is zero. Greedy may still select two zero-gain sensors; the representative exact optimum is empty.

Exercise: use disjoint masks (1,2,4,8). Is greedy exact? Solution: every selected sensor adds one new target, so greedy and optimum equal k. Equal gains do not imply a unique optimal set.

Exercise: does submodularity imply that adding a sensor always helps strictly? Solution: no. Monotonicity permits zero gain, and repeated coverage can be entirely redundant. Submodularity alone does not even require nonnegative gains; the present coverage function additionally is monotone.

## Research connection

[Nemhauser, Wolsey and Fisher's original analysis](https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf) establishes classical greedy bounds under stated assumptions. The four-sensor construction and conditional diagnostics here are explicit textbook examples. Extending this reasoning to an AI data-selection objective requires proving or testing the relevant structural assumptions, not transferring the coverage guarantee by name alone.
