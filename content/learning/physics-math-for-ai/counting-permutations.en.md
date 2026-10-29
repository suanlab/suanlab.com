## Define an outcome before counting it

A count depends on what makes two outcomes different. Selecting the features A, B and C is different from choosing the processing order A→B→C. The sets {A,B,C} and {C,B,A} agree; the corresponding ordered executions do not.

For n distinct types and k selections, first ask whether order matters and whether repetition is allowed. These questions define four different models. Counting outcomes alone does not establish a probability: equal likelihood requires a separate assumption.

## Product rule and permutations

Assigning three distinct roles to five candidates gives five choices, then four, then three. The total is 60. More generally,

$$
P(n,k)=n(n-1)\cdots(n-k+1)=\frac{n!}{(n-k)!},\quad 0\le k\le n.
$$

Choices may depend on earlier selections while the number available at each stage remains fixed. If restrictions instead leave different numbers of choices on different branches, split into cases or use a recurrence; a single product may no longer apply.

## Forgetting order gives combinations

A three-person committee has no role ordering. Each committee was counted 3!=6 times among the ordered selections. Since every committee has exactly six orderings, the unordered count is 60/6=10:

$$
\binom nk=\frac{n!}{k!(n-k)!}.
$$

Division is justified by equal-size equivalence classes. It is not a universal repair for overcounting: symmetries can make class sizes differ. Group actions and orbits later provide tools for those cases.

## Two kinds of repetition

An ordered length-k word over n symbols has nᵏ possibilities. Five symbols and three positions give 125 words, distinct from the 60 no-replacement permutations.

For unordered selections with repetition, record counts x₁,…,xₙ. These are nonnegative integers summing to k. When n≥1, a bijection with k stars and n−1 separators gives

$$
\#\{x_i\ge0:\sum_i x_i=k\}=\binom{n+k-1}{k}.
$$

For two types and three selections, the count vectors are (0,3),(1,2),(2,1),(3,0): four outcomes. Upper limits on individual counts invalidate this unrestricted stars-and-bars formula.

## Empty objects and impossible choices

There is exactly one empty selection. All four models therefore give one when k=0, including n=0. We also use 0!=1. For n=0 and positive k, there are no outcomes. The lab handles these cases directly; its combinatorial convention does not resolve every analytic question involving 0⁰.

Without replacement, k>n gives zero. Avoid applying a factorial quotient with negative factorial arguments to an impossible selection.

## Read the experiment and connect to AI

Mode zero counts ordered distinct selections; one counts subsets; two counts words with repetition; three counts multisets. The plot contains integer indices k=0,…,8 and exact counts. Its dots do not define a continuous density or interpolate fractional selection sizes.

For n=5,k=3 the four counts are 60,10,125,35. Compare them before changing n. Feature-subset search resembles combinations; token-sequence search resembles words with repetition. A candidate count is not a measured runtime or the number actually visited by a pruned search. State constraints, symmetry handling and any nonuniform weights.

## Exercises and answers

1. Choose two distinct items from four with order: **12**.
2. Remove order from that problem: **6**.
3. Choose three items from two types with repetition and no order: **4**.
4. Count all subsets of five features: **2⁵=32**. Explain the binary membership decisions.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). The small examples and bijections above are independently constructed.
