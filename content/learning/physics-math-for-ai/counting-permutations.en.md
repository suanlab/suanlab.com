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

## From an ordered batch to a count vector

Suppose the vocabulary is $V=\{A,B,C\}$ and a record contains four observations. The sequence AABC changes when its positions are permuted. The multiset records the count vector $(2,1,1)$ and retains the repeated A. The set $\{A,B,C\}$ retains membership. Decide which object carries the target before choosing a pooling operation.

There are $3^4=81$ sequences, $\binom{6}{4}=15$ multisets of size four, and $2^3=8$ subsets across all sizes. A multiset with counts $(c_A,c_B,c_C)$ has

$$
\frac{4!}{c_A!c_B!c_C!}
$$

distinct sequence orderings. For $(2,1,1)$ this is 12. For $(4,0,0)$ it is one. This explains why uniformly sampling sequences produces a different distribution on count vectors from uniformly sampling multisets. Keep this distinction when designing a dataset or reporting average error.

## Prove what a finite vocabulary sum preserves

Encode the symbols as three-dimensional one-hot vectors $\phi(A)=(1,0,0)$, $\phi(B)=(0,1,0)$ and $\phi(C)=(0,0,1)$. For a finite multiset $X$, define

$$
E(X)=\sum_{x\in X}\phi(x)=(c_A,c_B,c_C).
$$

Reordering terms preserves their sum. Equal encodings have equal counts of every symbol, so they identify the same multiset. Thus this particular encoding is both permutation invariant and injective on these multisets. For any target $f$ defined on them, define $\rho(c_A,c_B,c_C)$ as the target of the multiset with those counts. Then $f=\rho\circ E$. This is a direct finite-vocabulary construction, with exact integer arithmetic.

The zero vector encodes the empty multiset and its size is zero. Summing the three coordinates recovers size for every multiset. A learned scalar embedding can still collide. For $\phi(A)=0$, $\phi(B)=1$, $\phi(C)=2$, AC and BB both sum to two. Check the chosen embedding and target together.

## See the information lost by averaging

For a nonempty multiset of size $m$, mean pooling gives $E(X)/m$. It records proportions. Multiplying every count by a positive integer leaves those proportions unchanged.

| Input | Sum encoding | Mean encoding | Size |
|---|---|---|---|
| A | (1,0,0) | (1,0,0) | 1 |
| AA | (2,0,0) | (1,0,0) | 2 |
| AAA | (3,0,0) | (1,0,0) | 3 |
| AAAA | (4,0,0) | (1,0,0) | 4 |

A deterministic decoder of the mean must assign the same answer to these four inputs. It therefore cannot recover their different sizes. Under equal weights, its minimum squared-error prediction is 2.5, the mean of the four labels. The four squared errors sum to five. This is a representation obstruction that applies to every deterministic decoder of this mean. Supplying size alongside the mean recovers each count by multiplication. At the empty input, averaging needs an explicit convention or mask.

To prove the optimal decoder claim, let $y_1,\ldots,y_r$ be the labels sharing one encoding and let $\bar y$ be their mean. Expanding the squares gives

$$
\sum_{i=1}^r(y_i-a)^2
=\sum_{i=1}^r(y_i-\bar y)^2+r(a-\bar y)^2.
$$

The cross term vanishes because $\sum_i(y_i-\bar y)=0$. The second term is nonnegative and vanishes at $a=\bar y$. Apply this argument independently to each encoding group. Unequal sampling weights require the corresponding weighted mean.

## Run an exact representation audit

Enumerate all nonempty multisets over three symbols with sizes one through four. There are $3+6+10+15=34$. Give each multiset equal weight and use size as its target. Grouping equal mean encodings lets us compute the best possible decoder by averaging labels within each group. Rational arithmetic makes the equality comparison exact.

```python
from collections import defaultdict
from fractions import Fraction
from itertools import combinations_with_replacement, product
from math import factorial

def histogram(word):
    return tuple(word.count(symbol) for symbol in range(3))

groups = defaultdict(list)
multiset_count = sequence_count = 0
for size in range(1, 5):
    multisets = list(combinations_with_replacement(range(3), size))
    for word in multisets:
        counts = histogram(word)
        mean = tuple(Fraction(c, size) for c in counts)
        groups[mean].append(size)
        assert sum(counts) == size
    multiplicities = 0
    for word in multisets:
        denominator = 1
        for count in histogram(word):
            denominator *= factorial(count)
        multiplicities += factorial(size) // denominator
    assert multiplicities == 3**size
    multiset_count += len(multisets)
    for word in product(range(3), repeat=size):
        assert histogram(word) == histogram(word[::-1])
        sequence_count += 1

squared_error = Fraction(0)
for sizes in groups.values():
    prediction = Fraction(sum(sizes), len(sizes))
    squared_error += sum((size - prediction)**2 for size in sizes)
mse = squared_error / multiset_count
assert (multiset_count, sequence_count) == (34, 120)
assert mse == Fraction(21, 34)
print(multiset_count, sequence_count, mse)
```

The output is `34 120 21/34`. The three pure-symbol groups contribute 15 to total squared error. The three equal-proportion, two-symbol groups contribute six. All other groups have a single size. Sum-one-hot and mean-plus-size admit zero error on this task by direct decoding. These are exact representational results under the stated weights. Optimizing a neural network introduces training and generalization questions beyond this calculation.

## Read a research result and formulate a follow-up

Deep Sets studies permutation-invariant models with a sum between elementwise and output transformations. Its Theorem 2 distinguishes the countable-universe setting, and Appendix A discusses the continuous case with fixed size. Read those assumptions before extending a result to another input space. The finite-vocabulary proof and the 34-multiset audit here are independently constructed teaching examples. [Zaheer et al., Deep Sets, arXiv v3, 14 April 2018, §2.2, §3.1 and Appendix A](https://arxiv.org/pdf/1703.06114v3)

A follow-up can ask whether an embedding distinguishes the multisets required by a target under a smaller feature dimension. Record vocabulary, maximum size, arithmetic precision and equivalence rule. Use the exact histogram as a reference and mean pooling as a counterexample condition for the size task. Report encoding collisions separately from optimization error. Then hold out larger sizes to measure extrapolation under an explicit change in the evaluation domain.

## Exercises and answers

1. Choose two distinct items from four with order: **12**.
2. Remove order from that problem: **6**.
3. Choose three items from two types with repetition and no order: **4**.
4. Count all subsets of five features: **2⁵=32**. Explain the binary membership decisions.
5. Why does mean-one-hot fail on the size target for A and AA? **Both encodings are (1,0,0), so any decoder gives the same output although their labels are one and two.**
6. How many length-four sequences map to the multiset AABC? **12**, because exchanging the two A positions preserves the sequence, giving $4!/2!$ distinct orders.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). The small examples and bijections above are independently constructed.
