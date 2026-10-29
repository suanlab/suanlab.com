## Why a union needs a subtraction

Merging datasets A and B does not generally add their sizes. An observation present in both would be counted twice. In the lab U={0,1,2,3,4,5}, A={0,1,2}, B={2,3,4}. Their sizes sum to six, but there are five distinct elements:

$$
|A\cup B|=|A|+|B|-|A\cap B|=3+3-1=5.
$$

Prove this using indicators. An element in both contributes 1+1−1=1; an element in one contributes one; an element in neither contributes zero. Sum the pointwise equality over the universe.

## Alternating signs for more sets

For three sets, subtract the three pairwise intersections. An element in all three then contributes 3−3=0, so add the triple intersection:

$$
|A\cup B\cup C|=|A|+|B|+|C|-|A\cap B|-|A\cap C|-|B\cap C|+|A\cap B\cap C|.
$$

An element in r sets contributes Σⱼ₌₁ʳ(−1)ʲ⁺¹C(r,j)=1, by expanding (1−1)ʳ. This proves the finite inclusion–exclusion principle element by element. The interactive model handles two sets; the three-set identity is a separate argument here.

## Pascal’s identity by classification

Classify size-k subsets of n elements by membership of one distinguished element. Those excluding it number C(n−1,k); those including it number C(n−1,k−1). The classes are disjoint and exhaustive:

$$
\binom nk=\binom{n-1}k+\binom{n-1}{k-1}.
$$

Boundary values C(n,0)=C(n,n)=1, with out-of-range sizes counted as zero, generate Pascal’s triangle. This proof uses a partition of outcomes, independently of the factorial formula.

## Binomial theorem and all subsets

In the n factors of (a+b)ⁿ, choose k positions contributing b. There are C(n,k) choices, giving

$$
(a+b)^n=\sum_{k=0}^n\binom nk a^{n-k}b^k.
$$

Set a=b=1 to obtain ΣₖC(n,k)=2ⁿ. For n=6 the coefficients are 1,6,15,20,15,6,1, summing to 64. Twenty subsets have size three. These counts become probabilities after division by 64 only if all subsets are equally likely.

## Interpret the controls and plot

Bit i includes element i. Thus mask seven is {0,1,2}, and mask 28 is {2,3,4}. Check each indicator in the membership table. Identical sets union to themselves; union with the empty set changes nothing.

The plot shows C(6,k) against subset size k. It does not plot the positions of elements in A or B. The table illustrates overlap correction, while the plot classifies subsets by size.

## AI data and probability

Duplicate detection depends on the identity rule: matching IDs and near-identical text define different relations. Specify the rule before counting. Set overlap alone does not detect every form of data leakage.

The probability identity P(A∪B)=P(A)+P(B)−P(A∩B) requires no independence. Replacing the intersection with P(A)P(B) does require independence; do not insert it merely because the formula contains two events.

## Exercises and answers

1. If A=B and |A|=3, the union size is **three**.
2. Use Pascal’s identity: **C(6,2)=C(5,2)+C(5,1)=15**.
3. An element in three sets contributes **3−3+1=1**.
4. With unequal subset sampling weights, is the chance of size three necessarily 20/64? **No**; outcome counts alone are insufficient.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). The examples and indicator derivations are independently constructed.
