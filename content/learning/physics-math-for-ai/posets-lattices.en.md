## Comparability is optional

A partial order is reflexive, antisymmetric and transitive. A total order additionally compares every pair. Subset inclusion is generally partial: {0,1} and {0,2} are incomparable.

The lab’s B₃ consists of all eight subsets of U={0,1,2}. Incomparability is neither equality nor inconsistency; the order simply does not put one object above the other.

## Read a Hasse diagram

Higher sets contain lower ones. Lines show only cover relations, here adding exactly one element. Reflexive edges and transitive shortcuts are omitted.

A missing direct line does not imply lack of order: an upward sequence of lines can still establish inclusion. This differs from a full relation graph. Horizontal position and edge length have no numerical meaning. Height here follows subset cardinality.

## Greatest lower bounds: meet

A lower bound X of A and B satisfies X⊆A and X⊆B. The greatest such lower bound is their meet, A∧B. “Greatest” refers to the partial order, not a separate numerical size.

The intersection is a lower bound, and every common lower bound lies inside it. Therefore

$$
A\wedge B=A\cap B.
$$

For {0,1} and {0,2}, the meet is {0}. Incomparable elements can have a meet. Distinguish a greatest lower bound from merely maximal lower bounds: several incomparable maximal lower bounds need not yield a meet.

## Least upper bounds: join

An upper bound Y contains both A and B. Their union is an upper bound contained in every other upper bound, hence

$$
A\vee B=A\cup B.
$$

The reference join is {0,1,2}. A poset in which every pair has a meet and join is a lattice. Not every poset is a lattice: two incomparable elements with no common upper bound have no join.

## Bounds and additional laws

The bottom element is ∅ and top is U. Absorption gives A∩(A∪B)=A and A∪(A∩B)=A. Membership reasoning also proves

$$
A\cap(B\cup C)=(A\cap B)\cup(A\cap C).
$$

This lattice is distributive and has complements U\A, making it a set model of Boolean algebra. Arbitrary lattices need not be distributive or complemented. For vector subspaces, for example, join is the linear span of the union, not usually the set union itself.

## Decode the experiment

Mask three means {0,1}; five means {0,2}. Bitwise AND gives their intersection, 3&5=1; bitwise OR gives their union, 3|5=7. Inclusion is equivalent to (A&B)=A. The table displays masks and corresponding subsets.

With masks one and three, A⊆B, so meet=A and join=B. Masks zero and seven select bottom and top. Boolean truth operations and set operations act on different objects but connect through element-membership indicators.

## AI connections and exercises

Dataflow analysis, information precision and constraint inclusion use partial orders. A larger set may mean more information, or more possible states and therefore less precision, depending on the interpretation. Define the order direction. Fixed-point convergence claims require the relevant assumptions, such as monotonicity, completeness or finite height.

1. If A⊆B, meet and join are **A and B**.
2. Does incomparability exclude a meet? **No**, as the reference shows.
3. Does a linear extension preserve exactly the same comparisons? **No**; it adds comparisons between previously incomparable elements.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). This chapter gives direct finite-subset arguments.
