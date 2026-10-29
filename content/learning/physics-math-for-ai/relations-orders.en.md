## Grouping and ordering ask different questions

A relation can describe “has the same class,” “is a prerequisite of,” or “has at most this cost.” These meanings have different mathematical requirements. Treating every relation as an equivalence or every ranking as a total order hides failures that matter in data processing and research claims.

A binary relation on D is a subset of $D\times D$. We write xRy when the ordered pair belongs to it. A function is a special kind of relation with exactly one related output for each input, but a general relation need not have that uniqueness.

## Four properties with explicit quantifiers

| Property | Requirement for all indicated elements in D |
| --- | --- |
| Reflexive | xRx |
| Symmetric | xRy implies yRx |
| Antisymmetric | xRy and yRx imply x=y |
| Transitive | xRy and yRz imply xRz |

Antisymmetric does not mean “not symmetric.” Equality is both symmetric and antisymmetric. Asymmetry, by contrast, says xRy implies not yRx, and therefore prohibits xRx. Distinguishing these definitions prevents an apparently minor word change from changing a theorem.

## Equivalence and classes

An equivalence relation is reflexive, symmetric and transitive. Congruence modulo k gives an example: xRy when k divides x−y. On D={1,2,3,4} with k=2, the equivalence classes are {1,3} and {2,4}.

Reflexivity follows because k divides zero. Symmetry follows by negating a divisible difference. For transitivity, write x−y=ka and y−z=kb; then x−z=k(a+b). These arguments hold for all integers, independently of the lab’s finite table.

Equivalence classes partition the domain. If two classes overlap at z, transitivity and symmetry show every member of either class belongs to the other. Thus they are equal; otherwise they are disjoint. A partition records membership in a class, not an ordered ranking between classes.

## A partial order allows incomparability

A non-strict partial order is reflexive, antisymmetric and transitive. Divisibility on positive integers is an example. If x divides y and y divides x, positivity forces x=y. If x divides y and y divides z, multiplying the integer factors shows x divides z.

On {1,2,3,4}, 2 and 3 are incomparable: neither divides the other. The relation is still a partial order. A total order additionally requires that every pair be comparable. Numerical ≤ is total on this finite domain; divisibility is not.

Positivity is important. On all nonzero signed integers, 2 divides −2 and −2 divides 2 although they are not equal, so that relation fails antisymmetry. Changing the domain changes the property.

## Strict orders use another convention

The relation < is not reflexive, so it is not a partial order under the non-strict definition just used. It is a strict order: irreflexive and transitive. A non-strict partial order gives a strict one by requiring distinctness; a strict partial order gives a non-strict one by adjoining equality.

Statements from different conventions can agree after translation but should not be mixed mid-proof. The lab’s property checker always uses the explicit definitions in the table above.

## Find an actual failure witness

To refute reflexivity, exhibit x without xRx. To refute symmetry, exhibit xRy with not yRx. To refute antisymmetry, exhibit distinct x,y with both directions. To refute transitivity, a triple is required:

$$
xRy\land yRz\land\neg xRz.
$$

The lab associates witness tuples with the particular failed property. For congruence modulo two, antisymmetry fails at 1 and 3. For strict <, reflexivity fails at every element and symmetry fails at any increasing pair. A two-element failure is not automatically a transitivity counterexample.

The finite checker exhausts all pairs and triples of the stated domain. That establishes properties for that finite relation. It does not replace the integer arguments above when the claim concerns every positive integer.

## AI examples and misleading similarity

“Close in embedding distance by at most ε” is usually reflexive and symmetric, but not transitive: points at 0, 0.75ε and 1.5ε show two short links whose endpoints are too far apart. Thus a similarity threshold is not automatically an equivalence relation. Taking a transitive closure changes the relation and can merge long chains.

Pareto dominance produces partial comparisons across several objectives; forcing a single ranking introduces extra preferences. A prerequisite graph can represent a strict dependency relation only under appropriate acyclicity and transitive interpretation. Labels such as “equivalent models” and “better method” should state the relation being asserted.

## Read the experiment

Rows and columns contain domain values 1 through N. T/F records the exact relation, not a probability. The three presets compare congruence, divisibility and strict order. A separate code enables ≤. The modulus control affects only congruence, which is why changing it under another relation leaves that relation unchanged.

## Exercises and reference

1. Is equality symmetric and antisymmetric? **Answer:** both.
2. For divisibility on {1,2,3,4}, give a minimal element and maximal elements. **Answer:** 1 is least; 3 and 4 are maximal.
3. Construct a symmetric nontransitive relation using a distance threshold.
4. Explain why defining a ranking rule is part of a multiobjective experiment rather than a consequence of the data alone.

See the [MIT Mathematics for Computer Science readings](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/) for relations and partial orders. The lab uses explicitly bounded domains and displays its own witnesses.
