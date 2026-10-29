## Objects together with operations

A set alone does not specify an algebraic structure. We must also choose operations. A group has an associative closed binary operation, an identity and an inverse for every element. An abelian group also satisfies commutativity. The lab uses residue classes represented by integers from zero through n−1. Different representatives denote the same element when their difference is divisible by n.

$$
[x]=[y]\iff n\mid(x-y),\qquad [x]+[y]=[x+y].
$$

## A ring has two operations

In a ring, addition forms an abelian group, multiplication is associative and multiplication distributes over addition. We use rings with a multiplicative identity. An additive inverse cancels by addition; a multiplicative inverse cancels by multiplication. Modulo five, both inverses of two happen to be three, but this coincidence is not their definition.

$$
a+(-a)=0,\qquad au=ua=1\quad\text{for a multiplicative inverse }u.
$$

## Units, zero divisors and fields

A unit has a multiplicative inverse. Nonzero elements whose product is zero exhibit zero divisors. Modulo six, two times three is zero. If two had an inverse, multiplying this equation by it would force three to be zero, a contradiction. Modulo five, every nonzero residue is a unit. A commutative ring with distinct zero and one is a field when every nonzero element is a unit.

$$
a\in(\mathbb Z/n\mathbb Z)^\times\iff\gcd(a,n)=1.
$$

## Read the table and connect to AI

For n=5 and a=2, the multiplication column is 0,2,4,1,3. Changing n to six gives repeated outputs 0,2,4: distinct inputs collapse. Compare a permutation of residues with a map that loses information. Finite fields support coding and exact algebraic calculations, but ordinary real-valued division cannot be transferred blindly. Inputs larger than the modulus are normalized before calculation.

## Exercises with explanations

1. What is the inverse of three modulo seven? **Five, because fifteen reduces to one.**
2. Can zero be inverted? **No: every product with zero is zero.**
3. Does being an additive group imply being a field? **No; residues modulo six are a counterexample.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).
