## Name the objects before transforming them

A dataset filter selects a set of records. An encoder maps records into representations. A decoder may map representations back to objects. These are different mathematical operations, and their domains determine what guarantees are meaningful.

A set records membership, not order or multiplicity. The set {1,1,2} equals {1,2}, whereas the corresponding sequence and multiset preserve additional information. Do not silently replace an ordered batch or repeated observations by a set when counts matter.

## Set operations as predicates

For a fixed universe D, membership in a union is a logical “or,” while membership in an intersection is “and”:

$$
x\in A\cup B\iff(x\in A)\lor(x\in B),\qquad
x\in A\cap B\iff(x\in A)\land(x\in B).
$$

Set difference $A\setminus B$ requires membership in A and nonmembership in B. A complement is always relative to a stated universe. De Morgan’s laws follow by negating the membership predicates.

The lab uses D={0,…,m−1}, A as the even elements of D, and B={x∈D:x≥t}. At m=3,t=1, A={0,2}, B={1,2}, intersection={2}, union={0,1,2}, and A minus B={0}. The table checks each membership directly.

## A function includes its domain and codomain

A function $f:D\to C$ assigns exactly one output in C to each input in D. Distinct inputs may share an output. Some codomain elements may never be used. Neither situation invalidates the function definition.

| Object | Meaning |
| --- | --- |
| D | Domain: allowed inputs |
| C | Codomain: declared output space |
| f(D) | Image: outputs actually attained |
| f⁻¹(E) | Preimage of a codomain subset E |

In the lab C={0,…,n−1} and $f(x)=(ax+b)\bmod n$. Here b is an offset, unrelated to the set B. All parameters are integers; the domain consists of nonnegative indices, so the implementation’s remainder is the stated nonnegative residue.

## Injection, surjection and bijection

Injectivity means

$$
\forall x_1,x_2\in D:\quad f(x_1)=f(x_2)\Rightarrow x_1=x_2.
$$

Surjectivity means $\forall y\in C\,\exists x\in D:f(x)=y$. A bijection satisfies both. These definitions ask different quantified questions: collisions among inputs versus coverage of the declared codomain.

For m=n=3,a=1,b=0, the mapping is the identity and is bijective. For m=4,n=2, outputs are 0,1,0,1: it is surjective but not injective. For m=2,n=4, outputs 0 and 1 are distinct, so it is injective but not surjective.

When finite domain and codomain have the same cardinality, injection and surjection are equivalent. This finite counting fact must not be applied unchanged to infinite sets. On the natural numbers, x↦x+1 is injective but misses 0 under the convention ℕ={0,1,…}.

## Preimage is not an inverse function

For E⊆C, define

$$
f^{-1}(E)=\{x\in D:f(x)\in E\}.
$$

This notation does not require an inverse function. In the many-to-one preset, the preimage of the even-output set {0} is {0,2}. An inverse function C→D would have to choose a unique original input and cannot invert that collision.

A bijection has a genuine two-sided inverse. An injective map can be inverted on its image, which may be smaller than its codomain. A surjective noninjective map can have a chosen right inverse in finite settings, but such a selection is not recovery of every original input.

## Prove a useful preimage identity

For any E,F⊆C and x∈D,

$$
x\in f^{-1}(E\cap F)
\iff f(x)\in E\land f(x)\in F
\iff x\in f^{-1}(E)\cap f^{-1}(F).
$$

Thus preimages preserve intersections. The same membership method proves preservation of unions and complements relative to D and C. This is a direct proof for arbitrary elements, rather than a collection of sampled checks.

Images need more care: in general $f(A\cap B)\subseteq f(A)\cap f(B)$, but equality can fail under collisions. If A={0}, B={2} in the many-to-one preset, the left side is empty and the right side is {0}. Injectivity supplies the missing condition for equality.

## Read the correspondence diagram

Left nodes are domain elements and right nodes are codomain elements. One arrow leaves each input. Multiple arrows arriving together show a collision; outlined unused outputs demonstrate lack of surjectivity. Node positions encode no metric distance or probability.

The membership table and image/preimage readouts use the same mapping as the diagram. Changing a declared codomain can change surjectivity even if currently attained output values remain the same. State the whole function type when making a claim.

## AI interpretation and limits

A compression encoder with collisions loses information about which input produced a representation. A deterministic decoder cannot restore both distinct originals from the same code without extra information or a restricted input model. Low reconstruction loss on sampled data does not establish global invertibility.

Filtering records corresponds to a preimage of a condition on features. This helps express dataset selection, but says nothing by itself about sampling bias or causal validity. A function’s mathematical type also does not guarantee numerical stability, continuity or differentiability; those are additional properties.

## Exercises and reference

1. For m=4,t=2, list A, B and A∩B. **Answer:** {0,2}, {2,3}, {2}.
2. In the many-to-one preset, find the preimage of {1}. **Answer:** {1,3}.
3. Prove preservation of preimage unions by following one arbitrary input.
4. Give an encoding collision and explain precisely what a decoder cannot infer from it.

For sets, functions and relations, see the [MIT Mathematics for Computer Science readings](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/). The finite correspondence examples are independently specified in this textbook.
